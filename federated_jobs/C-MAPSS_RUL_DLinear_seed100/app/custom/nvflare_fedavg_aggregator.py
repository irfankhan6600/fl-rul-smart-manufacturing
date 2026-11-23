from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from fedavg_algorithm import ClientUpdate, FedAvgAggregator

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.fl_constant import ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.aggregator import Aggregator


def debug_print(message: str) -> None:
    print(f"[nvflare_fedavg_aggregator.py] {message}", flush=True)


class NVFlareFedAvgAggregator(Aggregator):
    """NVFlare wrapper around the local FedAvg implementation."""

    def __init__(self, expected_data_kind: str = DataKind.WEIGHT_DIFF, min_weight: float = 0.0):
        super().__init__()
        self.expected_data_kind = expected_data_kind
        self.fedavg = FedAvgAggregator(min_weight=min_weight)
        self.updates: list[ClientUpdate] = []
        debug_print(f"initialized: expected_data_kind={expected_data_kind}, min_weight={min_weight}")

    def accept(self, shareable: Shareable, fl_ctx: FLContext) -> bool:
        debug_print("accept called")
        try:
            dxo = from_shareable(shareable)
        except Exception as ex:
            debug_print(f"accept failed: could not convert shareable to DXO: {ex}")
            self.log_exception(fl_ctx, f"Failed to convert shareable to DXO: {ex}")
            return False

        if dxo.data_kind != self.expected_data_kind:
            debug_print(f"accept rejected: expected_kind={self.expected_data_kind}, received_kind={dxo.data_kind}")
            self.log_error(
                fl_ctx,
                f"Expected DXO kind {self.expected_data_kind} but received {dxo.data_kind}",
            )
            return False

        contributor_name = shareable.get_peer_prop(key=ReservedKey.IDENTITY_NAME, default="?")
        weight = self._get_contribution_weight(dxo)
        try:
            params = self._to_numpy_params(dxo.data)
        except Exception as ex:
            debug_print(f"accept failed for {contributor_name}: could not convert params: {ex}")
            self.log_exception(fl_ctx, f"Failed to convert params from {contributor_name}: {ex}")
            return False

        first_key = next(iter(params), None)
        first_shape = params[first_key].shape if first_key else None
        debug_print(
            f"accepting update: contributor={contributor_name}, weight={weight}, "
            f"tensors={len(params)}, first_key={first_key}, first_shape={first_shape}"
        )
        self.updates.append(
            ClientUpdate(
                params=params,
                weight=weight,
                client_id=str(contributor_name),
                metrics=dict(dxo.meta) if dxo.meta else None,
            )
        )
        self.log_info(fl_ctx, f"Accepted update from {contributor_name} with weight={weight}")
        return True

    def aggregate(self, fl_ctx: FLContext) -> Shareable:
        debug_print(f"aggregate called: accepted_updates={len(self.updates)}")
        if not self.updates:
            debug_print("aggregate returning empty DXO: no accepted updates")
            self.log_warning(fl_ctx, "No client updates were accepted for aggregation.")
            return DXO(data_kind=self.expected_data_kind, data={}).to_shareable()

        try:
            aggregated = self.fedavg.aggregate(self.updates)
        except Exception as ex:
            debug_print(f"aggregate failed: {ex}")
            self.log_exception(fl_ctx, f"FedAvg aggregation failed: {ex}")
            raise

        result_dxo = DXO(data_kind=self.expected_data_kind, data=aggregated)
        debug_print(f"aggregate finished: tensors={len(aggregated)}, contributors={len(self.updates)}")
        self.log_info(fl_ctx, f"Aggregated {len(self.updates)} client updates.")
        return result_dxo.to_shareable()

    def reset(self, fl_ctx: FLContext):
        debug_print(f"reset called: clearing {len(self.updates)} accepted updates")
        self.updates = []

    @staticmethod
    def _get_contribution_weight(dxo: DXO) -> float:
        meta = dxo.meta or {}
        weight = (
            meta.get(MetaKey.NUM_STEPS_CURRENT_ROUND)
            or meta.get("num_samples")
            or meta.get("n_samples")
            or 1.0
        )
        try:
            return float(weight)
        except Exception:
            return 1.0

    @staticmethod
    def _to_numpy_params(data: Dict[str, object]) -> Dict[str, np.ndarray]:
        params: Dict[str, np.ndarray] = {}
        for key, value in data.items():
            params[key] = np.asarray(value)
        return params
