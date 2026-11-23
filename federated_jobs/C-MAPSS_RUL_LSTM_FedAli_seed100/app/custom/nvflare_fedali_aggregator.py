"""NVFLARE wrapper for sample-weighted FedAli aggregation."""

from __future__ import annotations

from typing import Dict

import numpy as np

from fedali_algorithm import FedAliAggregator
from fedavg_algorithm import ClientUpdate
from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.fl_constant import ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.aggregator import Aggregator


def debug_print(message: str) -> None:
    print(f"[nvflare_fedali_aggregator.py] {message}", flush=True)


class NVFlareFedAliAggregator(Aggregator):
    """Aggregate full model weights and cluster uploaded local prototypes."""

    def __init__(
        self,
        expected_data_kind: str = DataKind.WEIGHTS,
        prototype_key: str = "alp.global_prototypes",
        min_weight: float = 0.0,
        random_state: int = 100,
        max_iter: int = 100,
    ) -> None:
        super().__init__()
        self.expected_data_kind = expected_data_kind
        self.fedali = FedAliAggregator(
            prototype_key=prototype_key,
            min_weight=min_weight,
            random_state=random_state,
            max_iter=max_iter,
        )
        self.updates: list[ClientUpdate] = []
        debug_print(
            f"initialized: expected_data_kind={expected_data_kind}, prototype_key={prototype_key}, "
            f"min_weight={min_weight}"
        )

    def accept(self, shareable: Shareable, fl_ctx: FLContext) -> bool:
        try:
            dxo = from_shareable(shareable)
        except Exception as ex:
            self.log_exception(fl_ctx, f"Failed to convert FedAli shareable to DXO: {ex}")
            return False
        if dxo.data_kind != self.expected_data_kind:
            self.log_error(
                fl_ctx,
                f"Expected DXO kind {self.expected_data_kind} but received {dxo.data_kind}",
            )
            return False

        contributor = shareable.get_peer_prop(key=ReservedKey.IDENTITY_NAME, default="?")
        weight = self._get_contribution_weight(dxo)
        try:
            params = {key: np.asarray(value) for key, value in dxo.data.items()}
            prototype_shape = params[self.fedali.prototype_key].shape
        except Exception as ex:
            self.log_exception(fl_ctx, f"Invalid FedAli update from {contributor}: {ex}")
            return False

        self.updates.append(
            ClientUpdate(
                params=params,
                weight=weight,
                client_id=str(contributor),
                metrics=dict(dxo.meta) if dxo.meta else None,
            )
        )
        debug_print(
            f"accepted contributor={contributor}, weight={weight}, tensors={len(params)}, "
            f"prototype_shape={prototype_shape}"
        )
        return True

    def aggregate(self, fl_ctx: FLContext) -> Shareable:
        if not self.updates:
            self.log_warning(fl_ctx, "No FedAli client updates were accepted")
            return DXO(data_kind=self.expected_data_kind, data={}).to_shareable()
        try:
            aggregated = self.fedali.aggregate(self.updates)
        except Exception as ex:
            self.log_exception(fl_ctx, f"FedAli aggregation failed: {ex}")
            raise
        self.log_info(fl_ctx, f"FedAli aggregated {len(self.updates)} client updates")
        debug_print(f"aggregated clients={len(self.updates)}, tensors={len(aggregated)}")
        return DXO(data_kind=self.expected_data_kind, data=aggregated).to_shareable()

    def reset(self, fl_ctx: FLContext) -> None:
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
        except (TypeError, ValueError):
            return 1.0
