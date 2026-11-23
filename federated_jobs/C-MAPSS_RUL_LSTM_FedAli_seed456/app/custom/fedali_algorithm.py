"""Framework-light FedAli aggregation.

Ordinary model tensors are aggregated by sample-weighted FedAvg.  Client-local
prototype banks are carried under the global prototype state-dict key and are
clustered into the next shared bank with weighted K-means.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.cluster import KMeans

from fedavg_algorithm import ClientUpdate, FedAvgAggregator, StateDict


class FedAliAggregator:
    def __init__(
        self,
        prototype_key: str = "alp.global_prototypes",
        min_weight: float = 0.0,
        random_state: int = 456,
        max_iter: int = 100,
    ) -> None:
        self.prototype_key = prototype_key
        self.min_weight = float(min_weight)
        self.random_state = int(random_state)
        self.max_iter = int(max_iter)
        self.fedavg = FedAvgAggregator(min_weight=min_weight)

    def _aggregate_prototypes(self, updates: Sequence[ClientUpdate]) -> np.ndarray:
        banks = []
        client_weights = []
        expected_shape = None
        for update in updates:
            if update.weight <= self.min_weight:
                continue
            if self.prototype_key not in update.params:
                raise KeyError(f"Client {update.client_id!r} did not provide {self.prototype_key!r}")
            bank = np.asarray(update.params[self.prototype_key])
            if bank.ndim != 2:
                raise ValueError(f"Prototype bank must be 2D, got {bank.shape}")
            if expected_shape is None:
                expected_shape = bank.shape
            elif bank.shape != expected_shape:
                raise ValueError(f"Prototype shape mismatch: expected {expected_shape}, got {bank.shape}")
            banks.append(bank.astype(np.float64, copy=False))
            client_weights.append(float(update.weight))

        if not banks or expected_shape is None:
            raise ValueError("No valid FedAli prototype banks were provided")

        weights = np.asarray(client_weights, dtype=np.float64)
        weights /= weights.sum()
        # Slot-wise weighted centroids provide stable, round-to-round cluster IDs.
        initialization = sum(weight * bank for weight, bank in zip(weights, banks))

        stacked = np.concatenate(banks, axis=0)
        kmeans = KMeans(
            n_clusters=expected_shape[0],
            init=initialization,
            n_init=1,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        # Algorithm 2 uses the sample-weighted centroid only as K-means
        # initialization; the pooled local prototypes are then clustered normally.
        kmeans.fit(stacked)
        return kmeans.cluster_centers_

    def aggregate(self, updates: Sequence[ClientUpdate]) -> StateDict:
        valid = [update for update in updates if update is not None and update.weight > self.min_weight]
        if not valid:
            raise ValueError("No valid client updates provided to FedAliAggregator")

        model_updates = []
        for update in valid:
            params = {key: value for key, value in update.params.items() if key != self.prototype_key}
            model_updates.append(
                ClientUpdate(
                    params=params,
                    weight=update.weight,
                    client_id=update.client_id,
                    metrics=update.metrics,
                )
            )

        result = self.fedavg.aggregate(model_updates)
        template = np.asarray(valid[0].params[self.prototype_key])
        prototypes = self._aggregate_prototypes(valid)
        result[self.prototype_key] = prototypes.astype(template.dtype, copy=False)
        return result


__all__ = ["FedAliAggregator"]
