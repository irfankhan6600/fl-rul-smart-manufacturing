"""FedAvg aggregation utilities for the federated-learning pipeline.

This module is intentionally framework-light so you can:
1) use it as a reference implementation of FedAvg,
2) wrap it inside an NVFLARE custom aggregator later, and
3) swap it out with a custom algorithm (FedProx, median, trimmed mean, etc.).

The core idea is simple:
- each client sends model parameters,
- the server aggregates them with a weighted average,
- the aggregated model becomes the next global model.

Supported update formats:
- mapping of parameter names to NumPy arrays / PyTorch tensors / Python lists
- optional per-client sample counts or explicit weights

The module does not depend on NVFLARE directly, so it is safe to reuse in unit tests
and in non-NVFLARE experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import copy
import numbers

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - torch is optional for this utility module
    torch = None  # type: ignore

ArrayLike = Union[np.ndarray, "torch.Tensor", Sequence[float], Sequence[int], float, int]
StateDict = Dict[str, ArrayLike]


@dataclass
class ClientUpdate:
    """Container for one client update.

    Attributes
    ----------
    params:
        Model parameters from the client.
    weight:
        Relative weight for aggregation, usually the number of local samples.
    client_id:
        Optional identifier for logging/debugging.
    metrics:
        Optional metrics returned by the client.
    """

    params: StateDict
    weight: float
    client_id: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None


class FedAvgAggregator:
    """Simple weighted FedAvg aggregator.

    Example
    -------
    >>> agg = FedAvgAggregator()
    >>> result = agg.aggregate([
    ...     ClientUpdate(params={"w": np.array([1.0, 2.0])}, weight=10),
    ...     ClientUpdate(params={"w": np.array([3.0, 4.0])}, weight=30),
    ... ])
    >>> result["w"]
    array([2.5, 3.5])
    """

    def __init__(self, min_weight: float = 0.0) -> None:
        self.min_weight = float(min_weight)

    @staticmethod
    def _to_numpy(value: ArrayLike) -> np.ndarray:
        """Convert a supported value into a NumPy array without modifying the input."""
        if torch is not None and isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, (list, tuple)):
            return np.asarray(value)
        if isinstance(value, numbers.Number):
            return np.asarray(value)
        raise TypeError(f"Unsupported parameter type: {type(value)!r}")

    @staticmethod
    def _from_numpy(template: ArrayLike, value: np.ndarray) -> ArrayLike:
        """Convert aggregated NumPy result back to the template type when possible."""
        if torch is not None and isinstance(template, torch.Tensor):
            return torch.as_tensor(value, dtype=template.dtype)
        if isinstance(template, np.ndarray):
            return value.astype(template.dtype, copy=False)
        if isinstance(template, (list, tuple)):
            return value.tolist()
        if isinstance(template, numbers.Number):
            return value.item()
        return value

    def aggregate(
        self,
        updates: Sequence[ClientUpdate],
        *,
        strict_keys: bool = True,
        normalize_weights: bool = True,
    ) -> StateDict:
        """Aggregate a sequence of client updates into a single state dict.

        Parameters
        ----------
        updates:
            Client updates to aggregate.
        strict_keys:
            If True, all updates must contain the same parameter keys.
            If False, keys present in the first valid update define the target set.
        normalize_weights:
            If True, weights are normalized to sum to 1.
            If False, raw weights are used directly, which is useful when weights are
            already normalized.

        Returns
        -------
        Dict[str, ArrayLike]
            Aggregated model parameters.
        """
        valid_updates = [u for u in updates if u is not None and u.weight > self.min_weight]
        if not valid_updates:
            raise ValueError("No valid client updates provided to FedAvgAggregator.")

        total_weight = float(sum(u.weight for u in valid_updates))
        if total_weight <= 0:
            raise ValueError("Total aggregation weight must be positive.")

        if normalize_weights:
            normalized = [(u, u.weight / total_weight) for u in valid_updates]
        else:
            normalized = [(u, u.weight) for u in valid_updates]

        base_keys = list(valid_updates[0].params.keys())
        if strict_keys:
            for u in valid_updates[1:]:
                if set(u.params.keys()) != set(base_keys):
                    raise ValueError("All client updates must have the same parameter keys.")
        else:
            # keep intersection only for non-strict mode
            common_keys = set(base_keys)
            for u in valid_updates[1:]:
                common_keys &= set(u.params.keys())
            base_keys = [k for k in base_keys if k in common_keys]

        aggregated: Dict[str, np.ndarray] = {}
        templates: Dict[str, ArrayLike] = {}

        for key in base_keys:
            templates[key] = valid_updates[0].params[key]
            weighted_sum: Optional[np.ndarray] = None
            for update, weight in normalized:
                arr = self._to_numpy(update.params[key])
                if weighted_sum is None:
                    weighted_sum = arr.astype(np.float64, copy=True) * weight
                else:
                    weighted_sum += arr.astype(np.float64, copy=False) * weight

            assert weighted_sum is not None
            aggregated[key] = weighted_sum

        # Convert back to the original container type when possible.
        result: StateDict = {}
        for key, arr in aggregated.items():
            result[key] = self._from_numpy(templates[key], arr)
        return result

    def aggregate_inplace(
        self,
        target_state: MutableMapping[str, ArrayLike],
        updates: Sequence[ClientUpdate],
        *,
        strict_keys: bool = True,
        normalize_weights: bool = True,
    ) -> MutableMapping[str, ArrayLike]:
        """Overwrite a target state dict with the aggregated result."""
        aggregated = self.aggregate(updates, strict_keys=strict_keys, normalize_weights=normalize_weights)
        target_state.clear()
        target_state.update(copy.deepcopy(aggregated))
        return target_state


# ----------------------------------------------------------------------------
# Convenience helpers for later NVFLARE integration
# ----------------------------------------------------------------------------

def fedavg_state_dict(
    client_states: Sequence[Mapping[str, ArrayLike]],
    sample_counts: Optional[Sequence[float]] = None,
) -> StateDict:
    """Aggregate a list of state dicts using FedAvg.

    Parameters
    ----------
    client_states:
        State dicts returned by clients.
    sample_counts:
        Optional sample counts for each client. If omitted, all clients are equally weighted.
    """
    if not client_states:
        raise ValueError("client_states cannot be empty")

    if sample_counts is None:
        sample_counts = [1.0] * len(client_states)
    if len(sample_counts) != len(client_states):
        raise ValueError("sample_counts must have the same length as client_states")

    updates = [
        ClientUpdate(params=dict(state), weight=float(weight), client_id=str(i))
        for i, (state, weight) in enumerate(zip(client_states, sample_counts))
    ]
    return FedAvgAggregator().aggregate(updates)


class AlgorithmRegistry:
    """Tiny swap-friendly registry for future algorithms.

    Later you can register FedProx, median, trimmed mean, robust aggregation, etc.
    """

    def __init__(self) -> None:
        self._algorithms: Dict[str, Any] = {"fedavg": FedAvgAggregator}

    def register(self, name: str, factory: Any) -> None:
        self._algorithms[name.lower()] = factory

    def create(self, name: str, **kwargs: Any) -> Any:
        key = name.lower()
        if key not in self._algorithms:
            raise KeyError(f"Unknown algorithm: {name}")
        factory = self._algorithms[key]
        return factory(**kwargs) if callable(factory) else factory

    def available(self) -> List[str]:
        return sorted(self._algorithms.keys())


__all__ = [
    "ClientUpdate",
    "FedAvgAggregator",
    "fedavg_state_dict",
    "AlgorithmRegistry",
]
