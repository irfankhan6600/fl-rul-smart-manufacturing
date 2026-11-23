"""
Federated learning data loader.

This version is designed for the raw per-client .npz files used in your
pipeline. It rebuilds unit-wise windows from the tabular data, then fits
scalers on the training split only.

Notebook alignment:
- Keep the same windowing approach as the centralized model.
- Keep USE_T50_HISTORY_AS_INPUT=True when you want autoregressive input.
- Fit input/target scalers on train windows only.
- Keep windows cycle-safe so they never cross cycle boundaries.

Expected raw columns (at minimum):
- unit
- cycle
- target/history column (default: T50)
- centralized baseline feature columns
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


DEFAULT_UNIT_COL = "unit"
DEFAULT_CYCLE_COL = "cycle"
DEFAULT_TARGET_COL = "RUL_capped"
DEFAULT_HISTORY_COL = "T50"
DEFAULT_RUL_CAP = 125
CENTRALIZED_FEATURE_COLS: Tuple[str, ...] = (
    "alt",
    "Mach",
    "TRA",
    "T2",
    "T24",
    "T30",
    "T48",
    "P15",
    "P21",
    "P24",
    "Ps30",
    "P40",
    "P50",
    "Nf",
    "Nc",
    "Wf",
    "P45",
    "P30",
    "W48",
    "W50",
    "W31",
    "W32",
    "W25",
    "W22",
    "T50",
)


@dataclass(frozen=True)
class LoaderConfig:
    unit_col: str = DEFAULT_UNIT_COL
    cycle_col: str = DEFAULT_CYCLE_COL
    target_col: str = DEFAULT_TARGET_COL
    history_col: str = DEFAULT_HISTORY_COL
    window_size: int = 96
    target_offset: int = 0
    batch_size: int = 256
    use_t50_history_as_input: bool = True
    num_workers: int = 0
    feature_cols: Tuple[str, ...] = CENTRALIZED_FEATURE_COLS


class WindowDataset(Dataset):
    """
    Dataset for already-windowed tensors.

    x: (num_samples, window_size, num_features)
    y: (num_samples, 1)
    """

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        if x.ndim != 3:
            raise ValueError(f"Expected x to have shape (n, window, features); got {x.shape}")
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if y.ndim != 2:
            raise ValueError(f"Expected y to have shape (n, 1) or (n,); got {y.shape}")
        if len(x) != len(y):
            raise ValueError("x and y must have the same number of samples")

        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class LazyWindowDataset(Dataset):
    """
    Dataset that builds cycle-safe windows on demand.

    It keeps the raw per-segment arrays in memory and computes the requested
    `(window_size, num_features)` slice only in `__getitem__`, which avoids
    materializing multi-GB `(num_windows, window_size, num_features)` tensors.
    """

    def __init__(
        self,
        segments: Sequence[Tuple[np.ndarray, np.ndarray]],
        window_size: int,
        target_offset: int,
        input_scaler: Optional[StandardScaler] = None,
        target_scaler: Optional[StandardScaler] = None,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if target_offset < 0:
            raise ValueError("target_offset must be non-negative")
        if not segments:
            raise ValueError("No valid segments were provided.")

        self.window_size = int(window_size)
        self.target_offset = int(target_offset)
        self.input_scaler = input_scaler
        self.target_scaler = target_scaler

        self.segments: List[Tuple[np.ndarray, np.ndarray]] = []
        self.window_counts: List[int] = []
        for x_src, y_src in segments:
            x_arr = np.asarray(x_src, dtype=np.float32)
            y_arr = np.asarray(y_src, dtype=np.float32).reshape(-1)
            if x_arr.ndim != 2:
                raise ValueError(f"Each segment x must be 2D, got shape {x_arr.shape}")
            if y_arr.ndim != 1:
                raise ValueError(f"Each segment y must be 1D, got shape {y_arr.shape}")
            if len(x_arr) != len(y_arr):
                raise ValueError("Feature and target segment lengths must match.")

            num_windows = len(x_arr) - self.window_size - self.target_offset + 1
            if num_windows <= 0:
                continue

            self.segments.append((x_arr, y_arr))
            self.window_counts.append(int(num_windows))

        if not self.segments:
            raise ValueError("No windows were created. Check window_size and the available cycles per unit.")

        self.cumulative_counts = np.cumsum(self.window_counts, dtype=np.int64)
        self.num_features = int(self.segments[0][0].shape[1])

    def __len__(self) -> int:
        return int(self.cumulative_counts[-1])

    def _resolve_index(self, idx: int) -> Tuple[int, int]:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        segment_idx = int(np.searchsorted(self.cumulative_counts, idx, side="right"))
        segment_start = 0 if segment_idx == 0 else int(self.cumulative_counts[segment_idx - 1])
        local_idx = idx - segment_start
        return segment_idx, int(local_idx)

    def __getitem__(self, idx: int):
        segment_idx, local_idx = self._resolve_index(idx)
        x_src, y_src = self.segments[segment_idx]

        start_idx = local_idx
        end_idx = start_idx + self.window_size
        target_idx = end_idx - 1 + self.target_offset

        x_win = x_src[start_idx:end_idx].astype(np.float32, copy=True)
        y_val = np.array([y_src[target_idx]], dtype=np.float32)

        if self.input_scaler is not None:
            x_win = self.input_scaler.transform(x_win).astype(np.float32)
        if self.target_scaler is not None:
            y_val = self.target_scaler.transform(y_val.reshape(1, 1)).reshape(1).astype(np.float32)

        return torch.from_numpy(x_win), torch.from_numpy(y_val)


def _to_dataframe_from_npz(npz_obj: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """
    Convert an NPZ mapping into a DataFrame.

    Supported layouts:
    1) One key containing a structured array / record array / table-like array
    2) Many 1D arrays, one per column
    3) A key named df / data / dataframe / table if it contains a table-like object
    """
    keys = list(npz_obj.keys())

    if len(keys) == 1:
        key = keys[0]
        arr = npz_obj[key]

        # structured / record array
        if getattr(arr, "dtype", None) is not None and arr.dtype.names:
            return pd.DataFrame.from_records(arr)

        # object array containing a single DataFrame
        if isinstance(arr, np.ndarray) and arr.dtype == object and arr.size == 1:
            item = arr.item()
            if isinstance(item, pd.DataFrame):
                return item.copy()

        # 2D matrix with no names is ambiguous; do not guess.
        if isinstance(arr, np.ndarray) and arr.ndim == 2:
            raise ValueError(
                "NPZ contains a single 2D array without column names. "
                "Please save columns separately or provide a different format."
            )

    # Many columns stored separately
    cols: Dict[str, np.ndarray] = {}
    lengths = set()

    for k, v in npz_obj.items():
        arr = np.asarray(v)
        if arr.dtype == object and arr.size == 1 and isinstance(arr.item(), pd.DataFrame):
            return arr.item().copy()
        cols[k] = arr
        if arr.ndim >= 1:
            lengths.add(len(arr))

    if len(lengths) == 1:
        return pd.DataFrame(cols)

    raise ValueError(
        "Could not convert NPZ to DataFrame. "
        "Expected either one structured array or multiple 1D arrays of equal length."
    )


def load_npz_frame(path: str | Path) -> pd.DataFrame:
    """
    Load a raw client .npz file into a DataFrame.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with np.load(path, allow_pickle=True) as data:
        df = _to_dataframe_from_npz({k: data[k] for k in data.files})

    # Normalize column names to plain strings
    df.columns = [str(c) for c in df.columns]
    if DEFAULT_UNIT_COL in df.columns and DEFAULT_CYCLE_COL in df.columns:
        max_cycle = df.groupby(DEFAULT_UNIT_COL)[DEFAULT_CYCLE_COL].transform("max")
        df["RUL"] = (max_cycle - df[DEFAULT_CYCLE_COL]).astype(np.float32)
        df["RUL_capped"] = np.minimum(df["RUL"], float(DEFAULT_RUL_CAP)).astype(np.float32)
    return df


def infer_feature_cols(
    df: pd.DataFrame,
    target_col: str = DEFAULT_TARGET_COL,
    unit_col: str = DEFAULT_UNIT_COL,
    cycle_col: str = DEFAULT_CYCLE_COL,
    history_col: str = DEFAULT_HISTORY_COL,
    use_t50_history_as_input: bool = True,
    feature_cols: Optional[Sequence[str]] = None,
) -> List[str]:
    """
    Infer input columns.

    Default behavior:
    - use all columns except unit/cycle/target
    - keep T50 as an input if it exists
    - if use_t50_history_as_input=False, drop the history_col when present
    """
    if feature_cols is not None:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(
                "Client NPZ is missing required centralized feature columns: "
                f"{missing}"
            )
        cols = list(feature_cols)
        return cols

    exclude = {unit_col, cycle_col, target_col}
    cols = [c for c in df.columns if c not in exclude]

    if not use_t50_history_as_input and history_col in cols:
        cols.remove(history_col)

    if not cols:
        raise ValueError("No feature columns left after applying exclusions.")

    return cols


def split_dataframe_by_units(
    df: pd.DataFrame,
    train_units: Sequence[int],
    val_units: Optional[Sequence[int]] = None,
    test_units: Optional[Sequence[int]] = None,
    unit_col: str = DEFAULT_UNIT_COL,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Split a dataframe by unit IDs.
    """
    if unit_col not in df.columns:
        raise KeyError(f"Missing required unit column: {unit_col}")

    train_set = set(int(u) for u in train_units)
    val_set = set(int(u) for u in (val_units or []))
    test_set = set(int(u) for u in (test_units or []))

    all_units = set(int(u) for u in df[unit_col].unique())

    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise ValueError("Train/val/test unit sets must be disjoint.")

    unknown = (train_set | val_set | test_set) - all_units
    if unknown:
        raise ValueError(f"Unknown unit IDs in split: {sorted(unknown)}")

    train_df = df[df[unit_col].isin(train_set)].copy()
    val_df = df[df[unit_col].isin(val_set)].copy() if val_set else None
    test_df = df[df[unit_col].isin(test_set)].copy() if test_set else None

    return train_df, val_df, test_df


def split_dataframe_by_cycles(
    df: pd.DataFrame,
    train_cycles: Sequence[int],
    val_cycles: Optional[Sequence[int]] = None,
    test_cycles: Optional[Sequence[int]] = None,
    cycle_col: str = DEFAULT_CYCLE_COL,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Split a dataframe by cycle IDs.
    """
    if cycle_col not in df.columns:
        raise KeyError(f"Missing required cycle column: {cycle_col}")

    train_set = set(int(c) for c in train_cycles)
    val_set = set(int(c) for c in (val_cycles or []))
    test_set = set(int(c) for c in (test_cycles or []))

    all_cycles = set(int(c) for c in df[cycle_col].unique())

    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise ValueError("Train/val/test cycle sets must be disjoint.")

    unknown = (train_set | val_set | test_set) - all_cycles
    if unknown:
        raise ValueError(f"Unknown cycle IDs in split: {sorted(unknown)}")

    train_df = df[df[cycle_col].isin(train_set)].copy()
    val_df = df[df[cycle_col].isin(val_set)].copy() if val_set else None
    test_df = df[df[cycle_col].isin(test_set)].copy() if test_set else None

    return train_df, val_df, test_df


def _build_windows_for_group(
    group: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    window_size: int,
    target_offset: int = 0,
    cycle_col: str = DEFAULT_CYCLE_COL,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build sliding windows for one unit-cycle segment.
    """
    if cycle_col in group.columns:
        group = group.sort_values(cycle_col).reset_index(drop=True)
    else:
        group = group.reset_index(drop=True)

    missing = [c for c in list(feature_cols) + [target_col] if c not in group.columns]
    if missing:
        raise KeyError(f"Missing columns in group: {missing}")

    x_src = group[list(feature_cols)].to_numpy(dtype=np.float32)
    y_src = group[target_col].to_numpy(dtype=np.float32)

    if len(group) < window_size + target_offset:
        return np.empty((0, window_size, len(feature_cols)), dtype=np.float32), np.empty((0, 1), dtype=np.float32)

    x_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []

    last_end = len(group) - 1 - target_offset
    for end_idx in range(window_size - 1, last_end + 1):
        start_idx = end_idx - window_size + 1
        x_win = x_src[start_idx : end_idx + 1]
        y_idx = end_idx + target_offset
        y_val = y_src[y_idx]

        x_list.append(x_win)
        y_list.append(np.array([y_val], dtype=np.float32))

    if not x_list:
        return np.empty((0, window_size, len(feature_cols)), dtype=np.float32), np.empty((0, 1), dtype=np.float32)

    x = np.stack(x_list, axis=0).astype(np.float32)
    y = np.stack(y_list, axis=0).astype(np.float32)
    return x, y


def _segment_from_group(
    group: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    cycle_col: str = DEFAULT_CYCLE_COL,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert one unit-cycle group into raw feature/target arrays.
    """
    if cycle_col in group.columns:
        group = group.sort_values(cycle_col).reset_index(drop=True)
    else:
        group = group.reset_index(drop=True)

    missing = [c for c in list(feature_cols) + [target_col] if c not in group.columns]
    if missing:
        raise KeyError(f"Missing columns in group: {missing}")

    x_src = group[list(feature_cols)].to_numpy(dtype=np.float32)
    y_src = group[target_col].to_numpy(dtype=np.float32)
    return x_src, y_src


def build_windows_from_dataframe(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str = DEFAULT_TARGET_COL,
    unit_col: str = DEFAULT_UNIT_COL,
    cycle_col: str = DEFAULT_CYCLE_COL,
    window_size: int = 96,
    target_offset: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a raw dataframe into windowed x/y tensors.

    Returns:
        x: (num_windows, window_size, num_features)
        y: (num_windows, 1)
    """
    if unit_col not in df.columns:
        raise KeyError(f"Missing unit column: {unit_col}")
    if target_col not in df.columns:
        raise KeyError(f"Missing target column: {target_col}")

    x_all: List[np.ndarray] = []
    y_all: List[np.ndarray] = []

    group_cols: List[str] = [unit_col]
    if cycle_col in df.columns:
        group_cols.append(cycle_col)

    for _, group in df.groupby(group_cols, sort=False):
        x_g, y_g = _build_windows_for_group(
            group=group,
            feature_cols=feature_cols,
            target_col=target_col,
            window_size=window_size,
            target_offset=target_offset,
            cycle_col=cycle_col,
        )
        if len(x_g):
            x_all.append(x_g)
            y_all.append(y_g)

    if not x_all:
        raise ValueError("No windows were created. Check window_size and the available cycles per unit.")

    x = np.concatenate(x_all, axis=0).astype(np.float32)
    y = np.concatenate(y_all, axis=0).astype(np.float32)
    return x, y


def build_segments_from_dataframe(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str = DEFAULT_TARGET_COL,
    unit_col: str = DEFAULT_UNIT_COL,
    cycle_col: str = DEFAULT_CYCLE_COL,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Convert a raw dataframe into raw per-segment arrays.
    """
    if unit_col not in df.columns:
        raise KeyError(f"Missing unit column: {unit_col}")
    if target_col not in df.columns:
        raise KeyError(f"Missing target column: {target_col}")

    segments: List[Tuple[np.ndarray, np.ndarray]] = []
    group_cols: List[str] = [unit_col]
    if cycle_col in df.columns:
        group_cols.append(cycle_col)

    for _, group in df.groupby(group_cols, sort=False):
        segments.append(
            _segment_from_group(
                group=group,
                feature_cols=feature_cols,
                target_col=target_col,
                cycle_col=cycle_col,
            )
        )

    return segments


def count_windows_in_segments(
    segments: Sequence[Tuple[np.ndarray, np.ndarray]],
    window_size: int,
    target_offset: int,
) -> int:
    total = 0
    for x_src, _ in segments:
        total += max(len(x_src) - window_size - target_offset + 1, 0)
    return int(total)


def fit_scalers(x: np.ndarray, y: np.ndarray) -> Tuple[StandardScaler, StandardScaler]:
    """
    Fit input and target scalers on training windows only.
    """
    if x.ndim != 3:
        raise ValueError(f"Expected x to be 3D, got {x.shape}")
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if y.ndim != 2:
        raise ValueError(f"Expected y to be 2D, got {y.shape}")

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_scaler.fit(x.reshape(-1, x.shape[-1]))
    y_scaler.fit(y)

    return x_scaler, y_scaler


def fit_scalers_on_segments(
    segments: Sequence[Tuple[np.ndarray, np.ndarray]],
    window_size: int,
    target_offset: int,
    batch_windows: int = 1024,
) -> Tuple[StandardScaler, StandardScaler]:
    """
    Fit scalers by iterating over windows lazily, without materializing all of them.
    """
    if batch_windows <= 0:
        raise ValueError("batch_windows must be positive")

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    saw_any = False

    for x_src, y_src in segments:
        num_windows = len(x_src) - window_size - target_offset + 1
        if num_windows <= 0:
            continue

        for batch_start in range(0, num_windows, batch_windows):
            batch_end = min(batch_start + batch_windows, num_windows)
            batch_size = batch_end - batch_start
            x_batch = np.empty((batch_size, window_size, x_src.shape[1]), dtype=np.float32)
            y_batch = np.empty((batch_size, 1), dtype=np.float32)

            for offset, local_idx in enumerate(range(batch_start, batch_end)):
                start_idx = local_idx
                end_idx = start_idx + window_size
                target_idx = end_idx - 1 + target_offset
                x_batch[offset] = x_src[start_idx:end_idx]
                y_batch[offset, 0] = y_src[target_idx]

            x_scaler.partial_fit(x_batch.reshape(-1, x_batch.shape[-1]))
            y_scaler.partial_fit(y_batch)
            saw_any = True

    if not saw_any:
        raise ValueError("No windows were created. Check window_size and the available cycles per unit.")

    return x_scaler, y_scaler


def apply_scalers(
    x: np.ndarray,
    y: np.ndarray,
    input_scaler: Optional[StandardScaler] = None,
    target_scaler: Optional[StandardScaler] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply scalers once up front.
    """
    x_out = np.asarray(x, dtype=np.float32)
    y_out = np.asarray(y, dtype=np.float32)

    if input_scaler is not None:
        n, t, f = x_out.shape
        x_out = input_scaler.transform(x_out.reshape(-1, f)).reshape(n, t, f).astype(np.float32)

    if target_scaler is not None:
        y_out = target_scaler.transform(y_out).astype(np.float32)

    return x_out, y_out


def make_dataset_from_raw_npz(
    path: str | Path,
    feature_cols: Optional[Sequence[str]] = None,
    cfg: Optional[LoaderConfig] = None,
    input_scaler: Optional[StandardScaler] = None,
    target_scaler: Optional[StandardScaler] = None,
    fit_scalers_flag: bool = False,
) -> Tuple[Dataset, List[str], Optional[StandardScaler], Optional[StandardScaler]]:
    """
    Load raw NPZ -> dataframe -> windows -> dataset.

    Returns:
        dataset, feature_cols, input_scaler, target_scaler
    """
    cfg = cfg or LoaderConfig()
    df = load_npz_frame(path)

    cols = infer_feature_cols(
        df=df,
        target_col=cfg.target_col,
        unit_col=cfg.unit_col,
        cycle_col=cfg.cycle_col,
        history_col=cfg.history_col,
        use_t50_history_as_input=cfg.use_t50_history_as_input,
        feature_cols=feature_cols or cfg.feature_cols,
    )

    segments = build_segments_from_dataframe(
        df=df,
        feature_cols=cols,
        target_col=cfg.target_col,
        unit_col=cfg.unit_col,
        cycle_col=cfg.cycle_col,
    )

    if fit_scalers_flag:
        input_scaler, target_scaler = fit_scalers_on_segments(
            segments=segments,
            window_size=cfg.window_size,
            target_offset=cfg.target_offset,
        )

    dataset = LazyWindowDataset(
        segments=segments,
        window_size=cfg.window_size,
        target_offset=cfg.target_offset,
        input_scaler=input_scaler,
        target_scaler=target_scaler,
    )
    return dataset, cols, input_scaler, target_scaler


def make_loader_from_raw_npz(
    path: str | Path,
    feature_cols: Optional[Sequence[str]] = None,
    cfg: Optional[LoaderConfig] = None,
    input_scaler: Optional[StandardScaler] = None,
    target_scaler: Optional[StandardScaler] = None,
    fit_scalers_flag: bool = False,
    shuffle: bool = True,
) -> Tuple[DataLoader, List[str], Optional[StandardScaler], Optional[StandardScaler]]:
    """
    Convenience wrapper to build a DataLoader from a raw NPZ file.
    """
    cfg = cfg or LoaderConfig()

    dataset, cols, input_scaler, target_scaler = make_dataset_from_raw_npz(
        path=path,
        feature_cols=feature_cols,
        cfg=cfg,
        input_scaler=input_scaler,
        target_scaler=target_scaler,
        fit_scalers_flag=fit_scalers_flag,
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return loader, cols, input_scaler, target_scaler


def make_train_val_test_loaders_from_raw_npz(
    path: str | Path,
    train_units: Sequence[int],
    val_units: Optional[Sequence[int]] = None,
    test_units: Optional[Sequence[int]] = None,
    feature_cols: Optional[Sequence[str]] = None,
    cfg: Optional[LoaderConfig] = None,
) -> Tuple[
    DataLoader,
    Optional[DataLoader],
    Optional[DataLoader],
    List[str],
    StandardScaler,
    StandardScaler,
]:
    """
    Full pipeline:
    raw NPZ -> dataframe -> split by unit -> window each split -> fit scalers on train -> loaders

    This is the function you will most likely call from train.py.
    """
    cfg = cfg or LoaderConfig()
    df = load_npz_frame(path)

    cols = infer_feature_cols(
        df=df,
        target_col=cfg.target_col,
        unit_col=cfg.unit_col,
        cycle_col=cfg.cycle_col,
        history_col=cfg.history_col,
        use_t50_history_as_input=cfg.use_t50_history_as_input,
        feature_cols=feature_cols or cfg.feature_cols,
    )

    train_df, val_df, test_df = split_dataframe_by_units(
        df=df,
        train_units=train_units,
        val_units=val_units,
        test_units=test_units,
        unit_col=cfg.unit_col,
    )

    train_segments = build_segments_from_dataframe(
        train_df,
        feature_cols=cols,
        target_col=cfg.target_col,
        unit_col=cfg.unit_col,
        cycle_col=cfg.cycle_col,
    )

    val_segments = None
    if val_df is not None and len(val_df):
        val_segments = build_segments_from_dataframe(
            val_df,
            feature_cols=cols,
            target_col=cfg.target_col,
            unit_col=cfg.unit_col,
            cycle_col=cfg.cycle_col,
        )

    test_segments = None
    if test_df is not None and len(test_df):
        test_segments = build_segments_from_dataframe(
            test_df,
            feature_cols=cols,
            target_col=cfg.target_col,
            unit_col=cfg.unit_col,
            cycle_col=cfg.cycle_col,
        )

    input_scaler, target_scaler = fit_scalers_on_segments(
        segments=train_segments,
        window_size=cfg.window_size,
        target_offset=cfg.target_offset,
    )

    train_ds = LazyWindowDataset(
        segments=train_segments,
        window_size=cfg.window_size,
        target_offset=cfg.target_offset,
        input_scaler=input_scaler,
        target_scaler=target_scaler,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = None
    if val_segments:
        val_loader = DataLoader(
            LazyWindowDataset(
                segments=val_segments,
                window_size=cfg.window_size,
                target_offset=cfg.target_offset,
                input_scaler=input_scaler,
                target_scaler=target_scaler,
            ),
            batch_size=cfg.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    test_loader = None
    if test_segments:
        test_loader = DataLoader(
            LazyWindowDataset(
                segments=test_segments,
                window_size=cfg.window_size,
                target_offset=cfg.target_offset,
                input_scaler=input_scaler,
                target_scaler=target_scaler,
            ),
            batch_size=cfg.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return train_loader, val_loader, test_loader, cols, input_scaler, target_scaler


def make_train_val_test_loaders_from_raw_npz_by_cycles(
    path: str | Path,
    train_cycles: Sequence[int],
    val_cycles: Optional[Sequence[int]] = None,
    test_cycles: Optional[Sequence[int]] = None,
    feature_cols: Optional[Sequence[str]] = None,
    cfg: Optional[LoaderConfig] = None,
) -> Tuple[
    DataLoader,
    Optional[DataLoader],
    Optional[DataLoader],
    List[str],
    StandardScaler,
    StandardScaler,
]:
    """
    Full pipeline using cycle IDs for train/validation/test splits.

    This is useful when a client owns one engine unit with multiple cycles.
    """
    cfg = cfg or LoaderConfig()
    df = load_npz_frame(path)

    cols = infer_feature_cols(
        df=df,
        target_col=cfg.target_col,
        unit_col=cfg.unit_col,
        cycle_col=cfg.cycle_col,
        history_col=cfg.history_col,
        use_t50_history_as_input=cfg.use_t50_history_as_input,
        feature_cols=feature_cols or cfg.feature_cols,
    )

    train_df, val_df, test_df = split_dataframe_by_cycles(
        df=df,
        train_cycles=train_cycles,
        val_cycles=val_cycles,
        test_cycles=test_cycles,
        cycle_col=cfg.cycle_col,
    )

    train_segments = build_segments_from_dataframe(
        train_df,
        feature_cols=cols,
        target_col=cfg.target_col,
        unit_col=cfg.unit_col,
        cycle_col=cfg.cycle_col,
    )

    val_segments = None
    if val_df is not None and len(val_df):
        val_segments = build_segments_from_dataframe(
            val_df,
            feature_cols=cols,
            target_col=cfg.target_col,
            unit_col=cfg.unit_col,
            cycle_col=cfg.cycle_col,
        )

    test_segments = None
    if test_df is not None and len(test_df):
        test_segments = build_segments_from_dataframe(
            test_df,
            feature_cols=cols,
            target_col=cfg.target_col,
            unit_col=cfg.unit_col,
            cycle_col=cfg.cycle_col,
        )

    input_scaler, target_scaler = fit_scalers_on_segments(
        segments=train_segments,
        window_size=cfg.window_size,
        target_offset=cfg.target_offset,
    )

    train_ds = LazyWindowDataset(
        segments=train_segments,
        window_size=cfg.window_size,
        target_offset=cfg.target_offset,
        input_scaler=input_scaler,
        target_scaler=target_scaler,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = None
    if val_segments:
        val_loader = DataLoader(
            LazyWindowDataset(
                segments=val_segments,
                window_size=cfg.window_size,
                target_offset=cfg.target_offset,
                input_scaler=input_scaler,
                target_scaler=target_scaler,
            ),
            batch_size=cfg.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    test_loader = None
    if test_segments:
        test_loader = DataLoader(
            LazyWindowDataset(
                segments=test_segments,
                window_size=cfg.window_size,
                target_offset=cfg.target_offset,
                input_scaler=input_scaler,
                target_scaler=target_scaler,
            ),
            batch_size=cfg.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return train_loader, val_loader, test_loader, cols, input_scaler, target_scaler


def validate_npz_file(
    path: str | Path,
    cfg: Optional[LoaderConfig] = None,
    feature_cols: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """
    Validate one client NPZ against the expected centralized schema.
    """
    cfg = cfg or LoaderConfig()
    df = load_npz_frame(path)
    cols = infer_feature_cols(
        df=df,
        target_col=cfg.target_col,
        unit_col=cfg.unit_col,
        cycle_col=cfg.cycle_col,
        history_col=cfg.history_col,
        use_t50_history_as_input=cfg.use_t50_history_as_input,
        feature_cols=feature_cols or cfg.feature_cols,
    )
    segments = build_segments_from_dataframe(
        df=df,
        feature_cols=cols,
        target_col=cfg.target_col,
        unit_col=cfg.unit_col,
        cycle_col=cfg.cycle_col,
    )
    num_windows = count_windows_in_segments(
        segments=segments,
        window_size=cfg.window_size,
        target_offset=cfg.target_offset,
    )
    return {
        "path": str(Path(path)),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "units": sorted(int(u) for u in df[cfg.unit_col].dropna().unique().tolist()),
        "cycles": sorted(int(c) for c in df[cfg.cycle_col].dropna().unique().tolist()) if cfg.cycle_col in df.columns else [],
        "feature_cols": list(cols),
        "num_features": len(cols),
        "num_windows": int(num_windows),
        "window_shape": [int(num_windows), int(cfg.window_size), int(len(cols))],
        "target_shape": [int(num_windows), 1],
    }


def validate_npz_directory(
    path: str | Path,
    cfg: Optional[LoaderConfig] = None,
    feature_cols: Optional[Sequence[str]] = None,
) -> List[Dict[str, object]]:
    """
    Validate every NPZ file in one client local data directory.
    """
    directory = Path(path)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")
    files = sorted(directory.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in directory: {directory}")
    return [
        validate_npz_file(file_path, cfg=cfg, feature_cols=feature_cols)
        for file_path in files
    ]


if __name__ == "__main__":
    import os

    path = os.environ.get("TRAIN_NPZ")
    if not path:
        raise SystemExit("Set TRAIN_NPZ to a raw client .npz file to run a smoke test.")

    cfg = LoaderConfig()

    loader, cols, x_scaler, y_scaler = make_loader_from_raw_npz(
        path=path,
        cfg=cfg,
        fit_scalers_flag=True,
        shuffle=True,
    )

    batch_x, batch_y = next(iter(loader))
    print("Feature cols:", cols)
    print("Batch X shape:", tuple(batch_x.shape))
    print("Batch Y shape:", tuple(batch_y.shape))
