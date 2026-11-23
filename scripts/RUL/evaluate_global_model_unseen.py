#!/usr/bin/env python3
"""Evaluate a federated RUL global model on the unseen test engines (units 11, 14, 15).

Predictions are made in standardized space and mapped back to cycles with a reference
scaler built only from the training data of the six federated clients:
build_pooled_reference_scalers() rebuilds each client's training-time scaler from its
local .npz, using the same cycle split as train.py (split_groups_for_validation with the
job's seed and 15%/15% validation/test fractions), and pools them by exact
sufficient-statistic aggregation weighted by each client's number of training windows.
The test engines are not used to fit any scaler.

Outputs window-level and cycle-level metrics. Cycle-level metrics average all window
predictions within each (unit, cycle) before computing RMSE, MAE and the NASA score.
The pooled scaler statistics are written to scaler_manifest.json.

Main arguments: --job-id, --checkpoint, --custom-dir (job app/custom folder),
--site-data-root, --seed, --val-fraction, --test-fraction, --results-dir, --model-name.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CUSTOM_DIR = PROJECT_ROOT / "federated_jobs" / "C-MAPSS_RUL_LSTM_FedAvg" / "app" / "custom"
FEDALI_CUSTOM_DIR = PROJECT_ROOT / "federated_jobs" / "C-MAPSS_RUL_LSTM_FedAli" / "app" / "custom"

# CUSTOM_DIR is consumed at import time (sys.path + MODEL_MODULES), so the
# override must be read before the data_loader import below. Only --custom-dir is
# pre-parsed here; every other argument is parsed by parse_args() as before.
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--custom-dir", default=str(DEFAULT_CUSTOM_DIR))
_pre_args, _ = _pre_parser.parse_known_args()
CUSTOM_DIR = Path(_pre_args.custom_dir).resolve()
if not (CUSTOM_DIR / "data_loader.py").exists():
    raise FileNotFoundError(f"--custom-dir does not contain data_loader.py: {CUSTOM_DIR}")
sys.path.insert(0, str(CUSTOM_DIR))

from data_loader import (  # noqa: E402
    CENTRALIZED_FEATURE_COLS,
    DEFAULT_RUL_CAP,
    LoaderConfig,
    build_segments_from_dataframe,
    count_windows_in_segments,
    fit_scalers_on_segments,
    load_npz_frame,
    split_dataframe_by_cycles,
    split_dataframe_by_units,
)
# split_groups_for_validation() lives in the job's train.py, not data_loader.py.
# Importing it from the training code guarantees the split reconstruction is the
# exact function that ran during federated training.
from train import TrainConfig, split_groups_for_validation  # noqa: E402


# Same helpers as scripts/RUL/evaluate_fedali_perspectives.py.
@dataclass
class ArrayScaler:
    mean_: np.ndarray
    scale_: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values) - self.mean_) / self.scale_

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values) * self.scale_ + self.mean_


def scaler_from_state(state: dict[str, Any]) -> ArrayScaler:
    mean = np.asarray(state["mean"], dtype=np.float64)
    scale = np.asarray(state["scale"], dtype=np.float64)
    return ArrayScaler(mean_=mean, scale_=np.where(scale == 0.0, 1.0, scale))


SITE_UNIT_MAP = {
    "site-1": 2,
    "site-2": 5,
    "site-3": 10,
    "site-4": 16,
    "site-5": 18,
    "site-6": 20,
}


# Reference scaler built from the clients' training splits only.
def build_pooled_reference_scalers(
    site_data_root,
    seed=42,
    val_fraction=0.15,
    test_fraction=0.15,
    window_size=512,
    target_offset=0,
    feature_cols=None,
    rul_cap=125,
):
    """
    Reconstruct each federated client's training-time scaler from its
    local NPZ file using the identical split logic as train.py (seed=42,
    val=0.15, test=0.15). Pool them into a single reference scaler using
    exact sufficient-statistic aggregation weighted by number of training
    windows (nk).

    Pooling formula:
        mu_pool  = sum(nk * mu_k)  / sum(nk)
        var_pool = sum(nk * (var_k + (mu_k - mu_pool)^2)) / sum(nk)
        std_pool = sqrt(var_pool)

    Applied separately for input scaler (shape 25,)
    and target scaler (shape 1,).

    Uses ZERO information from unseen test units 11, 14, 15.
    Returns (input_scaler, target_scaler, manifest) where the scalers are
    ArrayScaler objects and manifest is the dict written to scaler_manifest.json.
    """
    site_data_root = Path(site_data_root)
    feature_cols = list(feature_cols or CENTRALIZED_FEATURE_COLS)
    if float(rul_cap) != float(DEFAULT_RUL_CAP):
        # RUL_capped is derived inside load_npz_frame with DEFAULT_RUL_CAP; the
        # training pipeline never used any other cap, so refuse silently diverging.
        raise ValueError(f"rul_cap={rul_cap} differs from training DEFAULT_RUL_CAP={DEFAULT_RUL_CAP}")

    train_cfg = TrainConfig(
        input_window=int(window_size),
        target_offset=int(target_offset),
        seed=int(seed),
        val_fraction=float(val_fraction),
        test_fraction=float(test_fraction),
    )
    loader_cfg = LoaderConfig(
        target_col="RUL_capped",
        window_size=int(window_size),
        target_offset=int(target_offset),
        feature_cols=tuple(feature_cols),
    )

    client_stats = {}
    print("Reconstructing training-only scalers per federated client:")
    for site, expected_unit in SITE_UNIT_MAP.items():
        data_dir = site_data_root / site / "local" / "data"
        npz_files = sorted(data_dir.glob("*.npz"))
        if not npz_files:
            raise FileNotFoundError(f"No .npz found for {site} under {data_dir}")
        npz_path = npz_files[0]  # train.py resolve_npz_path() also takes the first sorted file

        df = load_npz_frame(npz_path)
        units_in_file = sorted(int(u) for u in df[loader_cfg.unit_col].dropna().unique().tolist())
        if units_in_file != [expected_unit]:
            raise ValueError(f"{site}: expected unit {expected_unit}, NPZ contains units {units_in_file}")

        split_name, train_groups, val_groups, test_groups = split_groups_for_validation(
            npz_path, train_cfg, loader_cfg
        )
        if split_name == "cycle":
            train_df, _, _ = split_dataframe_by_cycles(df, train_groups, val_groups, test_groups, loader_cfg.cycle_col)
        else:
            train_df, _, _ = split_dataframe_by_units(df, train_groups, val_groups, test_groups, loader_cfg.unit_col)

        train_segments = build_segments_from_dataframe(
            train_df,
            feature_cols=feature_cols,
            target_col=loader_cfg.target_col,
            unit_col=loader_cfg.unit_col,
            cycle_col=loader_cfg.cycle_col,
        )
        # Same function and same (training-only) segments as
        # make_train_val_test_loaders_from_raw_npz_by_cycles() in data_loader.py.
        x_scaler, y_scaler = fit_scalers_on_segments(train_segments, window_size, target_offset)
        n_k = count_windows_in_segments(train_segments, window_size, target_offset)
        if int(x_scaler.n_samples_seen_) != n_k * window_size or int(y_scaler.n_samples_seen_) != n_k:
            raise RuntimeError(f"{site}: scaler sample count does not match training window count")

        client_stats[site] = {
            "unit": expected_unit,
            "npz": str(npz_path),
            "split_by": split_name,
            "n_train_groups": len(train_groups),
            "n_val_groups": len(val_groups),
            "n_test_groups": len(test_groups),
            "n_train_windows": int(n_k),
            "input_mean": np.asarray(x_scaler.mean_, dtype=np.float64),
            "input_var": np.asarray(x_scaler.var_, dtype=np.float64),
            "target_mean": float(y_scaler.mean_[0]),
            "target_var": float(y_scaler.var_[0]),
            "target_std": float(np.sqrt(y_scaler.var_[0])),
        }
        print(
            f"  {site} | unit {expected_unit} | n_train={n_k} | "
            f"target_mean={client_stats[site]['target_mean']:.4f} | "
            f"target_std={client_stats[site]['target_std']:.4f}"
        )

    weights = np.asarray([s["n_train_windows"] for s in client_stats.values()], dtype=np.float64)
    n_total = float(weights.sum())

    def pool(means: np.ndarray, variances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        w = weights.reshape(-1, *([1] * (means.ndim - 1)))
        mu_pool = (w * means).sum(axis=0) / n_total
        var_pool = (w * (variances + (means - mu_pool) ** 2)).sum(axis=0) / n_total
        return mu_pool, np.sqrt(var_pool)

    input_mu, input_std = pool(
        np.stack([s["input_mean"] for s in client_stats.values()]),
        np.stack([s["input_var"] for s in client_stats.values()]),
    )
    target_mu, target_std = pool(
        np.asarray([s["target_mean"] for s in client_stats.values()]),
        np.asarray([s["target_var"] for s in client_stats.values()]),
    )
    print(f"  POOLED | n_total={int(n_total)} | target_mean={float(target_mu):.4f} | target_std={float(target_std):.4f}")

    input_scaler = scaler_from_state({"mean": input_mu, "scale": input_std})
    target_scaler = scaler_from_state({"mean": np.atleast_1d(target_mu), "scale": np.atleast_1d(target_std)})

    manifest = {
        "method": "pooled_sufficient_statistic_aggregation",
        "seed": int(seed),
        "val_fraction": float(val_fraction),
        "test_fraction": float(test_fraction),
        "window_size": int(window_size),
        "target_offset": int(target_offset),
        "rul_cap": float(rul_cap),
        "feature_cols": feature_cols,
        "note": "Zero information from unseen units 11, 14, 15 used in scaler construction",
        "clients": {
            site: {
                "unit": s["unit"],
                "npz": s["npz"],
                "split_by": s["split_by"],
                "n_train_groups": s["n_train_groups"],
                "n_val_groups": s["n_val_groups"],
                "n_test_groups": s["n_test_groups"],
                "n_train_windows": s["n_train_windows"],
                "target_mean": s["target_mean"],
                "target_std": s["target_std"],
            }
            for site, s in client_stats.items()
        },
        "pooled": {
            "target_mean": float(target_mu),
            "target_std": float(target_std),
            "input_means": [float(v) for v in input_mu],
            "input_stds": [float(v) for v in input_std],
            "total_train_windows": int(n_total),
        },
    }
    return input_scaler, target_scaler, manifest


def load_model_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


JOBS_DIR = PROJECT_ROOT / "federated_jobs"
CUSTOM_DIR_IS_FEDALI = (CUSTOM_DIR / "fedali_algorithm.py").exists()

# (module name, file name, fallback custom dir, whether the file is the FedAli LSTM variant).
# A job's custom/ dir only ships its own architecture file, so --custom-dir cannot supply
# every entry; each spec falls back to the architecture's own original job folder.
_MODEL_MODULE_SPECS = {
    "lstm": ("cmapps_rul_lstm_model", "LSTM_model.py", JOBS_DIR / "C-MAPSS_RUL_LSTM_FedAvg" / "app" / "custom", False),
    "fedali_lstm": ("cmapps_rul_fedali_lstm_model", "LSTM_model.py", FEDALI_CUSTOM_DIR, True),
    "gru": ("cmapps_rul_gru_model", "GRU_model.py", JOBS_DIR / "C-MAPSS_RUL_GRU" / "app" / "custom", False),
    "cnn1d": ("cmapps_rul_cnn1d_model", "CNN1D_model.py", JOBS_DIR / "C-MAPSS_RUL_1D-CNN" / "app" / "custom", False),
    "naive": ("cmapps_rul_naive_model", "NaiveRUL_model.py", JOBS_DIR / "C-MAPSS_RUL_Naive_Baseline" / "app" / "custom", False),
    "patchtst": ("cmapps_rul_patchtst_model", "PatchTST_model.py", JOBS_DIR / "C-MAPSS_RUL_PatchTST" / "app" / "custom", False),
    "dlinear": ("cmapps_rul_dlinear_model", "DLinear_model.py", JOBS_DIR / "C-MAPSS_RUL_DLinear" / "app" / "custom", False),
    "nbeats": ("cmapps_rul_nbeats_model", "NBEATS_model.py", JOBS_DIR / "C-MAPSS_RUL_NBeats" / "app" / "custom", False),
}


class _LazyModelModules:
    """Load an architecture module on first use.

    Building every module eagerly required --custom-dir to contain LSTM_model.py, which is
    false for any non-LSTM job folder. Only the architecture the checkpoint actually needs
    is imported now. The job's own copy of a file is preferred when --custom-dir ships it,
    so a seed variant uses its own architecture source; otherwise the original job folder
    supplies it. Both LSTM entries share a file name, so the FedAli variant is only taken
    from --custom-dir when that directory really is a FedAli job.
    """

    def __init__(self, specs):
        self._specs = specs
        self._cache = {}

    def path_for(self, key: str) -> Path:
        module_name, file_name, fallback_dir, is_fedali = self._specs[key]
        candidate = CUSTOM_DIR / file_name
        if candidate.exists() and CUSTOM_DIR_IS_FEDALI == is_fedali:
            return candidate
        return fallback_dir / file_name

    def __getitem__(self, key: str):
        if key not in self._cache:
            module_name = self._specs[key][0]
            self._cache[key] = load_model_module(module_name, self.path_for(key))
        return self._cache[key]


MODEL_MODULES = _LazyModelModules(_MODEL_MODULE_SPECS)


def load_checkpoint_from_job(job_id: str, checkpoint_name: str) -> dict[str, torch.Tensor]:
    workspace = PROJECT_ROOT / "Workspace" / "jobs-storage" / job_id / "workspace"
    member = f"app_server/{checkpoint_name}"
    if not workspace.exists():
        raise FileNotFoundError(f"Missing NVFlare job workspace archive or directory: {workspace}")

    if zipfile.is_zipfile(workspace):
        with zipfile.ZipFile(workspace) as zf:
            with zf.open(member) as src, tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
                tmp.write(src.read())
                tmp.flush()
                checkpoint = torch.load(tmp.name, map_location="cpu")
    else:
        checkpoint = torch.load(workspace / member, map_location="cpu")

    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    return dict(state_dict)


def build_model_from_state_dict(
    state_dict: dict[str, torch.Tensor],
    input_size: int,
    input_window: int,
) -> torch.nn.Module:
    if "alp.global_prototypes" in state_dict:
        module = MODEL_MODULES["fedali_lstm"]
    elif any(key.startswith("lstm.") for key in state_dict):
        module = MODEL_MODULES["lstm"]
    elif any(key.startswith("gru.") for key in state_dict):
        module = MODEL_MODULES["gru"]
    elif any(key.startswith("input_projection.") for key in state_dict):
        module = MODEL_MODULES["cnn1d"]
    elif any(key.startswith("patch_projection.") for key in state_dict):
        module = MODEL_MODULES["patchtst"]
    elif any(key.startswith(("decomposition.", "seasonal_linear.", "trend_linear.")) for key in state_dict):
        module = MODEL_MODULES["dlinear"]
    elif any(key.startswith("blocks.") for key in state_dict):
        module = MODEL_MODULES["nbeats"]
    elif set(state_dict.keys()) == {"bias"}:
        module = MODEL_MODULES["naive"]
    else:
        raise ValueError(f"Could not infer RUL model type from checkpoint keys: {list(state_dict)[:10]}")

    if module is MODEL_MODULES["fedali_lstm"]:
        prototypes = state_dict["alp.global_prototypes"]
        cfg = module.ModelConfig(
            input_size=input_size,
            hidden_size=int(prototypes.shape[1]),
            num_layers=2,
            dropout=0.2,
            num_prototypes=int(prototypes.shape[0]),
        )
    else:
        try:
            cfg = module.ModelConfig(
                input_size=input_size,
                input_window=input_window,
                hidden_size=128,
                num_layers=2,
                dropout=0.2,
            )
        except TypeError:
            cfg = module.ModelConfig(input_size=input_size, hidden_size=128, num_layers=2, dropout=0.2)
    return module.build_model(cfg)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    err = y_pred - y_true
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "mse": float(np.mean(err**2)),
        "bias": float(np.mean(err)),
        "nasa_score": nasa_score(y_true, y_pred),
    }


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    score = np.where(err < 0, np.exp(-err / 13.0) - 1.0, np.exp(err / 10.0) - 1.0)
    return float(np.sum(score))


def build_window_metadata(
    df: pd.DataFrame,
    window_size: int,
    target_offset: int,
    target_col: str,
) -> pd.DataFrame:
    rows = []
    for (unit, cycle), group in df.groupby(["unit", "cycle"], sort=False):
        group = group.reset_index(drop=True)
        num_windows = len(group) - window_size - target_offset + 1
        if num_windows <= 0:
            continue
        target_indices = np.arange(window_size - 1 + target_offset, window_size - 1 + target_offset + num_windows)
        rows.append(
            pd.DataFrame(
                {
                    "unit": int(unit),
                    "cycle": int(cycle),
                    "window_in_cycle": np.arange(num_windows, dtype=np.int64),
                    "target_row_in_cycle": target_indices.astype(np.int64),
                    "true_rul_from_frame": group.loc[target_indices, target_col].to_numpy(dtype=np.float32),
                }
            )
        )
    if not rows:
        raise ValueError("No window metadata was created. Check window_size and target_offset.")
    return pd.concat(rows, ignore_index=True)


@torch.no_grad()
def predict_segments(
    model: torch.nn.Module,
    segments: list[tuple[np.ndarray, np.ndarray]],
    input_scaler,
    target_scaler,
    window_size: int,
    target_offset: int,
    batch_size: int,
    device: torch.device,
    rul_cap: float,
) -> tuple[np.ndarray, np.ndarray]:
    preds = []
    trues = []
    model.eval()

    for x_src, y_src in segments:
        num_windows = len(x_src) - window_size - target_offset + 1
        if num_windows <= 0:
            continue

        for batch_start in range(0, num_windows, batch_size):
            batch_end = min(batch_start + batch_size, num_windows)
            current_size = batch_end - batch_start
            x_batch = np.empty((current_size, window_size, x_src.shape[1]), dtype=np.float32)
            y_batch = np.empty((current_size, 1), dtype=np.float32)

            for offset, local_idx in enumerate(range(batch_start, batch_end)):
                start_idx = local_idx
                end_idx = start_idx + window_size
                target_idx = end_idx - 1 + target_offset
                x_batch[offset] = x_src[start_idx:end_idx]
                y_batch[offset, 0] = y_src[target_idx]

            x_batch = input_scaler.transform(
                x_batch.reshape(-1, x_batch.shape[-1])
            ).reshape(current_size, window_size, x_src.shape[1]).astype(np.float32)

            pred_scaled = model(torch.from_numpy(x_batch).to(device)).detach().cpu().numpy()
            pred_raw = target_scaler.inverse_transform(pred_scaled).reshape(-1)
            pred_raw = np.clip(pred_raw, 0.0, rul_cap)
            preds.append(pred_raw)
            trues.append(y_batch.reshape(-1))

    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)


def write_single_row_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def save_cycle_outputs(out_dir: Path, prediction_df: pd.DataFrame) -> pd.DataFrame:
    cycle_df = (
        prediction_df.groupby(["unit", "cycle"], as_index=False)
        .agg(
            true_rul=("true_rul", "mean"),
            pred_rul_mean=("pred_rul", "mean"),
            pred_rul_median=("pred_rul", "median"),
            pred_rul_std=("pred_rul", "std"),
            windows=("pred_rul", "size"),
        )
    )
    cycle_df["error_mean"] = cycle_df["pred_rul_mean"] - cycle_df["true_rul"]
    cycle_df.to_csv(out_dir / "unseen_cycle_predictions.csv", index=False)

    rows = [
        {
            "unit": "overall_cycle_mean",
            "cycles": len(cycle_df),
            **regression_metrics(cycle_df["true_rul"], cycle_df["pred_rul_mean"]),
        }
    ]
    for unit, unit_df in cycle_df.groupby("unit"):
        rows.append(
            {
                "unit": int(unit),
                "cycles": len(unit_df),
                **regression_metrics(unit_df["true_rul"], unit_df["pred_rul_mean"]),
            }
        )
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(out_dir / "unseen_cycle_metrics.csv", index=False)
    return cycle_df


def plot_outputs(out_dir: Path, prediction_df: pd.DataFrame, cycle_df: pd.DataFrame, max_points: int = 20000) -> None:
    y_true = prediction_df["true_rul"].to_numpy()
    y_pred = prediction_df["pred_rul"].to_numpy()
    rng = np.random.default_rng(42)
    idx = rng.choice(len(y_true), size=min(max_points, len(y_true)), replace=False)

    plt.figure(figsize=(7, 7))
    plt.scatter(y_true[idx], y_pred[idx], s=5, alpha=0.25)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    plt.plot([lo, hi], [lo, hi], "r--", linewidth=1.2)
    plt.xlabel("True RUL")
    plt.ylabel("Predicted RUL")
    plt.title("Window-level predicted vs true RUL")
    plt.tight_layout()
    plt.savefig(out_dir / "window_predicted_vs_true.png", dpi=200)
    plt.close()

    cycle_true = cycle_df["true_rul"].to_numpy()
    cycle_pred = cycle_df["pred_rul_mean"].to_numpy()
    plt.figure(figsize=(7, 7))
    for unit in sorted(cycle_df["unit"].unique()):
        unit_df = cycle_df[cycle_df["unit"] == unit]
        plt.scatter(
            unit_df["true_rul"],
            unit_df["pred_rul_mean"],
            s=35,
            alpha=0.8,
            label=f"unit {unit}",
        )
    lo = float(min(cycle_true.min(), cycle_pred.min()))
    hi = float(max(cycle_true.max(), cycle_pred.max()))
    plt.plot([lo, hi], [lo, hi], "r--", linewidth=1.2)
    plt.xlabel("True cycle RUL")
    plt.ylabel("Predicted cycle RUL, mean over windows")
    plt.title("Cycle-level predicted vs true RUL")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "cycle_predicted_vs_true.png", dpi=200)
    plt.close()

    error = y_pred - y_true
    plt.figure(figsize=(9, 4.5))
    plt.hist(error, bins=80, alpha=0.85)
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Prediction error, predicted - true")
    plt.ylabel("Window count")
    plt.title("RUL Prediction Error Distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "window_error_distribution.png", dpi=200)
    plt.close()

    units = sorted(cycle_df["unit"].unique())
    fig, axes = plt.subplots(len(units), 1, figsize=(12, 3.5 * len(units)), sharex=False)
    if len(units) == 1:
        axes = [axes]
    for ax, unit in zip(axes, units):
        unit_df = cycle_df[cycle_df["unit"] == unit].sort_values("cycle")
        ax.plot(unit_df["cycle"], unit_df["true_rul"], marker="o", linewidth=1.8, label="True RUL")
        ax.plot(unit_df["cycle"], unit_df["pred_rul_mean"], marker="o", linewidth=1.5, label="Predicted RUL")
        ax.fill_between(
            unit_df["cycle"].to_numpy(),
            (unit_df["pred_rul_mean"] - unit_df["pred_rul_std"].fillna(0.0)).to_numpy(),
            (unit_df["pred_rul_mean"] + unit_df["pred_rul_std"].fillna(0.0)).to_numpy(),
            alpha=0.15,
            label="Prediction std across windows",
        )
        ax.set_title(f"Unit {unit}: RUL prediction by cycle")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("RUL")
        ax.grid(alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "cycle_true_vs_pred_by_unit.png", dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an FL RUL global model on unseen C-MAPSS units.")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--checkpoint", default="FL_global_model.pt", choices=["FL_global_model.pt", "best_FL_global_model.pt"])
    parser.add_argument("--test-npz", default=str(PROJECT_ROOT / "Workspace" / "data" / "test_units_11_14_15.npz"))
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--target-offset", type=int, default=0)
    parser.add_argument("--rul-cap", type=float, default=float(DEFAULT_RUL_CAP))
    parser.add_argument("--out-dir", default=None, help="Raw per-run output directory (default: <results-dir>/unseen_engines/raw/<model-name>/<checkpoint>).")
    parser.add_argument("--custom-dir", default=str(DEFAULT_CUSTOM_DIR), help="Job custom/ dir providing data_loader.py, train.py, LSTM_model.py.")
    parser.add_argument(
        "--site-data-root",
        default=str(PROJECT_ROOT / "Workspace" / "example_project" / "prod_00"),
        help="path to NVFlare provisioned workspace with site-N dirs",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for train/val/test split reconstruction")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument(
        "--results-dir",
        default=str(PROJECT_ROOT / "Workspace" / "fl_artifacts" / "unseen_eval"),
        help="directory where all results are saved",
    )
    parser.add_argument("--model-name", default=None, help="Model label used for result file names (default: job id).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_npz = Path(args.test_npz)
    model_name = args.model_name or args.job_id
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir / "unseen_engines" / "raw" / model_name / Path(args.checkpoint).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = LoaderConfig(
        target_col="RUL_capped",
        window_size=args.window_size,
        target_offset=args.target_offset,
        batch_size=args.batch_size,
        feature_cols=CENTRALIZED_FEATURE_COLS,
    )

    df = load_npz_frame(test_npz)
    units = sorted(int(u) for u in df[cfg.unit_col].dropna().unique().tolist())
    segments = build_segments_from_dataframe(
        df=df,
        feature_cols=cfg.feature_cols,
        target_col=cfg.target_col,
        unit_col=cfg.unit_col,
        cycle_col=cfg.cycle_col,
    )
    # Scalers are reconstructed from the six training clients only.
    input_scaler, target_scaler, scaler_manifest = build_pooled_reference_scalers(
        site_data_root=args.site_data_root,
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        window_size=cfg.window_size,
        target_offset=cfg.target_offset,
        feature_cols=cfg.feature_cols,
        rul_cap=args.rul_cap,
    )
    # CHECK 1: the pooled scalers were built exclusively from clients in SITE_UNIT_MAP.
    scaler_units = sorted(c["unit"] for c in scaler_manifest["clients"].values())
    assert not set(scaler_units) & set(units), f"Scaler built on unseen units: {set(scaler_units) & set(units)}"
    assert scaler_units == sorted(SITE_UNIT_MAP.values()), scaler_units
    print("SCALER INTEGRITY: PASS — no test data used in scaler fit")

    # One shared manifest at the results root plus a copy next to the raw run.
    results_dir.mkdir(parents=True, exist_ok=True)
    manifest_text = json.dumps(scaler_manifest, indent=2) + "\n"
    (results_dir / "scaler_manifest.json").write_text(manifest_text)
    (out_dir / "scaler_manifest.json").write_text(manifest_text)

    num_windows = count_windows_in_segments(segments, cfg.window_size, cfg.target_offset)
    metadata_df = build_window_metadata(df, cfg.window_size, cfg.target_offset, cfg.target_col)

    state_dict = load_checkpoint_from_job(args.job_id, args.checkpoint)
    model = build_model_from_state_dict(state_dict, input_size=len(cfg.feature_cols), input_window=cfg.window_size)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    num_parameters = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

    preds, trues = predict_segments(
        model=model,
        segments=segments,
        input_scaler=input_scaler,
        target_scaler=target_scaler,
        window_size=cfg.window_size,
        target_offset=cfg.target_offset,
        batch_size=args.batch_size,
        device=device,
        rul_cap=args.rul_cap,
    )
    if len(metadata_df) != len(preds):
        raise RuntimeError(f"Metadata/prediction length mismatch: {len(metadata_df)} vs {len(preds)}")

    prediction_df = metadata_df.copy()
    prediction_df["true_rul"] = trues
    prediction_df["pred_rul"] = preds
    prediction_df["error"] = prediction_df["pred_rul"] - prediction_df["true_rul"]
    prediction_df.to_csv(out_dir / "unseen_window_predictions.csv", index=False)

    cycle_df = save_cycle_outputs(out_dir, prediction_df)
    window_metrics = regression_metrics(trues, preds)
    metrics = {
        "job_id": args.job_id,
        "checkpoint": args.checkpoint,
        "test_npz": str(test_npz),
        "units": " ".join(str(u) for u in units),
        "window_size": cfg.window_size,
        "target_offset": cfg.target_offset,
        "rul_cap": args.rul_cap,
        "num_windows": num_windows,
        **window_metrics,
    }
    write_single_row_csv(out_dir / "unseen_window_metrics.csv", metrics)
    plot_outputs(out_dir, prediction_df, cycle_df)

    cycle_metrics = pd.read_csv(out_dir / "unseen_cycle_metrics.csv")
    overall_cycle = cycle_metrics.iloc[0].to_dict()

    # CHECK 2: cycle metrics must equal metrics over per-(unit, cycle) mean predictions.
    recomputed = prediction_df.groupby(["unit", "cycle"], as_index=False).agg(
        true_rul=("true_rul", "mean"), pred_rul=("pred_rul", "mean")
    )
    recomputed_metrics = regression_metrics(recomputed["true_rul"], recomputed["pred_rul"])
    assert len(recomputed) == int(overall_cycle["cycles"]), "cycle count mismatch"
    for key in ("rmse", "mae", "nasa_score"):
        assert np.isclose(recomputed_metrics[key], float(overall_cycle[key]), rtol=1e-9, atol=1e-9), key
    print("AGGREGATION ORDER: PASS")

    # CHECK 3: metrics must be in RUL cycle units.
    cycle_rmse = float(overall_cycle["rmse"])
    if not (0.1 <= cycle_rmse <= 100.0) or not (0.1 <= window_metrics["rmse"] <= 100.0):
        raise RuntimeError(
            f"UNITS CHECK: FAIL — cycle RMSE={cycle_rmse:.3f}, window RMSE={window_metrics['rmse']:.3f}; "
            "inverse transform is suspect"
        )
    print(f"UNITS CHECK: PASS — RMSE={cycle_rmse:.3f} cycles")

    # Results-tree copies used by the summary tables (STEP 4).
    window_level_dir = results_dir / "unseen_engines" / "window_level"
    cycle_level_dir = results_dir / "unseen_engines" / "cycle_level"
    window_level_dir.mkdir(parents=True, exist_ok=True)
    cycle_level_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.checkpoint == "best_FL_global_model.pt" else "__FL_global_model"
    window_row = {"model": model_name, "parameters": num_parameters, **metrics}
    write_single_row_csv(window_level_dir / f"{model_name}{suffix}_window_metrics.csv", window_row)
    cycle_out = cycle_metrics.copy()
    cycle_out.insert(0, "model", model_name)
    cycle_out.insert(1, "parameters", num_parameters)
    cycle_out.insert(2, "job_id", args.job_id)
    cycle_out.insert(3, "checkpoint", args.checkpoint)
    cycle_out.to_csv(cycle_level_dir / f"{model_name}{suffix}_cycle_metrics.csv", index=False)

    print(f"Saved RUL evaluation outputs to: {out_dir}")
    print(
        f"[{model_name} | {args.checkpoint} | params={num_parameters}] "
        f"Window RMSE={window_metrics['rmse']:.6f}, MAE={window_metrics['mae']:.6f}; "
        f"Cycle RMSE={cycle_rmse:.6f}, MAE={float(overall_cycle['mae']):.6f}, "
        f"NASA={float(overall_cycle['nasa_score']):.6f}; windows={num_windows}"
    )


if __name__ == "__main__":
    main()
