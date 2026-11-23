#!/usr/bin/env python3
"""Evaluate an FL RUL global model on each client's local held-out data.

This does not recover per-client local model weights. NVFlare usually does not
persist those by default. Instead, this script answers a useful comparison
question: how does the final global model perform on each client's local engine
data using the same cycle-based held-out split as train.py?
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
import tempfile
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CUSTOM_DIR = PROJECT_ROOT / "federated_jobs" / "C-MAPSS_RUL_LSTM_FedAvg" / "app" / "custom"
FEDALI_CUSTOM_DIR = PROJECT_ROOT / "federated_jobs" / "C-MAPSS_RUL_LSTM_FedAli" / "app" / "custom"
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


def load_model_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


MODEL_MODULES = {
    "lstm": load_model_module("cmapps_rul_lstm_model_local_eval", CUSTOM_DIR / "LSTM_model.py"),
    "fedali_lstm": load_model_module(
        "cmapps_rul_fedali_lstm_model_local_eval", FEDALI_CUSTOM_DIR / "LSTM_model.py"
    ),
    "gru": load_model_module(
        "cmapps_rul_gru_model_local_eval",
        PROJECT_ROOT / "federated_jobs" / "C-MAPSS_RUL_GRU" / "app" / "custom" / "GRU_model.py",
    ),
    "cnn1d": load_model_module(
        "cmapps_rul_cnn1d_model_local_eval",
        PROJECT_ROOT / "federated_jobs" / "C-MAPSS_RUL_1D-CNN" / "app" / "custom" / "CNN1D_model.py",
    ),
    "naive": load_model_module(
        "cmapps_rul_naive_model_local_eval",
        PROJECT_ROOT / "federated_jobs" / "C-MAPSS_RUL_Naive_Baseline" / "app" / "custom" / "NaiveRUL_model.py",
    ),
    "patchtst": load_model_module(
        "cmapps_rul_patchtst_model_local_eval",
        PROJECT_ROOT / "federated_jobs" / "C-MAPSS_RUL_PatchTST" / "app" / "custom" / "PatchTST_model.py",
    ),
    "dlinear": load_model_module(
        "cmapps_rul_dlinear_model_local_eval",
        PROJECT_ROOT / "federated_jobs" / "C-MAPSS_RUL_DLinear" / "app" / "custom" / "DLinear_model.py",
    ),
    "nbeats": load_model_module(
        "cmapps_rul_nbeats_model_local_eval",
        PROJECT_ROOT / "federated_jobs" / "C-MAPSS_RUL_NBeats" / "app" / "custom" / "NBEATS_model.py",
    ),
}


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


def build_model(state_dict: dict[str, torch.Tensor], input_size: int, input_window: int) -> torch.nn.Module:
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
    model = module.build_model(cfg)
    model.load_state_dict(state_dict, strict=True)
    return model


def split_groups_for_validation(df: pd.DataFrame, val_fraction: float, test_fraction: float, seed: int):
    units = sorted(int(u) for u in df["unit"].dropna().unique().tolist())
    split_name = "unit"
    groups = units
    if len(groups) < 3 and "cycle" in df.columns:
        cycles = sorted(int(c) for c in df["cycle"].dropna().unique().tolist())
        if len(cycles) >= 3:
            split_name = "cycle"
            groups = cycles

    rng = np.random.default_rng(seed)
    shuffled = list(groups)
    rng.shuffle(shuffled)
    if len(shuffled) < 3:
        return split_name, sorted(shuffled), [], []

    test_count = max(1, int(round(len(shuffled) * test_fraction))) if test_fraction > 0 else 0
    val_count = max(1, int(round(len(shuffled) * val_fraction))) if val_fraction > 0 else 0
    if test_count + val_count >= len(shuffled):
        test_count = 1
        val_count = 1

    # Match train.py exactly: reserve the test block first, followed by the
    # validation block. This ensures evaluation reconstructs the same groups
    # that were held out during federated training.
    test_groups = sorted(shuffled[:test_count])
    val_groups = sorted(shuffled[test_count : test_count + val_count])
    train_groups = sorted(shuffled[test_count + val_count :])
    return split_name, train_groups, val_groups, test_groups


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
            preds.append(np.clip(pred_raw, 0.0, rul_cap))
            trues.append(y_batch.reshape(-1))
    if not preds:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return np.concatenate(preds), np.concatenate(trues)


def cycle_metrics_from_df(df: pd.DataFrame) -> dict[str, float | int]:
    if df.empty:
        return {"cycle_rmse": "", "cycle_mae": "", "cycle_mse": "", "cycle_bias": "", "cycle_nasa_score": "", "cycles": 0}
    cycle_df = (
        df.groupby(["unit", "cycle"], as_index=False)
        .agg(true_rul=("true_rul", "mean"), pred_rul=("pred_rul", "mean"), windows=("pred_rul", "size"))
    )
    metrics = regression_metrics(cycle_df["true_rul"], cycle_df["pred_rul"])
    return {
        "cycle_rmse": metrics["rmse"],
        "cycle_mae": metrics["mae"],
        "cycle_mse": metrics["mse"],
        "cycle_bias": metrics["bias"],
        "cycle_nasa_score": metrics["nasa_score"],
        "cycles": len(cycle_df),
    }


def build_eval_predictions_df(
    source_df: pd.DataFrame,
    preds: np.ndarray,
    trues: np.ndarray,
    window_size: int,
    target_offset: int,
) -> pd.DataFrame:
    rows = []
    cursor = 0
    for (unit, cycle), group in source_df.groupby(["unit", "cycle"], sort=False):
        num_windows = len(group) - window_size - target_offset + 1
        if num_windows <= 0:
            continue
        rows.append(
            pd.DataFrame(
                {
                    "unit": int(unit),
                    "cycle": int(cycle),
                    "window_in_cycle": np.arange(num_windows, dtype=np.int64),
                    "true_rul": trues[cursor : cursor + num_windows],
                    "pred_rul": preds[cursor : cursor + num_windows],
                }
            )
        )
        cursor += num_windows
    if not rows:
        return pd.DataFrame(columns=["unit", "cycle", "window_in_cycle", "true_rul", "pred_rul", "error"])
    out = pd.concat(rows, ignore_index=True)
    out["error"] = out["pred_rul"] - out["true_rul"]
    return out


def find_client_npz_files(site_data_root: Path) -> list[tuple[str, Path]]:
    rows = []
    for site_dir in sorted(site_data_root.glob("site-*")):
        data_dir = site_dir / "local" / "data"
        if not data_dir.exists():
            continue
        for path in sorted(data_dir.glob("*.npz")):
            rows.append((site_dir.name, path))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a global RUL model on each client's local held-out data.")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--checkpoint", default="FL_global_model.pt", choices=["FL_global_model.pt", "best_FL_global_model.pt"])
    parser.add_argument(
        "--site-data-root",
        default=str(PROJECT_ROOT / "Workspace" / "example_project" / "prod_00"),
        help="Directory containing site-*/local/data/*.npz.",
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--target-offset", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rul-cap", type=float, default=float(DEFAULT_RUL_CAP))
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Evaluate only the held-out test split and skip validation inference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else PROJECT_ROOT / "Workspace" / "fl_artifacts" / "rul_local_eval" / args.job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = out_dir / "client_predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = load_checkpoint_from_job(args.job_id, args.checkpoint)
    model = build_model(state_dict, input_size=len(CENTRALIZED_FEATURE_COLS), input_window=args.window_size).to(device)
    cfg = LoaderConfig(
        target_col="RUL_capped",
        window_size=args.window_size,
        target_offset=args.target_offset,
        batch_size=args.batch_size,
        feature_cols=CENTRALIZED_FEATURE_COLS,
    )

    rows = []
    client_files = find_client_npz_files(Path(args.site_data_root))
    if not client_files:
        raise FileNotFoundError(f"No site local NPZ files found under {args.site_data_root}")

    for site_name, path in client_files:
        df = load_npz_frame(path)
        split_name, train_groups, val_groups, test_groups = split_groups_for_validation(
            df,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
        )
        if split_name == "cycle":
            train_df, val_df, test_df = split_dataframe_by_cycles(df, train_groups, val_groups, test_groups)
        else:
            train_df, val_df, test_df = split_dataframe_by_units(df, train_groups, val_groups, test_groups)

        train_segments = build_segments_from_dataframe(train_df, cfg.feature_cols, cfg.target_col)
        input_scaler, target_scaler = fit_scalers_on_segments(train_segments, cfg.window_size, cfg.target_offset)

        eval_splits = [("test", test_df)] if args.test_only else [
            ("validation", val_df),
            ("test", test_df),
        ]
        for split_label, split_df in eval_splits:
            if split_df is None or split_df.empty:
                continue
            segments = build_segments_from_dataframe(split_df, cfg.feature_cols, cfg.target_col)
            windows = count_windows_in_segments(segments, cfg.window_size, cfg.target_offset)
            preds, trues = predict_segments(
                model=model,
                segments=segments,
                input_scaler=input_scaler,
                target_scaler=target_scaler,
                window_size=cfg.window_size,
                target_offset=cfg.target_offset,
                batch_size=cfg.batch_size,
                device=device,
                rul_cap=args.rul_cap,
            )
            if len(preds) == 0:
                continue
            pred_df = build_eval_predictions_df(split_df, preds, trues, cfg.window_size, cfg.target_offset)
            pred_df.to_csv(predictions_dir / f"{site_name}_{path.stem}_{split_label}_predictions.csv", index=False)
            window_metrics = regression_metrics(trues, preds)
            cycle_metrics = cycle_metrics_from_df(pred_df)
            rows.append(
                {
                    "job_id": args.job_id,
                    "checkpoint": args.checkpoint,
                    "site": site_name,
                    "data_file": str(path),
                    "unit": " ".join(str(int(u)) for u in sorted(df["unit"].unique())),
                    "split": split_label,
                    "split_by": split_name,
                    "train_groups": " ".join(map(str, train_groups)),
                    "eval_groups": " ".join(map(str, val_groups if split_label == "validation" else test_groups)),
                    "windows": windows,
                    "window_rmse": window_metrics["rmse"],
                    "window_mae": window_metrics["mae"],
                    "window_mse": window_metrics["mse"],
                    "window_bias": window_metrics["bias"],
                    "window_nasa_score": window_metrics["nasa_score"],
                    **cycle_metrics,
                }
            )

    metrics_df = pd.DataFrame(rows)
    metrics_path = out_dir / "local_client_rul_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    if not metrics_df.empty:
        test_df = metrics_df[metrics_df["split"] == "test"].copy()
        if not test_df.empty:
            plt.figure(figsize=(9, 4.5))
            labels = test_df["site"] + " / " + test_df["unit"]
            plt.bar(labels, test_df["cycle_rmse"].astype(float))
            plt.xticks(rotation=35, ha="right")
            plt.ylabel("Cycle-level RUL RMSE")
            plt.title("Final Global Model on Local Client Test Splits")
            plt.tight_layout()
            plt.savefig(out_dir / "local_client_test_cycle_rmse.png", dpi=200)
            plt.close()

    print(f"Saved local client RUL evaluation outputs to: {out_dir}")
    print(metrics_df)


if __name__ == "__main__":
    main()
