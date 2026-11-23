#!/usr/bin/env python3
"""Evaluate FedAli global performance, personalization, and generalization.

The script uses each client's training-only scaler. For cross-client and unseen
evaluation, the source client's scaler is used because it is part of that
personalized model's input/output contract.
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import math
import re
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEDALI_CUSTOM_DIR = PROJECT_ROOT / "federated_jobs" / "C-MAPSS_RUL_LSTM_FedAli" / "app" / "custom"
sys.path.insert(0, str(FEDALI_CUSTOM_DIR))

from data_loader import (  # noqa: E402
    CENTRALIZED_FEATURE_COLS,
    DEFAULT_RUL_CAP,
    build_segments_from_dataframe,
    fit_scalers_on_segments,
    load_npz_frame,
    split_dataframe_by_cycles,
    split_dataframe_by_units,
)


@dataclass
class ArrayScaler:
    mean_: np.ndarray
    scale_: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values) - self.mean_) / self.scale_

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values) * self.scale_ + self.mean_


@dataclass
class ClientData:
    site: str
    path: Path
    test_df: pd.DataFrame
    test_segments: list[tuple[np.ndarray, np.ndarray]]
    input_scaler: Any
    target_scaler: Any
    feature_cols: list[str]
    test_groups: list[int]
    split_by: str


def torch_load(path_or_file) -> Any:
    try:
        return torch.load(path_or_file, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path_or_file, map_location="cpu")


def job_workspace(job_id: str) -> Path:
    path = PROJECT_ROOT / "Workspace" / "jobs-storage" / job_id / "workspace"
    if not path.exists():
        raise FileNotFoundError(f"Missing NVFlare job workspace: {path}")
    return path


def read_job_member(job_id: str, member: str) -> bytes:
    workspace = job_workspace(job_id)
    if zipfile.is_zipfile(workspace):
        with zipfile.ZipFile(workspace) as archive:
            return archive.read(member)
    path = workspace / member
    if not path.exists():
        raise FileNotFoundError(f"Missing job member: {path}")
    return path.read_bytes()


def load_job_model_module(job_id: str):
    source = read_job_member(job_id, "app_server/custom/LSTM_model.py")
    with tempfile.TemporaryDirectory(prefix="fedali_model_") as temp_dir:
        path = Path(temp_dir) / "LSTM_model.py"
        path.write_bytes(source)
        module_name = f"fedali_job_model_{job_id.replace('-', '_')}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not import archived FedAli model from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module


def load_job_environment(job_id: str) -> dict[str, str]:
    try:
        return json.loads(read_job_member(job_id, "app_server/environments.json").decode("utf-8"))
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return {}


def load_global_state(job_id: str, checkpoint_name: str) -> dict[str, torch.Tensor]:
    member = f"app_server/{checkpoint_name}"
    workspace = job_workspace(job_id)
    if zipfile.is_zipfile(workspace):
        with zipfile.ZipFile(workspace) as archive:
            checkpoint = torch_load(io.BytesIO(archive.read(member)))
    else:
        checkpoint = torch_load(workspace / member)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    return dict(state)


def infer_num_layers(state: dict[str, torch.Tensor]) -> int:
    indices = []
    for key in state:
        match = re.fullmatch(r"lstm\.weight_ih_l(\d+)", key)
        if match:
            indices.append(int(match.group(1)))
    return max(indices, default=0) + 1


def build_model(module, state: dict[str, torch.Tensor], environment: dict[str, str]) -> torch.nn.Module:
    prototype_key = "alp.global_prototypes"
    if prototype_key not in state:
        raise ValueError("Checkpoint is not a FedAli model: missing alp.global_prototypes")
    prototypes = state[prototype_key]
    input_size = int(state["lstm.weight_ih_l0"].shape[1])
    config = module.ModelConfig(
        input_size=input_size,
        hidden_size=int(prototypes.shape[1]),
        num_layers=infer_num_layers(state),
        dropout=0.2,
        num_prototypes=int(prototypes.shape[0]),
        beta=float(environment.get("FEDALI_BETA", 0.2)),
        gamma=float(environment.get("FEDALI_GAMMA", 0.999)),
        epsilon=float(environment.get("FEDALI_EPSILON", 0.05)),
        sinkhorn_iterations=int(environment.get("FEDALI_SINKHORN_ITERATIONS", 3)),
    )
    model = module.build_model(config)
    model.load_state_dict(state, strict=True)
    return model


def scaler_from_state(state: dict[str, Any]) -> ArrayScaler:
    mean = np.asarray(state["mean"], dtype=np.float64)
    scale = np.asarray(state["scale"], dtype=np.float64)
    return ArrayScaler(mean_=mean, scale_=np.where(scale == 0.0, 1.0, scale))


def split_groups(
    df: pd.DataFrame,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[str, list[int], list[int], list[int]]:
    """Match train.py exactly: test block first, then validation block."""
    units = sorted(int(value) for value in df["unit"].dropna().unique())
    split_by = "unit"
    groups = units
    if len(units) < 3 and "cycle" in df.columns:
        cycles = sorted(int(value) for value in df["cycle"].dropna().unique())
        if len(cycles) >= 3:
            split_by = "cycle"
            groups = cycles
    shuffled = list(groups)
    np.random.default_rng(seed).shuffle(shuffled)
    if len(shuffled) < 3:
        return split_by, sorted(shuffled), [], []
    num_test = max(1, int(round(len(shuffled) * test_fraction))) if test_fraction > 0 else 0
    num_val = max(1, int(round(len(shuffled) * val_fraction))) if val_fraction > 0 else 0
    if num_test + num_val >= len(shuffled):
        num_test = num_val = 1
    test_groups = sorted(shuffled[:num_test])
    val_groups = sorted(shuffled[num_test : num_test + num_val])
    train_groups = sorted(shuffled[num_test + num_val :])
    return split_by, train_groups, val_groups, test_groups


def split_frames(
    df: pd.DataFrame,
    split_by: str,
    train_groups: list[int],
    val_groups: list[int],
    test_groups: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if split_by == "cycle":
        return split_dataframe_by_cycles(df, train_groups, val_groups, test_groups)
    return split_dataframe_by_units(df, train_groups, val_groups, test_groups)


def find_client_files(site_data_root: Path) -> dict[str, Path]:
    result = {}
    for site_dir in sorted(site_data_root.glob("site-*")):
        files = sorted((site_dir / "local" / "data").glob("*.npz"))
        if files:
            result[site_dir.name] = files[0]
    if not result:
        raise FileNotFoundError(f"No site-*/local/data/*.npz files found under {site_data_root}")
    return result


def resolve_state_path(site_dir: Path, state_name: str, round_number: Optional[int]) -> Path:
    if round_number is not None:
        return site_dir / f"round_{round_number:04d}.pt"
    return site_dir / f"{state_name}.pt"


def find_site_workspace_state_dirs(
    site_data_root: Path,
    job_id: str,
    sites: list[str],
) -> dict[str, Path]:
    """Find states written inside NVFlare's per-site job workspaces."""
    result = {}
    for site in sites:
        job_dir = site_data_root / site / job_id
        candidates = sorted(
            job_dir.glob(f"app_*/fl_artifacts/personalized/{job_id}/{site}")
        )
        if candidates:
            result[site] = candidates[0]
    return result


def load_personalized_states(
    state_root: Path,
    sites: list[str],
    state_name: str,
    round_number: Optional[int],
    site_workspace_dirs: Optional[dict[str, Path]] = None,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    states = {}
    missing = []
    site_workspace_dirs = site_workspace_dirs or {}
    for site in sites:
        candidates = [resolve_state_path(state_root / site, state_name, round_number)]
        workspace_dir = site_workspace_dirs.get(site)
        if workspace_dir is not None:
            candidates.append(resolve_state_path(workspace_dir, state_name, round_number))
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            missing.append(site)
            continue
        payload = torch_load(path)
        payload["_path"] = str(path)
        states[site] = payload
    return states, missing


def prepare_clients(
    files: dict[str, Path],
    personalized_states: dict[str, dict[str, Any]],
    window_size: int,
    target_offset: int,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, ClientData]:
    clients = {}
    for site, path in files.items():
        frame = load_npz_frame(path)
        saved = personalized_states.get(site)
        if saved:
            split_by = str(saved["split_by"])
            train_groups = [int(value) for value in saved["train_groups"]]
            val_groups = [int(value) for value in saved["val_groups"]]
            test_groups = [int(value) for value in saved["test_groups"]]
            feature_cols = list(saved["feature_cols"])
            input_scaler = scaler_from_state(saved["input_scaler"])
            target_scaler = scaler_from_state(saved["target_scaler"])
        else:
            split_by, train_groups, val_groups, test_groups = split_groups(
                frame, val_fraction, test_fraction, seed
            )
            feature_cols = list(CENTRALIZED_FEATURE_COLS)
            train_df, _, _ = split_frames(frame, split_by, train_groups, val_groups, test_groups)
            train_segments = build_segments_from_dataframe(train_df, feature_cols, "RUL_capped")
            input_scaler, target_scaler = fit_scalers_on_segments(
                train_segments, window_size, target_offset
            )
        _, _, test_df = split_frames(frame, split_by, train_groups, val_groups, test_groups)
        if test_df is None or test_df.empty:
            raise ValueError(f"Client {site} has no held-out test data")
        test_segments = build_segments_from_dataframe(test_df, feature_cols, "RUL_capped")
        clients[site] = ClientData(
            site=site,
            path=path,
            test_df=test_df,
            test_segments=test_segments,
            input_scaler=input_scaler,
            target_scaler=target_scaler,
            feature_cols=feature_cols,
            test_groups=test_groups,
            split_by=split_by,
        )
    return clients


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    error = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    return float(np.sum(np.where(error < 0, np.exp(-error / 13.0) - 1.0, np.exp(error / 10.0) - 1.0)))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    error = pred - true
    mse = float(np.mean(error**2))
    return {
        "rmse": math.sqrt(mse),
        "mae": float(np.mean(np.abs(error))),
        "mse": mse,
        "bias": float(np.mean(error)),
        "nasa_score": nasa_score(true, pred),
    }


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
    predictions = []
    truths = []
    model.eval()
    for x_source, y_source in segments:
        num_windows = len(x_source) - window_size - target_offset + 1
        for start in range(0, max(num_windows, 0), batch_size):
            stop = min(start + batch_size, num_windows)
            size = stop - start
            x_batch = np.empty((size, window_size, x_source.shape[1]), dtype=np.float32)
            y_batch = np.empty((size, 1), dtype=np.float32)
            for offset, local_index in enumerate(range(start, stop)):
                end = local_index + window_size
                target_index = end - 1 + target_offset
                x_batch[offset] = x_source[local_index:end]
                y_batch[offset, 0] = y_source[target_index]
            transformed = input_scaler.transform(x_batch.reshape(-1, x_batch.shape[-1]))
            transformed = transformed.reshape(x_batch.shape).astype(np.float32)
            pred_scaled = model(torch.from_numpy(transformed).to(device)).cpu().numpy()
            pred_raw = target_scaler.inverse_transform(pred_scaled).reshape(-1)
            predictions.append(np.clip(pred_raw, 0.0, rul_cap))
            truths.append(y_batch.reshape(-1))
    if not predictions:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
    return np.concatenate(predictions), np.concatenate(truths)


def prediction_frame(
    source_df: pd.DataFrame,
    predictions: np.ndarray,
    truths: np.ndarray,
    window_size: int,
    target_offset: int,
) -> pd.DataFrame:
    rows = []
    cursor = 0
    for (unit, cycle), group in source_df.groupby(["unit", "cycle"], sort=False):
        count = max(len(group) - window_size - target_offset + 1, 0)
        if count == 0:
            continue
        rows.append(pd.DataFrame({
            "unit": int(unit),
            "cycle": int(cycle),
            "true_rul": truths[cursor : cursor + count],
            "pred_rul": predictions[cursor : cursor + count],
        }))
        cursor += count
    if cursor != len(predictions):
        raise RuntimeError(f"Prediction metadata mismatch: consumed {cursor}, received {len(predictions)}")
    return pd.concat(rows, ignore_index=True)


def evaluate(
    model: torch.nn.Module,
    data_frame: pd.DataFrame,
    segments: list[tuple[np.ndarray, np.ndarray]],
    input_scaler,
    target_scaler,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    predictions, truths = predict_segments(
        model, segments, input_scaler, target_scaler, args.window_size,
        args.target_offset, args.batch_size, device, args.rul_cap,
    )
    window_metrics = regression_metrics(truths, predictions)
    frame = prediction_frame(data_frame, predictions, truths, args.window_size, args.target_offset)
    cycles = frame.groupby(["unit", "cycle"], as_index=False).agg(
        true_rul=("true_rul", "mean"), pred_rul=("pred_rul", "mean")
    )
    cycle_metrics = regression_metrics(cycles["true_rul"], cycles["pred_rul"])
    metrics: dict[str, float | int] = {"windows": len(frame), "cycles": len(cycles)}
    metrics.update({f"window_{key}": value for key, value in window_metrics.items()})
    metrics.update({f"cycle_{key}": value for key, value in cycle_metrics.items()})
    return metrics, frame


def aggregate_rows(rows: list[dict[str, Any]], perspective: str) -> dict[str, Any]:
    if not rows:
        return {"perspective": perspective, "evaluations": 0}
    frame = pd.DataFrame(rows)
    return {
        "perspective": perspective,
        "evaluations": len(frame),
        "window_rmse_macro": float(frame.window_rmse.mean()),
        "window_mae_macro": float(frame.window_mae.mean()),
        "cycle_rmse_macro": float(frame.cycle_rmse.mean()),
        "cycle_mae_macro": float(frame.cycle_mae.mean()),
        "cycle_nasa_score_macro": float(frame.cycle_nasa_score.mean()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--checkpoint", default="FL_global_model.pt", choices=["FL_global_model.pt", "best_FL_global_model.pt"])
    parser.add_argument("--site-data-root", default=str(PROJECT_ROOT / "Workspace" / "example_project" / "prod_00"))
    parser.add_argument("--state-root", default=None, help="Directory containing one personalized-state directory per site.")
    parser.add_argument("--state", choices=["latest", "best"], default="latest")
    parser.add_argument("--round", type=int, default=None, help="Use the same saved communication round for every client.")
    parser.add_argument("--unseen-npz", default=None, help="Optional raw unseen-unit NPZ for personalized generalization.")
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--target-offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--rul-cap", type=float, default=float(DEFAULT_RUL_CAP))
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--require-personalized", action="store_true")
    parser.add_argument(
        "--local-only",
        action="store_true",
        help=(
            "Evaluate the global model and each personalized model on its own client only; "
            "skip cross-client and unseen-unit generalization."
        ),
    )
    parser.add_argument(
        "--global-only",
        action="store_true",
        help="Evaluate only the global model on each client's local test split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else (
        PROJECT_ROOT / "Workspace" / "fl_artifacts" / "fedali_perspectives" / args.job_id
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = out_dir / "predictions"
    if args.save_predictions:
        prediction_dir.mkdir(parents=True, exist_ok=True)

    module = load_job_model_module(args.job_id)
    environment = load_job_environment(args.job_id)
    global_state = load_global_state(args.job_id, args.checkpoint)
    global_model = build_model(module, global_state, environment)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_model.to(device)

    site_data_root = Path(args.site_data_root)
    files = find_client_files(site_data_root)
    default_state_root = (
        PROJECT_ROOT / "Workspace" / "fl_artifacts" / "C-MAPSS_RUL_LSTM_FedAli"
        / "personalized" / args.job_id
    )
    state_root = Path(args.state_root) if args.state_root else default_state_root
    site_workspace_dirs = {}
    if args.state_root is None:
        site_workspace_dirs = find_site_workspace_state_dirs(
            site_data_root, args.job_id, sorted(files)
        )
    personalized_states, missing = load_personalized_states(
        state_root,
        sorted(files),
        args.state,
        args.round,
        site_workspace_dirs=site_workspace_dirs,
    )
    if missing:
        message = (
            f"Missing personalized states for: {', '.join(missing)}; searched the central root "
            f"{state_root} and NVFlare per-site job workspaces"
        )
        if args.require_personalized:
            raise FileNotFoundError(message)
        print(f"WARNING: {message}")
        print("WARNING: global evaluation will run; personalized perspectives require a new FedAli job with state persistence.")

    clients = prepare_clients(
        files, personalized_states, args.window_size, args.target_offset,
        args.val_fraction, args.test_fraction, args.seed,
    )

    global_rows = []
    for target_site, target in clients.items():
        metrics, predictions = evaluate(
            global_model, target.test_df, target.test_segments,
            target.input_scaler, target.target_scaler, args, device,
        )
        row = {
            "perspective": "global",
            "source_site": "global",
            "target_site": target_site,
            "checkpoint": args.checkpoint,
            "split_by": target.split_by,
            "test_groups": " ".join(map(str, target.test_groups)),
            **metrics,
        }
        global_rows.append(row)
        if args.save_predictions:
            predictions.to_csv(prediction_dir / f"global_to_{target_site}.csv", index=False)
    pd.DataFrame(global_rows).to_csv(out_dir / "global_client_metrics.csv", index=False)

    if args.global_only:
        summary = [aggregate_rows(global_rows, "global")]
        pd.DataFrame(summary).to_csv(out_dir / "perspective_summary.csv", index=False)
        print(pd.DataFrame(summary).to_string(index=False))
        print(f"Saved FedAli global evaluation to: {out_dir}")
        return

    personalized_models = {}
    for site, payload in personalized_states.items():
        model = build_model(module, dict(payload["model_state"]), environment)
        model.set_local_prototypes(payload["local_prototypes"])
        personalized_models[site] = model.to(device)

    personalization_rows = []
    generalization_rows = []
    unseen_rows = []
    for source_site, model in personalized_models.items():
        source_state = personalized_states[source_site]
        source_input_scaler = scaler_from_state(source_state["input_scaler"])
        source_target_scaler = scaler_from_state(source_state["target_scaler"])
        for target_site, target in clients.items():
            if args.local_only and source_site != target_site:
                continue
            metrics, predictions = evaluate(
                model, target.test_df, target.test_segments,
                source_input_scaler, source_target_scaler, args, device,
            )
            perspective = "personalization" if source_site == target_site else "generalization"
            row = {
                "perspective": perspective,
                "source_site": source_site,
                "target_site": target_site,
                "state_path": source_state["_path"],
                "state_round": int(source_state["round"]),
                **metrics,
            }
            if perspective == "personalization":
                personalization_rows.append(row)
            else:
                generalization_rows.append(row)
            if args.save_predictions:
                predictions.to_csv(prediction_dir / f"{source_site}_to_{target_site}.csv", index=False)

        if args.unseen_npz and not args.local_only:
            unseen_df = load_npz_frame(args.unseen_npz)
            feature_cols = list(source_state["feature_cols"])
            unseen_segments = build_segments_from_dataframe(unseen_df, feature_cols, "RUL_capped")
            metrics, predictions = evaluate(
                model, unseen_df, unseen_segments,
                source_input_scaler, source_target_scaler, args, device,
            )
            unseen_rows.append({
                "perspective": "unseen_generalization",
                "source_site": source_site,
                "target_site": "unseen",
                "unseen_npz": str(args.unseen_npz),
                "state_round": int(source_state["round"]),
                **metrics,
            })
            if args.save_predictions:
                predictions.to_csv(prediction_dir / f"{source_site}_to_unseen.csv", index=False)

    pd.DataFrame(personalization_rows).to_csv(out_dir / "personalization_metrics.csv", index=False)
    pd.DataFrame(generalization_rows).to_csv(out_dir / "generalization_matrix.csv", index=False)
    pd.DataFrame(unseen_rows).to_csv(out_dir / "unseen_generalization_metrics.csv", index=False)
    if generalization_rows:
        pd.DataFrame(generalization_rows).pivot(
            index="source_site", columns="target_site", values="cycle_rmse"
        ).to_csv(out_dir / "generalization_cycle_rmse_matrix.csv")

    summary = [
        aggregate_rows(global_rows, "global"),
        aggregate_rows(personalization_rows, "personalization"),
        aggregate_rows(generalization_rows, "generalization"),
        aggregate_rows(unseen_rows, "unseen_generalization"),
    ]
    pd.DataFrame(summary).to_csv(out_dir / "perspective_summary.csv", index=False)

    manifest = {
        "job_id": args.job_id,
        "checkpoint": args.checkpoint,
        "personalized_state_root": str(state_root),
        "personalized_state_paths": {
            site: payload["_path"] for site, payload in personalized_states.items()
        },
        "personalized_state_selector": f"round {args.round}" if args.round is not None else args.state,
        "sites": sorted(clients),
        "missing_personalized_sites": missing,
        "window_size": args.window_size,
        "target_offset": args.target_offset,
        "rul_cap": args.rul_cap,
        "split_seed": args.seed,
        "local_only": args.local_only,
        "cross_client_scaling": "source-client training scaler",
    }
    (out_dir / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"Saved FedAli perspective evaluation to: {out_dir}")


if __name__ == "__main__":
    main()
