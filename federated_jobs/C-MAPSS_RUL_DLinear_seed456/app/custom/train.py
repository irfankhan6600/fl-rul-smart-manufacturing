"""Training utilities for the federated learning pipeline.

This module supports both:
- standalone local smoke testing, and
- NVFlare Client API execution via ``nvflare.client``.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW

from data_loader import (
    CENTRALIZED_FEATURE_COLS,
    LoaderConfig,
    DEFAULT_RUL_CAP,
    load_npz_frame,
    make_train_val_test_loaders_from_raw_npz_by_cycles,
    make_train_val_test_loaders_from_raw_npz,
    validate_npz_directory,
    validate_npz_file,
)
from DLinear_model import ModelConfig, build_model, count_parameters


try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


def debug_print(message: str) -> None:
    print(f"[train.py] {message}", flush=True)


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


@dataclass(frozen=True)
class TrainConfig:
    input_window: int = 512
    target_offset: int = 0
    input_size: int = 25
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    batch_size: int = 64
    epochs: int = 1
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    loss_name: str = "huber"
    seed: int = 456
    val_fraction: float = 0.15
    test_fraction: float = 0.15


def build_loader_config(cfg: TrainConfig) -> LoaderConfig:
    return LoaderConfig(
        target_col="RUL_capped",
        window_size=cfg.input_window,
        target_offset=cfg.target_offset,
        batch_size=cfg.batch_size,
        use_t50_history_as_input=True,
        feature_cols=CENTRALIZED_FEATURE_COLS,
    )


def with_input_size(cfg: TrainConfig, input_size: int) -> TrainConfig:
    if input_size == cfg.input_size:
        return cfg
    return TrainConfig(
        input_window=cfg.input_window,
        target_offset=cfg.target_offset,
        input_size=input_size,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        batch_size=cfg.batch_size,
        epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        grad_clip=cfg.grad_clip,
        loss_name=cfg.loss_name,
        seed=cfg.seed,
        val_fraction=cfg.val_fraction,
        test_fraction=cfg.test_fraction,
    )


def split_groups_for_validation(
    data_path: str | Path,
    cfg: TrainConfig,
    loader_cfg: LoaderConfig,
) -> Tuple[str, list[int], list[int], list[int]]:
    """Split a client's data into train/validation/test groups."""
    df = load_npz_frame(data_path)
    unit_col = loader_cfg.unit_col
    if unit_col not in df.columns:
        raise KeyError(f"Missing required unit column: {unit_col}")

    units = sorted(int(u) for u in df[unit_col].dropna().unique().tolist())
    if not units:
        raise ValueError("No units found for train/validation/test split.")

    split_name = "unit"
    groups = units
    if len(units) < 3 and loader_cfg.cycle_col in df.columns:
        cycles = sorted(int(c) for c in df[loader_cfg.cycle_col].dropna().unique().tolist())
        if len(cycles) >= 3:
            split_name = "cycle"
            groups = cycles

    rng = np.random.default_rng(cfg.seed)
    shuffled = groups.copy()
    rng.shuffle(shuffled)

    if len(shuffled) < 3:
        debug_print(
            f"only {len(shuffled)} {split_name}(s) available; using all groups for training and no held-out split"
        )
        return split_name, shuffled, [], []

    n_test = max(1, int(round(len(shuffled) * cfg.test_fraction))) if cfg.test_fraction > 0 else 0
    n_val = max(1, int(round(len(shuffled) * cfg.val_fraction))) if cfg.val_fraction > 0 else 0

    if n_val + n_test >= len(shuffled):
        n_test = 1
        n_val = 1

    test_units = sorted(shuffled[:n_test])
    val_units = sorted(shuffled[n_test:n_test + n_val])
    train_units = sorted(shuffled[n_test + n_val:])

    if not train_units:
        train_units = sorted(shuffled)
        val_units = []
        test_units = []

    return split_name, train_units, val_units, test_units


def resolve_npz_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    if candidate.is_dir():
        npz_files = sorted(candidate.glob("*.npz"))
        if not npz_files:
            raise FileNotFoundError(f"No .npz files found in directory: {candidate}")
        return npz_files[0]
    raise FileNotFoundError(f"Path does not exist: {candidate}")


def validate_data_path(path: str | Path) -> Dict[str, object]:
    candidate = Path(path)
    if candidate.is_dir():
        return {"directory": str(candidate), "files": validate_npz_directory(candidate)}
    return validate_npz_file(candidate)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_loss(name: str = "huber") -> nn.Module:
    name = name.lower().strip()
    if name == "huber":
        return nn.HuberLoss(delta=1.0)
    if name == "mse":
        return nn.MSELoss()
    if name == "l1":
        return nn.L1Loss()
    raise ValueError(f"Unsupported loss: {name}")


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    score = np.where(err < 0, np.exp(-err / 13.0) - 1.0, np.exp(err / 10.0) - 1.0)
    return float(np.sum(score))


def get_model(cfg: TrainConfig, device: Optional[torch.device] = None) -> nn.Module:
    model_cfg = ModelConfig(
        input_size=cfg.input_size,
        input_window=cfg.input_window,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    )
    model = build_model(model_cfg)
    return model.to(device or get_device())


def state_dict_to_numpy(state_dict: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    return {k: v.detach().cpu().numpy() for k, v in state_dict.items()}


def state_dict_to_cpu(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v).detach().cpu().clone() for k, v in state_dict.items()}


def state_dict_to_device(state_dict: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v, device=device) for k, v in state_dict.items()}


def numpy_to_state_dict(arrays: Dict[str, np.ndarray], device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
    device = device or get_device()
    return {k: torch.as_tensor(v, device=device) for k, v in arrays.items()}


def get_model_weights(model: nn.Module) -> Dict[str, np.ndarray]:
    return state_dict_to_numpy(model.state_dict())


def set_model_weights(model: nn.Module, weights: Dict[str, np.ndarray], device: Optional[torch.device] = None) -> None:
    tensor_state = numpy_to_state_dict(weights, device=device)
    model.load_state_dict(tensor_state, strict=True)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: Optional[nn.Module] = None,
    grad_clip: float = 1.0,
    epoch: Optional[int] = None,
    total_epochs: Optional[int] = None,
    log_every_batches: int = 250,
) -> float:
    model.train()
    loss_fn = loss_fn or build_loss("huber")

    total = 0.0
    count = 0
    num_batches = len(loader)
    epoch_label = f"epoch {epoch}/{total_epochs}" if epoch and total_epochs else "epoch"
    start_time = time.monotonic()
    debug_print(f"starting {epoch_label}: batches={num_batches}, device={device}")

    for batch_idx, (x, y) in enumerate(loader, start=1):
        if batch_idx == 1:
            debug_print(f"{epoch_label} first batch: x_shape={tuple(x.shape)}, y_shape={tuple(y.shape)}")
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        if grad_clip and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        total += float(loss.item()) * len(x)
        count += len(x)

        should_log_batch = batch_idx == num_batches or batch_idx % max(log_every_batches, 1) == 0
        if should_log_batch:
            running_loss = total / max(count, 1)
            elapsed = time.monotonic() - start_time
            sec_per_batch = elapsed / max(batch_idx, 1)
            eta = sec_per_batch * max(num_batches - batch_idx, 0)
            debug_print(
                f"{epoch_label} progress: batch={batch_idx}/{num_batches}, "
                f"samples={count}, running_loss={running_loss:.6f}, "
                f"elapsed={format_duration(elapsed)}, eta={format_duration(eta)}"
            )

    avg_loss = total / max(count, 1)
    debug_print(f"finished {epoch_label}: loss={avg_loss:.6f}, elapsed={format_duration(time.monotonic() - start_time)}")
    return avg_loss


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    loss_fn: Optional[nn.Module] = None,
    target_scaler=None,
    max_batches: Optional[int] = None,
) -> Dict[str, float | np.ndarray]:
    model.eval()
    loss_fn = loss_fn or build_loss("huber")

    total = 0.0
    count = 0
    preds = []
    trues = []
    start_time = time.monotonic()
    total_batches = len(loader)
    eval_batches = min(total_batches, max_batches) if max_batches else total_batches
    debug_print(f"starting evaluation: batches={eval_batches}/{total_batches}, device={device}")
    for batch_idx, (x, y) in enumerate(loader, start=1):
        if max_batches and batch_idx > max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        total += float(loss.item()) * len(x)
        count += len(x)
        preds.append(pred.detach().cpu().numpy())
        trues.append(y.detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0) if preds else np.empty((0, 1), dtype=np.float32)
    trues = np.concatenate(trues, axis=0) if trues else np.empty((0, 1), dtype=np.float32)

    if target_scaler is not None and len(preds):
        preds = target_scaler.inverse_transform(preds)
        trues = target_scaler.inverse_transform(trues)
        preds = np.clip(preds, 0.0, float(DEFAULT_RUL_CAP))

    mse = float(np.mean((preds - trues) ** 2)) if len(preds) else float("nan")
    mae = float(np.mean(np.abs(preds - trues))) if len(preds) else float("nan")
    rmse = float(np.sqrt(mse)) if len(preds) else float("nan")
    score = nasa_score(trues, preds) if len(preds) else float("nan")
    debug_print(
        f"finished evaluation: samples={count}, loss={total / max(count, 1):.6f}, "
        f"rmse={rmse:.6f}, mae={mae:.6f}, mse={mse:.6f}, nasa_score={score:.6f}, "
        f"elapsed={format_duration(time.monotonic() - start_time)}"
    )

    return {
        "loss": total / max(count, 1),
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "nasa_score": score,
        "preds": preds.squeeze(-1),
        "trues": trues.squeeze(-1),
    }


def fit_client_model(
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader] = None,
    cfg: Optional[TrainConfig] = None,
    initial_weights: Optional[Dict[str, np.ndarray]] = None,
    target_scaler=None,
) -> Tuple[nn.Module, Dict[str, float], Dict[str, np.ndarray]]:
    cfg = cfg or TrainConfig()
    set_seed(cfg.seed)
    device = get_device()
    debug_print(
        f"fit_client_model start: epochs={cfg.epochs}, lr={cfg.learning_rate}, "
        f"batch_size={cfg.batch_size}, train_samples={len(train_loader.dataset)}, device={device}"
    )
    model = get_model(cfg, device=device)
    if initial_weights is not None:
        debug_print("loading initial weights into local model")
        set_model_weights(model, initial_weights, device=device)

    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    loss_fn = build_loss(cfg.loss_name)

    best_val = float("inf")
    best_weights = get_model_weights(model)
    best_metrics: Dict[str, float] = {}

    for epoch_idx in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            loss_fn=loss_fn,
            grad_clip=cfg.grad_clip,
            epoch=epoch_idx,
            total_epochs=cfg.epochs,
        )
        metrics = {"train_loss": train_loss}
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, device=device, loss_fn=loss_fn, target_scaler=target_scaler)
            metrics.update({f"val_{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float))})
            if val_metrics["loss"] < best_val:
                best_val = float(val_metrics["loss"])
                best_weights = get_model_weights(model)
                best_metrics = metrics.copy()
        else:
            best_metrics = metrics.copy()
            best_weights = get_model_weights(model)

    set_model_weights(model, best_weights, device=device)
    debug_print(f"fit_client_model done: metrics={best_metrics}")
    return model, best_metrics, best_weights


def smoke_test(data_path: str | Path) -> None:
    """Run a quick local training sanity check."""
    cfg = TrainConfig()
    resolved_path = resolve_npz_path(data_path)
    loader_cfg = build_loader_config(cfg)
    split_name, train_groups, val_groups, test_groups = split_groups_for_validation(resolved_path, cfg, loader_cfg)
    if split_name == "cycle":
        loader, val_loader, test_loader, feature_cols, input_scaler, target_scaler = (
            make_train_val_test_loaders_from_raw_npz_by_cycles(
                path=resolved_path,
                cfg=loader_cfg,
                train_cycles=train_groups,
                val_cycles=val_groups,
                test_cycles=test_groups,
            )
        )
    else:
        loader, val_loader, test_loader, feature_cols, input_scaler, target_scaler = (
            make_train_val_test_loaders_from_raw_npz(
                path=resolved_path,
                cfg=loader_cfg,
                train_units=train_groups,
                val_units=val_groups,
                test_units=test_groups,
            )
        )
    cfg = with_input_size(cfg, len(feature_cols))
    model, metrics, weights = fit_client_model(loader, val_loader, cfg, target_scaler=target_scaler)
    eval_loader = test_loader or val_loader or loader
    eval_metrics = evaluate(model, eval_loader, device=get_device(), target_scaler=target_scaler, max_batches=200)
    print(
        json.dumps(
            {
                "data_path": str(resolved_path),
                "feature_cols": feature_cols,
                "params": count_parameters(model),
                "split_by": split_name,
                "train_groups": train_groups,
                "val_groups": val_groups,
                "test_groups": test_groups,
                **metrics,
                "eval_rmse": eval_metrics["rmse"],
                "eval_mae": eval_metrics["mae"],
                "eval_mse": eval_metrics["mse"],
                "eval_nasa_score": eval_metrics["nasa_score"],
            },
            indent=2,
        )
    )


def run_fl_client(data_path: str | Path) -> None:
    import nvflare.client as flare

    flare.init()
    debug_print("NVFlare client initialized")

    cfg = TrainConfig()
    resolved_path = resolve_npz_path(data_path)
    debug_print(f"resolved data path: {resolved_path}")
    loader_cfg = build_loader_config(cfg)
    debug_print(
        f"building data loader: window={loader_cfg.window_size}, target_offset={loader_cfg.target_offset}, "
        f"batch_size={loader_cfg.batch_size}, target={loader_cfg.target_col}, rul_cap={DEFAULT_RUL_CAP}"
    )
    split_name, train_groups, val_groups, test_groups = split_groups_for_validation(resolved_path, cfg, loader_cfg)
    debug_print(
        f"{split_name} split: train={train_groups}, val={val_groups if val_groups else 'none'}, "
        f"test={test_groups if test_groups else 'none'}"
    )
    if split_name == "cycle":
        train_loader, val_loader, test_loader, feature_cols, input_scaler, target_scaler = (
            make_train_val_test_loaders_from_raw_npz_by_cycles(
                path=resolved_path,
                cfg=loader_cfg,
                train_cycles=train_groups,
                val_cycles=val_groups,
                test_cycles=test_groups,
            )
        )
    else:
        train_loader, val_loader, test_loader, feature_cols, input_scaler, target_scaler = (
            make_train_val_test_loaders_from_raw_npz(
                path=resolved_path,
                cfg=loader_cfg,
                train_units=train_groups,
                val_units=val_groups,
                test_units=test_groups,
            )
        )
    eval_loader = val_loader or test_loader or train_loader
    final_eval_loader = test_loader or val_loader or train_loader
    cfg = with_input_size(cfg, len(feature_cols))
    device = get_device()
    model = get_model(cfg, device=device)
    debug_print(
        f"client ready: train_samples={len(train_loader.dataset)}, train_batches={len(train_loader)}, "
        f"val_samples={len(val_loader.dataset) if val_loader else 0}, "
        f"test_samples={len(test_loader.dataset) if test_loader else 0}, "
        f"features={len(feature_cols)}, params={count_parameters(model)}, device={device}"
    )
    task_count = 0
    tb_log_dir = Path(os.environ.get("TB_LOG_DIR", "tb")) / "client"
    writer = SummaryWriter(log_dir=str(tb_log_dir)) if SummaryWriter else None
    if writer:
        debug_print(f"TensorBoard logging enabled: {tb_log_dir}")
    else:
        debug_print("TensorBoard logging disabled: torch.utils.tensorboard is unavailable")

    while flare.is_running():
        debug_print("waiting for task from NVFlare")
        input_model = flare.receive()
        if input_model is None:
            debug_print("received no task; continuing")
            continue

        task_count += 1
        task_start = time.monotonic()
        debug_print(
            f"received task #{task_count}: train={flare.is_train()}, evaluate={flare.is_evaluate()}, "
            f"submit_model={flare.is_submit_model()}"
        )
        received_params = getattr(input_model, "params", None)
        if received_params is not None:
            debug_print(f"loading received params: tensors={len(received_params)}, target_device={device}")
            model.load_state_dict(state_dict_to_device(received_params, device), strict=True)
        else:
            debug_print("received task without params")

        if flare.is_train():
            debug_print("evaluating received global model before local training")
            global_eval_metrics = evaluate(
                model,
                eval_loader,
                device=get_device(),
                target_scaler=target_scaler,
                max_batches=200,
            )
            debug_print(
                f"global model before train metrics: rmse={float(global_eval_metrics['rmse']):.6f}, "
                f"mae={float(global_eval_metrics['mae']):.6f}, mse={float(global_eval_metrics['mse']):.6f}, "
                f"nasa_score={float(global_eval_metrics['nasa_score']):.6f}"
            )
            if writer:
                writer.add_scalar("global/rmse_before_train", float(global_eval_metrics["rmse"]), task_count)
                writer.add_scalar("global/mae_before_train", float(global_eval_metrics["mae"]), task_count)
                writer.add_scalar("global/mse_before_train", float(global_eval_metrics["mse"]), task_count)
                writer.add_scalar("global/nasa_score_before_train", float(global_eval_metrics["nasa_score"]), task_count)
                writer.flush()

            debug_print("train task started")
            model, train_metrics, _ = fit_client_model(
                train_loader=train_loader,
                val_loader=val_loader,
                cfg=cfg,
                initial_weights=get_model_weights(model),
                target_scaler=target_scaler,
            )
            debug_print("local training finished; evaluating trained model on held-out validation data")
            eval_metrics = evaluate(model, eval_loader, device=get_device(), target_scaler=target_scaler, max_batches=200)
            output_model = flare.FLModel(
                params=state_dict_to_cpu(model.state_dict()),
                metrics={
                    "train_loss": float(train_metrics.get("train_loss", float("nan"))),
                    "val_loss": float(eval_metrics["loss"]),
                    "rmse": float(eval_metrics["rmse"]),
                    "mae": float(eval_metrics["mae"]),
                    "mse": float(eval_metrics["mse"]),
                    "nasa_score": float(eval_metrics["nasa_score"]),
                },
                meta={
                    "num_samples": len(train_loader.dataset),
                    "num_train_samples": len(train_loader.dataset),
                    "num_val_samples": len(val_loader.dataset) if val_loader else 0,
                    "num_test_samples": len(test_loader.dataset) if test_loader else 0,
                    "data_path": str(resolved_path),
                    "split_by": split_name,
                    "train_groups": train_groups,
                    "val_groups": val_groups,
                    "test_groups": test_groups,
                    "target_col": loader_cfg.target_col,
                    "rul_cap": DEFAULT_RUL_CAP,
                    "window_size": cfg.input_window,
                    "target_offset": cfg.target_offset,
                },
            )
            debug_print(
                f"train task metrics: train_loss={float(train_metrics.get('train_loss', float('nan'))):.6f}, "
                f"rmse={float(eval_metrics['rmse']):.6f}, mae={float(eval_metrics['mae']):.6f}, "
                f"nasa_score={float(eval_metrics['nasa_score']):.6f}"
            )
            if writer:
                writer.add_scalar("train/loss", float(train_metrics.get("train_loss", float("nan"))), task_count)
                writer.add_scalar("val/loss_after_train", float(eval_metrics["loss"]), task_count)
                writer.add_scalar("val/rmse_after_train", float(eval_metrics["rmse"]), task_count)
                writer.add_scalar("val/mae_after_train", float(eval_metrics["mae"]), task_count)
                writer.add_scalar("val/mse_after_train", float(eval_metrics["mse"]), task_count)
                writer.add_scalar("val/nasa_score_after_train", float(eval_metrics["nasa_score"]), task_count)
                writer.add_scalar("eval/rmse", float(eval_metrics["rmse"]), task_count)
                writer.add_scalar("eval/mae", float(eval_metrics["mae"]), task_count)
                writer.add_scalar("eval/mse", float(eval_metrics["mse"]), task_count)
                writer.add_scalar("eval/nasa_score", float(eval_metrics["nasa_score"]), task_count)
                writer.flush()
        elif flare.is_evaluate():
            debug_print("evaluate task started")
            eval_metrics = evaluate(model, final_eval_loader, device=get_device(), target_scaler=target_scaler)
            output_model = flare.FLModel(
                params=state_dict_to_cpu(model.state_dict()),
                metrics={
                    "rmse": float(eval_metrics["rmse"]),
                    "mae": float(eval_metrics["mae"]),
                    "mse": float(eval_metrics["mse"]),
                    "nasa_score": float(eval_metrics["nasa_score"]),
                },
                meta={
                    "num_samples": len(final_eval_loader.dataset),
                    "num_train_samples": len(train_loader.dataset),
                    "num_val_samples": len(val_loader.dataset) if val_loader else 0,
                    "num_test_samples": len(test_loader.dataset) if test_loader else 0,
                    "data_path": str(resolved_path),
                    "split_by": split_name,
                    "train_groups": train_groups,
                    "val_groups": val_groups,
                    "test_groups": test_groups,
                    "target_col": loader_cfg.target_col,
                    "rul_cap": DEFAULT_RUL_CAP,
                    "window_size": cfg.input_window,
                    "target_offset": cfg.target_offset,
                },
            )
            debug_print(
                f"evaluate task metrics: rmse={float(eval_metrics['rmse']):.6f}, "
                f"mae={float(eval_metrics['mae']):.6f}, mse={float(eval_metrics['mse']):.6f}, "
                f"nasa_score={float(eval_metrics['nasa_score']):.6f}"
            )
            if writer:
                writer.add_scalar("test/rmse", float(eval_metrics["rmse"]), task_count)
                writer.add_scalar("test/mae", float(eval_metrics["mae"]), task_count)
                writer.add_scalar("test/mse", float(eval_metrics["mse"]), task_count)
                writer.add_scalar("test/nasa_score", float(eval_metrics["nasa_score"]), task_count)
                writer.flush()
        elif flare.is_submit_model():
            debug_print("submit_model task started")
            output_model = flare.FLModel(
                params=state_dict_to_cpu(model.state_dict()),
                meta={
                    "num_samples": len(train_loader.dataset),
                    "num_train_samples": len(train_loader.dataset),
                    "num_val_samples": len(val_loader.dataset) if val_loader else 0,
                    "num_test_samples": len(test_loader.dataset) if test_loader else 0,
                    "data_path": str(resolved_path),
                    "split_by": split_name,
                    "train_groups": train_groups,
                    "val_groups": val_groups,
                    "test_groups": test_groups,
                },
            )
        else:
            debug_print("unknown task type; returning current model params")
            output_model = flare.FLModel(
                params=state_dict_to_cpu(model.state_dict()),
                meta={
                    "num_samples": len(train_loader.dataset),
                    "num_train_samples": len(train_loader.dataset),
                    "num_val_samples": len(val_loader.dataset) if val_loader else 0,
                    "num_test_samples": len(test_loader.dataset) if test_loader else 0,
                    "data_path": str(resolved_path),
                    "split_by": split_name,
                    "train_groups": train_groups,
                    "val_groups": val_groups,
                    "test_groups": test_groups,
                },
            )
        debug_print("sending result to NVFlare")
        flare.send(output_model)
        debug_print(f"sent result for task #{task_count}; elapsed={format_duration(time.monotonic() - task_start)}")

    if writer:
        writer.close()
    debug_print("NVFlare client loop ended")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Client-local training smoke test and data validation.")
    parser.add_argument(
        "--data-path",
        default=None,
        help="Path to one client .npz file or to the client local data directory.",
    )
    parser.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "smoke", "flare"],
        help="Run standalone smoke test, NVFlare client loop, or auto-detect.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_path = args.data_path or os.environ.get("TRAIN_NPZ") or os.environ.get("TRAIN_DATA_DIR")
    if not data_path:
        raise SystemExit(
            "Provide --data-path or set TRAIN_NPZ or TRAIN_DATA_DIR."
        )
    if args.mode == "smoke":
        print(json.dumps({"validation": validate_data_path(data_path)}, indent=2))
        smoke_test(data_path)
    elif args.mode == "flare":
        run_fl_client(data_path)
    else:
        try:
            run_fl_client(data_path)
        except Exception as ex:
            if "nvflare" in str(ex).lower() or isinstance(ex, ModuleNotFoundError):
                print(json.dumps({"validation": validate_data_path(data_path)}, indent=2))
                smoke_test(data_path)
            else:
                raise
