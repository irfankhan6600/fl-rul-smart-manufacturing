#!/usr/bin/env python3
"""Centralized LSTM RUL baseline, matched to the federated experimental setup.

The original notebook (centralized_rul_lstm_study.ipynb) held unit 20 out entirely as a
validation engine, so it trained on 5 of the 6 federated training engines and fit its
scaler without unit 20. That makes it not directly comparable to the federated runs.

This script removes that mismatch:

  Training units : 2, 5, 10, 16, 18, 20  (all six federated clients)
  Validation     : the 15% validation cycles each federated client holds out, pooled.
                   Cycle IDs are split with the federated client's own
                   split_groups_for_validation(): shuffle with default_rng(seed), then
                   test block first, validation block second, train the remainder.
  Held out       : each unit's 15% local-test cycles are excluded from training and
                   validation, exactly as a federated client excludes its own local test
                   split. Centralized therefore trains on the same ~70% of cycles the
                   federation trains on - the only difference is pooled vs distributed.
  Test           : units 11, 14, 15, cycle-level aggregation identical to
                   scripts/RUL/evaluate_global_model_unseen.py.
  Scaler         : one scaler over the pooled training cycles, no test data. Computed by
                   pooling per-unit sufficient statistics, which is exactly a single
                   StandardScaler fit on the union and also yields the per-unit rows the
                   federated scaler_manifest.json carries.

Loader, split and model code are imported read-only from the federated job so the two
pipelines cannot drift apart. Nothing under federated_jobs/ is modified.

Usage:
  python Centralized_Study/RUL/centralized_lstm_matched.py --seed 42
  python Centralized_Study/RUL/centralized_lstm_matched.py --seed 42 --batch-size 64
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FED_CUSTOM_DIR = PROJECT_ROOT / "federated_jobs" / "C-MAPSS_RUL_LSTM_FedAvg" / "app" / "custom"
CENTRALIZED_DATA_DIR = PROJECT_ROOT / "Centralized_Study" / "data"
FALLBACK_SITE_ROOT = PROJECT_ROOT / "Workspace" / "example_project" / "prod_00"
FED_PAPER_TABLE = PROJECT_ROOT / "Workspace" / "fl_artifacts" / "multi_seed_summary" / "paper_table.csv"
FED_RESULTS_ROOT = PROJECT_ROOT / "Workspace" / "fl_artifacts"

sys.path.insert(0, str(FED_CUSTOM_DIR))

from data_loader import (  # noqa: E402
    CENTRALIZED_FEATURE_COLS,
    DEFAULT_RUL_CAP,
    LazyWindowDataset,
    LoaderConfig,
    build_segments_from_dataframe,
    count_windows_in_segments,
    fit_scalers_on_segments,
    load_npz_frame,
    split_dataframe_by_cycles,
    split_dataframe_by_units,
)
from LSTM_model import ModelConfig, build_model, count_parameters  # noqa: E402
from train import TrainConfig, split_groups_for_validation  # noqa: E402


TRAIN_UNITS = [2, 5, 10, 16, 18, 20]
TEST_UNITS = [11, 14, 15]
UNIT_TO_SITE = {2: "site-1", 5: "site-2", 10: "site-3", 16: "site-4", 18: "site-5", 20: "site-6"}
TEST_NPZ_NAME = "test_units_11_14_15.npz"

# Federated seed-42 pooled scaler, from Workspace/fl_artifacts/Seed_42/*/unseen_engines/
# scaler_manifest.json. Pooling the same per-unit statistics must reproduce these.
SEED42_EXPECTED = {"mean": 36.621691, "std": 22.704732, "n": 3581212}
SEED42_TOLERANCE = 1e-4


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    score = np.where(err < 0, np.exp(-err / 13.0) - 1.0, np.exp(err / 10.0) - 1.0)
    return float(np.sum(score))


def regression_metrics(y_true, y_pred) -> dict[str, float]:
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


def resolve_unit_npz(unit: int) -> tuple[Path, bool]:
    """Prefer Centralized_Study/data/, fall back to the federated site directory."""
    primary = CENTRALIZED_DATA_DIR / f"client_unit_{unit}.npz"
    if primary.exists():
        return primary, False
    fallback = FALLBACK_SITE_ROOT / UNIT_TO_SITE[unit] / "local" / "data" / f"client_unit_{unit}.npz"
    if fallback.exists():
        return fallback, True
    raise FileNotFoundError(f"unit {unit}: no NPZ at {primary} or {fallback}")


def resolve_test_npz() -> tuple[Path, bool]:
    primary = CENTRALIZED_DATA_DIR / TEST_NPZ_NAME
    if primary.exists():
        return primary, False
    fallback = PROJECT_ROOT / "Workspace" / "data" / TEST_NPZ_NAME
    if fallback.exists():
        return fallback, True
    raise FileNotFoundError(f"test set: no NPZ at {primary} or {fallback}")


def build_splits_and_scalers(args, feature_cols: list[str]):
    """Reproduce each client's cycle split, pool train/val segments, pool the scalers."""
    train_cfg = TrainConfig(
        input_window=args.window_size,
        target_offset=args.target_offset,
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
    )
    loader_cfg = LoaderConfig(
        target_col="RUL_capped",
        window_size=args.window_size,
        target_offset=args.target_offset,
        feature_cols=tuple(feature_cols),
    )

    train_segments, val_segments = [], []
    per_unit, fallbacks = {}, []
    print("Reproducing each federated client's cycle split (test block first, then validation):")
    for unit in TRAIN_UNITS:
        npz_path, used_fallback = resolve_unit_npz(unit)
        if used_fallback:
            fallbacks.append(str(npz_path))
        df = load_npz_frame(npz_path)

        split_by, tr_groups, va_groups, te_groups = split_groups_for_validation(npz_path, train_cfg, loader_cfg)
        if split_by == "cycle":
            tr_df, va_df, _ = split_dataframe_by_cycles(df, tr_groups, va_groups, te_groups, loader_cfg.cycle_col)
        else:
            tr_df, va_df, _ = split_dataframe_by_units(df, tr_groups, va_groups, te_groups, loader_cfg.unit_col)

        u_train = build_segments_from_dataframe(
            tr_df, feature_cols=feature_cols, target_col=loader_cfg.target_col,
            unit_col=loader_cfg.unit_col, cycle_col=loader_cfg.cycle_col,
        )
        u_val = build_segments_from_dataframe(
            va_df, feature_cols=feature_cols, target_col=loader_cfg.target_col,
            unit_col=loader_cfg.unit_col, cycle_col=loader_cfg.cycle_col,
        )
        # Per-unit fit gives the manifest rows; pooling them below is identical to one
        # StandardScaler fit over the union of these segments.
        x_scaler, y_scaler = fit_scalers_on_segments(u_train, args.window_size, args.target_offset)
        n_train = count_windows_in_segments(u_train, args.window_size, args.target_offset)
        n_val = count_windows_in_segments(u_val, args.window_size, args.target_offset)

        train_segments.extend(u_train)
        val_segments.extend(u_val)
        per_unit[unit] = {
            "unit": unit,
            "npz": str(npz_path),
            "split_by": split_by,
            "n_train_groups": len(tr_groups),
            "n_val_groups": len(va_groups),
            "n_test_groups": len(te_groups),
            "n_train_windows": int(n_train),
            "n_val_windows": int(n_val),
            "input_mean": np.asarray(x_scaler.mean_, dtype=np.float64),
            "input_var": np.asarray(x_scaler.var_, dtype=np.float64),
            "target_mean": float(y_scaler.mean_[0]),
            "target_var": float(y_scaler.var_[0]),
            "target_std": float(np.sqrt(y_scaler.var_[0])),
        }
        print(
            f"  unit {unit:<3} cycles train/val/local-test = {len(tr_groups)}/{len(va_groups)}/{len(te_groups)}"
            f" | train_windows={n_train:,} val_windows={n_val:,}"
            f" | target_mean={per_unit[unit]['target_mean']:.4f} std={per_unit[unit]['target_std']:.4f}"
        )

    weights = np.asarray([s["n_train_windows"] for s in per_unit.values()], dtype=np.float64)
    n_total = float(weights.sum())

    def pool(means: np.ndarray, variances: np.ndarray):
        w = weights.reshape(-1, *([1] * (means.ndim - 1)))
        mu = (w * means).sum(axis=0) / n_total
        var = (w * (variances + (means - mu) ** 2)).sum(axis=0) / n_total
        return mu, np.sqrt(var)

    input_mu, input_std = pool(
        np.stack([s["input_mean"] for s in per_unit.values()]),
        np.stack([s["input_var"] for s in per_unit.values()]),
    )
    target_mu, target_std = pool(
        np.asarray([s["target_mean"] for s in per_unit.values()]),
        np.asarray([s["target_var"] for s in per_unit.values()]),
    )
    print(f"  POOLED | n_total={int(n_total):,} | target_mean={float(target_mu):.4f} | target_std={float(target_std):.4f}")

    input_scaler = ArrayScaler(np.asarray(input_mu), np.where(np.asarray(input_std) == 0.0, 1.0, np.asarray(input_std)))
    target_scaler = ArrayScaler(np.atleast_1d(target_mu), np.atleast_1d(np.where(target_std == 0.0, 1.0, target_std)))

    manifest = {
        "method": "pooled_sufficient_statistic_aggregation (centralized; identical to one StandardScaler fit on the union of all training-unit cycles)",
        "paradigm": "centralized",
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "window_size": args.window_size,
        "target_offset": args.target_offset,
        "rul_cap": float(args.rul_cap),
        "feature_cols": list(feature_cols),
        "note": "Zero information from unseen units 11, 14, 15 used in scaler construction",
        "clients": {
            UNIT_TO_SITE[u]: {k: v for k, v in s.items() if k not in ("input_mean", "input_var", "target_var")}
            for u, s in per_unit.items()
        },
        "pooled": {
            "target_mean": float(target_mu),
            "target_std": float(target_std),
            "input_means": [float(v) for v in input_mu],
            "input_stds": [float(v) for v in input_std],
            "total_train_windows": int(n_total),
        },
    }
    return train_segments, val_segments, input_scaler, target_scaler, manifest, fallbacks


class ArrayScaler:
    """StandardScaler-compatible transform/inverse_transform over fixed statistics."""

    def __init__(self, mean_: np.ndarray, scale_: np.ndarray) -> None:
        self.mean_ = np.asarray(mean_, dtype=np.float64)
        self.scale_ = np.asarray(scale_, dtype=np.float64)

    def transform(self, values):
        return (np.asarray(values) - self.mean_) / self.scale_

    def inverse_transform(self, values):
        return np.asarray(values) * self.scale_ + self.mean_


def scaler_cross_check(manifest: dict, seed: int) -> None:
    pooled = manifest["pooled"]
    mean, std, n = pooled["target_mean"], pooled["target_std"], pooled["total_train_windows"]
    if seed != 42:
        print(f"SCALER (seed {seed}): pooled mean={mean:.6f} std={std:.6f} n={n:,}")
        fed = FED_RESULTS_ROOT / f"Seed_{seed}" / "LSTM_FedAvg" / "unseen_engines" / "scaler_manifest.json"
        if fed.exists():
            f = json.loads(fed.read_text())["pooled"]
            print(
                f"  federated seed-{seed} manifest: mean={f['target_mean']:.6f} std={f['target_std']:.6f} "
                f"n={f['total_train_windows']:,}  ->  match: "
                f"{abs(f['target_mean']-mean)<SEED42_TOLERANCE and abs(f['target_std']-std)<SEED42_TOLERANCE and f['total_train_windows']==n}"
            )
        return

    ok = (
        abs(mean - SEED42_EXPECTED["mean"]) < SEED42_TOLERANCE
        and abs(std - SEED42_EXPECTED["std"]) < SEED42_TOLERANCE
        and n == SEED42_EXPECTED["n"]
    )
    if ok:
        print("SCALER CROSS-CHECK: PASS — centralized pooled scaler matches federated seed-42 manifest exactly")
        return
    print("SCALER CROSS-CHECK: FAIL — STOP")
    print(f"  target_mean : actual {mean:.6f}  expected {SEED42_EXPECTED['mean']:.6f}")
    print(f"  target_std  : actual {std:.6f}  expected {SEED42_EXPECTED['std']:.6f}")
    print(f"  n_windows   : actual {n:,}  expected {SEED42_EXPECTED['n']:,}")
    raise SystemExit(1)


def train_one_epoch(model, loader, optimizer, loss_fn, device, grad_clip, log_every=2000):
    model.train()
    total, count = 0.0, 0
    start = time.monotonic()
    n_batches = len(loader)
    for batch_idx, (x, y) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        if grad_clip and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += float(loss.item()) * len(x)
        count += len(x)
        if batch_idx % log_every == 0 or batch_idx == n_batches:
            el = time.monotonic() - start
            eta = el / batch_idx * (n_batches - batch_idx)
            print(f"    batch {batch_idx}/{n_batches} loss={total/max(count,1):.6f} elapsed={el/60:.1f}m eta={eta/60:.1f}m", flush=True)
    return total / max(count, 1)


@torch.no_grad()
def evaluate(model, loader, target_scaler, device, rul_cap, loss_fn=None):
    model.eval()
    loss_fn = loss_fn or nn.HuberLoss(delta=1.0)
    total, count = 0.0, 0
    preds, trues = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        p = model(x)
        total += float(loss_fn(p, y).item()) * len(x)
        count += len(x)
        preds.append(p.detach().cpu().numpy())
        trues.append(y.detach().cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    preds_raw = np.clip(target_scaler.inverse_transform(preds).reshape(-1), 0.0, rul_cap)
    trues_raw = target_scaler.inverse_transform(trues).reshape(-1)
    metrics = regression_metrics(trues_raw, preds_raw)
    metrics["loss"] = total / max(count, 1)
    metrics["preds"] = preds_raw
    metrics["trues"] = trues_raw
    return metrics


def build_window_metadata(df, window_size, target_offset, target_col):
    """Identical to evaluate_global_model_unseen.py so cycle aggregation matches."""
    rows = []
    for (unit, cycle), group in df.groupby(["unit", "cycle"], sort=False):
        group = group.reset_index(drop=True)
        num_windows = len(group) - window_size - target_offset + 1
        if num_windows <= 0:
            continue
        target_indices = np.arange(window_size - 1 + target_offset, window_size - 1 + target_offset + num_windows)
        rows.append(pd.DataFrame({
            "unit": int(unit),
            "cycle": int(cycle),
            "window_in_cycle": np.arange(num_windows, dtype=np.int64),
            "true_rul_from_frame": group.loc[target_indices, target_col].to_numpy(dtype=np.float32),
        }))
    if not rows:
        raise ValueError("No window metadata was created.")
    return pd.concat(rows, ignore_index=True)


def federated_reference(seed: int) -> dict:
    ref = {}
    if FED_PAPER_TABLE.exists():
        t = pd.read_csv(FED_PAPER_TABLE)
        row = t[t["Model"] == "LSTM_FedAvg"]
        if not row.empty:
            r = row.iloc[0]
            ref["multi_seed"] = {
                "rmse_mean": float(r["RMSE_mean"]), "rmse_std": float(r["RMSE_std"]),
                "mae_mean": float(r["MAE_mean"]), "mae_std": float(r["MAE_std"]),
                "nasa_mean": float(r["NASA_mean"]), "nasa_std": float(r["NASA_std"]),
            }
            for s in (42, 100, 456):
                col = f"RMSE_seed{s}"
                if col in r and pd.notna(r[col]):
                    ref.setdefault("per_seed", {})[s] = {
                        "rmse": float(r[col]), "mae": float(r[f"MAE_seed{s}"]), "nasa": float(r[f"NASA_seed{s}"]),
                    }
    cyc = list((FED_RESULTS_ROOT / f"Seed_{seed}" / "LSTM_FedAvg" / "unseen_engines" / "cycle_level").glob("*__FL_global_model_cycle_metrics.csv")) if (FED_RESULTS_ROOT / f"Seed_{seed}").exists() else []
    if cyc:
        d = pd.read_csv(cyc[0])
        ref["per_unit"] = {str(r["unit"]): float(r["rmse"]) for _, r in d.iterrows()}
    return ref


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results-dir", default=None,
                   help="Default: Centralized_Study/RUL/outputs/matched/seed_<N>[ _batch64 ]")
    p.add_argument("--batch-size", type=int, default=256, help="256 matches the federated LSTM job; 64 is the secondary run")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--learning-rate", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--window-size", type=int, default=512)
    p.add_argument("--target-offset", type=int, default=0)
    p.add_argument("--rul-cap", type=float, default=float(DEFAULT_RUL_CAP))
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--test-fraction", type=float, default=0.15)
    p.add_argument("--num-workers", type=int, default=8,
                   help="DataLoader workers. Batch order is set by the seeded sampler, so this affects speed only.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.results_dir:
        out_dir = Path(args.results_dir)
    else:
        suffix = "" if args.batch_size == 256 else f"_batch{args.batch_size}"
        out_dir = PROJECT_ROOT / "Centralized_Study" / "RUL" / "outputs" / "matched" / f"seed_{args.seed}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_cols = list(CENTRALIZED_FEATURE_COLS)
    started = datetime.now()
    print(f"Centralized LSTM matched run | seed={args.seed} batch={args.batch_size} device={device}")

    train_segments, val_segments, input_scaler, target_scaler, manifest, fallbacks = build_splits_and_scalers(args, feature_cols)
    if fallbacks:
        print("NOTE: loaded from fallback site directories:")
        for f in fallbacks:
            print("   ", f)
    (out_dir.parent / "scaler_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out_dir / "scaler_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    scaler_cross_check(manifest, args.seed)

    train_ds = LazyWindowDataset(train_segments, args.window_size, args.target_offset, input_scaler, target_scaler)
    val_ds = LazyWindowDataset(val_segments, args.window_size, args.target_offset, input_scaler, target_scaler)
    dl_kwargs = dict(num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
                     persistent_workers=bool(args.num_workers))
    if args.num_workers:
        dl_kwargs["prefetch_factor"] = 4
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False, **dl_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False, **dl_kwargs)
    print(f"Pooled train windows: {len(train_ds):,} | pooled val windows: {len(val_ds):,} | batches/epoch: {len(train_loader):,}")

    set_seed(args.seed)
    model = build_model(ModelConfig(input_size=len(feature_cols), hidden_size=128, num_layers=2, dropout=0.2)).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.HuberLoss(delta=1.0)
    print(f"Trainable parameters: {count_parameters(model):,}")

    history = []
    best_rmse, best_epoch, improvements, stalled = float("inf"), 0, 0, 0
    early_stop_epoch = None
    ckpt_path = out_dir / "model_checkpoint.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.monotonic()
        print(f"  Epoch {epoch}/{args.epochs}", flush=True)
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, args.grad_clip)
        val = evaluate(model, val_loader, target_scaler, device, args.rul_cap, loss_fn)
        improved = val["rmse"] < best_rmse
        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_rmse": val["rmse"],
            "val_mae": val["mae"], "val_nasa": val["nasa_score"], "val_loss": val["loss"],
            "improved": bool(improved), "minutes": (time.monotonic() - t0) / 60.0,
        })
        print(
            f"    epoch {epoch}: train_loss={train_loss:.6f} val_rmse={val['rmse']:.4f} "
            f"val_mae={val['mae']:.4f} val_nasa={val['nasa_score']:.2f} "
            f"({(time.monotonic()-t0)/60:.1f} min){' *best*' if improved else ''}", flush=True
        )
        if improved:
            best_rmse, best_epoch, stalled = val["rmse"], epoch, 0
            improvements += 1
            torch.save({"model": model.state_dict(), "epoch": epoch, "best_val_rmse": best_rmse,
                        "seed": args.seed, "batch_size": args.batch_size,
                        "feature_cols": feature_cols}, ckpt_path)
        else:
            stalled += 1
            if stalled >= args.patience:
                early_stop_epoch = epoch
                print(f"    early stopping at epoch {epoch} (no val RMSE improvement for {args.patience} epochs)")
                break

    pd.DataFrame(history).to_csv(out_dir / "training_curve.csv", index=False)
    epochs_trained = len(history)

    # ---- test on unseen units, cycle-level aggregation identical to the federated eval ----
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    test_npz, test_fallback = resolve_test_npz()
    if test_fallback:
        print(f"NOTE: test set loaded from fallback {test_npz}")
    test_df = load_npz_frame(test_npz)
    test_segments = build_segments_from_dataframe(test_df, feature_cols=feature_cols, target_col="RUL_capped")
    test_ds = LazyWindowDataset(test_segments, args.window_size, args.target_offset, input_scaler, target_scaler)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False, **dl_kwargs)
    test = evaluate(model, test_loader, target_scaler, device, args.rul_cap, loss_fn)

    meta = build_window_metadata(test_df, args.window_size, args.target_offset, "RUL_capped")
    if len(meta) != len(test["preds"]):
        raise RuntimeError(f"metadata/prediction mismatch: {len(meta)} vs {len(test['preds'])}")
    window_df = meta.copy()
    window_df["true_rul"] = test["trues"]
    window_df["pred_rul"] = test["preds"]
    cycle_df = window_df.groupby(["unit", "cycle"], as_index=False).agg(
        true_rul=("true_rul", "mean"), predicted_rul=("pred_rul", "mean"), windows=("pred_rul", "size")
    )
    cycle_df["error"] = cycle_df["predicted_rul"] - cycle_df["true_rul"]
    cycle_df.to_csv(out_dir / "cycle_level_predictions.csv", index=False)

    overall = regression_metrics(cycle_df["true_rul"], cycle_df["predicted_rul"])
    per_unit = {str(int(u)): regression_metrics(d["true_rul"], d["predicted_rul"])
                for u, d in cycle_df.groupby("unit")}

    summary = {
        "seed": args.seed,
        "batch_size": args.batch_size,
        "train_units": TRAIN_UNITS,
        "val_split": "15% of each training unit cycles",
        "test_units": TEST_UNITS,
        "architecture": "LSTM hidden=128 layers=2 dropout=0.2",
        "parameters": count_parameters(model),
        "epochs_trained": epochs_trained,
        "early_stop_epoch": early_stop_epoch,
        "best_epoch": best_epoch,
        "best_val_rmse": best_rmse,
        "val_rmse_improved_epochs": improvements,
        "test_rmse": overall["rmse"],
        "test_mae": overall["mae"],
        "test_nasa": overall["nasa_score"],
        "test_bias": overall["bias"],
        "test_cycles": int(len(cycle_df)),
        "test_windows": int(len(window_df)),
        "unit_rmse": {u: m["rmse"] for u, m in per_unit.items()},
        "unit_mae": {u: m["mae"] for u, m in per_unit.items()},
        "unit_nasa": {u: m["nasa_score"] for u, m in per_unit.items()},
        "pooled_scaler": manifest["pooled"] | {"input_means": "see scaler_manifest.json", "input_stds": "see scaler_manifest.json"},
        "started": started.isoformat(timespec="seconds"),
        "finished": datetime.now().isoformat(timespec="seconds"),
        "runtime_minutes": round((datetime.now() - started).total_seconds() / 60.0, 1),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # ---- console report ----
    ref = federated_reference(args.seed)
    seed_ref = ref.get("per_seed", {}).get(args.seed)
    line = "=" * 60
    print()
    print(line)
    print("CENTRALIZED LSTM (MATCHED TO FEDERATED SETUP)")
    print(line)
    print(f"Seed          : {args.seed}")
    print(f"Batch size    : {args.batch_size}" + ("  (primary, matches federated LSTM)" if args.batch_size == 256 else "  (secondary)"))
    print(f"Train units   : {', '.join(map(str, TRAIN_UNITS))}")
    print(f"Val split     : 15% cycles per unit (seed {args.seed})")
    print(f"Test units    : {', '.join(map(str, TEST_UNITS))}")
    print(f"Epochs trained: {epochs_trained} / {args.epochs}")
    print(f"Early stop    : {f'epoch {early_stop_epoch} (patience {args.patience})' if early_stop_epoch else 'did not trigger'}")
    print(f"Best val RMSE : {best_rmse:.3f}  (epoch {best_epoch})")
    print(f"Val RMSE improved for {improvements} of {epochs_trained} epochs")
    print("-" * 60)
    print(f"TEST RESULTS (cycle-level, units {'/'.join(map(str, TEST_UNITS))}):")
    print(f"  RMSE       : {overall['rmse']:.3f} cycles")
    print(f"  MAE        : {overall['mae']:.3f} cycles")
    print(f"  NASA Score : {overall['nasa_score']:.2f}")
    print()
    print("Per-unit RMSE:")
    fed_units = ref.get("per_unit", {})
    for u in TEST_UNITS:
        c = per_unit[str(u)]["rmse"]
        f = fed_units.get(str(u))
        extra = f"   (federated seed {args.seed}: {f:.3f}, gap {c-f:+.3f})" if f is not None else ""
        print(f"  Unit {u} : {c:.3f}{extra}")
    print("-" * 60)
    if seed_ref:
        gap = overall["rmse"] - seed_ref["rmse"]
        pct = gap / seed_ref["rmse"] * 100.0
        print(f"Federated LSTM reference (seed {args.seed}) : {seed_ref['rmse']:.3f} RMSE")
        print(f"Centralized vs Federated gap       : {gap:+.3f} RMSE ({pct:+.1f}%)")
    else:
        print("Federated LSTM reference           : not found in paper_table.csv")
    print("-" * 60)
    print(f"Saved to: {out_dir}")
    print(line)


if __name__ == "__main__":
    main()
