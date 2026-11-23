#!/usr/bin/env python3
"""Aggregate the matched centralized LSTM runs and compare them to federated LSTM.

Reads Centralized_Study/RUL/outputs/matched/seed_<N>[_batch<B>]/summary.json and writes
summary_all_seeds.csv, then prints the centralized-vs-federated table for the paper.
Federated reference values are read from the multi-seed aggregation, never hardcoded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MATCHED_ROOT = PROJECT_ROOT / "Centralized_Study" / "RUL" / "outputs" / "matched"
FED_PAPER_TABLE = PROJECT_ROOT / "Workspace" / "fl_artifacts" / "multi_seed_summary" / "paper_table.csv"


def mean_std(values: list[float]) -> tuple[float, float]:
    vals = [v for v in values if v is not None and not np.isnan(v)]
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return float(vals[0]), float("nan")
    return float(np.mean(vals)), float(np.std(vals, ddof=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 100, 456])
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--matched-root", default=str(MATCHED_ROOT))
    args = p.parse_args()

    root = Path(args.matched_root)
    suffix = "" if args.batch_size == 256 else f"_batch{args.batch_size}"

    rows, missing = [], []
    for seed in args.seeds:
        path = root / f"seed_{seed}{suffix}" / "summary.json"
        if not path.exists():
            missing.append(str(path.relative_to(PROJECT_ROOT)))
            continue
        s = json.loads(path.read_text())
        rows.append({
            "Seed": s["seed"],
            "Batch_size": s["batch_size"],
            "Epochs": s["epochs_trained"],
            "Early_stop_epoch": s["early_stop_epoch"],
            "Best_epoch": s["best_epoch"],
            "Best_val_RMSE": round(s["best_val_rmse"], 4),
            "Val_RMSE_improved_epochs": s["val_rmse_improved_epochs"],
            "Test_RMSE": round(s["test_rmse"], 4),
            "Test_MAE": round(s["test_mae"], 4),
            "Test_NASA": round(s["test_nasa"], 4),
            "Unit11_RMSE": round(s["unit_rmse"]["11"], 4),
            "Unit14_RMSE": round(s["unit_rmse"]["14"], 4),
            "Unit15_RMSE": round(s["unit_rmse"]["15"], 4),
            "Runtime_minutes": s.get("runtime_minutes"),
        })
    if missing:
        for m in missing:
            print(f"WARNING: missing {m}")
    if not rows:
        raise SystemExit("No matched runs found.")

    table = pd.DataFrame(rows).sort_values("Seed").reset_index(drop=True)
    out_csv = root / (f"summary_all_seeds{suffix}.csv" if suffix else "summary_all_seeds.csv")
    table.to_csv(out_csv, index=False)

    c_rmse = mean_std(table["Test_RMSE"].tolist())
    c_mae = mean_std(table["Test_MAE"].tolist())
    c_nasa = mean_std(table["Test_NASA"].tolist())

    fed = None
    if FED_PAPER_TABLE.exists():
        t = pd.read_csv(FED_PAPER_TABLE)
        r = t[t["Model"] == "LSTM_FedAvg"]
        if not r.empty:
            r = r.iloc[0]
            fed = {
                "rmse": (float(r["RMSE_mean"]), float(r["RMSE_std"])),
                "mae": (float(r["MAE_mean"]), float(r["MAE_std"])),
                "nasa": (float(r["NASA_mean"]), float(r["NASA_std"])),
                "per_seed": {s: float(r[f"RMSE_seed{s}"]) for s in (42, 100, 456) if f"RMSE_seed{s}" in r.index},
            }

    print("=== CENTRALIZED LSTM MATCHED — ALL SEEDS ===")
    print(f"Seed  | Epochs | Best ep | Val RMSE | Test RMSE | Test MAE | NASA")
    print(f"------|--------|---------|----------|-----------|----------|------")
    for _, r in table.iterrows():
        print(f"{int(r['Seed']):<5} |   {int(r['Epochs']):>2}   |   {int(r['Best_epoch']):>2}    |  {r['Best_val_RMSE']:>6.3f}  |   {r['Test_RMSE']:>5.3f}   |  {r['Test_MAE']:>5.3f}   | {r['Test_NASA']:>5.1f}")
    print(f"{'Mean':<5} |        |         |          |   {c_rmse[0]:>5.3f}   |  {c_mae[0]:>5.3f}   | {c_nasa[0]:>5.1f}")
    print(f"{'Std':<5} |        |         |          |   {c_rmse[1]:>5.3f}   |  {c_mae[1]:>5.3f}   | {c_nasa[1]:>5.1f}")
    print()

    if fed:
        print("Federated LSTM reference (from multi_seed_summary/paper_table.csv):")
        print(f"  Mean RMSE : {fed['rmse'][0]:.3f} ± {fed['rmse'][1]:.3f}")
        print(f"  Mean MAE  : {fed['mae'][0]:.3f} ± {fed['mae'][1]:.3f}")
        print(f"  Mean NASA : {fed['nasa'][0]:.1f} ± {fed['nasa'][1]:.1f}")
        print()
        print("Centralized vs Federated gap:")
        for label, cen, f, digits in (("RMSE", c_rmse, fed["rmse"], 3), ("MAE", c_mae, fed["mae"], 3), ("NASA", c_nasa, fed["nasa"], 1)):
            d = cen[0] - f[0]
            print(f"  {label:<5}: {d:+.{digits}f} ({d / f[0] * 100:+.1f}% vs federated)")
        print()
        print("=== TABLE FOR PAPER — CENTRALIZED vs FEDERATED ===")
        print()
        print(f"{'':<20}| RMSE mean±std  | MAE mean±std   | NASA mean±std")
        print(f"{'-'*20}|----------------|----------------|----------------")
        print(f"{'Centralized LSTM':<20}| {c_rmse[0]:.3f} ± {c_rmse[1]:.3f}  | {c_mae[0]:.3f} ± {c_mae[1]:.3f}  | {c_nasa[0]:.1f} ± {c_nasa[1]:.1f}")
        print(f"{'Federated LSTM':<20}| {fed['rmse'][0]:.3f} ± {fed['rmse'][1]:.3f}  | {fed['mae'][0]:.3f} ± {fed['mae'][1]:.3f}  | {fed['nasa'][0]:.1f} ± {fed['nasa'][1]:.1f}")
        dr, dm, dn = c_rmse[0] - fed["rmse"][0], c_mae[0] - fed["mae"][0], c_nasa[0] - fed["nasa"][0]
        print(f"{'FL performance gap':<20}| Δ{dr:+.3f} ({dr/fed['rmse'][0]*100:+.1f}%)| Δ{dm:+.3f} ({dm/fed['mae'][0]*100:+.1f}%)| Δ{dn:+.1f} ({dn/fed['nasa'][0]*100:+.1f}%)")
        print()
        print("Note: Both use identical architecture (LSTM hidden=128, layers=2), identical test set")
        print("(units 11/14/15), identical evaluation protocol (cycle-level aggregation, pooled")
        print("training scaler), and the identical 70/15/15 cycle partition per engine. The only")
        print("difference is the training paradigm: centralized pooled training vs. federated")
        print("distributed training.")
        print()
        latex = [
            f"Centralized LSTM & {c_rmse[0]:.3f} $\\pm$ {c_rmse[1]:.3f} & {c_mae[0]:.3f} $\\pm$ {c_mae[1]:.3f} & {c_nasa[0]:.1f} $\\pm$ {c_nasa[1]:.1f} \\\\",
            f"Federated LSTM & {fed['rmse'][0]:.3f} $\\pm$ {fed['rmse'][1]:.3f} & {fed['mae'][0]:.3f} $\\pm$ {fed['mae'][1]:.3f} & {fed['nasa'][0]:.1f} $\\pm$ {fed['nasa'][1]:.1f} \\\\",
        ]
        (root / ("paper_table_latex.txt" if not suffix else f"paper_table_latex{suffix}.txt")).write_text("\n".join(latex) + "\n")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
