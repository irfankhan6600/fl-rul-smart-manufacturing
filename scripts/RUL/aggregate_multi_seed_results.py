#!/usr/bin/env python3
"""Aggregate per-seed unseen-engine results into the paper table (mean ± std).

Reads <base>/Seed_<seed>/<model>/unseen_engines/summary_table.csv for every
model × seed and writes, under <base>/multi_seed_summary/:
  paper_table.csv, per_model_variance.csv, paper_table_latex.txt

Usage:
  python scripts/RUL/aggregate_multi_seed_results.py \
      --base-results-dir .../Workspace/fl_artifacts \
      --seeds 42 100 456 \
      --models DLinear GRU LSTM_FedAvg 1D-CNN PatchTST N-BEATS LSTM_FedAli Naive
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS = ["DLinear", "GRU", "LSTM_FedAvg", "1D-CNN", "PatchTST", "N-BEATS", "LSTM_FedAli", "Naive"]
METRICS = {"RMSE": "RMSE", "MAE": "MAE", "NASA": "NASA_Score"}


def read_cell(base: Path, seed: int, model: str) -> dict | None:
    path = base / f"Seed_{seed}" / model / "unseen_engines" / "summary_table.csv"
    if not path.exists():
        print(f"WARNING: missing {path.relative_to(base)}")
        return None
    df = pd.read_csv(path)
    if df.empty:
        print(f"WARNING: empty {path.relative_to(base)}")
        return None
    row = df.iloc[0]
    return {
        "RMSE": float(row["RMSE"]),
        "MAE": float(row["MAE"]),
        "NASA": float(row["NASA_Score"]),
        "Parameters": int(row["Parameters"]),
        "Checkpoint": str(row.get("Checkpoint_used", "")),
    }


def mean_std(values: list[float]) -> tuple[float, float, int]:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan"), 0
    if n == 1:
        return vals[0], float("nan"), 1
    return float(np.mean(vals)), float(np.std(vals, ddof=1)), n


def stability(cv: float) -> str:
    if math.isnan(cv):
        return "n/a (single seed)"
    if cv < 5:
        return "High"
    if cv <= 10:
        return "Medium"
    return "Low"


def fmt_pm(mean: float, std: float, digits: int) -> str:
    if math.isnan(mean):
        return "—"
    if math.isnan(std):
        return f"{mean:.{digits}f} ± n/a"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def latex_pm(mean: float, std: float, digits: int) -> str:
    if math.isnan(mean):
        return "--"
    if math.isnan(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def latex_int(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-results-dir", default=str(PROJECT_ROOT / "Workspace" / "fl_artifacts"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 100, 456])
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--out-dir", default=None, help="Defaults to <base-results-dir>/multi_seed_summary")
    args = parser.parse_args()

    base = Path(args.base_results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else base / "multi_seed_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, variance_rows, missing, two_seed_models = [], [], [], []
    checkpoints = set()
    for model in args.models:
        cells = {seed: read_cell(base, seed, model) for seed in args.seeds}
        done = [s for s, c in cells.items() if c]
        absent = [s for s, c in cells.items() if not c]
        missing.extend(f"{model} / seed {s}" for s in absent)
        params = next((c["Parameters"] for c in cells.values() if c), None)
        checkpoints.update(c["Checkpoint"] for c in cells.values() if c)

        row = {"Model": model, "Parameters": params}
        stats = {}
        for short, _ in METRICS.items():
            for seed in args.seeds:
                row[f"{short}_seed{seed}"] = cells[seed][short] if cells[seed] else float("nan")
            mean, std, n = mean_std([cells[s][short] for s in args.seeds if cells[s]])
            row[f"{short}_mean"] = mean
            row[f"{short}_std"] = std
            stats[short] = (mean, std, n)
        row["Seeds_completed"] = " ".join(map(str, done))
        row["Seeds_missing"] = " ".join(map(str, absent))
        row["N_seeds"] = len(done)
        if len(done) == 2:
            two_seed_models.append(model)
        rows.append(row)

        cvs = {short: (stats[short][1] / stats[short][0] * 100 if stats[short][0] and not math.isnan(stats[short][1]) else float("nan")) for short in METRICS}
        variance_rows.append({
            "Model": model,
            "N_seeds": len(done),
            "RMSE_cv": round(cvs["RMSE"], 2) if not math.isnan(cvs["RMSE"]) else float("nan"),
            "MAE_cv": round(cvs["MAE"], 2) if not math.isnan(cvs["MAE"]) else float("nan"),
            "NASA_cv": round(cvs["NASA"], 2) if not math.isnan(cvs["NASA"]) else float("nan"),
            "Stability_rating": stability(cvs["RMSE"]),
        })

    table = pd.DataFrame(rows)
    ordered_cols = ["Model", "Parameters"]
    for short in METRICS:
        ordered_cols += [f"{short}_seed{s}" for s in args.seeds] + [f"{short}_mean", f"{short}_std"]
    ordered_cols += ["Seeds_completed", "Seeds_missing", "N_seeds"]
    table = table[ordered_cols]
    table = table.sort_values("RMSE_mean", na_position="last").reset_index(drop=True)
    table.to_csv(out_dir / "paper_table.csv", index=False)

    variance = pd.DataFrame(variance_rows).set_index("Model").loc[table["Model"]].reset_index()
    variance.to_csv(out_dir / "per_model_variance.csv", index=False)

    latex_lines = []
    for _, r in table.iterrows():
        name = str(r["Model"]).replace("_", "\\_")
        params = latex_int(int(r["Parameters"])) if pd.notna(r["Parameters"]) else "--"
        latex_lines.append(
            f"{name} & {params} & {latex_pm(r['RMSE_mean'], r['RMSE_std'], 3)} & "
            f"{latex_pm(r['MAE_mean'], r['MAE_std'], 3)} & {latex_pm(r['NASA_mean'], r['NASA_std'], 1)} \\\\"
        )
    header = (
        "% Columns: Model & Params & RMSE (mean $\\pm$ std) & MAE (mean $\\pm$ std) & NASA score (mean $\\pm$ std)\n"
        f"% Seeds: {', '.join(map(str, args.seeds))}; std uses ddof=1; checkpoint FL_global_model.pt; pooled training scaler.\n"
    )
    (out_dir / "paper_table_latex.txt").write_text(header + "\n".join(latex_lines) + "\n")

    # ---- console -------------------------------------------------------------
    ckpt = ", ".join(sorted(checkpoints)) or "FL_global_model.pt"
    print("=== MULTI-SEED RESULTS (Paper Table) ===")
    print(f"Seeds: {', '.join(map(str, args.seeds))}  |  Checkpoint: {ckpt} (final)")
    print("Evaluation: pooled training scaler")
    print()
    print(f"{'Model':<11} | {'Params':>11} | {'RMSE mean±std':<17} | {'MAE mean±std':<17} | NASA mean±std")
    print(f"{'-'*11}-|-{'-'*11}-|-{'-'*17}-|-{'-'*17}-|-{'-'*15}")
    for _, r in table.iterrows():
        params = f"{int(r['Parameters']):,}" if pd.notna(r["Parameters"]) else "—"
        print(
            f"{r['Model']:<11} | {params:>11} | {fmt_pm(r['RMSE_mean'], r['RMSE_std'], 3):<17} | "
            f"{fmt_pm(r['MAE_mean'], r['MAE_std'], 3):<17} | {fmt_pm(r['NASA_mean'], r['NASA_std'], 1)}"
        )
    print()
    print("Variance Analysis:")
    print(f"{'Model':<11} | {'RMSE CV%':>9} | Stability")
    print(f"{'-'*11}-|-{'-'*9}-|----------")
    for _, r in variance.iterrows():
        cv = f"{r['RMSE_cv']:.1f}%" if pd.notna(r["RMSE_cv"]) else "n/a"
        print(f"{r['Model']:<11} | {cv:>9} | {r['Stability_rating']}")
    print()
    if two_seed_models:
        print(f"Note: std computed from only 2 seeds (ddof=1) for: {', '.join(two_seed_models)}")
    single = [r["Model"] for r in rows if r["N_seeds"] == 1]
    if single:
        print(f"Note: only 1 seed available (std = NaN) for: {', '.join(single)}")
    print("Missing results:")
    if missing:
        for m in missing:
            print(f"  {m}")
    else:
        print("  (none — all model/seed combinations complete)")
    print()
    print(f"Files saved to: {out_dir}")
    print("===")


if __name__ == "__main__":
    main()
