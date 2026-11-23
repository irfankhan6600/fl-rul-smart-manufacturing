#!/usr/bin/env python3
"""Build Figure 8 data with across-seed error bars (seeds 42 / 100 / 456).

Three groups, all on each client's held-out local TEST split, cycle-level metrics,
FL_global_model.pt (final checkpoint) throughout:

  FedAvg Global        - evaluate_global_model_local_clients.py output per seed
  FedAli Global        - evaluate_fedali_perspectives.py, "global" perspective
  FedAli Personalized  - evaluate_fedali_perspectives.py, "personalization" perspective

Per client and for the 6-client macro-average, reports mean and std (ddof=1) across
the three seeds. Matches the multi-seed methodology used for Table 3/4.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_EVAL = PROJECT_ROOT / "Workspace" / "fl_artifacts" / "rul_local_eval"
SEEDS = [42, 100, 456]
SITES = ["site-1", "site-2", "site-3", "site-4", "site-5", "site-6"]
SITE_UNIT = {"site-1": 2, "site-2": 5, "site-3": 10, "site-4": 16, "site-5": 18, "site-6": 20}
METRICS = [("RMSE", "cycle_rmse"), ("MAE", "cycle_mae"), ("NASA Score", "cycle_nasa_score")]
GROUPS = ["FedAvg Global", "FedAli Global", "FedAli Personalized"]
OUTLIER_RMSE = 15.0


def load_fedavg(job_id: str) -> pd.DataFrame:
    path = LOCAL_EVAL / job_id / "local_client_rul_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    d = pd.read_csv(path)
    d = d[d["split"] == "test"]           # test split only, never validation
    assert set(d["checkpoint"]) == {"FL_global_model.pt"}, f"unexpected checkpoint in {path}"
    return d.set_index("site")


def load_fedali(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    g = pd.read_csv(out_dir / "global_client_metrics.csv").set_index("target_site")
    p = pd.read_csv(out_dir / "personalization_metrics.csv").set_index("target_site")
    return g, p


def mean_std(vals: list[float]) -> tuple[float, float, int]:
    v = [x for x in vals if x is not None and not np.isnan(x)]
    if not v:
        return float("nan"), float("nan"), 0
    if len(v) == 1:
        return float(v[0]), float("nan"), 1
    return float(np.mean(v)), float(np.std(v, ddof=1)), len(v)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fedavg-jobs", nargs=3, required=True, metavar=("SEED42", "SEED100", "SEED456"))
    ap.add_argument("--fedali-dirs", nargs=3, required=True, metavar=("SEED42", "SEED100", "SEED456"))
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "Centralized_Study" / "RUL" / "outputs" / "figures"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fedavg = {s: load_fedavg(j) for s, j in zip(SEEDS, args.fedavg_jobs)}
    fedali = {s: load_fedali(Path(d)) for s, d in zip(SEEDS, args.fedali_dirs)}
    warnings: list[str] = []

    # ---- per-seed values -------------------------------------------------------
    # per_seed[group][metric][site][seed]
    per_seed: dict = {g: {m: {} for m, _ in METRICS} for g in GROUPS}
    for s in SEEDS:
        fa, (fg, fp) = fedavg[s], fedali[s]
        for site in SITES:
            for label, col in METRICS:
                per_seed["FedAvg Global"][label].setdefault(site, {})[s] = float(fa.loc[site, col])
                per_seed["FedAli Global"][label].setdefault(site, {})[s] = float(fg.loc[site, col])
                per_seed["FedAli Personalized"][label].setdefault(site, {})[s] = float(fp.loc[site, col])
        # cycle-count sanity: all three evaluations must use the same test split
        for site in SITES:
            counts = {int(fa.loc[site, "cycles"]), int(fg.loc[site, "cycles"]), int(fp.loc[site, "cycles"])}
            if len(counts) != 1:
                warnings.append(f"seed {s} {site}: differing cycle counts {counts} across groups")

    # macro-average per group/metric/seed
    macro: dict = {g: {m: {} for m, _ in METRICS} for g in GROUPS}
    for g in GROUPS:
        for label, _ in METRICS:
            for s in SEEDS:
                macro[g][label][s] = float(np.mean([per_seed[g][label][site][s] for site in SITES]))

    for g in GROUPS:
        for site in SITES:
            for s in SEEDS:
                v = per_seed[g]["RMSE"][site][s]
                if v > OUTLIER_RMSE:
                    warnings.append(f"{g} seed {s} {site} (Unit {SITE_UNIT[site]}): RMSE {v:.4f} > {OUTLIER_RMSE}")

    # ---- console ---------------------------------------------------------------
    def row(vals):
        return "|".join(f"{v:^11.4f}" if isinstance(v, float) else f"{v:^11}" for v in vals)

    print("=== LEFT PANEL DATA WITH ERROR BARS ===")
    print("(Across-seed mean ± std, ddof=1, seeds 42/100/456)")
    print()
    print("Client        |FedAvg_mean|FedAvg_std|FedAliG_mean|FedAliG_std|FedAliP_mean|FedAliP_std")
    print("--------------|-----------|----------|------------|-----------|------------|----------")
    left_rows = []
    for site in SITES:
        cells = []
        for g in GROUPS:
            m, sd, n = mean_std([per_seed[g]["RMSE"][site][s] for s in SEEDS])
            cells += [m, sd]
            left_rows.append((g, site, m, sd, n))
        lbl = f"Client {SITES.index(site)+1} (U{SITE_UNIT[site]})"
        print(f"{lbl:<14}| {cells[0]:9.4f} | {cells[1]:8.4f} |  {cells[2]:9.4f} | {cells[3]:9.4f} |  {cells[4]:9.4f} | {cells[5]:8.4f}")
    print()
    print("=== RIGHT PANEL DATA WITH ERROR BARS ===")
    print("(Macro-average across 6 clients, then mean ± std across seeds)")
    print()
    print("Metric    |FedAvg_mean|FedAvg_std|FedAliG_mean|FedAliG_std|FedAliP_mean|FedAliP_std")
    print("----------|-----------|----------|------------|-----------|------------|----------")
    right_rows = []
    for label, _ in METRICS:
        cells = []
        for g in GROUPS:
            m, sd, n = mean_std([macro[g][label][s] for s in SEEDS])
            cells += [m, sd]
            right_rows.append((g, label, m, sd, n))
        print(f"{label:<10}| {cells[0]:9.4f} | {cells[1]:8.4f} |  {cells[2]:9.4f} | {cells[3]:9.4f} |  {cells[4]:9.4f} | {cells[5]:8.4f}")

    print()
    print("=== PER-SEED RAW VALUES (for verification) ===")
    for g in GROUPS:
        print()
        print(f"LEFT PANEL — {g} per seed (cycle RMSE):")
        print("Client        | Seed 42 | Seed 100 | Seed 456")
        print("--------------|---------|----------|----------")
        for site in SITES:
            v = per_seed[g]["RMSE"][site]
            print(f"Client {SITES.index(site)+1} (U{SITE_UNIT[site]})".ljust(14) + f"| {v[42]:7.4f} |  {v[100]:7.4f} |  {v[456]:7.4f}")
    print()
    print("RIGHT PANEL — macro-average per seed:")
    for g in GROUPS:
        print(f"  {g}:")
        print("  Metric    | Seed 42 | Seed 100 | Seed 456")
        for label, _ in METRICS:
            v = macro[g][label]
            print(f"  {label:<10}| {v[42]:7.4f} |  {v[100]:7.4f} |  {v[456]:7.4f}")

    # ---- CSVs ------------------------------------------------------------------
    main_rows, raw_rows = [], []
    for panel, items in (("left", [(g, site, "RMSE") for g in GROUPS for site in SITES]),
                         ("right", [(g, label, label) for g in GROUPS for label, _ in METRICS])):
        for g, key, metric in items:
            if panel == "left":
                vals = {s: per_seed[g]["RMSE"][key][s] for s in SEEDS}
                name = f"Client {SITES.index(key)+1} (Unit {SITE_UNIT[key]})"
            else:
                vals = {s: macro[g][key][s] for s in SEEDS}
                name = key
            m, sd, n = mean_std(list(vals.values()))
            main_rows.append({"Panel": panel, "Group": g, "Client_or_Metric": name,
                              "Mean_seed42": round(vals[42], 4), "Mean_seed100": round(vals[100], 4),
                              "Mean_seed456": round(vals[456], 4), "Mean_across_seeds": round(m, 4),
                              "Std_across_seeds": round(sd, 4), "N_seeds": n})
            for s in SEEDS:
                raw_rows.append({"Panel": panel, "Group": g, "Client_or_Metric": name,
                                 "Metric": metric, "Seed": s, "Value": round(vals[s], 6)})
    pd.DataFrame(main_rows).to_csv(out_dir / "Figure8_data_with_errorbars.csv", index=False)
    pd.DataFrame(raw_rows).to_csv(out_dir / "Figure8_raw_per_seed.csv", index=False)

    print()
    if warnings:
        print("=== WARNINGS ===")
        for w in warnings:
            print("  " + w)
    else:
        print("=== WARNINGS === none (no client RMSE > 15; cycle counts consistent across groups)")
    print()
    print("=== CAPTION TEXT TO INCLUDE IN PAPER ===")
    print('"Error bars show standard deviation across three independent random seeds (42, 100, 456)."')
    print()
    print(f"Saved: {out_dir/'Figure8_data_with_errorbars.csv'} ({len(main_rows)} rows)")
    print(f"Saved: {out_dir/'Figure8_raw_per_seed.csv'} ({len(raw_rows)} rows)")


if __name__ == "__main__":
    main()
