#!/usr/bin/env python3
"""Build the per-client and held-out test .npz files from N-CMAPSS DS02.

Script version of the notebook used to create the data for the paper
(Create_dataset_for_Clents.ipynb). The logic is unchanged:

  * development block (A_dev, W_dev, X_s_dev, X_v_dev) -> one file per training unit
      client_unit_{2,5,10,16,18,20}.npz
  * test block (A_test, W_test, X_s_test, X_v_test)     -> one combined file
      test_units_11_14_15.npz
  * manifest.csv with row and cycle counts

Each .npz stores one 1-D array per column (unit, cycle, W vars, X_s vars, X_v vars,
T50_target). The RUL target is NOT stored; data_loader.py derives it at load time.

Usage:
    python data_preparation/create_client_datasets.py \
        --h5 /path/to/N-CMAPSS_DS02-006.h5 \
        --out-dir data/fl_dataset_ds02_006
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

CLIENT_UNITS = [2, 5, 10, 16, 18, 20]
TEST_UNITS = [11, 14, 15]
TARGET_NAME = "T50"  # kept as T50_target for compatibility with the loader schema


def decode_names(arr) -> list[str]:
    return [x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x) for x in arr]


def save_npz_dataframe(df: pd.DataFrame, filepath: Path) -> None:
    np.savez_compressed(filepath, **{col: df[col].to_numpy() for col in df.columns})


def build_block_dataframe(A, W, Xs, Xv, A_var, W_var, Xs_var, Xv_var) -> pd.DataFrame:
    df = pd.DataFrame()
    df["unit"] = A[:, A_var.index("unit")].astype(int)
    df["cycle"] = A[:, A_var.index("cycle")].astype(int)
    for col in W_var:
        df[col] = W[:, W_var.index(col)]
    for col in Xs_var:
        df[col] = Xs[:, Xs_var.index(col)]
    for col in Xv_var:
        df[col] = Xv[:, Xv_var.index(col)]
    if TARGET_NAME not in df.columns:
        raise KeyError(f"{TARGET_NAME} not found in X_s variables.")
    df["T50_target"] = df[TARGET_NAME].astype(np.float32)
    return df


def select_units(df: pd.DataFrame, units: list[int]) -> pd.DataFrame:
    out = df[df["unit"].isin(units)].copy()
    out["unit"] = out["unit"].astype(int)
    out["cycle"] = out["cycle"].astype(int)
    return out.sort_values(["unit", "cycle"], kind="mergesort").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", required=True, help="Path to N-CMAPSS_DS02-006.h5")
    ap.add_argument("--out-dir", default="data/fl_dataset_ds02_006")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.h5, "r") as hdf:
        blocks = {split: [np.array(hdf.get(f"{k}_{split}")) for k in ("A", "W", "X_s", "X_v")]
                  for split in ("dev", "test")}
        names = [decode_names(np.array(hdf.get(f"{k}_var"))) for k in ("A", "W", "X_s", "X_v")]

    df_dev = build_block_dataframe(*blocks["dev"], *names)
    df_test = build_block_dataframe(*blocks["test"], *names)
    print("Dev shape:", df_dev.shape, "units:", sorted(df_dev["unit"].unique().tolist()))
    print("Test shape:", df_test.shape, "units:", sorted(df_test["unit"].unique().tolist()))

    rows = []
    for u in CLIENT_UNITS:
        unit_df = select_units(df_dev, [u])
        path = out_dir / f"client_unit_{u}.npz"
        save_npz_dataframe(unit_df, path)
        rows.append({"unit": u, "file": str(path), "rows": len(unit_df),
                     "cycles": int(unit_df["cycle"].nunique()), "split": "train"})
        print(f"saved {path} rows={len(unit_df)}")

    test_df = select_units(df_test, TEST_UNITS)
    test_path = out_dir / "test_units_11_14_15.npz"
    save_npz_dataframe(test_df, test_path)
    rows.append({"unit_group": "11_14_15", "file": str(test_path), "rows": len(test_df),
                 "cycles": int(test_df["cycle"].nunique()), "split": "test"})
    print(f"saved {test_path} rows={len(test_df)}")

    manifest = pd.DataFrame(rows)
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    print(manifest)


if __name__ == "__main__":
    main()
