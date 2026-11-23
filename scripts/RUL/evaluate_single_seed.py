#!/usr/bin/env python3
"""Evaluate ONE model at ONE seed after its FL job completes.

Wraps scripts/RUL/evaluate_global_model_unseen.py (pooled
training scaler) so that:
  * the scaler is reconstructed with the seed the job actually trained with,
  * the job's own app/custom directory supplies data_loader.py / train.py,
  * results land in <base-results-dir>/Seed_<seed>/<model>/unseen_engines/,
  * a single-row unseen_engines/summary_table.csv is written for the aggregator,
  * the folder README.md is updated with status, job id and metrics.

Usage:
  python scripts/RUL/evaluate_single_seed.py \
      --model DLinear --seed 100 --job-id <job_id_from_fl_admin> \
      --base-results-dir .../Workspace/fl_artifacts
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_SCRIPT = PROJECT_ROOT / "scripts" / "RUL" / "evaluate_global_model_unseen.py"
JOBS_STORAGE = PROJECT_ROOT / "Workspace" / "jobs-storage"

# --model value -> original job folder (seed 42). Other seeds append _seed<N>.
MODEL_TO_JOB = {
    "DLinear": "C-MAPSS_RUL_DLinear",
    "GRU": "C-MAPSS_RUL_GRU",
    "LSTM_FedAvg": "C-MAPSS_RUL_LSTM_FedAvg",
    "1D-CNN": "C-MAPSS_RUL_1D-CNN",
    "PatchTST": "C-MAPSS_RUL_PatchTST",
    "N-BEATS": "C-MAPSS_RUL_NBeats",
    "LSTM_FedAli": "C-MAPSS_RUL_LSTM_FedAli",
    "Naive": "C-MAPSS_RUL_Naive_Baseline",
}


def job_folder(model: str, seed: int) -> str:
    base = MODEL_TO_JOB[model]
    return base if seed == 42 else f"{base}_seed{seed}"


def resolve_job_id(job_id: str) -> str:
    """Accept a full NVFlare job id or an unambiguous prefix."""
    if (JOBS_STORAGE / job_id).is_dir():
        return job_id
    matches = [p.name for p in JOBS_STORAGE.iterdir() if p.is_dir() and p.name.startswith(job_id)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No job under {JOBS_STORAGE} matches {job_id!r}")
    raise ValueError(f"Job id prefix {job_id!r} is ambiguous: {matches}")


def checkpoint_exists(job_id: str, checkpoint: str) -> bool:
    workspace = JOBS_STORAGE / job_id / "workspace"
    member = f"app_server/{checkpoint}"
    if zipfile.is_zipfile(workspace):
        with zipfile.ZipFile(workspace) as zf:
            return member in zf.namelist()
    return (workspace / member).exists()


def write_readme(path: Path, *, model: str, seed: int, job_name: str, job_id: str, completed: str,
                 rmse: float, mae: float, nasa: float, notes: str, submitted: str = "") -> None:
    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith("Submitted  :"):
                submitted = line.split(":", 1)[1].strip() or submitted
    path.write_text(
        f"# {model} — Seed {seed}\n"
        f"Status     : COMPLETE\n"
        f"Job name   : {job_name}\n"
        f"Submitted  : {submitted}\n"
        f"Job ID     : {job_id}\n"
        f"Completed  : {completed}\n"
        f"RMSE       : {rmse:.3f}\n"
        f"MAE        : {mae:.3f}\n"
        f"NASA Score : {nasa:.3f}\n"
        f"Notes      : {notes}\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_TO_JOB))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--job-id", required=True, help="NVFlare job id (full, or an unambiguous prefix)")
    parser.add_argument("--base-results-dir", default=str(PROJECT_ROOT / "Workspace" / "fl_artifacts"))
    parser.add_argument("--checkpoint", default="FL_global_model.pt", choices=["FL_global_model.pt", "best_FL_global_model.pt"])
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--python", default=sys.executable, help="Interpreter used to run the evaluation script")
    parser.add_argument("--dry-run", action="store_true", help="Resolve everything and print the command without running it")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, seed = args.model, args.seed

    results_dir = Path(args.base_results_dir) / f"Seed_{seed}" / model
    results_dir.mkdir(parents=True, exist_ok=True)

    job_name = job_folder(model, seed)
    custom_dir = PROJECT_ROOT / "federated_jobs" / job_name / "app" / "custom"
    if not (custom_dir / "data_loader.py").exists():
        raise FileNotFoundError(f"Job custom dir not found for {model} seed {seed}: {custom_dir}")

    job_id = resolve_job_id(args.job_id)
    if not checkpoint_exists(job_id, args.checkpoint):
        raise FileNotFoundError(f"{args.checkpoint} not found in {JOBS_STORAGE / job_id / 'workspace'}")

    cmd = [
        args.python, str(EVAL_SCRIPT),
        "--job-id", job_id,
        "--checkpoint", args.checkpoint,
        "--seed", str(seed),
        "--results-dir", str(results_dir),
        "--custom-dir", str(custom_dir),
        "--model-name", model,
        "--batch-size", str(args.batch_size),
    ]
    print(f"Model      : {model}\nSeed       : {seed}\nJob folder : {job_name}\nJob ID     : {job_id}")
    print(f"Custom dir : {custom_dir}\nResults    : {results_dir}\nCommand    : {' '.join(cmd)}")
    if args.dry_run:
        print("(dry run — evaluation not executed)")
        return

    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))

    suffix = "" if args.checkpoint == "best_FL_global_model.pt" else "__FL_global_model"
    cycle_csv = results_dir / "unseen_engines" / "cycle_level" / f"{model}{suffix}_cycle_metrics.csv"
    cycle = pd.read_csv(cycle_csv)
    overall = cycle[cycle["unit"] == "overall_cycle_mean"].iloc[0]

    summary = pd.DataFrame([{
        "Seed": seed,
        "Model": model,
        "Parameters": int(overall["parameters"]),
        "RMSE": round(float(overall["rmse"]), 3),
        "MAE": round(float(overall["mae"]), 3),
        "NASA_Score": round(float(overall["nasa_score"]), 3),
        "Bias": round(float(overall["bias"]), 3),
        "Cycles": int(overall["cycles"]),
        "Checkpoint_used": args.checkpoint,
        "Job_ID": job_id,
        "Job_name": job_name,
    }])
    summary_path = results_dir / "unseen_engines" / "summary_table.csv"
    summary.to_csv(summary_path, index=False)

    # Keep the manifest alongside the metrics it applies to, matching the Seed_42 layout.
    manifest = results_dir / "scaler_manifest.json"
    if manifest.exists():
        shutil.copy2(manifest, results_dir / "unseen_engines" / "scaler_manifest.json")

    completed = datetime.now().strftime("%Y-%m-%d %H:%M")
    write_readme(
        results_dir / "README.md", model=model, seed=seed, job_name=job_name, job_id=job_id, completed=completed,
        rmse=float(overall["rmse"]), mae=float(overall["mae"]), nasa=float(overall["nasa_score"]),
        notes=(f"Evaluation with pooled training scaler (split seed {seed}), checkpoint {args.checkpoint}, "
               f"cycle-level metrics on unseen units 11/14/15."),
    )

    print("================================")
    print("EVALUATION COMPLETE")
    print("================================")
    print(f"Model      : {model}")
    print(f"Seed       : {seed}")
    print(f"Job ID     : {job_id}")
    print(f"RMSE       : {float(overall['rmse']):.3f} cycles")
    print(f"MAE        : {float(overall['mae']):.3f} cycles")
    print(f"NASA Score : {float(overall['nasa_score']):.2f}")
    print(f"Saved to   : {results_dir}")
    print("================================")


if __name__ == "__main__":
    main()
