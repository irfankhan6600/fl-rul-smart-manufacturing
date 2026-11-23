#!/usr/bin/env python3
"""Export NVFlare job metrics to CSV files.

This reads a completed job from Workspace/jobs-storage/<job_id>/workspace,
which is a zip archive, and writes CSVs with round timing, per-client
validation metrics, and communication events.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Iterable


TS_RE = r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})"
ROUND_STARTED_RE = re.compile(TS_RE + r".*Round (?P<round>\d+) started\.")
ROUND_FINISHED_RE = re.compile(TS_RE + r".*Round (?P<round>\d+) finished\.")
CONTRIB_RE = re.compile(
    TS_RE
    + r".*peer=(?P<client>site-\d+).*task_id=(?P<task_id>[a-f0-9-]+)\] - "
    + r"Contribution from (?P=client) ACCEPTED by the aggregator at round (?P<round>\d+)\."
)
VAL_RE = re.compile(
    TS_RE
    + r".*peer=(?P<client>site-\d+).*task_id=(?P<task_id>[a-f0-9-]+)\] - "
    + r"validation metric (?P<metric>[0-9.eE+-]+) from client (?P=client)"
)
BEST_RE = re.compile(TS_RE + r".*new best validation metric at round (?P<round>\d+): (?P<metric>[0-9.eE+-]+)")
TX_RE = re.compile(
    TS_RE
    + r".*download tx (?P<tx_id>\S+) done: status=(?P<status>\w+) "
    + r"elapsed=(?P<elapsed>[0-9.]+)s size=(?P<size_mb>[0-9.]+)MB \((?P<bytes>[0-9,]+) bytes\)"
)
CLIENT_DOWNLOAD_RE = re.compile(
    TS_RE
    + r".*\[client\] download ref=(?P<ref_id>\S+) done: "
    + r"elapsed=(?P<elapsed>[0-9.]+)s size=(?P<size_mb>[0-9.]+)MB \((?P<bytes>[0-9,]+) bytes\)"
)


def parse_ts(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S,%f")


def read_zip_text(zf: zipfile.ZipFile, name: str) -> str:
    try:
        return zf.read(name).decode("utf-8", errors="replace")
    except KeyError:
        return ""


def read_workspace_text(workspace_path: Path, name: str) -> str:
    if workspace_path.is_dir():
        path = workspace_path / name
        return path.read_text(errors="replace") if path.exists() else ""
    with zipfile.ZipFile(workspace_path) as zf:
        return read_zip_text(zf, name)


def list_workspace_names(workspace_path: Path) -> set[str]:
    if workspace_path.is_dir():
        return {str(path.relative_to(workspace_path)) for path in workspace_path.rglob("*") if path.is_file()}
    with zipfile.ZipFile(workspace_path) as zf:
        return set(zf.namelist())


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def find_round_for_timestamp(ts: datetime, round_rows: list[dict]) -> int | None:
    for row in round_rows:
        if not row["started_at"] or not row["finished_at"]:
            continue
        started = datetime.fromisoformat(row["started_at"])
        finished = datetime.fromisoformat(row["finished_at"])
        if started <= ts <= finished:
            return int(row["round"])
    return None


def export_job_metrics(job_id: str, workspace_root: Path, output_dir: Path) -> None:
    job_dir = workspace_root / "Workspace" / "jobs-storage" / job_id
    workspace_path = job_dir / "workspace"
    meta_path = job_dir / "meta"
    if not workspace_path.exists():
        raise FileNotFoundError(f"Could not find job workspace archive or directory: {workspace_path}")

    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())

    log_fl = read_workspace_text(workspace_path, "log_fl.txt")
    log_txt = read_workspace_text(workspace_path, "log.txt")
    log_error = read_workspace_text(workspace_path, "log_error.txt")
    names = list_workspace_names(workspace_path)

    round_started: dict[int, datetime] = {}
    round_finished: dict[int, datetime] = {}
    client_rows: list[dict] = []
    best_rows: list[dict] = []
    val_by_task: dict[tuple[int, str, str], float] = {}

    for line in log_fl.splitlines():
        if m := ROUND_STARTED_RE.search(line):
            round_started[int(m.group("round"))] = parse_ts(m.group("ts"))
            continue
        if m := ROUND_FINISHED_RE.search(line):
            round_finished[int(m.group("round"))] = parse_ts(m.group("ts"))
            continue
        if m := VAL_RE.search(line):
            # The current round is inferred from the nearest contribution line
            # when possible, so this map is keyed by client/task for joining.
            val_by_task[(-1, m.group("client"), m.group("task_id"))] = float(m.group("metric"))
            continue
        if m := CONTRIB_RE.search(line):
            round_id = int(m.group("round"))
            client = m.group("client")
            task_id = m.group("task_id")
            metric = val_by_task.get((-1, client, task_id))
            client_rows.append(
                {
                    "job_id": job_id,
                    "round": round_id,
                    "client": client,
                    "task_id": task_id,
                    "accepted_at": parse_ts(m.group("ts")).isoformat(sep=" "),
                    "validation_metric": metric if metric is not None else "",
                }
            )
            continue
        if m := BEST_RE.search(line):
            best_rows.append(
                {
                    "job_id": job_id,
                    "round": int(m.group("round")),
                    "best_validation_metric": float(m.group("metric")),
                    "timestamp": parse_ts(m.group("ts")).isoformat(sep=" "),
                }
            )

    metrics_by_round: dict[int, list[float]] = {}
    contribs_by_round: dict[int, int] = {}
    for row in client_rows:
        round_id = int(row["round"])
        contribs_by_round[round_id] = contribs_by_round.get(round_id, 0) + 1
        if row["validation_metric"] != "":
            metrics_by_round.setdefault(round_id, []).append(float(row["validation_metric"]))

    round_rows = []
    for round_id in sorted(set(round_started) | set(round_finished) | set(contribs_by_round)):
        started = round_started.get(round_id)
        finished = round_finished.get(round_id)
        vals = metrics_by_round.get(round_id, [])
        round_rows.append(
            {
                "job_id": job_id,
                "round": round_id,
                "started_at": started.isoformat(sep=" ") if started else "",
                "finished_at": finished.isoformat(sep=" ") if finished else "",
                "duration_sec": round((finished - started).total_seconds(), 3) if started and finished else "",
                "num_client_contributions": contribs_by_round.get(round_id, 0),
                "avg_validation_metric": round(mean(vals), 6) if vals else "",
                "min_validation_metric": round(min(vals), 6) if vals else "",
                "max_validation_metric": round(max(vals), 6) if vals else "",
            }
        )

    comm_rows = []
    for line in log_txt.splitlines():
        if m := TX_RE.search(line):
            ts = parse_ts(m.group("ts"))
            comm_rows.append(
                {
                    "job_id": job_id,
                    "timestamp": ts.isoformat(sep=" "),
                    "round": find_round_for_timestamp(ts, round_rows),
                    "direction": "server_to_client_tensor_download",
                    "tx_id": m.group("tx_id"),
                    "status": m.group("status"),
                    "elapsed_sec": float(m.group("elapsed")),
                    "size_mb": float(m.group("size_mb")),
                    "bytes": int(m.group("bytes").replace(",", "")),
                }
            )
            continue
        if m := CLIENT_DOWNLOAD_RE.search(line):
            ts = parse_ts(m.group("ts"))
            comm_rows.append(
                {
                    "job_id": job_id,
                    "timestamp": ts.isoformat(sep=" "),
                    "round": find_round_for_timestamp(ts, round_rows),
                    "direction": "client_download_ref_logged",
                    "tx_id": m.group("ref_id"),
                    "status": "finished",
                    "elapsed_sec": float(m.group("elapsed")),
                    "size_mb": float(m.group("size_mb")),
                    "bytes": int(m.group("bytes").replace(",", "")),
                }
            )

    comm_by_round_rows = []
    all_round_ids = sorted({int(row["round"]) for row in round_rows})
    for round_id in all_round_ids:
        rows = [row for row in comm_rows if row["round"] not in (None, "") and int(row["round"]) == round_id]
        server_download_mb = sum(
            float(row["size_mb"]) for row in rows if row["direction"] == "server_to_client_tensor_download"
        )
        client_download_mb = sum(
            float(row["size_mb"]) for row in rows if row["direction"] == "client_download_ref_logged"
        )
        comm_by_round_rows.append(
            {
                "job_id": job_id,
                "round": round_id,
                "server_to_client_tensor_download_mb": round(server_download_mb, 6),
                "client_download_ref_logged_mb": round(client_download_mb, 6),
                "total_logged_communication_mb": round(server_download_mb + client_download_mb, 6),
                "num_communication_events": len(rows),
            }
        )

    total_server_download_mb = sum(
        float(row["size_mb"]) for row in comm_rows if row["direction"] == "server_to_client_tensor_download"
    )
    total_client_download_mb = sum(
        float(row["size_mb"]) for row in comm_rows if row["direction"] == "client_download_ref_logged"
    )
    finished_rounds = [row for row in round_rows if row["finished_at"]]
    metric_rounds = [row for row in round_rows if row["avg_validation_metric"] != ""]
    final_metric_row = metric_rounds[-1] if metric_rounds else {}
    best_metric = min((float(row["avg_validation_metric"]) for row in metric_rounds), default="")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "round_summary.csv",
        round_rows,
        [
            "job_id",
            "round",
            "started_at",
            "finished_at",
            "duration_sec",
            "num_client_contributions",
            "avg_validation_metric",
            "min_validation_metric",
            "max_validation_metric",
        ],
    )
    write_csv(
        output_dir / "client_round_metrics.csv",
        client_rows,
        ["job_id", "round", "client", "task_id", "accepted_at", "validation_metric"],
    )
    write_csv(
        output_dir / "communication_events.csv",
        comm_rows,
        ["job_id", "timestamp", "round", "direction", "tx_id", "status", "elapsed_sec", "size_mb", "bytes"],
    )
    write_csv(
        output_dir / "communication_by_round.csv",
        comm_by_round_rows,
        [
            "job_id",
            "round",
            "server_to_client_tensor_download_mb",
            "client_download_ref_logged_mb",
            "total_logged_communication_mb",
            "num_communication_events",
        ],
    )
    write_csv(
        output_dir / "best_model_events.csv",
        best_rows,
        ["job_id", "round", "best_validation_metric", "timestamp"],
    )

    summary_rows = [
        {
            "job_id": job_id,
            "name": meta.get("name", ""),
            "status": meta.get("status", ""),
            "start_time": meta.get("start_time", ""),
            "duration": meta.get("duration", ""),
            "num_rounds_finished": sum(1 for row in round_rows if row["finished_at"]),
            "avg_round_duration_sec": round(
                mean(float(row["duration_sec"]) for row in finished_rounds if row["duration_sec"] != ""), 3
            )
            if finished_rounds
            else "",
            "final_round": final_metric_row.get("round", ""),
            "final_avg_validation_metric": final_metric_row.get("avg_validation_metric", ""),
            "best_avg_validation_metric": round(best_metric, 6) if best_metric != "" else "",
            "total_server_to_client_tensor_download_mb": round(total_server_download_mb, 6),
            "total_client_download_ref_logged_mb": round(total_client_download_mb, 6),
            "total_logged_communication_mb": round(total_server_download_mb + total_client_download_mb, 6),
            "has_global_model": "app_server/FL_global_model.pt" in names,
            "has_best_global_model": "app_server/best_FL_global_model.pt" in names,
            "log_error_bytes": len(log_error.encode("utf-8")),
        }
    ]
    write_csv(
        output_dir / "job_summary.csv",
        summary_rows,
        [
            "job_id",
            "name",
            "status",
            "start_time",
            "duration",
            "num_rounds_finished",
            "avg_round_duration_sec",
            "final_round",
            "final_avg_validation_metric",
            "best_avg_validation_metric",
            "total_server_to_client_tensor_download_mb",
            "total_client_download_ref_logged_mb",
            "total_logged_communication_mb",
            "has_global_model",
            "has_best_global_model",
            "log_error_bytes",
        ],
    )

    print(f"Wrote CSV files to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export NVFlare job metrics to CSV.")
    parser.add_argument("job_id", help="NVFlare job ID, e.g. 1a8d762d-...")
    parser.add_argument(
        "--workspace-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Federated_Learning workspace root. Defaults to current directory.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to Workspace/fl_artifacts/job_metrics/<job_id>.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    output_dir = (
        Path(args.out_dir).resolve()
        if args.out_dir
        else workspace_root / "Workspace" / "fl_artifacts" / "job_metrics" / args.job_id
    )
    export_job_metrics(args.job_id, workspace_root, output_dir)


if __name__ == "__main__":
    main()
