#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <run_label_or_job_id> [interval_seconds]" >&2
  exit 1
fi

label="$1"
interval="${2:-5}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/../.." && pwd)"
out_dir="${project_root}/Workspace/fl_artifacts/resource_logs"
out_file="${out_dir}/${label}_gpu.csv"

mkdir -p "${out_dir}"

echo "Writing GPU metrics to: ${out_file}" >&2
echo "Press Ctrl+C to stop." >&2

nvidia-smi \
  --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw \
  --format=csv \
  -l "${interval}" \
  > "${out_file}"
