"""Launch a site process on a deterministic GPU.

NVFlare runs one external training process per site. When several local sites
run on the same workstation, each process may otherwise see all GPUs and crowd
onto cuda:0. This wrapper sets CUDA_VISIBLE_DEVICES before train.py imports
PyTorch, then replaces itself with the real training command.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, Optional


DEFAULT_SITE_GPU_MAP: Dict[str, str] = {
    "site-1": "0",
    "site-2": "0",
    "site-3": "0",
    "site-4": "1",
    "site-5": "1",
    "site-6": "1",
}


def parse_gpu_map(value: Optional[str]) -> Dict[str, str]:
    if not value:
        return DEFAULT_SITE_GPU_MAP

    mapping: Dict[str, str] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid FL_GPU_MAP entry: {item!r}. Expected site-1:0")
        site, gpu = item.split(":", 1)
        mapping[site.strip()] = gpu.strip()
    return mapping or DEFAULT_SITE_GPU_MAP


def infer_site_name(raw_site_name: Optional[str]) -> Optional[str]:
    candidates = [
        raw_site_name,
        os.environ.get("NVFLARE_SITE_NAME"),
        os.environ.get("SITE_NAME"),
        os.environ.get("CLIENT_NAME"),
        os.environ.get("NVFL_CLIENT_NAME"),
    ]

    cwd_parts = Path.cwd().parts
    candidates.extend(part for part in cwd_parts if re.fullmatch(r"site-\d+", part))

    for candidate in candidates:
        if candidate and not candidate.startswith("{"):
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Pin an NVFlare site process to a GPU.")
    parser.add_argument("--site-name", default=None)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("No command provided to site_gpu_launcher.py")

    site_name = infer_site_name(args.site_name)
    gpu_map = parse_gpu_map(os.environ.get("FL_GPU_MAP"))
    gpu_id = gpu_map.get(site_name or "")

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        print(
            f"[site_gpu_launcher.py] site={site_name} CUDA_VISIBLE_DEVICES={gpu_id}",
            flush=True,
        )
    else:
        print(
            f"[site_gpu_launcher.py] site={site_name} has no GPU mapping; leaving CUDA_VISIBLE_DEVICES unchanged",
            flush=True,
        )

    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
