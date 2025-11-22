#!/usr/bin/env python3
"""
Run a single instance with SCIP 9.2.4 using default parameters for tree-size estimation.

- Uses default SCIP parameters (no overrides).
- Sets a 1200s time limit (override with --time-limit if needed).
- Streams output directly to the terminal; no log files are written.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from utilities.scip_cli import ensure_version, _scip_bin


def build_scip_script(instance_path: str, time_limit: float) -> str:
    cmds = [
        f"set limits time {float(time_limit)}",
        f"read {instance_path}",
        "optimize",
        "quit",
    ]
    return "\n".join(cmds) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SCIP 9.2.4 on an instance for tree-size estimation without writing logs."
    )
    parser.add_argument("instance", help="Path to the instance file")
    parser.add_argument(
        "--time-limit",
        type=float,
        default=1200.0,
        help="Time limit in seconds (default: 1200)",
    )
    args = parser.parse_args()

    instance_path = os.path.abspath(args.instance)
    if not os.path.isfile(instance_path):
        parser.error(f"Instance not found: {instance_path}")

    # Ensure SCIP CLI version is exactly 9.2.4
    ensure_version("9.2.4")
    scip_bin = _scip_bin()

    script = build_scip_script(instance_path, args.time_limit)

    print(f"Running SCIP 9.2.4 on {instance_path} (time limit: {args.time_limit}s)...", flush=True)
    proc = subprocess.run(
        [scip_bin],
        input=script,
        text=True,
    )

    if proc.returncode != 0:
        print(f"SCIP exited with return code {proc.returncode}", file=sys.stderr)
        sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
