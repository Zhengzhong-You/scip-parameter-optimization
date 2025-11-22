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
import tempfile

from utilities.scip_cli import ensure_version, _scip_bin


def create_scip_settings_file(time_limit: float) -> str:
    """Create .set file with parameters (matching runner.py approach)"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".set", delete=False) as f:
        f.write(f"limits/time = {float(time_limit)}\n")  # .set file uses / syntax
        return f.name


def build_scip_script(instance_path: str) -> str:
    """Build script commands (no time limit here, it's in .set file)"""
    cmds = [
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

    # Create .set file with time limit (matching runner.py approach)
    set_file_path = create_scip_settings_file(args.time_limit)
    script = build_scip_script(instance_path)

    print(f"Running SCIP 9.2.4 on {instance_path} (time limit: {args.time_limit}s)...", flush=True)

    try:
        proc = subprocess.run(
            [scip_bin, "-s", set_file_path],  # Use -s flag like runner.py
            input=script,
            text=True,
        )

        if proc.returncode != 0:
            print(f"SCIP exited with return code {proc.returncode}", file=sys.stderr)
            sys.exit(proc.returncode)

    finally:
        # Clean up temporary .set file
        try:
            os.unlink(set_file_path)
        except OSError:
            pass


if __name__ == "__main__":
    main()
