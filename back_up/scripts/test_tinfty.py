#!/usr/bin/env python3
"""
Quick T_infty test utility.

Usage:
  PYTHONPATH=./src python3 scripts/test_tinfty.py \
    --log runs/single_quick/.../log/..._scip_trial_2.log \
    --tau 10

This reads the SCIP log, parses a minimal summary, and computes:
  - compute_T_infty(log_text, tau, summary)
  - per_instance_T_infty({name: metrics}, tau)

It prints both results and optional diagnostics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from utilities.logs import (
    compute_T_infty,
    per_instance_T_infty,
    diagnose_t_infty,
    format_t_infty_diagnostic,
)

# Private util, but safe to import for testing
from utilities.runner import _parse_summary as _parse_summary_from_log


def main() -> int:
    ap = argparse.ArgumentParser(description="Compute T_infty from a SCIP log")
    ap.add_argument("--log", required=True, help="Path to SCIP log file")
    ap.add_argument(
        "--tau",
        type=float,
        required=True,
        help="Time limit (seconds) used during the run (needed for extrapolation)",
    )
    ap.add_argument(
        "--diagnostics",
        action="store_true",
        help="Print detailed diagnostics for the T_infty computation",
    )
    args = ap.parse_args()

    log_path = args.log
    if not os.path.exists(log_path):
        print(f"ERROR: Log file not found: {log_path}", file=sys.stderr)
        return 1

    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            log_text = f.read()
    except Exception as e:
        print(f"ERROR: Failed to read log file: {e}", file=sys.stderr)
        return 1

    # Build minimal summary from the log to pass into compute_T_infty
    summary = _parse_summary_from_log(log_path)
    # Fill a few defaults to be safe
    if "solve_time" not in summary:
        summary["solve_time"] = args.tau

    # Direct computation from raw log
    comp = compute_T_infty(log_text, tau=float(args.tau), summary=summary)

    # Per-instance API emulation (as used by pipelines)
    metrics = {
        "log_path": log_path,
        "status": summary.get("status"),
        "solve_time": summary.get("solve_time"),
        "primal": summary.get("primal"),
        "dual": summary.get("dual"),
        "gap": summary.get("gap"),
        "n_nodes": summary.get("n_nodes"),
    }
    tinf_map = per_instance_T_infty({"test": metrics}, tau=float(args.tau))

    out = {
        "compute_T_infty": comp,
        "per_instance_T_infty": {"test": tinf_map.get("test")},
    }
    print(json.dumps(out, indent=2))

    if args.diagnostics:
        try:
            diag = diagnose_t_infty(log_text, tau=float(args.tau), summary=summary)
            print("\n=== Diagnostics ===")
            print(format_t_infty_diagnostic(diag))
        except Exception as e:
            print(f"WARNING: Failed to produce diagnostics: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

