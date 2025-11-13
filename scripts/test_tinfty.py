#!/usr/bin/env python3
"""
Quick T_infty tester for a single SCIP log file.

Usage:
  source scip_env/bin/activate
  PYTHONPATH=./src python3 scripts/test_tinfty.py --log /path/to/log.log --tau 60

Prints the computed T_infty and key diagnostics (root-dual, terminal-primal,
theta_late, estimated b_hat, etc.). The parser is restart-aware and uses the
last row with left==2 after the final restart as the root dual bound anchor.
"""
from __future__ import annotations

import argparse
import json
import os

from utilities.logs import compute_T_infty, diagnose_t_infty


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Path to a SCIP .log file")
    ap.add_argument("--tau", type=float, default=60.0, help="Time budget tau (seconds)")
    args = ap.parse_args()

    if not os.path.exists(args.log):
        raise SystemExit(f"Log not found: {args.log}")

    with open(args.log, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # Minimal summary needed: solve_time, primal, dual, n_nodes if available
    # For a raw log file we do not necessarily have a parsed summary; we pass blanks
    summary = {}

    res = compute_T_infty(text, tau=float(args.tau), summary=summary)
    diag = diagnose_t_infty(text, tau=float(args.tau), summary=summary)

    print("T_infty:", res.get("T_infty"))
    print("Solved?", res.get("solved"))
    print("Gap_end:", res.get("gap"))
    print("Details:")
    print(json.dumps(res.get("details", {}), indent=2))
    print("\nDiagnostics:")
    print(json.dumps(diag, indent=2, default=lambda o: float(o) if hasattr(o, '__float__') else str(o)))


if __name__ == "__main__":
    main()

