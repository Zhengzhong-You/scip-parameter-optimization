#!/usr/bin/env python3
"""
Process a folder of SCIP logs and report the SVB-based T_infty per log.

Usage:
  source scip_env/bin/activate
  PYTHONPATH=./src python3 scripts/process.py \
    --logdir runs/smac_<inst>/<inst>/log \
    --tau 30 \
    --out runs/smac_<inst>/tinf_report.xlsx

For comprehensive analysis with all T_infty intermediate values, use:
  python3 scripts/process_enhanced.py --help

Notes:
  - Parser is restart-aware and uses the last segment after the final restart.
  - Root dual bound is taken as the last row with left == 2 in the final segment.
  - For unsolved runs, T_infty = max(tau, theta_late * b_hat);
    if there is insufficient signal, falls back to 1e9.
"""
from __future__ import annotations

import argparse
import glob
import os
from typing import Dict, Any, List

import pandas as pd

from utilities.logs import compute_T_infty, diagnose_t_infty


def _summarize_one(path: str, tau: float, allow_invalid: bool = False) -> Dict[str, Any]:
    try:
        text = open(path, "r", encoding="utf-8", errors="ignore").read()
    except Exception:
        text = ""
    # Provide empty summary; compute_T_infty and diagnose_t_infty parse progress table
    summary: Dict[str, Any] = {}
    try:
        # Strict policy: if SVB fit is invalid, compute_T_infty raises
        res = compute_T_infty(text, tau=float(tau), summary=summary)
        diag = diagnose_t_infty(text, tau=float(tau), summary=summary)
        details = res.get("details", {})
        return {
            "filename": os.path.basename(path),
            "path": os.path.abspath(path),
            "tau": float(tau),
            "T_infty": float(res.get("T_infty", float("nan"))),
            "solved": bool(res.get("solved", False)),
            "gap_end": details.get("gap" if "gap" in details else "gap_end", None) or res.get("gap"),
            "theta_late": details.get("theta"),
            "b_hat": details.get("b_hat"),
            "G_anchor": details.get("G_anchor"),
            "samples": details.get("samples"),
            "z_du_root": details.get("z_du_root"),
            "z_pr_terminal": details.get("z_pr_terminal"),
            "status": (diag.get("error") or ("ok" if str(details.get("fallback", "")) == "" else f"fallback:{details.get('fallback')}")),
        }
    except Exception as e:
        if not allow_invalid:
            raise
        # Record an error row and continue
        diag = diagnose_t_infty(text, tau=float(tau), summary=summary)
        return {
            "filename": os.path.basename(path),
            "path": os.path.abspath(path),
            "tau": float(tau),
            "T_infty": float('nan'),
            "solved": False,
            "gap_end": None,
            "theta_late": diag.get("theta"),
            "b_hat": None,
            "G_anchor": diag.get("G_anchor"),
            "samples": diag.get("kept_pairs_count"),
            "z_du_root": diag.get("z_du_root"),
            "z_pr_terminal": diag.get("z_pr_terminal"),
            "status": str(diag.get("error") or e),
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default="log", help="Path to folder containing *.log files")
    ap.add_argument("--tau", type=float, default=60.0, help="Time budget tau (seconds)")
    ap.add_argument("--out", default=None, help="Output Excel path (.xlsx). Default: <logdir>/tinf_report.xlsx")
    ap.add_argument("--allow-invalid", action="store_true", help="Do not abort on invalid SVB fits; record a row with status instead.")
    args = ap.parse_args()

    logdir = os.path.abspath(args.logdir)
    if not os.path.isdir(logdir):
        raise SystemExit(f"Log directory not found: {logdir}")

    log_paths = sorted(glob.glob(os.path.join(logdir, "*.log")))
    if not log_paths:
        raise SystemExit(f"No .log files found in: {logdir}")

    rows: List[Dict[str, Any]] = []
    import sys
    for lp in log_paths:
        try:
            rows.append(_summarize_one(lp, tau=float(args.tau), allow_invalid=bool(args.allow_invalid)))
        except Exception as e:
            print(f"ERROR: invalid SVB fit while processing {lp}: {e}")
            sys.exit(2)

    df = pd.DataFrame(rows)
    out_path = args.out or os.path.join(os.path.dirname(logdir), "tinf_report.xlsx")
    # Write Excel; require openpyxl if not present
    try:
        with pd.ExcelWriter(out_path, engine="openpyxl") as w:
            df.to_excel(w, index=False, sheet_name="T_infty")
    except Exception as e:
        # Fallback to CSV if openpyxl is missing
        csv_path = os.path.splitext(out_path)[0] + ".csv"
        df.to_csv(csv_path, index=False)
        print(f"openpyxl not available or Excel write failed ({e}); wrote CSV instead: {csv_path}")
        return
    print(f"Wrote T_infty report: {out_path}")


if __name__ == "__main__":
    main()
