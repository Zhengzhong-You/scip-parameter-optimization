#!/usr/bin/env python3
"""
Concise T_infty analysis with only essential intermediate values.

Usage:
  PYTHONPATH=./src python scripts/process_concise.py \
    --logdir path/to/logs --paramdir path/to/params --out results.xlsx
"""

import argparse
import glob
import os
import re
import pandas as pd

def extract_svb_essentials(log_path: str, param_path: str, tau: float = 60.0):
    """Extract only essential T_infty intermediate values."""
    import sys
    sys.path.insert(0, 'src')
    from utilities.logs import compute_T_infty

    # Parse parameter file for key params
    key_params = {}
    if param_path and os.path.exists(param_path):
        try:
            with open(param_path, 'r') as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        key = key.strip()
                        # Only keep key parameters
                        if any(k in key for k in ['presolving', 'branching', 'limits/time']):
                            try:
                                key_params[key] = float(value.strip()) if '.' in value else int(value.strip())
                            except:
                                key_params[key] = value.strip()
        except Exception:
            pass

    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            log_text = f.read()

        summary = {}
        result = compute_T_infty(log_text, tau=tau, summary=summary)
        details = result.get("details", {})

        return {
            "filename": os.path.basename(log_path),
            "trial": re.search(r'trial_(\d+)', log_path).group(1) if re.search(r'trial_(\d+)', log_path) else "0",

            # Essential T_infty components
            "a_hat": details.get("a"),           # â = log(Ĉ * log φ̂)
            "kappa_hat": details.get("kappa"),   # κ̂ = log φ̂
            "C_hat": details.get("C"),           # Ĉ = e^â / κ̂
            "phi_hat": details.get("varphi"),       # φ̂ = e^κ̂
            "r_squared": details.get("r_squared"),   # Regression quality

            # Time estimates
            "theta_hat": details.get("theta"),       # θ̂: time per node
            "b_hat": details.get("b_hat"),           # b̂: estimated nodes
            "T_infty": result.get("T_infty"),        # T̂_∞: final estimate

            # Bounds & gap
            "z_du_root": details.get("z_du_root"),   # Root dual bound
            "z_pr_ter": details.get("z_pr_terminal"), # Terminal primal
            "target_gap": details.get("gap", details.get("gap_end")), # Target gap

            # Status
            "solved": result.get("solved", False),
            "status": details.get("fallback", "ok" if result.get("T_infty") else "failed"),

            # Key parameters (flattened)
            **{f"param_{k.split('/')[-1]}": v for k, v in key_params.items()},
        }

    except Exception as e:
        return {
            "filename": os.path.basename(log_path),
            "trial": re.search(r'trial_(\d+)', log_path).group(1) if re.search(r'trial_(\d+)', log_path) else "0",
            "error": str(e)
        }

def main():
    ap = argparse.ArgumentParser(description="Concise T_infty analysis")
    ap.add_argument("--logdir", required=True, help="Log directory")
    ap.add_argument("--paramdir", help="Parameter directory")
    ap.add_argument("--tau", type=float, default=60.0, help="Time budget")
    ap.add_argument("--out", default="tinf_essentials.xlsx", help="Output file")
    args = ap.parse_args()

    log_paths = sorted(glob.glob(os.path.join(args.logdir, "*.log")))
    if not log_paths:
        raise SystemExit(f"No logs found in {args.logdir}")

    results = []
    for log_path in log_paths:
        trial_num = re.search(r'trial_(\d+)', log_path).group(1) if re.search(r'trial_(\d+)', log_path) else "0"
        param_path = None
        if args.paramdir:
            param_path = os.path.join(args.paramdir, f"params_{trial_num}.set")

        results.append(extract_svb_essentials(log_path, param_path, args.tau))

    df = pd.DataFrame(results)

    try:
        df.to_excel(args.out, index=False)
        print(f"✅ Results: {args.out}")
        print(f"📊 Files: {len(results)}")
    except:
        csv_path = args.out.replace('.xlsx', '.csv')
        df.to_csv(csv_path, index=False)
        print(f"✅ Results: {csv_path}")

if __name__ == "__main__":
    main()