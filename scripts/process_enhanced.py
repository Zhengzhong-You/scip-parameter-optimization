#!/usr/bin/env python3
"""
Enhanced process script for comprehensive T_infty analysis with all intermediate values.

Extracts all mathematical components from the SVB formulation:
- SVB parameters: Ĉ, φ̂, κ̂, â
- Regression data: log(Δb_j/ΔG_j) vs Ḡ_j pairs
- Time estimates: θ̂(p,i;E,τ), b̂(p,i;E,τ), T̂_∞(p;i,E,τ)
- Dual/primal bounds: z^du_root, z^pr_ter
- Parameter values from .set files

Usage:
  source scip_env/bin/activate
  PYTHONPATH=./src python3 scripts/process_enhanced.py \
    --logdir back_up/runs_back/single_quick/XML100_3346_25_50.mps/XML100_3346_25_50.mps/log \
    --paramdir back_up/runs_back/single_quick/XML100_3346_25_50.mps/XML100_3346_25_50.mps \
    --tau 60 \
    --out comprehensive_analysis.xlsx
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from typing import Dict, Any, List, Optional
import math

import pandas as pd

from utilities.logs import compute_T_infty
from utilities.logs import parse_progress_series


def parse_set_file(path: str) -> Dict[str, Any]:
    """Parse SCIP .set file to extract parameter values."""
    params = {}
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()

                    # Try to parse value as number
                    try:
                        if '.' in value:
                            params[key] = float(value)
                        else:
                            params[key] = int(value)
                    except ValueError:
                        params[key] = value
    except Exception as e:
        print(f"Warning: Could not parse {path}: {e}")

    return params


def extract_svb_detailed_analysis(log_text: str, tau: float = 60.0) -> Dict[str, Any]:
    """
    Detailed SVB analysis extracting all intermediate mathematical components.

    Returns comprehensive breakdown of T_infty calculation including:
    - SVB regression parameters (â, κ̂, Ĉ, φ̂)
    - All regression pairs (Δb_j, ΔG_j, Ḡ_j)
    - Time estimates (θ̂, b̂, T̂_∞)
    - Dual/primal bounds
    """

    result = {
        # Basic info
        "log_length": len(log_text),
        "tau": tau,

        # SVB regression parameters
        "a_hat": None,           # intercept: â = log(Ĉ * log φ̂)
        "kappa_hat": None,       # slope: κ̂ = log φ̂
        "C_hat": None,           # SVB constant: Ĉ = e^â / κ̂
        "phi_hat": None,         # SVB base: φ̂ = e^κ̂

        # Regression quality
        "n_regression_pairs": 0,
        "r_squared": None,
        "residual_sum_squares": None,

        # Gap/node analysis
        "z_du_root": None,       # Root dual bound
        "z_pr_ter": None,        # Terminal primal bound
        "target_gap": None,      # |z^pr_ter - z^du_root|

        # Time analysis
        "theta_hat": None,       # Time per node: θ̂
        "t_late": None,          # Late-stage time
        "b_obs_late": None,      # Late-stage nodes observed

        # SVB extrapolation
        "b_hat": None,           # Estimated total nodes: b̂(G) = Ĉ * φ̂^G
        "T_infty": None,         # Final time estimate

        # Status
        "solved": False,
        "status": "unknown",
        "error_message": None,

        # Regression details
        "regression_pairs": [],  # List of {Delta_G_j, Delta_b_j, G_bar_j, log_ratio}
        "samples_used": 0,

        # Final solver state
        "final_gap": None,
        "final_nodes": None,
        "final_time": None,
    }

    try:
        # First use existing T_infty computation
        summary = {}
        t_infty_result = compute_T_infty(log_text, tau=tau, summary=summary)

        # Extract basic results
        result["T_infty"] = t_infty_result.get("T_infty", None)
        result["solved"] = t_infty_result.get("solved", False)

        details = t_infty_result.get("details", {})
        result["theta_hat"] = details.get("theta")
        result["b_hat"] = details.get("b_hat")
        result["target_gap"] = details.get("gap", details.get("gap_end"))
        result["z_du_root"] = details.get("z_du_root")
        result["z_pr_terminal"] = details.get("z_pr_terminal")

        # Now do detailed SVB analysis
        columns, rows = parse_progress_series(log_text)

        if len(rows) < 3:
            result["status"] = "insufficient_data"
            result["error_message"] = f"Only {len(rows)} progress rows found, need at least 3"
            return result

        # Extract valid data points for SVB analysis
        valid_data = []
        for row in rows:
            if all(x is not None for x in [row.get('node'), row.get('primal'), row.get('dual'), row.get('time')]):
                try:
                    node = float(row['node'])
                    primal = float(row['primal'])
                    dual = float(row['dual'])
                    time = float(row['time'])
                    gap = abs(primal - dual)

                    if all(math.isfinite(x) for x in [node, primal, dual, time, gap]):
                        valid_data.append({
                            'node': node,     # b_j in paper notation
                            'gap': gap,       # G_j in paper notation
                            'time': time,
                            'primal': primal,
                            'dual': dual
                        })
                except (ValueError, TypeError):
                    continue

        result["samples_used"] = len(valid_data)

        if len(valid_data) < 2:
            result["status"] = "insufficient_valid_data"
            result["error_message"] = f"Only {len(valid_data)} valid data points"
            return result

        # Build SVB regression pairs according to Theorem 1
        # For checkpoints j=0,1,...,k, define:
        # - ΔG_j := G_j - G_{j+1} > 0 (gap decrement)
        # - Δb_j := b_{j+1} - b_j > 0 (node increment)
        # - Ḡ_j := (G_j + G_{j+1})/2 (average gap)

        X = []  # Average gaps: Ḡ_j
        Y = []  # log(Δb_j / ΔG_j)
        regression_pairs = []

        for j in range(len(valid_data) - 1):
            curr_sample = valid_data[j]     # j-th checkpoint
            next_sample = valid_data[j+1]   # (j+1)-th checkpoint

            # Following paper notation:
            G_j = curr_sample['gap']         # Gap at checkpoint j
            G_j_plus_1 = next_sample['gap']  # Gap at checkpoint j+1
            b_j = curr_sample['node']        # Nodes at checkpoint j
            b_j_plus_1 = next_sample['node'] # Nodes at checkpoint j+1

            # Calculate increments per paper definitions
            Delta_G_j = G_j - G_j_plus_1     # Gap decrement (should be > 0)
            Delta_b_j = b_j_plus_1 - b_j     # Node increment (should be > 0)
            G_bar_j = (G_j + G_j_plus_1) / 2 # Average gap: Ḡ_j

            # Check validity (both increments should be positive)
            if Delta_G_j > 1e-12 and Delta_b_j > 0:
                ratio = Delta_b_j / Delta_G_j
                log_ratio = math.log(ratio)

                X.append(G_bar_j)      # X-variable: average gap
                Y.append(log_ratio)    # Y-variable: log(Δb_j/ΔG_j)

                regression_pairs.append({
                    'j': j,
                    'G_j': G_j,
                    'G_j_plus_1': G_j_plus_1,
                    'b_j': b_j,
                    'b_j_plus_1': b_j_plus_1,
                    'Delta_G_j': Delta_G_j,
                    'Delta_b_j': Delta_b_j,
                    'G_bar_j': G_bar_j,
                    'ratio': ratio,
                    'log_ratio': log_ratio
                })

        result["regression_pairs"] = regression_pairs
        result["n_regression_pairs"] = len(regression_pairs)

        if len(X) < 2:
            result["status"] = "insufficient_regression_pairs"
            result["error_message"] = f"Only {len(X)} valid regression pairs, need at least 2"
            return result

        # Perform linear regression: Y = â + κ̂ * X
        # Following Theorem 1: log(Δb_j/ΔG_j) = a + κ * Ḡ_j
        import numpy as np

        X_array = np.array(X)
        Y_array = np.array(Y)

        # Linear regression: Y = slope * X + intercept = κ̂ * X + â
        A = np.vstack([X_array, np.ones(len(X_array))]).T
        coeffs, residuals, rank, s = np.linalg.lstsq(A, Y_array, rcond=None)

        kappa_hat = coeffs[0]      # Slope = κ̂ = log φ̂
        a_hat = coeffs[1]          # Intercept = â = log(Ĉ * log φ̂)

        # Calculate R-squared
        y_mean = np.mean(Y_array)
        ss_tot = np.sum((Y_array - y_mean) ** 2)
        if len(residuals) > 0:
            ss_res = residuals[0]
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        else:
            y_pred = kappa_hat * X_array + a_hat
            ss_res = np.sum((Y_array - y_pred) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        result["r_squared"] = r_squared
        result["residual_sum_squares"] = ss_res if len(residuals) > 0 else ss_res

        # Extract SVB parameters according to paper
        if kappa_hat <= 0:
            result["status"] = "invalid_kappa"
            result["error_message"] = f"Invalid κ̂ = {kappa_hat:.6f} ≤ 0"
            return result

        phi_hat = math.exp(kappa_hat)                    # φ̂ = e^κ̂
        C_hat = math.exp(a_hat) / kappa_hat              # Ĉ = e^â / κ̂

        result["a_hat"] = a_hat
        result["kappa_hat"] = kappa_hat
        result["C_hat"] = C_hat
        result["phi_hat"] = phi_hat

        # Extract bounds for gap calculation
        if result["z_du_root"] is not None and result["z_pr_terminal"] is not None:
            result["z_pr_ter"] = result["z_pr_terminal"]
            target_gap = abs(result["z_pr_terminal"] - result["z_du_root"])
            result["target_gap"] = target_gap

            # Calculate SVB extrapolation: b̂(G) = Ĉ * φ̂^G
            if target_gap > 0:
                b_hat_calculated = C_hat * (phi_hat ** target_gap)
                result["b_hat"] = b_hat_calculated

        # Final state from last sample
        if valid_data:
            last_sample = valid_data[-1]
            result["final_gap"] = last_sample['gap']
            result["final_nodes"] = last_sample['node']
            result["final_time"] = last_sample['time']

        result["status"] = "success"

    except Exception as e:
        result["status"] = "error"
        result["error_message"] = str(e)

    return result


def _analyze_comprehensive(log_path: str, param_path: str, tau: float) -> Dict[str, Any]:
    """Comprehensive analysis of a single log with parameters."""

    # Read log file
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            log_text = f.read()
    except Exception as e:
        return {
            "filename": os.path.basename(log_path),
            "error": f"Cannot read log: {e}"
        }

    # Parse parameter file
    params = {}
    if param_path and os.path.exists(param_path):
        params = parse_set_file(param_path)

    # Extract detailed SVB analysis
    svb_analysis = extract_svb_detailed_analysis(log_text, tau)

    # Combine everything
    result = {
        "filename": os.path.basename(log_path),
        "log_path": os.path.abspath(log_path),
        "param_path": os.path.abspath(param_path) if param_path else None,
        "tau": tau,

        # SVB Analysis Results
        **svb_analysis,

        # Parameter values (flattened)
        **{f"param_{k}": v for k, v in params.items()},
    }

    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Comprehensive T_infty analysis with all intermediate values")
    ap.add_argument("--logdir", required=True, help="Path to folder containing *.log files")
    ap.add_argument("--paramdir", help="Path to folder containing params_*.set files")
    ap.add_argument("--tau", type=float, default=60.0, help="Time budget tau (seconds)")
    ap.add_argument("--out", default="comprehensive_analysis.xlsx", help="Output Excel path")
    ap.add_argument("--include-regression-details", action="store_true",
                   help="Include separate sheet with detailed regression pairs")
    args = ap.parse_args()

    logdir = os.path.abspath(args.logdir)
    if not os.path.isdir(logdir):
        raise SystemExit(f"Log directory not found: {logdir}")

    paramdir = args.paramdir
    if paramdir:
        paramdir = os.path.abspath(paramdir)
        if not os.path.isdir(paramdir):
            raise SystemExit(f"Parameter directory not found: {paramdir}")

    # Find log files
    log_paths = sorted(glob.glob(os.path.join(logdir, "*.log")))
    if not log_paths:
        raise SystemExit(f"No .log files found in: {logdir}")

    print(f"Found {len(log_paths)} log files to analyze...")

    # Process each log file
    results = []
    regression_details = []

    for log_path in log_paths:
        # Find corresponding parameter file
        basename = os.path.basename(log_path)
        # Extract trial number from filename like "XML100_3346_25_50.mps_scip_trial_1.log"
        trial_match = re.search(r'trial_(\d+)\.log$', basename)
        trial_num = trial_match.group(1) if trial_match else "0"

        param_path = None
        if paramdir:
            param_file = f"params_{trial_num}.set"
            candidate_param_path = os.path.join(paramdir, param_file)
            if os.path.exists(candidate_param_path):
                param_path = candidate_param_path
            else:
                print(f"Warning: Parameter file not found: {candidate_param_path}")

        print(f"Processing: {basename}")
        result = _analyze_comprehensive(log_path, param_path, args.tau)
        results.append(result)

        # Extract regression details for separate sheet
        if args.include_regression_details and "regression_pairs" in result:
            for pair in result["regression_pairs"]:
                regression_details.append({
                    "filename": result["filename"],
                    "trial": trial_num,
                    **pair
                })

    # Create comprehensive Excel report
    try:
        with pd.ExcelWriter(args.out, engine="openpyxl") as writer:
            # Main analysis sheet
            df_main = pd.DataFrame(results)
            df_main.to_excel(writer, sheet_name="Comprehensive_Analysis", index=False)

            # Regression details sheet (if requested)
            if args.include_regression_details and regression_details:
                df_regression = pd.DataFrame(regression_details)
                df_regression.to_excel(writer, sheet_name="Regression_Pairs", index=False)

        print(f"✅ Comprehensive analysis written to: {args.out}")
        print(f"   📊 Analyzed {len(results)} log files")
        if args.include_regression_details:
            print(f"   📈 Captured {len(regression_details)} regression pairs")

    except Exception as e:
        # Fallback to CSV
        csv_path = os.path.splitext(args.out)[0] + ".csv"
        pd.DataFrame(results).to_csv(csv_path, index=False)
        print(f"Excel write failed ({e}); wrote CSV instead: {csv_path}")


if __name__ == "__main__":
    main()