#!/usr/bin/env python3
"""
Debug script for T_infty calculation with detailed intermediate steps.
Shows complete calculation process for SVB model fitting and T_infty estimation.
"""

import sys
import os
import math
import json
from typing import Dict, Any, List, Tuple

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utilities.logs import (
    parse_progress_series,
    estimate_svb_from_log,
    estimate_remaining_time,
    compute_T_infty
)
from utilities.runner import _parse_summary


def debug_t_infty_calculation(log_path: str, tau: float = 10.0):
    """Debug T_infty calculation with complete intermediate output."""

    print("=" * 80)
    print(f"DEBUGGING T_INFTY CALCULATION")
    print(f"Log file: {log_path}")
    print(f"Time limit (tau): {tau}")
    print("=" * 80)

    # Read log file
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            log_text = f.read()
    except Exception as e:
        print(f"ERROR: Could not read log file: {e}")
        return

    print(f"\nLog file size: {len(log_text)} characters")
    print(f"Log file lines: {len(log_text.splitlines())}")

    # Parse summary from log
    print("\n" + "="*60)
    print("STEP 1: PARSING LOG SUMMARY")
    print("="*60)

    summary = _parse_summary(log_path)
    print("Summary extracted from log:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    # Parse progress series
    print("\n" + "="*60)
    print("STEP 2: PARSING PROGRESS TABLE")
    print("="*60)

    columns, rows = parse_progress_series(log_text)
    print(f"Progress table columns: {columns}")
    print(f"Number of progress rows: {len(rows)}")

    if rows:
        print("\nFirst few progress rows:")
        for i, row in enumerate(rows[:5]):
            print(f"  Row {i+1}: {row}")

        print(f"\nLast few progress rows:")
        for i, row in enumerate(rows[-3:]):
            print(f"  Row {len(rows)-2+i}: {row}")
    else:
        print("No progress rows found!")

    # Build pairs for SVB estimation
    print("\n" + "="*60)
    print("STEP 3: BUILDING PAIRS FOR SVB ESTIMATION")
    print("="*60)

    pairs_x: List[float] = []
    pairs_y: List[float] = []
    pair_details: List[Dict[str, Any]] = []
    last = None

    for i, r in enumerate(rows):
        node = r.get("node")
        primal = r.get("primal")
        dual = r.get("dual")

        if node is None or primal is None or dual is None:
            print(f"Row {i+1}: Skipping - missing data (node={node}, primal={primal}, dual={dual})")
            continue

        if not (math.isfinite(node) and math.isfinite(primal) and math.isfinite(dual)):
            print(f"Row {i+1}: Skipping - non-finite values (node={node}, primal={primal}, dual={dual})")
            continue

        G = abs(primal - dual)
        cur = (float(node), float(G))

        if last is not None:
            db = cur[0] - last[0]  # Delta nodes
            dG = last[1] - cur[1]  # Delta gap (should be positive if gap decreases)

            print(f"Row {i+1}: node={node}, primal={primal}, dual={dual}, G={G}")
            print(f"  Previous: node={last[0]}, G={last[1]}")
            print(f"  Deltas: db={db}, dG={dG}")

            if db > 0 and dG > 1e-12 and math.isfinite(dG):
                ratio = db / dG
                y = math.log(ratio)
                x = 0.5 * (last[1] + cur[1])  # G_bar

                print(f"  ✓ Valid pair: db/dG={ratio}, log(db/dG)={y}, G_bar={x}")
                pairs_x.append(x)
                pairs_y.append(y)

                pair_details.append({
                    "row": i+1,
                    "node_prev": last[0],
                    "node_curr": cur[0],
                    "G_prev": last[1],
                    "G_curr": cur[1],
                    "db": db,
                    "dG": dG,
                    "ratio": ratio,
                    "log_ratio": y,
                    "G_bar": x
                })
            else:
                print(f"  ✗ Invalid pair: db={db}, dG={dG}")

        last = cur

    print(f"\nTotal valid pairs: {len(pairs_x)}")

    # SVB estimation
    print("\n" + "="*60)
    print("STEP 4: SVB MODEL ESTIMATION")
    print("="*60)

    est = estimate_svb_from_log(log_text)
    print("SVB estimation result:")
    for key, value in est.items():
        print(f"  {key}: {value}")

    # OLS regression details
    if len(pairs_x) >= 2:
        print(f"\nOLS Regression Details:")

        # Trimming
        n = len(pairs_x)
        order = sorted(range(n), key=lambda i: pairs_y[i])
        keep_idx = order
        if n >= 20:
            k = max(1, int(0.05 * n))
            keep_idx = order[k : n - k]
            print(f"Trimming: Keeping indices {k} to {n-k-1} out of {n}")

        X = [pairs_x[i] for i in keep_idx]
        Y = [pairs_y[i] for i in keep_idx]

        print(f"After trimming: {len(X)} pairs")
        print(f"X (G_bar) range: [{min(X):.6f}, {max(X):.6f}]")
        print(f"Y (log ratio) range: [{min(Y):.6f}, {max(Y):.6f}]")

        # Regression calculations
        x_mean = sum(X) / len(X)
        y_mean = sum(Y) / len(Y)
        sxx = sum((x - x_mean) ** 2 for x in X)
        sxy = sum((x - x_mean) * (y - y_mean) for x, y in zip(X, Y))

        print(f"Regression statistics:")
        print(f"  x_mean = {x_mean:.6f}")
        print(f"  y_mean = {y_mean:.6f}")
        print(f"  Sxx = {sxx:.6f}")
        print(f"  Sxy = {sxy:.6f}")

        if sxx > 1e-18:
            kappa = sxy / sxx
            a = y_mean - kappa * x_mean
            varphi = math.exp(kappa)
            C = math.exp(a) / (kappa if abs(kappa) > 1e-12 else 1e-12)

            print(f"  kappa = {kappa:.6f}")
            print(f"  a = {a:.6f}")
            print(f"  varphi = exp(kappa) = {varphi:.6f}")
            print(f"  C = exp(a)/kappa = {C:.6f}")

    # Remaining time estimation
    print("\n" + "="*60)
    print("STEP 5: REMAINING TIME ESTIMATION")
    print("="*60)

    remaining_est = estimate_remaining_time(log_text, tau=tau, summary=summary)
    print("Remaining time estimation:")
    for key, value in remaining_est.items():
        print(f"  {key}: {value}")

    # Find final gap and left nodes
    print(f"\nDetailed breakdown:")

    # Find last row with left nodes
    last_left = None
    for r in rows[::-1]:
        if r.get("left") is not None:
            last_left = int(r.get("left"))
            print(f"Last left nodes from progress: {last_left}")
            break

    # Find final gap
    G_used = None
    for r in rows[::-1]:
        pr = r.get("primal")
        du = r.get("dual")
        if pr is not None and du is not None and math.isfinite(float(pr)) and math.isfinite(float(du)):
            G_used = abs(float(pr) - float(du))
            print(f"Final gap from progress: {G_used}")
            break

    if G_used is None:
        pr = summary.get("primal")
        du = summary.get("dual")
        if pr is not None and du is not None:
            G_used = abs(float(pr) - float(du))
            print(f"Final gap from summary: {G_used}")

    # Theta calculation
    t = float(summary.get("solve_time", tau))
    b_obs = float(summary.get("n_nodes", 0.0) or 0.0)
    theta = max(t, 1e-9) / max(b_obs, 1.0)
    print(f"Time per node (theta): {t} / {b_obs} = {theta}")

    # Subtree estimation
    if remaining_est.get("C") and remaining_est.get("varphi") and G_used is not None:
        C = float(remaining_est["C"])
        varphi = float(remaining_est["varphi"])

        try:
            b_sub = C * (varphi ** float(G_used))
            print(f"Subtree nodes (b_sub): {C} * {varphi}^{G_used} = {b_sub}")
        except OverflowError:
            b_sub = float("inf")
            print(f"Subtree nodes (b_sub): overflow -> inf")

        if last_left is not None:
            b_rem = float(last_left) * float(b_sub)
            T_rem = float(theta) * float(b_rem)
            print(f"Remaining nodes (b_rem): {last_left} * {b_sub} = {b_rem}")
            print(f"Remaining time (T_rem): {theta} * {b_rem} = {T_rem}")

    # Final T_infty calculation
    print("\n" + "="*60)
    print("STEP 6: FINAL T_INFTY CALCULATION")
    print("="*60)

    t_infty_result = compute_T_infty(log_text, tau=tau, summary=summary)
    print("Final T_infty result:")
    for key, value in t_infty_result.items():
        if key == "details":
            print(f"  {key}: {type(value).__name__} with {len(value)} keys")
        else:
            print(f"  {key}: {value}")

    # Show final results
    print("\n" + "="*60)
    print("STEP 7: FINAL RESULTS SUMMARY")
    print("="*60)

    print("T_infty calculation completed successfully!")
    print(f"Final T_infty: {t_infty_result.get('T_infty')}")
    print(f"Solved: {t_infty_result.get('solved')}")
    print(f"Gap: {t_infty_result.get('gap')}")
    if 'warning' in t_infty_result:
        print(f"Warning: {t_infty_result['warning']}")
    print(f"Details keys: {list(t_infty_result.get('details', {}).keys())}")

    # Show raw pairs sample
    if pair_details:
        print(f"\n" + "="*60)
        print("RAW PAIRS SAMPLE (First 10)")
        print("="*60)
        for i, detail in enumerate(pair_details[:10]):
            print(f"Pair {i+1}:")
            for key, value in detail.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.6f}")
                else:
                    print(f"  {key}: {value}")
            print()


if __name__ == "__main__":
    log_file = "/Users/ricky_you/Desktop/gpt/runs/single_quick/XML100_3346_25_50.mps/XML100_3346_25_50.mps/log/XML100_3346_25_50.mps_scip_trial_2.log"

    if not os.path.exists(log_file):
        print(f"ERROR: Log file not found: {log_file}")
        print("Available files in the directory:")
        log_dir = os.path.dirname(log_file)
        if os.path.exists(log_dir):
            for f in os.listdir(log_dir):
                print(f"  {f}")
        else:
            print(f"Directory does not exist: {log_dir}")
        sys.exit(1)

    debug_t_infty_calculation(log_file)