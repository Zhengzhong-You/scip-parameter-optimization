#!/usr/bin/env python3
"""
Fixed comprehensive T_infty debug script with proper error handling.
"""

import sys
import os
import math
import json
import re
from typing import Dict, Any, List, Tuple, Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utilities.logs import (
    parse_progress_series,
    estimate_svb_from_log,
    estimate_remaining_time,
    compute_T_infty,
    diagnose_t_infty
)
from utilities.runner import _parse_summary


def print_header(title: str, char: str = "=", width: int = 80):
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def comprehensive_t_infty_debug(log_file: str, tau: float = 10.0):
    """Comprehensive T_infty debug with proper error handling."""

    print_header("🔍 COMPREHENSIVE T_INFTY DEBUG 🔍")
    print(f"Log file: {log_file}")
    print(f"Time limit (tau): {tau}")

    # Read log file
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            log_text = f.read()
    except Exception as e:
        print(f"❌ ERROR reading log file: {e}")
        return

    print(f"✅ Log file read: {len(log_text):,} chars, {len(log_text.splitlines()):,} lines")

    # Parse summary - handle potential None
    print_header("STEP 1: SUMMARY PARSING")
    try:
        summary = _parse_summary(log_file)
        if summary is None:
            summary = {}
        print("✅ Summary parsed successfully:")
        for k, v in summary.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"❌ Summary parsing error: {e}")
        summary = {}

    # Parse progress table
    print_header("STEP 2: PROGRESS TABLE PARSING")
    try:
        columns, rows = parse_progress_series(log_text)
        print(f"✅ Progress table parsed: {len(columns)} columns, {len(rows)} rows")
        print(f"Columns: {columns}")

        if rows:
            print(f"First row: {rows[0]}")
            print(f"Last row: {rows[-1]}")

            # Count unique node values
            node_values = set()
            for row in rows:
                node = row.get('node')
                if node is not None:
                    node_values.add(node)
            print(f"Unique node values: {sorted(node_values)}")

    except Exception as e:
        print(f"❌ Progress parsing error: {e}")
        columns, rows = [], []

    # SVB estimation analysis
    print_header("STEP 3: SVB ESTIMATION DETAILED ANALYSIS")

    # Manual pair building to understand the issue
    valid_data_rows = []
    print("Analyzing rows for SVB pair building...")

    for i, row in enumerate(rows):
        node = row.get('node')
        primal = row.get('primal')
        dual = row.get('dual')

        has_all_data = all(x is not None for x in [node, primal, dual])
        if has_all_data:
            all_finite = all(math.isfinite(float(x)) for x in [node, primal, dual])
            if all_finite:
                gap = abs(float(primal) - float(dual))
                valid_data_rows.append({
                    'row_idx': i+1,
                    'node': float(node),
                    'primal': float(primal),
                    'dual': float(dual),
                    'gap': gap
                })

    print(f"Valid data rows: {len(valid_data_rows)}")

    # Show why pairs fail
    if len(valid_data_rows) >= 2:
        print("\nPair building analysis (first 5 attempts):")
        for i in range(min(5, len(valid_data_rows)-1)):
            curr = valid_data_rows[i+1]
            prev = valid_data_rows[i]

            delta_nodes = curr['node'] - prev['node']
            delta_gap = prev['gap'] - curr['gap']

            print(f"  Pair {i+1}: rows {prev['row_idx']}→{curr['row_idx']}")
            print(f"    Node: {prev['node']} → {curr['node']} (Δ={delta_nodes})")
            print(f"    Gap:  {prev['gap']:.2f} → {curr['gap']:.2f} (Δ={delta_gap:.2f})")
            print(f"    Valid? db>0: {delta_nodes > 0}, dG>1e-12: {delta_gap > 1e-12}")

    else:
        print("❌ Insufficient valid data rows for pair building")

    # Try SVB estimation
    try:
        svb_result = estimate_svb_from_log(log_text)
        print(f"✅ SVB estimation completed:")
        for k, v in svb_result.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"❌ SVB estimation error: {e}")
        svb_result = {'error': str(e)}

    # Try remaining time estimation
    print_header("STEP 4: REMAINING TIME ESTIMATION")
    try:
        remaining_time = estimate_remaining_time(log_text, tau=tau, summary=summary)
        print("✅ Remaining time estimation:")
        for k, v in remaining_time.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"❌ Remaining time error: {e}")
        remaining_time = {'error': str(e)}

    # Final T_infty calculation
    print_header("STEP 5: FINAL T_INFTY CALCULATION")
    try:
        t_infty_result = compute_T_infty(log_text, tau=tau, summary=summary)
        print("✅ T_infty calculation completed:")
        for k, v in t_infty_result.items():
            if k == 'details':
                print(f"  {k}: {type(v).__name__} with {len(v) if isinstance(v, dict) else '?'} entries")
            else:
                print(f"  {k}: {v}")
    except Exception as e:
        print(f"❌ T_infty calculation error: {e}")
        t_infty_result = {'error': str(e)}

    # Diagnostic analysis
    print_header("STEP 6: DIAGNOSTIC ANALYSIS")
    try:
        diagnostic = diagnose_t_infty(log_text, tau=tau, summary=summary)
        print("✅ Diagnostic completed:")

        key_metrics = [
            'progress_rows', 'raw_pairs_count', 'kept_pairs_count',
            'left_nodes', 'G_used', 'theta', 'T_rem', 'T_infty'
        ]

        for key in key_metrics:
            if key in diagnostic:
                print(f"  {key}: {diagnostic[key]}")

        # Show OLS details if available
        if 'ols' in diagnostic:
            print(f"  OLS: {diagnostic['ols']}")

    except Exception as e:
        print(f"❌ Diagnostic error: {e}")

    # Root cause analysis
    print_header("🎯 ROOT CAUSE ANALYSIS")

    if len(valid_data_rows) == 0:
        print("🔴 NO VALID DATA ROWS")
        print("   - All progress rows missing node/primal/dual data or have non-finite values")

    elif len(set(row['node'] for row in valid_data_rows)) == 1:
        unique_node = valid_data_rows[0]['node']
        print(f"🔴 ALL NODES IDENTICAL: {unique_node}")
        print("   - No node progression means no tree exploration")
        print("   - SVB model requires node count increases (db > 0)")
        print("   - Solver stuck at root node - likely due to:")
        print("     * Very short time limit")
        print("     * Complex root relaxation")
        print("     * Parameter settings preventing branching")

    else:
        node_values = sorted(set(row['node'] for row in valid_data_rows))
        print(f"✅ Node progression detected: {node_values}")
        print("   - But SVB estimation still failed - investigate pair building logic")

    # Show gap progression
    if valid_data_rows:
        gaps = [row['gap'] for row in valid_data_rows]
        print(f"\n📊 Gap progression:")
        print(f"   Initial gap: {gaps[0]:.2f}")
        print(f"   Final gap: {gaps[-1]:.2f}")
        print(f"   Gap reduction: {gaps[0] - gaps[-1]:.2f}")
        print(f"   Reduction rate: {(gaps[0] - gaps[-1])/gaps[0]*100:.1f}%")

    # Recommendations
    print_header("💡 RECOMMENDATIONS")

    if 'error' in svb_result or svb_result.get('samples', 0) == 0:
        print("1. 🚀 INCREASE TIME LIMIT")
        print("   - Current limit too short for tree exploration")
        print("   - Try 60+ seconds to allow meaningful branching")

        print("\n2. 🔧 CHECK SCIP PARAMETERS")
        print("   - Verify parameters aren't forcing root-only solving")
        print("   - Check branching and node selection settings")

        print("\n3. 📊 IMPROVE FALLBACK LOGIC")
        print("   - Current fallback: T_infty = tau + solve_time")
        print("   - Consider gap-based estimates when SVB fails")

    else:
        print("✅ SVB estimation working - issue may be elsewhere")

    return {
        'summary': summary,
        'progress_rows': len(rows),
        'svb_result': svb_result,
        'remaining_time': remaining_time,
        't_infty_result': t_infty_result
    }


if __name__ == "__main__":
    log_file = "/Users/ricky_you/Desktop/gpt/runs/single_quick/XML100_3346_25_50.mps/XML100_3346_25_50.mps/log/XML100_3346_25_50.mps_scip_trial_2.log"

    if not os.path.exists(log_file):
        print(f"❌ Log file not found: {log_file}")

        # Search for alternatives
        print("\n🔍 Searching for available log files...")
        for root, dirs, files in os.walk("/Users/ricky_you/Desktop/gpt"):
            for file in files:
                if file.endswith('.log') and 'scip_trial' in file:
                    full_path = os.path.join(root, file)
                    size = os.path.getsize(full_path)
                    print(f"  📄 {full_path} ({size:,} bytes)")
        sys.exit(1)

    result = comprehensive_t_infty_debug(log_file)
    print(f"\n🏁 Debug completed. Results: {len(result)} sections analyzed.")