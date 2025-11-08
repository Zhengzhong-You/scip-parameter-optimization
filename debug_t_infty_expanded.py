#!/usr/bin/env python3
"""
ULTRA-COMPREHENSIVE T_infty Debug Script
Shows EVERYTHING in maximum detail for complete debugging visibility.
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
    diagnose_t_infty,
    format_t_infty_diagnostic,
    parse_scip_log_lines,
    shrink_scip_log_for_gpt,
    _normalize_col,
    _parse_float_cell
)
from utilities.runner import _parse_summary


def print_separator(title: str, char: str = "=", width: int = 100):
    """Print a formatted separator line."""
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}")


def print_subsection(title: str, char: str = "-", width: int = 80):
    """Print a formatted subsection header."""
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def debug_raw_log_content(log_path: str):
    """Debug raw log file content analysis."""
    print_separator("RAW LOG FILE ANALYSIS")

    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            log_text = f.read()
    except Exception as e:
        print(f"ERROR: Could not read log file: {e}")
        return ""

    lines = log_text.splitlines()

    print(f"File path: {log_path}")
    print(f"File size: {len(log_text):,} characters")
    print(f"Number of lines: {len(lines):,}")
    print(f"File encoding detected: utf-8 (with error ignore)")

    # Character frequency analysis
    char_counts = {}
    for char in log_text[:1000]:  # Sample first 1000 chars
        char_counts[char] = char_counts.get(char, 0) + 1

    print(f"\nCharacter frequency (first 1000 chars):")
    for char, count in sorted(char_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        if char.isprintable():
            print(f"  '{char}': {count}")
        else:
            print(f"  {repr(char)}: {count}")

    # Line length analysis
    line_lengths = [len(line) for line in lines]
    print(f"\nLine length statistics:")
    print(f"  Min length: {min(line_lengths) if line_lengths else 0}")
    print(f"  Max length: {max(line_lengths) if line_lengths else 0}")
    print(f"  Avg length: {sum(line_lengths) / len(line_lengths) if line_lengths else 0:.2f}")

    # Show first and last few lines
    print_subsection("FIRST 10 LINES")
    for i, line in enumerate(lines[:10]):
        print(f"Line {i+1:3d}: {repr(line)}")

    print_subsection("LAST 10 LINES")
    for i, line in enumerate(lines[-10:]):
        line_num = len(lines) - 10 + i + 1
        print(f"Line {line_num:3d}: {repr(line)}")

    # Search for key patterns
    print_subsection("KEY PATTERN SEARCH")
    patterns = {
        "SCIP Status": re.compile(r"SCIP Status", re.I),
        "Progress table header": re.compile(r"time.*node.*left", re.I),
        "Progress rows": re.compile(r"^\s*(?:\*|d)?\d+(?:\.\d+)?s\|"),
        "Solving Time": re.compile(r"Solving Time", re.I),
        "Primal/Dual Bound": re.compile(r"(Primal|Dual) Bound", re.I),
        "Original problem": re.compile(r"original problem has", re.I),
    }

    for pattern_name, pattern in patterns.items():
        matches = [(i+1, line) for i, line in enumerate(lines) if pattern.search(line)]
        print(f"\n{pattern_name} matches ({len(matches)} found):")
        for line_num, line in matches[:5]:  # Show first 5 matches
            print(f"  Line {line_num:3d}: {line.strip()}")
        if len(matches) > 5:
            print(f"  ... and {len(matches) - 5} more matches")

    return log_text


def debug_log_parsing_detailed(log_text: str):
    """Debug detailed log parsing with every step."""
    print_separator("DETAILED LOG PARSING ANALYSIS")

    # Parse using the scip log lines parser
    parsed = parse_scip_log_lines(log_text.splitlines())

    print_subsection("PARSED LOG STRUCTURE")
    for key, value in parsed.items():
        if isinstance(value, list):
            print(f"  {key}: {len(value)} items")
            if value and isinstance(value[0], list):  # Nested list
                print(f"    Sample: {value[0][:3] if len(value[0]) > 3 else value[0]}")
            elif value:
                print(f"    Sample: {value[:3] if len(value) > 3 else value}")
        else:
            print(f"  {key}: {repr(value)}")

    # Show original problem detection
    if parsed["original_problem"]:
        print_subsection("ORIGINAL PROBLEM")
        print(f"  Detected: {parsed['original_problem']}")

    # Show presolve blocks
    if parsed["presolve_blocks"]:
        print_subsection("PRESOLVE BLOCKS")
        for i, block in enumerate(parsed["presolve_blocks"]):
            print(f"  Block {i+1}: {len(block)} lines")
            for j, line in enumerate(block[:5]):
                print(f"    Line {j+1}: {line}")
            if len(block) > 5:
                print(f"    ... and {len(block) - 5} more lines")

    # Show progress blocks in detail
    if parsed["progress_blocks"]:
        print_subsection("PROGRESS BLOCKS")
        for i, block in enumerate(parsed["progress_blocks"]):
            print(f"  Block {i+1}:")
            print(f"    Header: {repr(block['header'])}")
            print(f"    Rows: {len(block['rows'])}")
            for j, row in enumerate(block["rows"]):
                print(f"      Row {j+1}: {repr(row)}")

    # Show restarts
    if parsed["restarts"]:
        print_subsection("RESTARTS")
        for i, restart in enumerate(parsed["restarts"]):
            print(f"  Restart {i+1}: {restart}")

    # Show summary lines
    if parsed["summary_lines"]:
        print_subsection("SUMMARY LINES")
        for i, line in enumerate(parsed["summary_lines"]):
            print(f"  Summary {i+1}: {repr(line)}")


def debug_progress_parsing_ultra_detailed(log_text: str):
    """Ultra-detailed progress table parsing."""
    print_separator("ULTRA-DETAILED PROGRESS TABLE PARSING")

    lines = log_text.splitlines() if isinstance(log_text, str) else list(log_text)

    # Header detection
    print_subsection("HEADER DETECTION")
    HDR_RE = re.compile(r"^\s*time\s*\|\s*node\s*\|\s*left\s*\|", re.I)

    header_candidates = []
    for i, line in enumerate(lines):
        if HDR_RE.match(line):
            header_candidates.append((i+1, line))

    print(f"Header pattern: {HDR_RE.pattern}")
    print(f"Header candidates found: {len(header_candidates)}")
    for line_num, line in header_candidates:
        print(f"  Line {line_num}: {repr(line)}")

    # Parse actual progress series
    columns, rows = parse_progress_series(log_text)

    print_subsection("COLUMN PARSING")
    print(f"Detected columns: {len(columns)}")
    for i, col in enumerate(columns):
        print(f"  Column {i+1}: '{col}' (normalized)")

    # Show column normalization process
    if header_candidates:
        sample_header = header_candidates[0][1]
        raw_parts = [p.strip() for p in sample_header.split("|")]
        print(f"\nRaw header parts: {raw_parts}")
        normalized_parts = [_normalize_col(p) for p in raw_parts if p.strip()]
        print(f"Normalized parts: {normalized_parts}")

    print_subsection("ROW PARSING DETAILED")
    ROW_RE = re.compile(r"^\s*(?:\*|d)?\d+(?:\.\d+)?s\|")

    row_candidates = []
    for i, line in enumerate(lines):
        if ROW_RE.match(line):
            row_candidates.append((i+1, line))

    print(f"Row pattern: {ROW_RE.pattern}")
    print(f"Row candidates found: {len(row_candidates)}")
    print(f"Actually parsed rows: {len(rows)}")

    # Show first 5 raw row parsing
    print(f"\nFirst 5 row parsing examples:")
    for i, (line_num, line) in enumerate(row_candidates[:5]):
        print(f"\nRaw line {line_num}: {repr(line)}")

        # Manual parsing to show process
        parts = [p.strip() for p in line.split("|")]
        vals = [p for p in parts if p != ""]
        print(f"  Split parts: {parts}")
        print(f"  Non-empty vals: {vals}")

        # Show field mapping
        row_dict = {}
        for idx, name in enumerate(columns[:len(vals)]):
            raw_val = vals[idx] if idx < len(vals) else ""
            parsed_val = _parse_float_cell(raw_val) if raw_val else None
            row_dict[name] = parsed_val

            print(f"    Column '{name}' <- '{raw_val}' -> {parsed_val}")

        print(f"  Final row dict: {row_dict}")

        if i < len(rows):
            print(f"  Actual parsed: {rows[i]}")

    # Show detailed cell parsing examples
    print_subsection("CELL PARSING EXAMPLES")
    test_cells = ["10.5s", "1.0", "inf", "-inf", "25.59", "10.42%", "", "7370.0", "6038.78"]
    for cell in test_cells:
        parsed = _parse_float_cell(cell)
        print(f"  '{cell}' -> {parsed} ({type(parsed).__name__})")


def debug_summary_parsing_detailed(log_path: str):
    """Debug detailed summary parsing."""
    print_separator("DETAILED SUMMARY PARSING")

    summary = _parse_summary(log_path)

    print_subsection("REGEX PATTERNS")
    # Show the actual regex patterns used
    patterns = {
        "Version": re.compile(r"^\s*SCIP version\s+([0-9]+\.[0-9]+\.[0-9]+)\b", re.I),
        "Status": re.compile(r"^\s*SCIP Status\s*:\s*(.+?)\s*$", re.I),
        "Time": re.compile(r"^\s*(?:Solving|Total)\s+Time\s*\(sec\)\s*:\s*([0-9]+(?:\.[0-9]+)?)", re.I),
        "Nodes": re.compile(r"^\s*Solving Nodes\s*:\s*([0-9]+)", re.I),
        "Primal": re.compile(r"^\s*Primal Bound\s*:\s*([+-]?(?:\d+\.\d*|\d+|\.\d+)(?:[eE][+-]?\d+)?)", re.I),
        "Dual": re.compile(r"^\s*Dual Bound\s*:\s*([+-]?(?:\d+\.\d*|\d+|\.\d+)(?:[eE][+-]?\d+)?)", re.I),
        "Gap": re.compile(r"^\s*Gap\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%", re.I),
        "LP Iter": re.compile(r"^\s*LP Iterations\s*:\s*([0-9]+)", re.I),
    }

    # Read file and test each pattern
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"ERROR reading file: {e}")
        return summary

    for pattern_name, pattern in patterns.items():
        print(f"\n{pattern_name} pattern: {pattern.pattern}")
        matches = []
        for i, line in enumerate(lines):
            match = pattern.match(line)
            if match:
                matches.append((i+1, line.strip(), match.groups()))

        print(f"  Matches found: {len(matches)}")
        for line_num, line, groups in matches:
            print(f"    Line {line_num}: {repr(line)}")
            print(f"      Groups: {groups}")

    print_subsection("FINAL PARSED SUMMARY")
    for key, value in summary.items():
        print(f"  {key}: {repr(value)} ({type(value).__name__})")


def debug_svb_estimation_step_by_step(log_text: str):
    """Debug SVB estimation with every calculation step."""
    print_separator("STEP-BY-STEP SVB ESTIMATION")

    # Get progress rows
    columns, rows = parse_progress_series(log_text)

    print_subsection("INPUT DATA PREPARATION")
    print(f"Total progress rows: {len(rows)}")

    # Show data preparation
    valid_rows = []
    for i, r in enumerate(rows):
        node = r.get("node")
        primal = r.get("primal")
        dual = r.get("dual")

        print(f"\nRow {i+1}:")
        print(f"  Raw: {r}")
        print(f"  node={node} ({type(node).__name__})")
        print(f"  primal={primal} ({type(primal).__name__})")
        print(f"  dual={dual} ({type(dual).__name__})")

        # Check validity conditions
        has_data = node is not None and primal is not None and dual is not None
        print(f"  Has all data: {has_data}")

        if has_data:
            is_finite = math.isfinite(node) and math.isfinite(primal) and math.isfinite(dual)
            print(f"  All finite: {is_finite}")

            if is_finite:
                G = abs(primal - dual)
                print(f"  Gap (G): |{primal} - {dual}| = {G}")
                valid_rows.append((i+1, float(node), float(G)))
                print(f"  ✓ VALID for pair building")
            else:
                print(f"  ✗ INVALID: Non-finite values")
        else:
            print(f"  ✗ INVALID: Missing data")

    print_subsection("PAIR BUILDING PROCESS")
    print(f"Valid rows for pairing: {len(valid_rows)}")

    pairs_x = []
    pairs_y = []
    pair_details = []

    for i in range(1, len(valid_rows)):
        prev_row_idx, prev_node, prev_G = valid_rows[i-1]
        curr_row_idx, curr_node, curr_G = valid_rows[i]

        print(f"\nPair attempt {i}:")
        print(f"  Previous row {prev_row_idx}: node={prev_node}, G={prev_G}")
        print(f"  Current row {curr_row_idx}: node={curr_node}, G={curr_G}")

        # Calculate deltas
        db = curr_node - prev_node
        dG = prev_G - curr_G  # Should be positive if gap decreases

        print(f"  Delta nodes (db): {curr_node} - {prev_node} = {db}")
        print(f"  Delta gap (dG): {prev_G} - {curr_G} = {dG}")

        # Check validity conditions
        db_valid = db > 0
        dG_valid = dG > 1e-12
        dG_finite = math.isfinite(dG)

        print(f"  db > 0: {db_valid}")
        print(f"  dG > 1e-12: {dG_valid}")
        print(f"  dG finite: {dG_finite}")

        if db_valid and dG_valid and dG_finite:
            ratio = db / dG
            y = math.log(ratio)
            x = 0.5 * (prev_G + curr_G)

            print(f"  ✓ VALID PAIR:")
            print(f"    Ratio (db/dG): {db} / {dG} = {ratio}")
            print(f"    log(db/dG): ln({ratio}) = {y}")
            print(f"    G_bar (x): 0.5 * ({prev_G} + {curr_G}) = {x}")

            pairs_x.append(x)
            pairs_y.append(y)
            pair_details.append({
                "pair_idx": i,
                "prev_row": prev_row_idx,
                "curr_row": curr_row_idx,
                "prev_node": prev_node,
                "curr_node": curr_node,
                "prev_G": prev_G,
                "curr_G": curr_G,
                "db": db,
                "dG": dG,
                "ratio": ratio,
                "log_ratio": y,
                "G_bar": x
            })
        else:
            print(f"  ✗ INVALID PAIR")

    print_subsection("REGRESSION DATA PREPARATION")
    n = len(pairs_x)
    print(f"Total valid pairs: {n}")

    if n == 0:
        print("No valid pairs - SVB estimation impossible")
        return {"error": "no_valid_pairs", "details": pair_details}

    print(f"\nRaw pairs (X=G_bar, Y=log(db/dG)):")
    for i, (x, y) in enumerate(zip(pairs_x, pairs_y)):
        print(f"  Pair {i+1}: X={x:.6f}, Y={y:.6f}")

    # Trimming process
    print_subsection("OUTLIER TRIMMING")
    order = sorted(range(n), key=lambda i: pairs_y[i])
    print(f"Y-values sorted order (indices): {order}")
    print(f"Y-values sorted: {[pairs_y[i] for i in order]}")

    keep_idx = order
    if n >= 20:
        k = max(1, int(0.05 * n))
        keep_idx = order[k : n - k]
        print(f"Trimming enabled (n={n} >= 20): removing {k} from each tail")
        print(f"Keeping indices: {keep_idx}")
    else:
        print(f"No trimming (n={n} < 20): keeping all pairs")

    X = [pairs_x[i] for i in keep_idx]
    Y = [pairs_y[i] for i in keep_idx]

    print(f"After trimming: {len(X)} pairs")
    if X:
        print(f"X range: [{min(X):.6f}, {max(X):.6f}]")
        print(f"Y range: [{min(Y):.6f}, {max(Y):.6f}]")

    # Regression calculation
    print_subsection("LINEAR REGRESSION CALCULATION")
    if len(X) < 2:
        print("Insufficient data for regression (need at least 2 points)")
        return {"error": "insufficient_data", "details": pair_details}

    # Calculate means
    x_mean = sum(X) / len(X)
    y_mean = sum(Y) / len(Y)
    print(f"Means: x_mean={x_mean:.6f}, y_mean={y_mean:.6f}")

    # Calculate sums
    print(f"\nSum calculations:")
    sxx_terms = [(x - x_mean) ** 2 for x in X]
    sxy_terms = [(x - x_mean) * (y - y_mean) for x, y in zip(X, Y)]

    for i, (x, y) in enumerate(zip(X, Y)):
        sxx_term = sxx_terms[i]
        sxy_term = sxy_terms[i]
        print(f"  Point {i+1}: x={x:.6f}, y={y:.6f}")
        print(f"    (x-x̄)² = ({x:.6f}-{x_mean:.6f})² = {sxx_term:.6f}")
        print(f"    (x-x̄)(y-ȳ) = ({x:.6f}-{x_mean:.6f})({y:.6f}-{y_mean:.6f}) = {sxy_term:.6f}")

    sxx = sum(sxx_terms)
    sxy = sum(sxy_terms)
    print(f"\nSums: Sxx={sxx:.6f}, Sxy={sxy:.6f}")

    # Check for singularity
    if sxx <= 1e-18:
        print(f"Sxx too small ({sxx:.2e} <= 1e-18) - singular system")
        return {"error": "singular_system", "sxx": sxx, "details": pair_details}

    # Calculate regression coefficients
    kappa = sxy / sxx
    a = y_mean - kappa * x_mean
    print(f"Regression coefficients:")
    print(f"  kappa (slope) = Sxy/Sxx = {sxy:.6f}/{sxx:.6f} = {kappa:.6f}")
    print(f"  a (intercept) = ȳ - κx̄ = {y_mean:.6f} - {kappa:.6f}*{x_mean:.6f} = {a:.6f}")

    # Transform to physical parameters
    varphi = math.exp(kappa)
    C = math.exp(a) / (kappa if abs(kappa) > 1e-12 else 1e-12)

    print(f"Physical parameters:")
    print(f"  varphi = exp(kappa) = exp({kappa:.6f}) = {varphi:.6f}")
    print(f"  C = exp(a)/kappa = exp({a:.6f})/{kappa:.6f} = {C:.6f}")

    return {
        "a": a,
        "kappa": kappa,
        "C": C,
        "varphi": varphi,
        "samples": n,
        "used_samples": len(X),
        "x_mean": x_mean,
        "y_mean": y_mean,
        "sxx": sxx,
        "sxy": sxy,
        "raw_pairs": [(pairs_x[i], pairs_y[i]) for i in range(n)],
        "kept_pairs": [(X[i], Y[i]) for i in range(len(X))],
        "details": pair_details
    }


def debug_remaining_time_calculation(log_text: str, tau: float, summary: Dict[str, Any]):
    """Debug remaining time calculation in extreme detail."""
    print_separator("ULTRA-DETAILED REMAINING TIME CALCULATION")

    # Get SVB estimation
    est = debug_svb_estimation_step_by_step(log_text)

    print_subsection("SVB ESTIMATION INPUT TO REMAINING TIME")
    if est.get("error"):
        print(f"SVB estimation failed: {est['error']}")
        return {"error": est["error"], "svb_details": est}

    varphi = float(est["varphi"])
    C = float(est["C"])
    kappa = float(est["kappa"])

    print(f"From SVB estimation:")
    print(f"  C = {C}")
    print(f"  varphi = {varphi}")
    print(f"  kappa = {kappa}")

    # Find last progress row with left nodes
    print_subsection("FINDING LEFT NODES")
    columns, rows = parse_progress_series(log_text)

    last_row_with_left = None
    for i in range(len(rows) - 1, -1, -1):  # Reverse order
        r = rows[i]
        left = r.get("left")
        print(f"Row {i+1}: left={left}")
        if left is not None:
            last_row_with_left = r
            print(f"  ✓ Found last row with left nodes: {left}")
            break

    b_left = int(last_row_with_left.get("left", 0)) if last_row_with_left else 0
    print(f"Final b_left: {b_left}")

    # Find latest gap
    print_subsection("FINDING CURRENT GAP")
    G = None

    print("Searching progress rows (newest first):")
    for i in range(len(rows) - 1, -1, -1):
        r = rows[i]
        pr = r.get("primal")
        du = r.get("dual")
        print(f"Row {i+1}: primal={pr}, dual={du}")

        if pr is not None and du is not None and math.isfinite(float(pr)) and math.isfinite(float(du)):
            G = abs(float(pr) - float(du))
            print(f"  ✓ Found valid gap: |{pr} - {du}| = {G}")
            break
        else:
            print(f"  ✗ Invalid or missing bounds")

    if G is None:
        print("No gap from progress rows, checking summary:")
        pr = summary.get("primal")
        du = summary.get("dual")
        print(f"Summary: primal={pr}, dual={du}")

        if pr is not None and du is not None and math.isfinite(float(pr)) and math.isfinite(float(du)):
            G = abs(float(pr) - float(du))
            print(f"  ✓ Gap from summary: |{pr} - {du}| = {G}")
        else:
            print(f"  ✗ No valid gap available")
            return {"error": "no_gap_available", "svb_details": est}

    # Calculate theta (time per node)
    print_subsection("THETA CALCULATION")
    t = float(summary.get("solve_time", tau))
    b_obs = float(summary.get("n_nodes", 0.0) or 0.0)

    print(f"From summary:")
    print(f"  solve_time = {t}")
    print(f"  n_nodes = {b_obs}")
    print(f"  time_limit (tau) = {tau}")

    theta = max(t, 1e-9) / max(b_obs, 1.0)
    print(f"Theta calculation:")
    print(f"  theta = max({t}, 1e-9) / max({b_obs}, 1.0)")
    print(f"        = {max(t, 1e-9)} / {max(b_obs, 1.0)}")
    print(f"        = {theta}")

    # Subtree nodes estimation
    print_subsection("SUBTREE NODES ESTIMATION")
    print(f"Calculating b_sub = C * varphi^G")
    print(f"  C = {C}")
    print(f"  varphi = {varphi}")
    print(f"  G = {G}")

    try:
        exponent = float(G)
        base = float(varphi)
        print(f"Computing {base}^{exponent}...")

        if exponent == 0:
            power_result = 1.0
            print(f"  Special case: {base}^0 = 1.0")
        elif base == 1.0:
            power_result = 1.0
            print(f"  Special case: 1.0^{exponent} = 1.0")
        elif exponent > 700:  # Prevent overflow
            power_result = float("inf")
            print(f"  Exponent too large ({exponent} > 700): result = inf")
        else:
            power_result = base ** exponent
            print(f"  {base}^{exponent} = {power_result}")

        b_sub = C * power_result
        print(f"b_sub = {C} * {power_result} = {b_sub}")

    except OverflowError:
        b_sub = float("inf")
        print(f"  OverflowError: b_sub = inf")
    except Exception as e:
        print(f"  Error in calculation: {e}")
        b_sub = float("inf")

    # Remaining nodes calculation
    print_subsection("REMAINING NODES CALCULATION")
    b_rem = float(b_left) * float(b_sub)
    print(f"b_rem = b_left * b_sub = {b_left} * {b_sub} = {b_rem}")

    # Remaining time calculation
    print_subsection("REMAINING TIME CALCULATION")
    T_rem = float(theta) * float(b_rem)
    print(f"T_rem = theta * b_rem = {theta} * {b_rem} = {T_rem}")

    return {
        "theta": theta,
        "b_left": b_left,
        "G": float(G),
        "C": C,
        "kappa": kappa,
        "varphi": varphi,
        "b_sub": b_sub,
        "b_rem": b_rem,
        "T_rem": T_rem,
        "svb_details": est
    }


def debug_final_t_infty_calculation(log_text: str, tau: float, summary: Dict[str, Any]):
    """Debug the final T_infty calculation."""
    print_separator("FINAL T_INFTY CALCULATION")

    # Check if problem is solved
    print_subsection("SOLUTION STATUS CHECK")
    pr = summary.get("primal")
    du = summary.get("dual")

    print(f"Primal bound: {pr}")
    print(f"Dual bound: {du}")

    gap = None
    try:
        if pr is not None and du is not None:
            prf = float(pr)
            duf = float(du)
            if math.isfinite(prf) and math.isfinite(duf):
                gap = abs(prf - duf)
                print(f"Gap: |{prf} - {duf}| = {gap}")
            else:
                print(f"Non-finite bounds: primal finite={math.isfinite(float(pr))}, dual finite={math.isfinite(float(du))}")
        else:
            print("Missing primal or dual bound")
    except Exception as e:
        print(f"Error calculating gap: {e}")

    t = float(summary.get("solve_time", tau))
    print(f"Solve time: {t}")
    print(f"Time limit (tau): {tau}")

    # Check for solved status
    if gap is not None and gap <= 0.0:
        print(f"✓ SOLVED: gap={gap} <= 0")
        result = {"T_infty": t, "solved": True, "gap": 0.0, "details": {}}
        print(f"Returning: {result}")
        return result

    print(f"Not solved (gap={gap}), proceeding with remaining time estimation")

    # Get remaining time estimation
    print_subsection("REMAINING TIME ESTIMATION")
    det = debug_remaining_time_calculation(log_text, tau=tau, summary=summary)

    if det.get("error"):
        print(f"Remaining time estimation failed: {det['error']}")
        fallback_t_infty = t if gap is None else (tau + t)
        print(f"Using fallback T_infty: {fallback_t_infty}")
        result = {"T_infty": fallback_t_infty, "solved": False, "gap": gap, "details": det}
        print(f"Returning: {result}")
        return result

    T_rem = float(det.get("T_rem", 0.0))
    T_infty = float(tau) + T_rem

    print(f"Final T_infty calculation:")
    print(f"  T_infty = tau + T_rem = {tau} + {T_rem} = {T_infty}")

    result = {"T_infty": T_infty, "solved": False, "gap": gap, "details": det}
    print(f"Returning: {result}")
    return result


def ultra_comprehensive_debug(log_file: str, tau: float = 10.0):
    """The ultimate comprehensive debug function."""
    print_separator("🔍 ULTRA-COMPREHENSIVE T_INFTY DEBUG ANALYSIS 🔍", "🔍", 120)
    print(f"Log file: {log_file}")
    print(f"Time limit (tau): {tau}")
    print(f"Script: {__file__}")
    print(f"Python version: {sys.version}")

    if not os.path.exists(log_file):
        print(f"❌ ERROR: Log file not found: {log_file}")
        return

    # Step 1: Raw log analysis
    log_text = debug_raw_log_content(log_file)

    # Step 2: Detailed log parsing
    debug_log_parsing_detailed(log_text)

    # Step 3: Progress parsing
    debug_progress_parsing_ultra_detailed(log_text)

    # Step 4: Summary parsing
    summary = debug_summary_parsing_detailed(log_file)

    # Step 5: SVB estimation
    svb_result = debug_svb_estimation_step_by_step(log_text)

    # Step 6: Remaining time (if SVB successful)
    if not svb_result.get("error"):
        remaining_time_result = debug_remaining_time_calculation(log_text, tau, summary)

    # Step 7: Final T_infty
    final_result = debug_final_t_infty_calculation(log_text, tau, summary)

    # Summary
    print_separator("🎯 FINAL SUMMARY 🎯", "🎯", 120)
    print(f"Final T_infty: {final_result.get('T_infty')}")
    print(f"Solved: {final_result.get('solved')}")
    print(f"Gap: {final_result.get('gap')}")
    print(f"SVB estimation successful: {not svb_result.get('error')}")

    if svb_result.get("error"):
        print(f"SVB error: {svb_result['error']}")
    else:
        print(f"SVB parameters: C={svb_result['C']:.6f}, varphi={svb_result['varphi']:.6f}")


if __name__ == "__main__":
    log_file = "/Users/ricky_you/Desktop/gpt/runs/single_quick/XML100_3346_25_50.mps/XML100_3346_25_50.mps/log/XML100_3346_25_50.mps_scip_trial_2.log"

    # Check if file exists and show alternatives if not
    if not os.path.exists(log_file):
        print(f"❌ ERROR: Log file not found: {log_file}")
        print("\n🔍 Searching for available log files...")

        # Search in common locations
        search_dirs = [
            "/Users/ricky_you/Desktop/gpt/runs",
            "/Users/ricky_you/Desktop/gpt/runs_back",
            "/Users/ricky_you/Desktop/gpt/back_up"
        ]

        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                for root, dirs, files in os.walk(search_dir):
                    for file in files:
                        if file.endswith(".log"):
                            full_path = os.path.join(root, file)
                            size = os.path.getsize(full_path)
                            print(f"  📄 {full_path} ({size:,} bytes)")

        sys.exit(1)

    ultra_comprehensive_debug(log_file)