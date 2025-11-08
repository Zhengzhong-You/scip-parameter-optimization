from __future__ import annotations

import re
from typing import Dict, Any, List, Tuple
from collections import deque
import math
import re


def parse_scip_log_lines(lines: List[str]) -> Dict[str, Any]:
    HEADER_RE = re.compile(r'^\s*time\s*\|\s*node\s*\|\s*left\s*\|', re.I)
    TIMEROW_RE = re.compile(r'^\s*(?:\*|d)?\d+(?:\.\d+)?s\|')
    ORIGINAL_RE = re.compile(r'original problem has .*? and \d+\s+constraints', re.I)
    PRESOLVE_SUMMARY_RE = re.compile(r'^\s*presolving\s*\(', re.I)
    PRESOLVE_TIME_RE = re.compile(r'^\s*Presolving Time\s*:', re.I)
    RESTART_RE = re.compile(r'(Restart triggered.*|performing user restart|\(restart\).*)', re.I)

    SUMMARY_KEYS = [
        re.compile(r'^\s*SCIP Status\s*:', re.I),
        re.compile(r'^\s*Solving Time\s*\(sec\)\s*:', re.I),
        re.compile(r'^\s*Solving Nodes\s*:', re.I),
        re.compile(r'^\s*Primal Bound\s*:', re.I),
        re.compile(r'^\s*Dual Bound\s*:', re.I),
        re.compile(r'^\s*Gap\s*:', re.I),
    ]

    res = {
        "original_problem": None,
        "presolve_blocks": [],
        "progress_blocks": [],
        "restarts": [],
        "summary_lines": []
    }
    in_presolve = False
    curr_presolve = None
    curr_prog = None

    def end_presolve():
        nonlocal in_presolve, curr_presolve
        if in_presolve and curr_presolve:
            res["presolve_blocks"].append(curr_presolve)
        in_presolve = False
        curr_presolve = None

    def end_progress():
        nonlocal curr_prog
        if curr_prog and curr_prog["rows"]:
            res["progress_blocks"].append({
                "header": curr_prog["header"],
                "rows": list(curr_prog["rows"])
            })
        curr_prog = None

    for line in lines:
        line = line.rstrip()
        if res["original_problem"] is None and ORIGINAL_RE.search(line):
            res["original_problem"] = line.strip()
        if RESTART_RE.search(line):
            res["restarts"].append(line.strip())
        if PRESOLVE_SUMMARY_RE.match(line):
            end_presolve(); in_presolve = True; curr_presolve = [line]; continue
        if in_presolve and curr_presolve is not None:
            curr_presolve.append(line)
            if PRESOLVE_TIME_RE.match(line):
                end_presolve(); continue
        if HEADER_RE.match(line):
            end_progress();
            curr_prog = {"header": line, "rows": deque(maxlen=5)}
            continue
        if curr_prog is not None and TIMEROW_RE.match(line):
            curr_prog["rows"].append(line)
        if any(pat.match(line) for pat in SUMMARY_KEYS):
            res["summary_lines"].append(line.strip())

    end_presolve(); end_progress()
    return res


def shrink_scip_log_for_gpt(log_text: str, max_length: int = 1500) -> str:
    if not log_text:
        return ""
    parsed = parse_scip_log_lines(log_text.splitlines())
    out = []
    if parsed["original_problem"]:
        out.append("PROBLEM SIZE:")
        out.append(parsed["original_problem"]); out.append("")
    if parsed["presolve_blocks"]:
        out.append("PRESOLVING:")
        out.extend(parsed["presolve_blocks"][0][:8]); out.append("")
    if parsed["progress_blocks"]:
        last_block = parsed["progress_blocks"][-1]
        out.append("SOLVING PROGRESS:")
        out.append(last_block["header"])
        out.extend(list(last_block["rows"])[-3:]); out.append("")
    if parsed["restarts"]:
        out.append(f"RESTARTS: {len(parsed['restarts'])} restart(s)"); out.append("")
    if parsed["summary_lines"]:
        out.append("FINAL RESULTS:")
        out.extend(parsed["summary_lines"])
    result = "\n".join(out)
    if len(result) > max_length:
        result = result[:max_length-15] + "\n...[truncated]"
    return result


# -----------------------------
# Progress table parsing (rich)
# -----------------------------

_HDR_RE = re.compile(r"^\s*time\s*\|\s*node\s*\|\s*left\s*\|", re.I)
_ROW_RE = re.compile(r"^\s*(?:\*|d)?\d+(?:\.\d+)?s\|")


def _normalize_col(name: str) -> str:
    n = re.sub(r"\s+", " ", name.strip().lower())
    n = n.replace("dual bound", "dual").replace("primal bound", "primal")
    n = n.replace("dualbound", "dual").replace("primalbound", "primal")
    return n


def _parse_float_cell(s: str) -> float | None:
    t = s.strip()
    if t.endswith("s") and t[:-1].replace(".", "", 1).replace("-", "", 1).isdigit():
        t = t[:-1]
    t = t.replace("%", "")
    try:
        return float(t)
    except Exception:
        tl = t.lower()
        if tl in ("inf", "+inf", "infinite", "infinity"):
            return float("inf")
        if tl in ("-inf", "-infinity"):
            return float("-inf")
    return None


def parse_progress_series(log_text: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Parse the SCIP progress table into a series of rows with named columns.

    Returns (columns, rows), where each row has possible keys among:
      time, node, left, dual, primal, gap
    """
    lines = log_text.splitlines() if isinstance(log_text, str) else list(log_text)
    cols: List[str] = []
    rows: List[Dict[str, Any]] = []
    parsing = False
    for ln in lines:
        if _HDR_RE.match(ln):
            # Parse header columns by splitting on '|'
            parts = [p.strip() for p in ln.split("|")]
            cols = [_normalize_col(p) for p in parts if p.strip()]
            parsing = True
            continue
        if parsing and _ROW_RE.match(ln):
            parts = [p.strip() for p in ln.split("|")]
            # align to header count (skip empty fragments at ends)
            vals = [p for p in parts if p != ""]
            row: Dict[str, Any] = {}
            for idx, name in enumerate(cols[: len(vals)]):
                key = None
                if name.startswith("time"):
                    key = "time"
                elif name.startswith("node"):
                    key = "node"
                elif name.startswith("left"):
                    key = "left"
                elif name.startswith("dual"):
                    key = "dual"
                elif name.startswith("primal"):
                    key = "primal"
                elif name.startswith("gap"):
                    key = "gap"
                if key:
                    row[key] = _parse_float_cell(vals[idx])
            rows.append(row)
        # Stop parsing on a blank line after table or summary begins
        if parsing and (ln.strip().startswith("SCIP Status") or ln.strip().startswith("Presolving Time") or ln.strip() == ""):
            parsing = False
    return cols, rows


def estimate_svb_from_log(log_text: str) -> Dict[str, Any]:
    """Estimate (a, kappa, C, varphi) per Theorem, from SCIP progress rows.

    Uses adjacent progress rows to form pairs (Δb/ΔG) vs \overline{G} with
    G = |primal - dual|, b = processed nodes, and fits y = a + kappa * x
    where y = log(Δb/ΔG), x = \overline{G}.
    """
    _, rows = parse_progress_series(log_text)
    pairs_x: List[float] = []
    pairs_y: List[float] = []
    last = None
    for r in rows:
        # Need finite node, primal, dual
        node = r.get("node"); primal = r.get("primal"); dual = r.get("dual")
        if node is None or primal is None or dual is None:
            continue
        if not (math.isfinite(node) and math.isfinite(primal) and math.isfinite(dual)):
            continue
        G = abs(primal - dual)
        cur = (float(node), float(G))
        if last is not None:
            db = cur[0] - last[0]
            dG = last[1] - cur[1]  # should be positive if gap decreases
            if db > 0 and dG > 1e-12 and math.isfinite(dG):
                y = math.log(db / dG)
                x = 0.5 * (last[1] + cur[1])
                if math.isfinite(x) and math.isfinite(y):
                    pairs_x.append(x); pairs_y.append(y)
        last = cur

    n = len(pairs_x)
    if n < 2:
        return {"a": None, "kappa": None, "C": None, "varphi": None, "samples": 0}

    # Simple robust trimming: drop extreme 5% tails if enough samples
    order = sorted(range(n), key=lambda i: pairs_y[i])
    keep_idx = order
    if n >= 20:
        k = max(1, int(0.05 * n))
        keep_idx = order[k : n - k]
    X = [pairs_x[i] for i in keep_idx]
    Y = [pairs_y[i] for i in keep_idx]

    x_mean = sum(X) / len(X)
    y_mean = sum(Y) / len(Y)
    sxx = sum((x - x_mean) ** 2 for x in X)
    sxy = sum((x - x_mean) * (y - y_mean) for x, y in zip(X, Y))
    if sxx <= 1e-18:
        return {"a": None, "kappa": None, "C": None, "varphi": None, "samples": n}
    kappa = sxy / sxx
    a = y_mean - kappa * x_mean
    varphi = math.exp(kappa)
    C = math.exp(a) / (kappa if abs(kappa) > 1e-12 else 1e-12)
    return {"a": a, "kappa": kappa, "C": C, "varphi": varphi, "samples": n}


def estimate_remaining_time(log_text: str, tau: float, summary: Dict[str, Any]) -> Dict[str, Any]:
    """Estimate remaining time using SVB model at timeout.

    summary expects keys: 'solve_time', 'primal', 'dual', 'n_nodes'.
    Returns dict with fields: theta, b_left, G, C, kappa, varphi, b_sub, b_rem, T_rem.
    """
    est = estimate_svb_from_log(log_text)
    if not est.get("varphi"):
        return {"error": "insufficient_samples"}
    varphi = float(est["varphi"]); C = float(est["C"]); kappa = float(est["kappa"])

    # Extract last progress row for left and G
    _, rows = parse_progress_series(log_text)
    last_row = None
    for r in rows[::-1]:
        if r.get("left") is not None:
            last_row = r; break
    b_left = int(last_row.get("left", 0)) if last_row else 0
    # Use latest available primal/dual for G
    G = None
    for r in rows[::-1]:
        pr = r.get("primal"); du = r.get("dual")
        if pr is not None and du is not None and math.isfinite(pr) and math.isfinite(du):
            G = abs(float(pr) - float(du)); break
    if G is None:
        pr = summary.get("primal"); du = summary.get("dual")
        if pr is not None and du is not None and math.isfinite(float(pr)) and math.isfinite(float(du)):
            G = abs(float(pr) - float(du))
    if G is None:
        return {"error": "no_gap_available", **est}

    t = float(summary.get("solve_time", tau))
    b_obs = float(summary.get("n_nodes", 0.0) or 0.0)
    theta = max(t, 1e-9) / max(b_obs, 1.0)

    try:
        b_sub = C * (varphi ** float(G))
    except OverflowError:
        b_sub = float("inf")
    b_rem = float(b_left) * float(b_sub)
    T_rem = float(theta) * float(b_rem)
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
    }


def compute_T_infty(log_text: str, tau: float, summary: Dict[str, Any]) -> Dict[str, Any]:
    """Compute the extrapolated time-to-optimality surrogate T_infty.

    Returns dict with keys: T_infty, solved (bool), gap, details (dict from estimate_remaining_time if used).
    """
    pr = summary.get("primal"); du = summary.get("dual")
    gap = None
    try:
        if pr is not None and du is not None:
            prf = float(pr); duf = float(du)
            if math.isfinite(prf) and math.isfinite(duf):
                gap = abs(prf - duf)
    except Exception:
        gap = None
    t = float(summary.get("solve_time", tau))
    if gap is not None and gap <= 0.0:
        return {"T_infty": t, "solved": True, "gap": 0.0, "details": {}}
    det = estimate_remaining_time(log_text, tau=tau, summary=summary)
    if det.get("error"):
        return {"T_infty": t if gap is None else (tau + t), "solved": False, "gap": gap, "details": det}
    return {"T_infty": float(tau) + float(det.get("T_rem", 0.0)), "solved": False, "gap": gap, "details": det}


def per_instance_T_infty(per_m: Dict[str, Dict[str, Any]], tau: float) -> Dict[str, float]:
    """Compute T_infty for each instance given per-instance summary dicts with 'log_path'."""
    out: Dict[str, float] = {}
    for name, m in per_m.items():
        lp = m.get("log_path")
        try:
            text = open(lp, "r", encoding="utf-8", errors="ignore").read() if lp and isinstance(lp, str) and lp else ""
        except Exception:
            text = ""
        res = compute_T_infty(text, tau=tau, summary=m)
        out[name] = float(res.get("T_infty", float("inf")))
    return out


def diagnose_t_infty(log_text: str, tau: float, summary: Dict[str, Any]) -> Dict[str, Any]:
    """Return a rich diagnostic record for the T_infty computation.

    Includes:
      - parsed progress rows count
      - raw pairs (x=G_bar, y=log(Δb/ΔG)) [truncated]
      - OLS stats (means, sxx, sxy), (a, kappa, varphi, C)
      - last-row left nodes, chosen G, theta, b_sub, b_rem, T_rem, T_infty
    """
    cols, rows = parse_progress_series(log_text)
    # Build pairs as in estimate_svb_from_log
    pairs_x: List[float] = []
    pairs_y: List[float] = []
    usable_pairs = 0
    last = None
    for r in rows:
        node = r.get("node"); pr = r.get("primal"); du = r.get("dual")
        if node is None or pr is None or du is None:
            continue
        if not (math.isfinite(node) and math.isfinite(pr) and math.isfinite(du)):
            continue
        G = abs(float(pr) - float(du))
        cur = (float(node), float(G))
        if last is not None:
            db = cur[0] - last[0]
            dG = last[1] - cur[1]
            if db > 0 and dG > 1e-12 and math.isfinite(dG):
                y = math.log(db / dG)
                x = 0.5 * (last[1] + cur[1])
                if math.isfinite(x) and math.isfinite(y):
                    pairs_x.append(x); pairs_y.append(y); usable_pairs += 1
        last = cur

    # Trimming behavior
    n = len(pairs_x)
    order = sorted(range(n), key=lambda i: pairs_y[i])
    keep_idx = order
    if n >= 20:
        k = max(1, int(0.05 * n))
        keep_idx = order[k : n - k]
    X = [pairs_x[i] for i in keep_idx]
    Y = [pairs_y[i] for i in keep_idx]
    x_mean = (sum(X) / len(X)) if X else float("nan")
    y_mean = (sum(Y) / len(Y)) if Y else float("nan")
    sxx = sum((x - x_mean) ** 2 for x in X) if X else float("nan")
    sxy = sum((x - x_mean) * (y - y_mean) for x, y in zip(X, Y)) if X else float("nan")
    if not X or sxx <= 1e-18 or not math.isfinite(sxx):
        ols = {"a": None, "kappa": None, "varphi": None, "C": None}
    else:
        kappa = sxy / sxx
        a = y_mean - kappa * x_mean
        varphi = math.exp(kappa)
        C = math.exp(a) / (kappa if abs(kappa) > 1e-12 else 1e-12)
        ols = {"a": a, "kappa": kappa, "varphi": varphi, "C": C}

    # Final T_infty bits
    # Left nodes and G selection
    last_left = None
    for r in rows[::-1]:
        if r.get("left") is not None:
            last_left = int(r.get("left")); break
    G_used = None
    for r in rows[::-1]:
        pr = r.get("primal"); du = r.get("dual")
        if pr is not None and du is not None and math.isfinite(float(pr)) and math.isfinite(float(du)):
            G_used = abs(float(pr) - float(du)); break
    if G_used is None:
        pr = summary.get("primal"); du = summary.get("dual")
        if pr is not None and du is not None and math.isfinite(float(pr)) and math.isfinite(float(du)):
            G_used = abs(float(pr) - float(du))
    t = float(summary.get("solve_time", tau))
    b_obs = float(summary.get("n_nodes", 0.0) or 0.0)
    theta = max(t, 1e-9) / max(b_obs, 1.0)
    if ols["C"] is None or G_used is None or last_left is None:
        T_rem = None; T_inf = (t if (summary.get("gap") == 0) else tau + t)
    else:
        try:
            b_sub = float(ols["C"]) * (float(ols["varphi"]) ** float(G_used))
        except Exception:
            b_sub = float("inf")
        b_rem = float(last_left) * float(b_sub)
        T_rem = float(theta) * float(b_rem)
        T_inf = float(tau) + float(T_rem)

    # Truncate raw pairs for readability
    max_show = 50
    raw_pairs = [
        {"G_bar": float(pairs_x[i]), "log_db_over_dg": float(pairs_y[i])}
        for i in range(min(n, max_show))
    ]
    kept_pairs = [
        {"G_bar": float(X[j]), "log_db_over_dg": float(Y[j])}
        for j in range(min(len(X), max_show))
    ]
    return {
        "progress_rows": len(rows),
        "raw_pairs_count": n,
        "raw_pairs_sample": raw_pairs,
        "kept_pairs_count": len(X),
        "kept_pairs_sample": kept_pairs,
        "ols": ols,
        "ols_stats": {"x_mean": x_mean, "y_mean": y_mean, "sxx": sxx, "sxy": sxy},
        "left_nodes": last_left,
        "G_used": G_used,
        "theta": theta,
        "T_rem": T_rem,
        "T_infty": T_inf,
    }


def format_t_infty_diagnostic(diag: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"progress_rows: {diag.get('progress_rows')}")
    lines.append(f"raw_pairs_count: {diag.get('raw_pairs_count')}")
    lines.append(f"kept_pairs_count: {diag.get('kept_pairs_count')}")
    ols = diag.get("ols", {})
    lines.append("ols: a={a}, kappa={k}, varphi={v}, C={c}".format(a=ols.get("a"), k=ols.get("kappa"), v=ols.get("varphi"), c=ols.get("C")))
    lines.append(f"left_nodes: {diag.get('left_nodes')}, G_used: {diag.get('G_used')}")
    lines.append(f"theta: {diag.get('theta')}")
    lines.append(f"T_rem: {diag.get('T_rem')}")
    lines.append(f"T_infty: {diag.get('T_infty')}")
    # Show a small sample of pairs
    rp = diag.get("raw_pairs_sample") or []
    kp = diag.get("kept_pairs_sample") or []
    if rp:
        lines.append("raw_pairs_sample (G_bar, log_db_over_dg):")
        for e in rp[:5]:
            lines.append(f"  {e['G_bar']:.6g}, {e['log_db_over_dg']:.6g}")
    if kp:
        lines.append("kept_pairs_sample (after trimming):")
        for e in kp[:5]:
            lines.append(f"  {e['G_bar']:.6g}, {e['log_db_over_dg']:.6g}")
    return "\n".join(lines)
