"""
Time Estimation Module for SCIP Logs (Paper Implementation)
===========================================================

This module implements the complete Single Variable Branching (SVB) model
from the paper's problem formulation.

Paper Implementation:
--------------------
Section 2: Problem Formulation
- Leverages residual-gap information from timeouts
- Uses SVB model: b(G) ≈ φ^G for tree size estimation
- Computes T_infinity surrogate: T̂_∞(p;i,E,τ)
- Calculates geometric mean time ratio: R̂(p,q|I,E,τ)

Section 2.2: Estimating Tree Size from Solver Logs
- Implements L1 MINLP optimization (equation 4)
- Fits power-law relation: b̂_i(x) = e_i + u_i * x^g_i
- Minimizes L1 discrepancy with linearized absolute deviations
- Uses terminal-phase samples for robust estimation

Functions:
- extract_log_samples: Parse log into (t,e,u,z_pr,z_du,g) samples
- fit_svb_growth_factor_l1: L1 MINLP optimization for φ and b
- compute_t_infinity_surrogate: T̂_∞ surrogate calculation
- compute_geometric_mean_ratio: R̂(p,q) ratio calculation
- estimate_theta_hat: Empirical time per processed node θ̂(p,i;E,τ)
"""

from __future__ import annotations
import math
from typing import Dict, Any, List, Tuple
from .log_utils import parse_progress_series, _slice_after_last_restart

# Try importing PySCIPOpt for L1 MINLP optimization
try:
    from pyscipopt import Model, quicksum
    PYSCIPOPT_AVAILABLE = True
except ImportError:
    PYSCIPOPT_AVAILABLE = False


def extract_log_samples(log_text: str, tau_c: float = None) -> List[Dict[str, Any]]:
    """
    Extract time-ordered sequence of log samples as defined in paper:
    s_i = (t_i, e_i, u_i, z^pr_i, z^du_i, g_i)

    where:
    - t_i ∈ ℝ_≥0: wall-clock time
    - e_i ∈ ℕ: cumulative number of processed nodes
    - u_i ∈ ℕ: number of open (remaining) nodes
    - z^pr_i, z^du_i ∈ ℝ: incumbent and dual bounds
    - g_i = |z^pr_i - z^du_i|: instantaneous absolute gap

    Args:
        log_text: SCIP solver log content
        tau_c: cutoff time for sample extraction (if None, use all samples)

    Returns:
        List of samples following paper's sample format
    """
    # Respect restarts: only consider segment after final restart
    log_after_restart = _slice_after_last_restart(log_text)
    _, rows = parse_progress_series(log_after_restart)

    samples = []
    branching_started = False

    for r in rows:
        time_val = r.get("time")
        node = r.get("node")
        left = r.get("left")
        primal = r.get("primal")
        dual = r.get("dual")

        # Check for required fields
        if any(x is None for x in [time_val, node, primal, dual]):
            continue
        if not all(math.isfinite(float(x)) for x in [time_val, node, primal, dual]):
            continue

        time_val, node, primal, dual = map(float, [time_val, node, primal, dual])

        # Apply cutoff time if specified
        if tau_c is not None and time_val > tau_c:
            break

        # Skip root processing phase: wait for node > 1 to start sampling
        if not branching_started:
            if node > 1:
                branching_started = True
            else:
                continue

        # Extract u_i (open nodes)
        if left is not None and math.isfinite(float(left)):
            u_i = float(left)
        else:
            # Fallback estimation if open nodes not available
            gap = abs(primal - dual)
            if abs(dual) > 1e-6:
                u_i = max(1.0, gap / abs(dual) * node * 0.1)
            else:
                u_i = 1.0

        # Calculate instantaneous absolute gap
        g_i = abs(primal - dual)

        samples.append({
            't_i': time_val,     # wall-clock time
            'e_i': node,         # cumulative processed nodes
            'u_i': u_i,          # open/remaining nodes
            'z_pr_i': primal,    # incumbent bound
            'z_du_i': dual,      # dual bound
            'g_i': g_i           # instantaneous absolute gap
        })

    return samples


def fit_svb_growth_factor_l1(samples: List[Dict[str, Any]],
                             tau_c: float = 60.0,
                             eps: float = 1e-6,
                             debug: bool = False,
                             range_factor: float = 2.0,
                             silent: bool = False) -> Dict[str, Any]:
    """
    Implement L1 MINLP optimization from paper equation (4):

    min Σ_{i∈T} z_i
    s.t. z_i ≥ b - b̂_i(x)     ∀i ∈ T
         z_i ≥ b̂_i(x) - b     ∀i ∈ T
         z_i ≥ 0              ∀i ∈ T
         b̂_i(x) bounded within range_factor across samples
         b ≥ underline_b (empirical lower bound), b ≥ 3
         1+ε ≤ x ≤ √2-ε

    where b̂_i(x) = e_i + u_i * x^{g_i} is the per-sample proxy for final processed-node count.

    Args:
        samples: List of log samples following paper format
        tau_c: Time limit for optimization (default 60s as in paper)
        eps: Numerical margin parameter ε > 0

    Returns:
        Dict containing fitted parameters {x_star, b_star, objective, status}
    """
    if not PYSCIPOPT_AVAILABLE:
        return {"error": "PySCIPOpt not available for L1 MINLP optimization"}

    if len(samples) < 2:
        return {"error": "insufficient_samples", "samples": len(samples)}

    def _compute_lower_bound_b(sample_seq: List[Dict[str, Any]]) -> float:
        """
        Data-driven lower bound underline_b = e_k + u_k + g_k / delta_bar where
        delta_bar is the observed average gap reduction per processed node
        between the earliest larger-gap sample and the tail sample.
        """
        if len(sample_seq) < 2:
            return 3.0

        k_idx = len(sample_seq) - 1
        k_sample = sample_seq[k_idx]

        try:
            e_k = float(k_sample.get("e_i", 0.0))
            u_k = float(k_sample.get("u_i", 0.0))
            g_k = float(k_sample.get("g_i", 0.0))
        except Exception:
            return 3.0

        # Earliest index p < k such that g_p > g_k
        p_idx = None
        for idx, sample in enumerate(sample_seq[:-1]):
            try:
                if float(sample.get("g_i", 0.0)) > g_k:
                    p_idx = idx
                    break
            except Exception:
                continue

        if p_idx is None:
            return 3.0

        try:
            e_p = float(sample_seq[p_idx].get("e_i", 0.0))
            g_p = float(sample_seq[p_idx].get("g_i", 0.0))
        except Exception:
            return 3.0

        delta_gap = g_p - g_k
        delta_nodes = e_k - e_p
        if delta_gap <= 0 or delta_nodes <= 0:
            return 3.0

        bar_delta = delta_gap / delta_nodes
        if bar_delta <= 0:
            return 3.0

        lower = e_k + u_k + (g_k / bar_delta if bar_delta > 0 else 0.0)
        if not math.isfinite(lower):
            return 3.0

        return max(3.0, lower)

    # Filter samples by unique gaps (from last to first)
    n = len(samples)
    unique_gap_samples = []
    seen_gaps = set()

    # Go from last to first, only pick samples with unique gaps
    for sample in reversed(samples):
        gap = sample.get('g_i', 0)
        if gap not in seen_gaps:
            unique_gap_samples.append(sample)
            seen_gaps.add(gap)

    # Reverse to maintain chronological order
    unique_gap_samples.reverse()

    # Check if we have enough unique gap samples (at least 10% of total)
    min_required = max(int(0.1 * n), 2)  # At least 10% or minimum 2 samples
    if len(unique_gap_samples) < min_required:
        if not silent:
            print(f"L1 MINLP: Insufficient unique gap samples {len(unique_gap_samples)}/{n} (need ≥{min_required})")
        return {"error": "insufficient_unique_gaps", "samples": len(unique_gap_samples), "required": min_required}

    # Use up to 10% of total samples, but from the unique gap samples
    max_samples = min(max(int(0.1 * n), 2), len(unique_gap_samples), 100)
    selected_samples = unique_gap_samples[-max_samples:]  # Take most recent unique gap samples

    if not silent:
        print(f"L1 MINLP: Using {len(selected_samples)}/{len(unique_gap_samples)} unique gap samples ({len(unique_gap_samples)}/{n} total unique)")
        sample_times_str = ', '.join(f"{s.get('t_i', 0):.1f}" for s in selected_samples)
        print(f"Sample times: [{sample_times_str}]")

    # Empirical gap-based floor underline_b to avoid overly optimistic b
    b_lower_bound = _compute_lower_bound_b(selected_samples)
    if not silent:
        print(f"Lower bound underline_b: {b_lower_bound:.3f}")

    if debug:
        print(f"\n=== L1 MINLP Debug: Sample Details ===")
        print(f"Sample | Time(s) | Processed | Remaining | Gap    | b̂_i formula")
        print(f"-------|---------|-----------|-----------|--------|-------------")
        for i, sample in enumerate(selected_samples):
            t_i = sample.get('t_i', 0)
            e_i = sample['e_i']
            u_i = sample['u_i']
            g_i = sample['g_i']
            if abs(g_i) < 1e-12:
                b_hat_formula = f"e_i + u_i = {e_i:.0f} + {u_i:.0f} = {e_i + u_i:.0f}"
            else:
                b_hat_formula = f"e_i + u_i*x^{g_i:.3f} = {e_i:.0f} + {u_i:.0f}*x^{g_i:.3f}"
            print(f"   {i:3} | {t_i:7.1f} | {e_i:9.0f} | {u_i:9.0f} | {g_i:6.3f} | {b_hat_formula}")
        print(f"=== End Sample Details ===\n")

    try:
        # Create SCIP model for L1 MINLP optimization
        m = Model("L1_SVB_Growth_Factor")

        # Variables: x (growth factor), b (final tree size)
        x_lb = 1.0 + eps
        x_ub = math.sqrt(2.0) - eps
        x = m.addVar(name="x", vtype="C", lb=x_lb, ub=x_ub)
        b_lb = max(3.0, b_lower_bound)
        b = m.addVar(name="b", vtype="C", lb=b_lb)

        # Auxiliary variables z_i for L1 norm linearization
        z = [m.addVar(name=f"z_{i}", vtype="C", lb=0.0) for i in range(len(selected_samples))]

        # Compute b̂_i expressions for all samples
        b_hat_exprs = []
        for i, sample in enumerate(selected_samples):
            e_i = sample['e_i']
            u_i = sample['u_i']
            g_i = sample['g_i']

            if abs(g_i) < 1e-12:
                # Special case: x^0 = 1, so b̂_i(x) = e_i + u_i
                b_hat_i = e_i + u_i
            else:
                # General case: b̂_i(x) = e_i + u_i * x^{g_i}
                b_hat_i = e_i + u_i * (x ** g_i)

            b_hat_exprs.append(b_hat_i)

        # L1 norm constraints: z_i ≥ |b - b̂_i(x)|
        for i, b_hat_i in enumerate(b_hat_exprs):
            m.addCons(z[i] >= b - b_hat_i, name=f"abs_pos_{i}")
            m.addCons(z[i] >= b_hat_i - b, name=f"abs_neg_{i}")

        # Range constraints: prevent b̂_i estimates from being too different
        # For each pair (i,j): b̂_j ≤ range_factor * b̂_i and b̂_j ≥ b̂_i / range_factor
        for i in range(len(b_hat_exprs)):
            for j in range(i + 1, len(b_hat_exprs)):
                b_hat_i = b_hat_exprs[i]
                b_hat_j = b_hat_exprs[j]
                # b̂_j ≤ range_factor * b̂_i
                m.addCons(b_hat_j <= range_factor * b_hat_i, name=f"range_upper_{i}_{j}")
                # b̂_j ≥ b̂_i / range_factor  =>  range_factor * b̂_j ≥ b̂_i
                m.addCons(range_factor * b_hat_j >= b_hat_i, name=f"range_lower_{i}_{j}")

        # Objective: minimize Σ z_i (L1 discrepancy)
        m.setObjective(quicksum(z), "minimize")

        # Solver settings following paper's numerical requirements
        m.setRealParam("numerics/feastol", 1e-6)
        m.setRealParam("numerics/epsilon", 1e-6)
        m.setRealParam("numerics/sumepsilon", 1e-6)
        m.setRealParam("limits/gap", 1e-6)        # Much tighter gap tolerance
        m.setRealParam("limits/absgap", 1e-3)     # Much tighter absolute gap
        m.setRealParam("limits/time", tau_c)  # 60s time limit as in paper
        m.setIntParam("display/verblevel", 4 if debug else 0)  # Show detailed output when debug=True
        m.setBoolParam("misc/catchctrlc", False)
        m.setIntParam("presolving/maxrounds", 10)  # More presolving rounds for better accuracy

        # Solve L1 MINLP
        m.optimize()
        status = m.getStatus()

        if status in ["optimal", "timelimit", "gaplimit"] and m.getNSols() > 0:
            x_star = m.getVal(x)
            b_star = m.getVal(b)
            obj_val = m.getObjVal()

            # Compute growth factor φ = x_star
            phi_star = x_star

            if debug:
                print(f"\n=== L1 MINLP Debug: Optimization Results ===")
                print(f"Status: {status}")
                print(f"x_star (growth factor): {x_star:.6f}")
                print(f"b_star (predicted total nodes): {b_star:.1f}")
                print(f"obj_val (L1 discrepancy): {obj_val:.3f}")
                print(f"\n=== |b - b̂_i(x*)| terms ===")
                print(f"Sample | b̂_i(x*) | |b - b̂_i(x*)| | Contribution")
                print(f"-------|---------|-------------|-------------")
                total_discrepancy = 0
                for i, sample in enumerate(selected_samples):
                    e_i = sample['e_i']
                    u_i = sample['u_i']
                    g_i = sample['g_i']

                    if abs(g_i) < 1e-12:
                        b_hat_i = e_i + u_i
                    else:
                        b_hat_i = e_i + u_i * (x_star ** g_i)

                    discrepancy = abs(b_star - b_hat_i)
                    total_discrepancy += discrepancy
                    print(f"   {i:3} | {b_hat_i:7.1f} | {discrepancy:11.3f} | {discrepancy/obj_val*100:8.1f}%")

                print(f"       |         | {total_discrepancy:11.3f} | {100.0:8.1f}%")
                print(f"=== End L1 Debug ===\n")

            return {
                "x_star": x_star,      # Fitted growth parameter
                "b_star": b_star,      # Fitted final tree size
                "phi_star": phi_star,  # Growth factor φ = x
                "objective": obj_val,  # L1 discrepancy
                "status": status,
                "b_lower_bound": b_lower_bound,
                "samples_used": len(selected_samples),
                "solver": "L1_MINLP"
            }
        else:
            return {
                "error": "optimization_failed",
                "status": status,
                "samples_used": len(selected_samples)
            }

    except Exception as e:
        return {"error": "solver_exception", "message": str(e)}


def estimate_theta_hat(log_text: str, summary: Dict[str, Any], window: int = None) -> float:
    """
    Estimate θ̂(p,i;E,τ) - empirical time per processed node from terminal phase.

    Following paper: "terminal-phase samples are more informative than early-stage samples
    for estimating θ̂(p,i;E,τ)... we construct T from the tail of the log"

    Args:
        log_text: SCIP solver log content
        summary: Summary dict with solve_time, n_nodes
        window: Number of terminal rows for estimation (None = use last 10% of samples)

    Returns:
        θ̂ (time per node) estimate
    """
    # Respect restarts
    log_text = _slice_after_last_restart(log_text)
    _, rows = parse_progress_series(log_text)

    # Filter valid time/node rows
    valid_rows = [r for r in rows
                  if r.get("time") is not None and r.get("node") is not None
                  and math.isfinite(float(r["time"])) and math.isfinite(float(r["node"]))]

    if len(valid_rows) >= 2:
        # Use last 10% of terminal samples or specified window for θ̂ estimation
        if window is None:
            # Use last 10% of valid rows (terminal phase samples)
            n = len(valid_rows)
            max_samples = min(int(0.1 * n), 100)
            tail = valid_rows[-max_samples:] if n >= 10 else valid_rows
        else:
            # Use specified window (backward compatibility)
            k = max(2, min(window, len(valid_rows)))
            tail = valid_rows[-k:]

        t0 = float(tail[0]["time"])
        tn = float(tail[-1]["time"])
        b0 = float(tail[0]["node"])
        bn = float(tail[-1]["node"])

        dt = max(tn - t0, 0.0)
        db = max(bn - b0, 0.0)

        if db > 0:
            return dt / db

    # Fallback: overall average
    t = float(summary.get("solve_time", 0.0) or 0.0)
    b = float(summary.get("n_nodes", 0.0) or 0.0)
    return (t / b) if b > 0 else max(t, 1e-9)


def compute_t_infinity_surrogate(log_text: str, tau: float, summary: Dict[str, Any], debug: bool = False, silent: bool = False) -> Dict[str, Any]:
    """
    Compute T̂_∞(p;i,E,τ) surrogate from paper definition:

    T̂_∞(p;i,E,τ) = {
        t(p,i;E,τ),                                    if g(p,i;E,τ) = 0
        max{τ, θ̂(p,i;E,τ) · b̂(p,i;E,τ)},            if g(p,i;E,τ) > 0
    }

    Where θ̂ ties overall effort to dual-bound progress and b̂ is estimated using SVB model.

    Args:
        log_text: SCIP solver log content
        tau: timeout parameter τ
        summary: Summary dict with solve_time, primal, dual, n_nodes

    Returns:
        Dict with T_infinity value and computation details
    """
    # Check if instance was solved (gap = 0)
    primal = summary.get("primal")
    dual = summary.get("dual")
    gap_final = None

    try:
        if primal is not None and dual is not None:
            gap_final = abs(float(primal) - float(dual))
            if gap_final <= 1e-12:  # Solved case
                t_obs = float(summary.get("solve_time", tau))
                # Instance solved optimally - return exact time
                return {
                    "T_infinity": t_obs,
                    "solved": True,
                    "gap_final": 0.0,
                    "method": "exact_time"
                }
    except Exception as e:
        # Error checking solution status
        pass

    # Extract samples for SVB fitting
    samples = extract_log_samples(log_text)

    if len(samples) < 2:
        # Insufficient samples - using fallback
        return {
            "T_infinity": 1e9,
            "solved": False,
            "error": "insufficient_samples",
            "method": "fallback_constant"
        }

    # Fit SVB growth factor using L1 MINLP
    svb_result = fit_svb_growth_factor_l1(samples, debug=debug, range_factor=2.0, silent=silent)
    if svb_result.get("error"):
        # SVB fitting failed - return T_infinity = 1e9 as requested
        return {
            "T_infinity": 1e9,
            "solved": False,
            "error": svb_result.get("error"),
            "method": "fallback_constant"
        }

    # Get b̂ from SVB result
    b_hat = svb_result["b_star"]

    # Get b_last_sample (final processed nodes) and u_last_sample (actual remaining nodes from log)
    samples = extract_log_samples(log_text)
    if samples:
        b_last_sample = samples[-1]['e_i']  # e_i is cumulative processed nodes
        u_last_sample = samples[-1]['u_i']  # u_i is actual remaining nodes from log
    else:
        b_last_sample = float(summary.get("n_nodes", 0.0) or 0.0)
        u_last_sample = 0.0

    # Calculate θ̂ (theta_hat) - empirical time per processed node
    theta_hat = estimate_theta_hat(log_text, summary)

    # Compute T̂_∞ = τ + θ̂ × max{b̂ - b_last_sample, 0}
    remaining_nodes = max(b_hat - b_last_sample, 0.0)
    T_infinity = float(tau) + theta_hat * remaining_nodes

    # Compact output exactly as requested (only if not silent)
    if not silent:
        gap_str = f", Final gap: {gap_final:.1f}" if gap_final is not None else ""
        print(f"Solution bounds: primal={primal}, dual={dual}{gap_str}", flush=True)
        print(f"  Extracted {len(samples)} log samples for SVB fitting, samples_used: {svb_result.get('samples_used', 'N/A')}", flush=True)
        print(f"  b_hat (predicted total nodes): {b_hat:.6f}", flush=True)
        b_lb_val = svb_result.get("b_lower_bound")
        if b_lb_val is not None:
            print(f"  underline_b (lower bound on b): {b_lb_val:.6f}", flush=True)
        print(f"  phi_star (growth factor): {svb_result.get('phi_star', 'N/A')}", flush=True)
        print(f"  Current processed nodes: {b_last_sample}, Remaining nodes (from log): {u_last_sample:.1f}", flush=True)
        print(f"  theta_hat: {theta_hat:.6f} seconds per node", flush=True)
        print(f"  Final T_infinity computation:", flush=True)
        print(f"    Remaining nodes: max({b_hat:.1f} - {b_last_sample}, 0) = {remaining_nodes:.1f}", flush=True)
        print(f"    T_infinity = {tau} + {theta_hat:.6f} * {remaining_nodes:.1f} = {T_infinity:.2f}s", flush=True)

    return {
        "T_infinity": T_infinity,
        "solved": False,
        "b_hat": b_hat,
        "b_last_sample": b_last_sample,
        "remaining_nodes": remaining_nodes,
        "theta_hat": theta_hat,
        "b_lower_bound": svb_result.get("b_lower_bound"),
        "phi_star": svb_result.get("phi_star"),
        "x_star": svb_result.get("x_star"),
        "svb_objective": svb_result.get("objective"),
        "samples_used": svb_result.get("samples_used"),
        "method": "new_formula_tau_plus_theta_times_remaining"
    }


def compute_geometric_mean_ratio(T_p: Dict[str, float], T_q: Dict[str, float],
                                cap: float = 1e3) -> float:
    """
    Compute geometric mean time ratio R̂(p,q|I,E,τ) from paper equation:

    R̂(p,q|I,E,τ) = exp(1/|I| Σ_{i∈I} min{log T̂_∞(p;i,E,τ) - log T̂_∞(q;i,E,τ), log 10³})

    This preserves residual-gap information, converges to true runtime ratio as τ→∞,
    and mitigates outlier influence via upper cap at 10³.

    Args:
        T_p: Dict {instance: T_infinity_value} for configuration p
        T_q: Dict {instance: T_infinity_value} for configuration q
        cap: Upper cap (default 10³ as in paper)

    Returns:
        Geometric mean ratio R̂(p,q)
    """
    print("Computing R_hat geometric mean ratio...", flush=True)
    print(f"  T_p (candidate): {T_p}", flush=True)
    print(f"  T_q (baseline): {T_q}", flush=True)
    print(f"  Cap: {cap}", flush=True)

    log_ratios = []

    for instance in T_p:
        if instance in T_q:
            t_p = T_p[instance]
            t_q = T_q[instance]

            if t_p > 0 and t_q > 0 and math.isfinite(t_p) and math.isfinite(t_q):
                # Apply cap inside logarithm as per paper
                raw_log_ratio = math.log(t_p) - math.log(t_q)
                log_ratio = min(raw_log_ratio, math.log(cap))
                print(f"  Instance {instance}: T_p={t_p:.2f}, T_q={t_q:.2f}, ratio={t_p/t_q:.4f}, log_ratio={log_ratio:.6f}", flush=True)
                log_ratios.append(log_ratio)

    if not log_ratios:
        print("  No valid comparisons found - returning infinity", flush=True)
        return float('inf')  # No valid comparisons

    # Geometric mean: exp(1/|I| Σ log_ratios)
    mean_log_ratio = sum(log_ratios) / len(log_ratios)
    r_hat = math.exp(mean_log_ratio)
    print(f"  Mean log ratio: {mean_log_ratio:.6f}", flush=True)
    print(f"  Final R_hat = exp({mean_log_ratio:.6f}) = {r_hat:.6f}", flush=True)
    return r_hat


# Legacy compatibility functions (keep existing API) - L1 ONLY
def estimate_svb_from_log(log_text: str) -> Dict[str, Any]:
    """Legacy function using L1 formulation only."""
    samples = extract_log_samples(log_text)
    if len(samples) < 2:
        return {"a": None, "kappa": None, "C": None, "varphi": None, "samples": 0}

    svb_result = fit_svb_growth_factor_l1(samples, silent=True)
    if svb_result.get("error"):
        return {"a": None, "kappa": None, "C": None, "varphi": None, "samples": 0}

    # Only use L1 MINLP results
    phi = svb_result.get("phi_star", 1.0)
    b_star = svb_result.get("b_star", 1.0)  # This is z from L1 formulation

    return {
        "a": None,  # Not used in L1 formulation
        "kappa": math.log(phi) if phi > 1.0 else None,
        "C": b_star,  # Use z from L1 MINLP
        "varphi": phi,  # Use x from L1 MINLP
        "samples": svb_result.get("samples_used", 0)
    }


def compute_T_infty(log_text: str, tau: float, summary: Dict[str, Any]) -> Dict[str, Any]:
    """Legacy function - redirects to new implementation."""
    result = compute_t_infinity_surrogate(log_text, tau, summary)
    return {
        "T_infty": result["T_infinity"],
        "solved": result.get("solved", False),
        "gap": result.get("gap_final", None),
        "details": {k: v for k, v in result.items() if k not in ["T_infinity", "solved"]}
    }


def per_instance_T_infty(per_m: Dict[str, Dict[str, Any]], tau: float) -> Dict[str, float]:
    """Compute T_infinity for each instance given per-instance summary dicts."""
    out: Dict[str, float] = {}
    for name, m in per_m.items():
        lp = m.get("log_path")
        try:
            text = open(lp, "r", encoding="utf-8", errors="ignore").read() if lp else ""
        except Exception:
            text = ""
        result = compute_t_infinity_surrogate(text, tau=tau, summary=m)
        out[name] = float(result["T_infinity"])
    return out


# Legacy functions that logs.py expects - L1 ONLY
def estimate_total_nodes_svb(log_text: str, summary: Dict[str, Any]) -> Dict[str, Any]:
    """Legacy function using L1 MINLP only - returns z from L1 formulation."""
    samples = extract_log_samples(log_text)
    if len(samples) < 2:
        return {"error": "insufficient_samples"}

    svb_result = fit_svb_growth_factor_l1(samples, silent=True)
    if svb_result.get("error"):
        return {"error": svb_result.get("error")}

    # Only use z (b_star) from L1 MINLP
    b_star = svb_result.get("b_star")  # This is z from L1 formulation
    phi_star = svb_result.get("phi_star")  # This is x from L1 formulation

    return {
        "b_hat": b_star,  # Use z from L1 MINLP
        "G_anchor": None,  # Not used in L1 formulation
        "a": None,  # Not used in L1 formulation
        "kappa": math.log(phi_star) if phi_star and phi_star > 1.0 else None,
        "C": b_star,  # Use z from L1 MINLP
        "varphi": phi_star,  # Use x from L1 MINLP
        "samples": svb_result.get("samples_used", 0)
    }


def estimate_remaining_time(log_text: str, tau: float, summary: Dict[str, Any]) -> Dict[str, Any]:
    """Legacy function using L1 MINLP only."""
    result = compute_t_infinity_surrogate(log_text, tau, summary)

    # Only use results from L1 MINLP
    b_hat = result.get("b_hat", 1.0)  # z from L1 formulation
    phi_star = result.get("phi_star", 1.0)  # x from L1 formulation

    return {
        "theta": result.get("theta_hat", 0.0),
        "b_left": 0,  # Not computed in L1 formulation
        "G": 0.0,     # Not computed in L1 formulation
        "C": b_hat,   # Use z from L1 MINLP
        "kappa": math.log(phi_star) if phi_star and phi_star > 1.0 else 0.0,
        "varphi": phi_star,  # Use x from L1 MINLP
        "b_sub": b_hat,  # Use z from L1 MINLP
        "b_rem": b_hat,  # Use z from L1 MINLP
        "T_rem": result.get("T_infinity", tau)
    }


# Additional utility functions for diagnostics
def format_t_infty_diagnostic(diag: Dict[str, Any]) -> str:
    """Format diagnostic output for T_infinity computation."""
    return f"T_infinity: {diag.get('T_infinity', 'N/A')}\nMethod: {diag.get('method', 'unknown')}"


def diagnose_t_infty(log_text: str, tau: float, summary: Dict[str, Any]) -> Dict[str, Any]:
    """Provide diagnostics for T_infinity computation."""
    result = compute_t_infinity_surrogate(log_text, tau, summary)
    samples = extract_log_samples(log_text)

    return {
        "T_infty": result["T_infinity"],
        "method": result.get("method"),
        "samples_extracted": len(samples),
        "solved": result.get("solved", False),
        "error": result.get("error"),
        "details": result
    }
