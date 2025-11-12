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
                             eps: float = 1e-4) -> Dict[str, Any]:
    """
    Implement L1 MINLP optimization from paper equation (4):

    min Σ_{i∈T} z_i
    s.t. z_i ≥ b - b̂_i(x)     ∀i ∈ T
         z_i ≥ b̂_i(x) - b     ∀i ∈ T
         z_i ≥ 0              ∀i ∈ T
         1+ε ≤ x ≤ √2-ε,  b ≥ 3

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

    # Select valid samples T ⊆ {1,...,n} from terminal phase
    # Following paper: use last samples, cap at |T| = min{⌊0.1*n⌋, 100}
    n = len(samples)
    max_samples = min(int(0.1 * n), 100)
    selected_samples = samples[-max_samples:] if n >= 10 else samples

    print(f"📊 L1 MINLP: Using {len(selected_samples)}/{n} terminal-phase samples")

    try:
        # Create SCIP model for L1 MINLP optimization
        m = Model("L1_SVB_Growth_Factor")

        # Variables: x (growth factor), b (final tree size)
        x_lb = 1.0 + eps
        x_ub = math.sqrt(2.0) - eps
        x = m.addVar(name="x", vtype="C", lb=x_lb, ub=x_ub)
        b = m.addVar(name="b", vtype="C", lb=3.0)

        # Auxiliary variables z_i for L1 norm linearization
        z = [m.addVar(name=f"z_{i}", vtype="C", lb=0.0) for i in range(len(selected_samples))]

        # L1 norm constraints: z_i ≥ |b - b̂_i(x)|
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

            # Linearized absolute value constraints
            m.addCons(z[i] >= b - b_hat_i, name=f"abs_pos_{i}")
            m.addCons(z[i] >= b_hat_i - b, name=f"abs_neg_{i}")

        # Objective: minimize Σ z_i (L1 discrepancy)
        m.setObjective(quicksum(z), "minimize")

        # Solver settings following paper's numerical requirements
        m.setRealParam("numerics/feastol", 1e-3)
        m.setRealParam("numerics/epsilon", 1e-4)
        m.setRealParam("numerics/sumepsilon", 1e-3)
        m.setRealParam("limits/gap", 1e-2)
        m.setRealParam("limits/absgap", 1e-1)
        m.setRealParam("limits/time", tau_c)  # 60s time limit as in paper
        m.setIntParam("display/verblevel", 0)
        m.setBoolParam("misc/catchctrlc", False)
        m.setIntParam("presolving/maxrounds", 5)

        # Solve L1 MINLP
        m.optimize()
        status = m.getStatus()

        if status in ["optimal", "timelimit", "gaplimit"] and m.getNSols() > 0:
            x_star = m.getVal(x)
            b_star = m.getVal(b)
            obj_val = m.getObjVal()

            # Compute growth factor φ = x_star
            phi_star = x_star

            return {
                "x_star": x_star,      # Fitted growth parameter
                "b_star": b_star,      # Fitted final tree size
                "phi_star": phi_star,  # Growth factor φ = x
                "objective": obj_val,  # L1 discrepancy
                "status": status,
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


def estimate_theta_hat(log_text: str, summary: Dict[str, Any], window: int = 5) -> float:
    """
    Estimate θ̂(p,i;E,τ) - empirical time per processed node from terminal phase.

    Following paper: "terminal-phase samples are more informative than early-stage samples
    for estimating θ̂(p,i;E,τ)... we construct T from the tail of the log"

    Args:
        log_text: SCIP solver log content
        summary: Summary dict with solve_time, n_nodes
        window: Number of terminal rows for estimation

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
        # Use terminal window for θ̂ estimation
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


def compute_t_infinity_surrogate(log_text: str, tau: float, summary: Dict[str, Any]) -> Dict[str, Any]:
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

    try:
        if primal is not None and dual is not None:
            gap_final = abs(float(primal) - float(dual))
            if gap_final <= 1e-12:  # Solved case
                t_obs = float(summary.get("solve_time", tau))
                return {
                    "T_infinity": t_obs,
                    "solved": True,
                    "gap_final": 0.0,
                    "method": "exact_time"
                }
    except Exception:
        pass

    # Unsolved case: estimate T̂_∞ using SVB model

    # Extract samples for SVB fitting
    samples = extract_log_samples(log_text)
    if len(samples) < 2:
        # Insufficient data: return large constant to keep ratios finite
        return {
            "T_infinity": 1e9,
            "solved": False,
            "error": "insufficient_samples",
            "method": "fallback_constant"
        }

    # Fit SVB growth factor using L1 MINLP
    svb_result = fit_svb_growth_factor_l1(samples)
    if svb_result.get("error"):
        return {
            "T_infinity": 1e9,
            "solved": False,
            "error": svb_result.get("error"),
            "method": "fallback_constant"
        }

    # Estimate θ̂ from terminal phase
    theta_hat = estimate_theta_hat(log_text, summary)

    # Compute T̂_∞ = max{τ, θ̂ · b̂}
    b_hat = svb_result["b_star"]
    T_raw = theta_hat * b_hat
    T_infinity = max(float(tau), T_raw)

    return {
        "T_infinity": T_infinity,
        "solved": False,
        "theta_hat": theta_hat,
        "b_hat": b_hat,
        "phi_star": svb_result.get("phi_star"),
        "x_star": svb_result.get("x_star"),
        "svb_objective": svb_result.get("objective"),
        "samples_used": svb_result.get("samples_used"),
        "method": "svb_extrapolation"
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
    log_ratios = []

    for instance in T_p:
        if instance in T_q:
            t_p = T_p[instance]
            t_q = T_q[instance]

            if t_p > 0 and t_q > 0 and math.isfinite(t_p) and math.isfinite(t_q):
                # Apply cap inside logarithm as per paper
                log_ratio = min(math.log(t_p) - math.log(t_q), math.log(cap))
                log_ratios.append(log_ratio)

    if not log_ratios:
        return float('inf')  # No valid comparisons

    # Geometric mean: exp(1/|I| Σ log_ratios)
    mean_log_ratio = sum(log_ratios) / len(log_ratios)
    return math.exp(mean_log_ratio)


# Legacy compatibility functions (keep existing API) - L1 ONLY
def estimate_svb_from_log(log_text: str) -> Dict[str, Any]:
    """Legacy function using L1 formulation only."""
    samples = extract_log_samples(log_text)
    if len(samples) < 2:
        return {"a": None, "kappa": None, "C": None, "varphi": None, "samples": 0}

    svb_result = fit_svb_growth_factor_l1(samples)
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

    svb_result = fit_svb_growth_factor_l1(samples)
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