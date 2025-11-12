"""
SolverMind Baseline Integration
==============================

Provides unified interface for SolverMind LLM-based optimization
to work with the same infrastructure as SMAC and RBFOpt.
"""

from typing import Dict, Any, List, Tuple
import os
import json
from pathlib import Path

from .pipeline import run_tuning
from utilities.logs import per_instance_T_infty


def run_solvermind(whitelist: List[Dict[str, Any]], instances: List[str], runner_fn, tau: float,
                   tinf_base: Dict[str, float],
                   n_trials: int = 10, seed: int = 0, out_dir: str = "./runs"
                   ) -> Tuple[Dict[str, Any], float, None, Dict[str, float]]:
    """
    SolverMind optimization function compatible with unified runner interface.

    This wrapper adapts the SolverMind pipeline to work with the same interface
    as SMAC and RBFOpt optimizers.

    Args:
        whitelist: List of parameter definitions (not used directly by SolverMind)
        instances: List of instance file paths
        runner_fn: Runner function (not used - SolverMind uses its own runner)
        tau: Time limit per evaluation
        tinf_base: Baseline T_infinity values
        n_trials: Number of optimization trials
        seed: Random seed
        out_dir: Output directory

    Returns:
        Tuple of (best_config, best_rhat, None, tinf_best)
    """
    # SolverMind configuration
    solvermind_config = {
        "max_trials": n_trials,
        "max_edits": 3,  # Default for SolverMind
        "gpt_model": "gpt-4",  # Default model
        "early_stop_patience": 3,
        "early_stop_delta": 0.01,
        "whitelist_regime": "curated"  # Use curated parameters
    }

    # Run SolverMind tuning
    try:
        result = run_tuning(
            instances=instances,
            time_limit=tau,
            max_trials=solvermind_config["max_trials"],
            max_edits=solvermind_config["max_edits"],
            outdir=out_dir,
            gpt_model=solvermind_config["gpt_model"],
            seed=seed,
            early_stop_patience=solvermind_config["early_stop_patience"],
            early_stop_delta=solvermind_config["early_stop_delta"],
            whitelist_regime=solvermind_config["whitelist_regime"]
        )

        # Extract best configuration and metrics
        best_cfg = result.get("best_config", {})
        best_rhat = result.get("best_r_hat", float('inf'))

        # Compute T_infinity for best configuration
        # SolverMind should provide per-instance metrics in result
        best_metrics = result.get("best_metrics", {})
        if best_metrics:
            tinf_best = per_instance_T_infty(best_metrics, tau=tau)
        else:
            # Fallback: use baseline values
            tinf_best = tinf_base.copy()

    except Exception as e:
        print(f"❌ SolverMind optimization failed: {e}")
        # Return fallback values
        best_cfg = {}
        best_rhat = float('inf')
        tinf_best = tinf_base.copy()

    return best_cfg, best_rhat, None, tinf_best