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
from utilities.log_utils import per_instance_T_infty


def run_solvermind(whitelist: List[Dict[str, Any]], instance: str, runner_fn, tau: float,
                   tinf_base: Dict[str, float],
                   n_trials: int = 10, seed: int = 0, out_dir: str = "./runs", config: Dict[str, Any] = None
                   ) -> Tuple[Dict[str, Any], float, None, Dict[str, float]]:
    """
    SolverMind optimization function compatible with unified runner interface.

    This wrapper adapts the SolverMind pipeline to work with the same interface
    as SMAC and RBFOpt optimizers.

    Args:
        whitelist: List of parameter definitions (not used directly by SolverMind)
        instance: Instance file path
        runner_fn: Runner function (not used - SolverMind uses its own runner)
        tau: Time limit per evaluation
        tinf_base: Baseline T_infinity values
        n_trials: Number of optimization trials
        seed: Random seed
        out_dir: Output directory

    Returns:
        Tuple of (best_config, best_rhat, None, tinf_best)
    """
    # SolverMind configuration with config file support
    config = config or {}
    solvermind_section = config.get("solvermind", {})

    solvermind_config = {
        "max_trials": n_trials,
        "max_edits": solvermind_section.get("max_edits", 3),  # Default 3, configurable
        "gpt_model": solvermind_section.get("gpt_model", "gpt-5-nano"),  # Default model
        "whitelist_regime": "curated"  # Use curated parameters
    }

    # Run SolverMind tuning
    try:
        result = run_tuning(
            instance=instance,
            time_limit=tau,
            max_trials=solvermind_config["max_trials"],
            max_edits=solvermind_config["max_edits"],
            outdir=out_dir,
            gpt_model=solvermind_config["gpt_model"],
            seed=seed,
            whitelist_regime=solvermind_config["whitelist_regime"]
        )

        # Read the actual results from the log file
        log_path = os.path.join(out_dir, "solvermind.log")
        best_cfg = {}
        best_rhat = float('inf')

        if os.path.exists(log_path):
            try:
                with open(log_path, 'r') as f:
                    log_content = f.read()

                # Look for the tuning session summary
                if "TUNING SESSION COMPLETED" in log_content:
                    lines = log_content.split('\n')
                    in_summary = False

                    for line in lines:
                        if "TUNING SESSION COMPLETED" in line:
                            in_summary = True
                            continue

                        if in_summary:
                            if line.startswith("Best trial:"):
                                trial_num = int(line.split(":")[1].strip())
                            elif line.startswith("Best R_hat:"):
                                best_rhat = float(line.split(":")[1].strip())
                            elif line.startswith("Best parameters:"):
                                # Parse the parameters from following lines
                                best_cfg = {}
                            elif line.strip().startswith("•") and ":" in line and "→" in line:
                                # Parse parameter line like "• separating/clique/freq: 0 → 3"
                                param_line = line.strip()[1:].strip()  # Remove bullet
                                param_name = param_line.split(":")[0].strip()
                                new_value = param_line.split("→")[1].strip()
                                # Try to convert to appropriate type
                                try:
                                    if new_value.lower() == "true":
                                        best_cfg[param_name] = True
                                    elif new_value.lower() == "false":
                                        best_cfg[param_name] = False
                                    elif "." in new_value:
                                        best_cfg[param_name] = float(new_value)
                                    else:
                                        best_cfg[param_name] = int(new_value)
                                except:
                                    best_cfg[param_name] = new_value
                            elif line.startswith("Total trials completed:"):
                                break
            except Exception:
                pass

        # If no improvements found, return baseline as best
        if best_rhat == float('inf') or (best_rhat >= 1.0 and not best_cfg):
            best_cfg = "default"
            best_rhat = 1.0

        # Compute T_infinity for best configuration
        # SolverMind should provide per-instance metrics in result
        best_metrics = result.get("best_metrics", {})
        if best_metrics:
            tinf_best = per_instance_T_infty(best_metrics, tau=tau)
        else:
            # Fallback: use baseline values
            tinf_best = tinf_base.copy()

    except Exception as e:
        print(f"SolverMind optimization failed: {e}")
        # Return fallback values
        best_cfg = "default"
        best_rhat = 1.0
        tinf_best = tinf_base.copy()

    return best_cfg, best_rhat, None, tinf_best