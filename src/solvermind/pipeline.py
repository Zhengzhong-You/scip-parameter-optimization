from __future__ import annotations

import os
import json
from typing import Dict, Any, List
from pathlib import Path

from utilities.whitelist import get_whitelist
from utilities.scip_cli import get_default_params
from utilities.log_utils import shrink_scip_log_for_gpt, per_instance_T_infty
from utilities.scoring import r_hat_ratio
from utilities.runner import run_instance
from .components.validator import ParamValidator


def instance_name(path: str) -> str:
    return Path(path).stem


def run_solvermind_optimization(
    instance_path: str,
    output_dir: str,
    iterations: int = 10,
    trial_timeout: float = 600,
    max_edits: int = 3,
    gpt_model: str = "gpt-4-mini",
    seed: int = 0,
    whitelist_regime: str = "curated",
) -> Dict[str, Any]:
    """
    Run SolverMind optimization on a single instance.
    """
    os.makedirs(output_dir, exist_ok=True)

    whitelist = get_whitelist(regime=whitelist_regime)
    all_defaults = get_default_params()
    defaults = {k: all_defaults[k] for k in whitelist.get("params", []) if k in all_defaults}

    name = instance_name(instance_path)
    log_path = os.path.join(output_dir, "solvermind.log")

    with open(log_path, "w", encoding="utf-8") as log:
        log.write("# SolverMind Single Instance Optimization\n")
        log.write(f"# Instance: {name}\n")
        log.write(f"# Model: {gpt_model}\n\n")

        # Baseline run
        log.write("Running baseline with default parameters...\n")
        baseline_result = run_instance(
            instance_path, params={}, time_limit=trial_timeout,
            outdir=output_dir, trial_id=0, seed=seed
        )

        tinf_base = per_instance_T_infty({name: baseline_result}, tau=trial_timeout)
        baseline_tinf = tinf_base[name]

        log.write(f"Baseline T_infinity: {baseline_tinf:.2f}s\n\n")

        # Initialize optimization
        validator = ParamValidator(whitelist=whitelist, defaults=defaults)

        best_config = {}
        best_rhat = 1.0
        best_tinf = baseline_tinf

        for trial in range(1, iterations + 1):
            log.write(f"=== TRIAL {trial} ===\n")

            # Simple parameter suggestion (for now, just return empty - can be enhanced later)
            from .gpt.call_gpt import call_gpt

            prompt = f"""You are tuning SCIP parameters for instance {name}.
Current best R_hat: {best_rhat:.4f}
Suggest up to {max_edits} parameter changes from: {list(whitelist.get('params', []))}
Return JSON: {{"params": {{"param/name": {{"value": <val>, "reason": "<why>"}}}}}}"""

            try:
                response = call_gpt([{"role": "user", "content": prompt}], model=gpt_model)
                suggested_params = response.get("params", {})
            except Exception as e:
                log.write(f"LLM call failed: {e}\n")
                suggested_params = {}

            if not suggested_params:
                log.write("No parameter suggestions, stopping.\n")
                break

            # Validate parameters
            params = {}
            for param_name, param_info in suggested_params.items():
                if isinstance(param_info, dict) and 'value' in param_info:
                    params[param_name] = param_info['value']

            applied, rejected = validator.validate(params=params, max_edits=max_edits)

            if not applied:
                log.write("No valid parameters to apply, stopping.\n")
                break

            log.write(f"Applying: {applied}\n")

            # Run trial
            trial_result = run_instance(
                instance_path, params=applied, time_limit=trial_timeout,
                outdir=output_dir, trial_id=trial, seed=seed
            )

            # Compute metrics
            tinf_trial = per_instance_T_infty({name: trial_result}, tau=trial_timeout)
            trial_tinf = tinf_trial[name]
            rhat = r_hat_ratio(tinf_trial, tinf_base, cap=1e3)

            log.write(f"Trial {trial} R_hat: {rhat:.4f} (T_infinity: {trial_tinf:.2f}s)\n")

            # Update best
            if rhat < best_rhat:
                best_rhat = rhat
                best_config = applied.copy()
                best_tinf = trial_tinf
                log.write("*** NEW BEST ***\n")

            log.write("\n")


        log.write("\n=== OPTIMIZATION COMPLETE ===\n")
        log.write(f"Best R_hat: {best_rhat:.4f}\n")
        log.write(f"Best T_infinity: {best_tinf:.2f}s\n")
        log.write(f"Best config: {best_config}\n")

    return {
        "best_config": best_config,
        "best_rhat": best_rhat,
        "best_metrics": {name: {"T_infty": best_tinf}},
        "baseline_metrics": {name: baseline_result}
    }


# For backward compatibility
def run_tuning(instance: str, time_limit: float, max_trials: int, max_edits: int,
               outdir: str, gpt_model: str, seed: int, whitelist_regime: str = "curated") -> Dict[str, Any]:
    """Legacy function for single instance optimization."""
    return run_solvermind_optimization(
        instance_path=instance,
        output_dir=outdir,
        iterations=max_trials,
        trial_timeout=time_limit,
        max_edits=max_edits,
        gpt_model=gpt_model,
        seed=seed,
        whitelist_regime=whitelist_regime,
    )