from __future__ import annotations

import os
import json
from typing import Dict, Any, List

from utilities.whitelist import get_whitelist
from utilities.scip_cli import get_default_params
from utilities.logs import shrink_scip_log_for_gpt
from utilities.scoring import r_hat_ratio, gm
from utilities.logs import per_instance_T_infty
from utilities.runner import run_instance
from .components.features_batch import collect_batch_features, instance_name
from .components.validator import ParamValidator
from .components.prompt_builder import build_prompt
from .components.stopping import EarlyStopping


def _build_defaults(whitelist: Dict[str, Any]) -> Dict[str, Any]:
    all_defaults = get_default_params()
    keys = whitelist.get("params", []) or []
    return {k: all_defaults[k] for k in keys if k in all_defaults}


def _aggregate_summary(per_instance_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    solved = [m for m in per_instance_metrics.values() if str(m.get("status", "")).lower().find("optimal") >= 0
              or str(m.get("status", "")).lower().find("infeasible") >= 0
              or str(m.get("status", "")).lower().find("unbounded") >= 0]
    unsolved = [m for m in per_instance_metrics.values() if m not in solved]

    # Include all instances in time calculation, not just solved ones
    all_times = [float(m.get("solve_time") or 0.0) for m in per_instance_metrics.values() if m.get("solve_time") is not None]
    gm_time = gm(all_times) if all_times else 0.0
    import math
    nodes_vals = [float(m.get("n_nodes")) for m in per_instance_metrics.values() if m.get("n_nodes") is not None and math.isfinite(float(m.get("n_nodes")))]
    gm_nodes = gm(nodes_vals) if nodes_vals else 0.0
    # Approximate gap GM over unsolved if available
    gaps = [float(m.get("gap")) for m in unsolved if m.get("gap") is not None]
    gm_gap = gm(gaps) if gaps else 0.0

    return {
        "count": len(per_instance_metrics),
        "solved_pct": 100.0 * (len(solved) / max(1, len(per_instance_metrics))),
        "gm_time": gm_time,
        "gm_gap": gm_gap,
        "gm_nodes": gm_nodes,
    }


def run_tuning(
    instances: List[str],
    time_limit: float,
    max_trials: int,
    max_edits: int,
    outdir: str,
    gpt_model: str,
    seed: int,
    early_stop_patience: int,
    early_stop_delta: float,
    whitelist_regime: str = "curated",
) -> Dict[str, Any]:
    os.makedirs(outdir, exist_ok=True)

    whitelist = get_whitelist(regime=whitelist_regime)
    defaults = _build_defaults(whitelist)

    # Clean per-instance subdirectories in batch root
    try:
        base = os.path.basename(os.path.normpath(outdir))
        for p in (instances or []):
            nm = instance_name(p)
            if nm and base != nm:
                per_dir = os.path.join(outdir, nm)
                if os.path.exists(per_dir):
                    import shutil as _sh
                    _sh.rmtree(per_dir, ignore_errors=True)
    except Exception:
        pass

    feats = collect_batch_features(instances)

    solvermind_log_path = os.path.join(outdir, "solvermind.log")
    solvermind_log = open(solvermind_log_path, "w", encoding="utf-8")
    solvermind_log.write("# SolverMind Reasoning Log\n")
    import time as time_module
    solvermind_log.write(f"# Time: {time_module.strftime('%Y-%m-%d %H:%M:%S')}\n")
    solvermind_log.write(f"# Model: {gpt_model}\n")
    from .gpt.call_gpt import get_api_params
    api_params = get_api_params()
    param_str = ", ".join([f"{k}={v}" for k, v in api_params.items()])
    solvermind_log.write(f"# Reproducibility: {param_str}\n\n")


    trials_csv = os.path.join(outdir, "batch_trials.csv")
    per_inst_csv = os.path.join(outdir, "per_instance_metrics.csv")

    # Baseline trial 0 (defaults) acts as fixed baseline q
    baseline_metrics: Dict[str, Dict[str, Any]] = {}
    for p in instances:
        name = instance_name(p)
        outdir_i = os.path.join(outdir, name)
        baseline_metrics[name] = run_instance(p, params={}, time_limit=time_limit, outdir=outdir_i, seed=seed, trial_id=0)

    # Compute baseline T_infty per instance
    tinf_base = per_instance_T_infty(baseline_metrics, tau=time_limit)
    r0 = 1.0  # ratio to itself
    summ0 = _aggregate_summary(baseline_metrics)

    with open(trials_csv, "w", encoding="utf-8", newline="") as f:
        import csv as _csv
        w = _csv.DictWriter(f, fieldnames=["trial", "r_hat", "solved_pct", "gm_time", "gm_gap", "gm_nodes", "applied_param_count", "param_file"])
        w.writeheader(); w.writerow({"trial": 0, "r_hat": r0, "solved_pct": summ0["solved_pct"], "gm_time": summ0["gm_time"], "gm_gap": summ0["gm_gap"], "gm_nodes": summ0["gm_nodes"], "applied_param_count": 0, "param_file": ""})

    with open(per_inst_csv, "w", encoding="utf-8", newline="") as f:
        import csv as _csv
        w = _csv.DictWriter(f, fieldnames=["trial", "instance", "T_infty", "status", "solve_time", "primal", "dual", "gap", "n_nodes", "log_path"])
        w.writeheader()
        for name, m in baseline_metrics.items():
            w.writerow({"trial": 0, "instance": name, "T_infty": tinf_base.get(name), "status": m.get("status"), "solve_time": m.get("solve_time"), "primal": m.get("primal"), "dual": m.get("dual"), "gap": m.get("gap"), "n_nodes": m.get("n_nodes"), "log_path": m.get("log_path")})

    incumbent = {"trial": 0, "r_hat": r0, "params": {}, "param_file": ""}
    stopper = EarlyStopping(patience=early_stop_patience, min_delta=early_stop_delta)
    last_trial = 0

    validator = ParamValidator(whitelist=whitelist, defaults=defaults)
    history_data: List[Dict[str, Any]] = [{"trial": 0, "params": {}, "param_defaults": {}, "r_hat": r0, "summary": summ0}]

    # Attach baseline log digest to history
    name_to_idx0 = {instance_name(p): idx for idx, p in enumerate(instances, start=1)}
    logs0: Dict[str, str] = {}
    logs0_indexed: List[Dict[str, Any]] = []
    for nm, m in baseline_metrics.items():
        lp = m.get("log_path")
        if lp and os.path.exists(lp):
            with open(lp, "r", encoding="utf-8", errors="ignore") as fh:
                snippet = shrink_scip_log_for_gpt(fh.read(), max_length=10000)
                logs0[nm] = snippet
                logs0_indexed.append({"index": int(name_to_idx0.get(nm, 0)), "instance": nm, "snippet": snippet})
    combined0 = "\n".join(f"[Instance {e.get('index',0)}: {e.get('instance','?')}]\n{e.get('snippet','')}\n" for e in sorted(logs0_indexed, key=lambda x: x.get("index", 0)))
    history_data[0]["log_snippets"] = logs0
    history_data[0]["scip_logs_indexed"] = logs0_indexed
    history_data[0]["combined_log_digest"] = combined0
    history_data[0]["param_file"] = ""
    history_data[0]["tau"] = time_limit


    for trial_idx in range(1, int(max_trials) + 1):
        prompt = build_prompt(features_batch=feats, history=history_data, whitelist=whitelist, max_changes=max_edits)
        # For trial 1, we need to create the complete prompt that becomes the file content
        if trial_idx == 1:
            # Reset the file to write the complete prompt format
            solvermind_log.seek(0)
            solvermind_log.truncate()
            solvermind_log.write("You are a SCIP parameter tuning assistant.\n")
            solvermind_log.write("Prefer mathematically impactful solver knobs (branching, presolving, heuristics, separators).\n")
            solvermind_log.write("Do NOT propose I/O, display, logging/verbosity, output formatting, file, or time measurement parameters.\n")
            solvermind_log.write("Optimize the distribution-level ratio objective R_hat across the batch, unless the instance batch is singleton.\n")
            solvermind_log.write("Analyze solver log snippets from previous trials (and any provided metadata) to understand solving behavior.\n")
            solvermind_log.write("Use log information (presolving, progress, timing) to guide parameter choices.\n")
            solvermind_log.write(f"LIMIT: Change at most {max_edits} parameters per trial.\n")
            solvermind_log.write("ALWAYS return strict JSON with keys: params (dict of name->dict with 'value' and 'reason' keys). Example: {'params': {'branching/relpscost/minreliable': {'value': 2, 'reason': 'reduce strong branching overhead'}}}.\n")
            solvermind_log.write("IMPORTANT: Parameter changes cannot be accumulated. Each trial starts from default parameters. If you want to keep successful changes from previous trials, you can say 'I choose to keep the changes from previous trials: param1=value1, param2=value2, and this does not count to the max edits, and you can do still max edits changes for the new round'. If parameters were bad, just don't mention them since all parameter changes will be from the default ones.\n\n")

            solvermind_log.write("Scoring: we minimize a batch-level time ratio R_hat computed from extrapolated time-to-optimality surrogates (T_infty).\n")
            solvermind_log.write("Treat improvements in time, gap as beneficial signals. Actually the T_infty is a uniform metric which you should consider. Less time and less gap means less T_infty.\n\n")

            solvermind_log.write("USER REQUEST:\n")
            solvermind_log.write("Task: propose_parameters\n")
            solvermind_log.write("Problem instances:\n")
            for p in instances:
                name = instance_name(p)
                solvermind_log.write(f"  • {name}\n")

            solvermind_log.write(f"Parameter whitelist ({whitelist.get('regime', '')}): ")
            param_list = whitelist.get('params', [])
            solvermind_log.write(f"{len(param_list)} parameters available\n")
            for param in param_list:
                default_val = defaults.get(param, "unknown")
                solvermind_log.write(f"  • {param} (default: {default_val})\n")

            solvermind_log.write("Previous trials:\n")
            solvermind_log.write("  Trial 0: R_hat=1.000000, 0 parameter(s) changed\n")
            solvermind_log.write("    SCIP solver logs:\n")

            # Add baseline logs
            for nm, m in baseline_metrics.items():
                lp = m.get("log_path")
                if lp and os.path.exists(lp):
                    with open(lp, "r", encoding="utf-8", errors="ignore") as fh:
                        snippet = shrink_scip_log_for_gpt(fh.read(), max_length=10000)
                        solvermind_log.write(f"    [Instance: {nm}]\n")
                        for line in snippet.split('\n'):
                            if line.strip():
                                solvermind_log.write(f"    {line}\n")
                        solvermind_log.write(f"\n")

        # Flush and close the file to ensure it's written completely
        solvermind_log.flush()
        solvermind_log.close()

        # Call LLM with the entire file content as input
        with open(solvermind_log_path, "r", encoding="utf-8") as f:
            file_content = f.read()

        # Create simple prompt structure for the API
        simple_prompt = [
            {"role": "user", "content": file_content}
        ]

        from .gpt.call_gpt import call_gpt
        reply = call_gpt(simple_prompt, model=gpt_model)
        params_data = reply.get("params", {}) or {}

        # Reopen the file in append mode to add results
        solvermind_log = open(solvermind_log_path, "a", encoding="utf-8")

        # Append SolverMind's recommendations to the file
        solvermind_log.write(f"\nSolverMind recommends:\n")
        params = {}
        if params_data:
            for param_name, param_info in params_data.items():
                if isinstance(param_info, dict) and 'value' in param_info and 'reason' in param_info:
                    # New format: {"value": val, "reason": "text"}
                    value = param_info['value']
                    reason = param_info['reason']
                    solvermind_log.write(f"  • {param_name} = {value} ({reason})\n")
                    params[param_name] = value
                else:
                    # Fallback for old format
                    solvermind_log.write(f"  • {param_name} = {param_info}\n")
                    params[param_name] = param_info
        else:
            solvermind_log.write("  • No parameter changes recommended\n")
        solvermind_log.write(f"\n")

        applied, rejected = validator.validate(params=params, max_edits=max_edits)

        param_file = os.path.join(outdir, "param", f"params_trial_{trial_idx}.set")
        os.makedirs(os.path.dirname(param_file), exist_ok=True)
        with open(param_file, "w", encoding="utf-8") as f:
            f.write("# SCIP parameter set file generated by SolverMind\n")
            f.write(f"# Trial {trial_idx}\n\n")
            for k, v in applied.items():
                f.write(f"{k} = {v}\n")
            f.write(f"\nlimits/time = {time_limit}\n")
            if seed is not None:
                try: f.write(f"randomization/randomseedshift = {int(seed)}\n")
                except Exception: pass

        per_m = {}
        for p in instances:
            name = instance_name(p)
            outdir_i = os.path.join(outdir, name)
            per_m[name] = run_instance(p, params=applied, time_limit=time_limit, outdir=outdir_i, seed=seed, trial_id=trial_idx)

        # Compute T_infty per instance for this trial
        tinf_p = per_instance_T_infty(per_m, tau=time_limit)
        rhat = r_hat_ratio(tinf_p, tinf_base, cap=1e3)
        summary = _aggregate_summary(per_m)

        # Append trial results to the file in the new format
        try:
            solvermind_log.write(f"This is the result of trial {trial_idx}:\n")
            solvermind_log.write(f"R_hat: {rhat:.6f} (")
            # Compare with current best trial, not baseline
            best_rhat = incumbent["r_hat"]
            if rhat < best_rhat:
                solvermind_log.write("success - improved over best trial)")
            elif rhat > best_rhat:
                solvermind_log.write("failure - worse than best trial)")
            else:
                solvermind_log.write("neutral - same as best trial)")
            # Calculate geometric mean of T_infty values for display
            tinf_values = [float(v) for v in tinf_p.values() if v is not None]
            from utilities.scoring import gm
            gm_tinf = gm(tinf_values) if tinf_values else 0.0
            solvermind_log.write(f"\nT_infty: {gm_tinf:.2f}s\n")
            solvermind_log.write(f"Nodes: {summary.get('gm_nodes', 0):.0f}\n\n")

            solvermind_log.write("Here is the log:\n")
            # Add SCIP logs for this trial
            for name, mtr in per_m.items():
                lp = mtr.get("log_path")
                if lp and os.path.exists(lp):
                    with open(lp, "r", encoding="utf-8", errors="ignore") as fh:
                        snippet = shrink_scip_log_for_gpt(fh.read(), max_length=10000)
                        solvermind_log.write(f"[Instance: {name}]\n")
                        for line in snippet.split('\n'):
                            if line.strip():
                                solvermind_log.write(f"{line}\n")
                        solvermind_log.write(f"\n")

            # Ask for analysis of why parameters succeeded or failed
            if trial_idx < max_trials:  # Don't ask on the last trial
                # Always compare with the best trial, not the previous trial
                best_rhat = incumbent["r_hat"]

                if rhat < best_rhat:
                    solvermind_log.write("Why did this succeed? Analyze the performance improvement and use this information to make better decisions for the next trial.\n\n")
                else:
                    solvermind_log.write("Why did this fail? Analyze what went wrong and use this information to make better decisions for the next trial.\n\n")

        except Exception:
            pass

        with open(per_inst_csv, "a", encoding="utf-8", newline="") as f:
            import csv as _csv
            w = _csv.DictWriter(f, fieldnames=["trial", "instance", "T_infty", "status", "solve_time", "primal", "dual", "gap", "n_nodes", "log_path"])
            for name, m in per_m.items():
                w.writerow({"trial": trial_idx, "instance": name, "T_infty": tinf_p.get(name), "status": m.get("status"), "solve_time": m.get("solve_time"), "primal": m.get("primal"), "dual": m.get("dual"), "gap": m.get("gap"), "n_nodes": m.get("n_nodes"), "log_path": m.get("log_path")})

        with open(trials_csv, "a", encoding="utf-8", newline="") as f:
            import csv as _csv
            w = _csv.DictWriter(f, fieldnames=["trial", "r_hat", "solved_pct", "gm_time", "gm_gap", "gm_nodes", "applied_param_count", "param_file"])
            w.writerow({"trial": trial_idx, "r_hat": rhat, "solved_pct": summary["solved_pct"], "gm_time": summary["gm_time"], "gm_gap": summary["gm_gap"], "gm_nodes": summary["gm_nodes"], "applied_param_count": len(applied), "param_file": param_file})

        # Collect logs digest for history
        name_to_idx = {instance_name(p): idx for idx, p in enumerate(instances, start=1)}
        logs_indexed: List[Dict[str, Any]] = []
        logs: Dict[str, str] = {}
        for name, mtr in per_m.items():
            lp = mtr.get("log_path")
            if lp and os.path.exists(lp):
                with open(lp, "r", encoding="utf-8", errors="ignore") as fh:
                    snippet = shrink_scip_log_for_gpt(fh.read(), max_length=10000)
                    logs[name] = snippet
                    logs_indexed.append({"index": int(name_to_idx.get(name, 0)), "instance": name, "snippet": snippet})
        combined_digest = "\n".join(
            f"[Instance {e.get('index',0)}: {e.get('instance','?')}]\n{e.get('snippet','')}\n" for e in sorted(logs_indexed, key=lambda x: x.get("index", 0))
        )


        # Collect reasons for history
        reasons_collected = ""
        if params_data:
            reason_parts = []
            for param_name, param_info in params_data.items():
                if isinstance(param_info, dict) and 'reason' in param_info:
                    reason_parts.append(f"{param_name}: {param_info['reason']}")
            reasons_collected = "; ".join(reason_parts)

        history_data.append({
            "trial": trial_idx,
            "params": dict(applied),
            "param_defaults": {k: defaults.get(k) for k in applied.keys()},
            "r_hat": rhat,
            "summary": summary,
            "reasons": reasons_collected,
            "log_snippets": logs,
            "scip_logs_indexed": logs_indexed,
            "combined_log_digest": combined_digest,
            "param_file": param_file,
            "tau": time_limit,
        })

        last_trial = trial_idx
        if stopper.update(rhat):
            break
        if rhat + early_stop_delta < incumbent["r_hat"]:
            incumbent = {
                "trial": trial_idx,
                "r_hat": rhat,
                "params": dict(applied),
                "param_file": param_file,
            }

    # Write final summary in human-readable format
    try:
        solvermind_log.write("=" * 50 + "\n")
        solvermind_log.write("TUNING SESSION COMPLETED\n")
        solvermind_log.write("=" * 50 + "\n")
        solvermind_log.write(f"Best trial: {incumbent['trial']}\n")
        solvermind_log.write(f"Best R_hat: {incumbent['r_hat']:.6f}\n")
        if incumbent['params']:
            solvermind_log.write("Best parameters:\n")
            for param, value in incumbent['params'].items():
                default_val = defaults.get(param, "unknown")
                solvermind_log.write(f"  • {param}: {default_val} → {value}\n")
        else:
            solvermind_log.write("Best configuration: default parameters\n")
        solvermind_log.write(f"Total trials completed: {last_trial}\n")
    except Exception:
        pass

    solvermind_log.close()

    return {
        "trials_csv": trials_csv,
        "per_instance_csv": per_inst_csv,
        "incumbent": incumbent,
        "outdir": outdir,
        "completed_trials": last_trial,
    }
