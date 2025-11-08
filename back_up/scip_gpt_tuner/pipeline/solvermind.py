from __future__ import annotations

import os
import json
from typing import Dict, Any, List

from ..params.get_whitelist import get_whitelist
from ..components.features_batch import collect_batch_features, instance_name
from ..components.validator import ParamValidator
from ..components.prompt_builder import build_prompt
from ..components.runner_cli import run_batch_cli
from ..components.objective import compute_j_hat, gm
from ..components.stopping import EarlyStopping
from ..log.parse_scip_log import shrink_scip_log_for_gpt
from ..utils.scip_cli import get_default_params

def _human_prompt_summary(messages: list) -> str:
    import json as _json, textwrap as _tw
    try:
        user = next((m for m in messages if m.get("role") == "user"), None)
        data = _json.loads(user.get("content", "{}")) if user else {}
    except Exception:
        return "(unparsed prompt)"

    lines = []
    # Instances
    try:
        batch = (data.get("features", {}) or {}).get("batch", [])
        names = [str(x.get("instance")) for x in batch if isinstance(x, dict)]
        lines.append(f"Instances: {len(names)} -> {', '.join(names[:5])}{' ...' if len(names) > 5 else ''}")
    except Exception:
        pass
    # Whitelist size
    try:
        wl = data.get("reinforced_whitelist", {})
        wln = len(wl.get("params", []) or [])
        lines.append(f"Whitelist parameters: {wln}")
    except Exception:
        pass
    # Constraints
    try:
        cons = data.get("constraints", {})
        lines.append(f"Constraints: max_changes={cons.get('max_changes')}, avoid_unlisted={cons.get('avoid_unlisted')}")
    except Exception:
        pass
    # History J_hat summary
    try:
        hist = data.get("trial_history", [])
        jh = [h.get("j_hat") for h in hist if isinstance(h, dict) and "j_hat" in h]
        if jh:
            tail = jh[-5:]
            lines.append("J_hat trajectory (tail): " + ", ".join(f"{float(x):.4f}" for x in tail))
    except Exception:
        pass
    # Combined log digest excerpt
    try:
        hist = data.get("trial_history", [])
        digests = [h.get("scip_logs_batch_digest") for h in hist if isinstance(h, dict) and h.get("scip_logs_batch_digest")]
        if digests:
            excerpt = digests[-1][:500].rstrip()
            lines.append("Batch log digest (excerpt):\n" + _tw.indent(excerpt, prefix="  "))
    except Exception:
        pass
    return "\n".join(lines)

def _human_reply_summary(reply: dict) -> str:
    import textwrap as _tw
    params = reply.get("params", {}) or {}
    reasons = reply.get("reasons", "") or ""
    lines = []
    lines.append(f"Proposed parameters: {len(params)} change(s)")
    for k, v in params.items():
        lines.append(f"  - {k} = {v}")
    if reasons:
        lines.append("Reasons:")
        lines.append(_tw.indent(str(reasons).strip(), prefix="  "))
    return "\n".join(lines)

def _human_saturation_prompt_summary(messages: list) -> str:
    import json as _json
    try:
        user = next((m for m in messages if m.get("role") == "user"), None)
        data = _json.loads(user.get("content", "{}")) if user else {}
        traj = [float(x) for x in data.get("jhat_trajectory", [])][-5:]
        md = float(data.get("min_delta", 0.0))
        pat = int(data.get("patience", 0))
        return f"Saturation check: min_delta={md}, patience={pat}, J_hat tail={traj}"
    except Exception:
        return "(unparsed saturation prompt)"

def _human_saturation_reply_summary(reply: dict) -> str:
    try:
        cont = bool(reply.get("continue", True))
        exp = float(reply.get("expected_improvement", 0.0))
        reason = reply.get("reason", "")
        return f"Decision: continue={cont}, expected_improvement={exp}\nReason: {reason}"
    except Exception:
        return "(unparsed saturation reply)"

def _llm_input_plain(messages: list) -> str:
    import json as _json
    import textwrap as _tw
    out = []
    try:
        # System message
        sysm = next((m for m in messages if m.get("role") == "system"), None)
        if sysm:
            out.append("SYSTEM MESSAGE:")
            out.append(_tw.indent(str(sysm.get("content", "")).strip(), prefix="  "))
        # User message
        user = next((m for m in messages if m.get("role") == "user"), None)
        if user:
            out.append("")
            out.append("USER PAYLOAD:")
            try:
                data = _json.loads(user.get("content", "{}"))
            except Exception:
                out.append(_tw.indent(str(user.get("content", "")).strip(), prefix="  "))
                return "\n".join(out)
            # Task
            task = data.get("task")
            if task:
                out.append(f"  task: {task}")
            # Instances
            batch = (data.get("features", {}) or {}).get("batch", [])
            names = [str(x.get("instance")) for x in batch if isinstance(x, dict)]
            if names:
                out.append("  instances:")
                for nm in names:
                    out.append(f"    - {nm}")
            # Instance features (what LLM sees)
            if batch:
                out.append("  features:")
                for x in batch:
                    nm = str(x.get("instance"))
                    feats = x.get("features", {}) or {}
                    out.append(f"    - {nm}:")
                    for k, v in feats.items():
                        out.append(f"        {k} = {v}")
            # Constraints
            cons = data.get("constraints", {}) or {}
            if cons:
                out.append("  constraints:")
                for k, v in cons.items():
                    out.append(f"    - {k} = {v}")
            # Whitelist (full)
            wl = data.get("reinforced_whitelist", {}) or {}
            regime = wl.get("regime")
            params = wl.get("params", []) or []
            if regime:
                out.append(f"  whitelist_regime: {regime}")
            # For 'full' regime (very large), do not enumerate; provide clear instruction instead
            if str(regime).lower() == "full":
                out.append("  whitelist: all SCIP parameters (list omitted)")
                out.append("  note: you may propose any valid parameter; do NOT include meta settings; do NOT propose limits/* changes (time budget is fixed).")
            else:
                out.append(f"  whitelist: {len(params)} parameters")
                for p in params:
                    out.append(f"    - {p}")
            # Full trial history summary with a human-first ordering:
            # show the LLM's reasons for trial k (based on k-1 logs) first,
            # then the chosen parameters, and finally highlight J_hat.
            hist = data.get("trial_history", []) or []
            if hist:
                out.append("  trial_history:")
                # Build map from trial -> reasons (as produced by that trial's LLM)
                reasons_by_trial: Dict[int, str] = {}
                try:
                    for hh in hist:
                        ti = int(hh.get('trial', -1))
                        rtext = hh.get('reasons')
                        if ti >= 0 and rtext:
                            reasons_by_trial[ti] = str(rtext)
                except Exception:
                    pass
                for h in hist:
                    try:
                        tr = h.get('trial')
                        jh = h.get('j_hat')
                        try:
                            tr_i = int(tr)
                        except Exception:
                            tr_i = -1
                        label = " (baseline)" if tr_i == 0 else ""
                        out.append(f"    - trial {tr}{label}:")
                        # Reasons for this trial (derived from previous trial's logs)
                        if tr_i > 0:
                            prev_label = f"trial {tr_i-1}'s log"
                            out.append(
                                f"        after reading {prev_label}, the LLM gives the reasons for improvement:"
                            )
                            rcur = reasons_by_trial.get(tr_i)
                            if rcur:
                                out.append(_tw.indent(rcur.strip(), prefix="          "))
                            else:
                                out.append("          none")
                        else:
                            out.append("        baseline run (no prior log)")
                        # Chosen parameters for this run
                        params_h = h.get('params') or {}
                        params_defaults = h.get('param_defaults') or {}
                        if params_h:
                            out.append("        and choose the following parameters:")
                            for k, v in params_h.items():
                                if k in params_defaults and params_defaults.get(k) is not None:
                                    out.append(f"          - {k} = {v}  (default: {params_defaults.get(k)})")
                                else:
                                    out.append(f"          - {k} = {v}")
                        else:
                            out.append("        and choose the following parameters: (defaults only)")
                        # Attach concise log summary for this run
                        dig = h.get('log_summary') or h.get('scip_logs_batch_digest') or h.get('combined_log_digest')
                        if dig:
                            out.append("        log_summary:")
                            out.append(_tw.indent(str(dig).rstrip(), prefix="          "))
                        # Highlight J_hat at the end for emphasis
                        try:
                            out.append(f"        IMPORTANT: J_hat={float(jh):.4f}")
                        except Exception:
                            pass
                    except Exception:
                        pass
            # No separate batch_log_digest section; trial entries show reasons/params/J_hat
            # No defaults report included (by request)
            # Guidance
            guidance = data.get("guidance", {}) or {}
            if guidance:
                out.append("  guidance:")
                for k, v in guidance.items():
                    out.append(f"    - {k}: {v}")
            # Return format (schema hint)
            rf = data.get("return_format", {}) or {}
            if rf:
                out.append("  return_format:")
                for k, v in rf.items():
                    out.append(f"    - {k}: {v}")
    except Exception:
        return "(unparsed llm input)"
    return "\n".join(out)


def _build_defaults(whitelist: Dict[str, Any]) -> Dict[str, Any]:
    """Return default values for whitelisted parameters via SCIP CLI."""
    all_defaults = get_default_params()
    keys = whitelist.get("params", []) or []
    return {k: all_defaults[k] for k in keys if k in all_defaults}


def _aggregate_summary(per_instance_metrics: Dict[str, Dict[str, Any]], tau: float) -> Dict[str, Any]:
    solved = [m for m in per_instance_metrics.values() if str(m.get("status", "")).lower().find("optimal") >= 0
              or str(m.get("status", "")).lower().find("infeasible") >= 0
              or str(m.get("status", "")).lower().find("unbounded") >= 0]
    unsolved = [m for m in per_instance_metrics.values() if m not in solved]

    gm_time = gm([float(m.get("solve_time") or 0.0) for m in solved]) if solved else 0.0
    from solvermind.core.scoring import gap_from_bounds as _gap_from_bounds  # type: ignore
    # Prefer SCIP-reported gaps for unsolved; fall back to computing from bounds
    gaps = []
    if unsolved:
        from solvermind.core.scoring import gap_from_bounds as _gap_from_bounds  # type: ignore
        for m in unsolved:
            g = m.get("gap")
            if g is None:
                g = _gap_from_bounds(m.get("primal"), m.get("dual"))
            gaps.append(g)
    gm_gap = gm(gaps) if gaps else 0.0
    import math
    nodes_vals = [float(m.get("n_nodes")) for m in per_instance_metrics.values() if m.get("n_nodes") is not None and math.isfinite(float(m.get("n_nodes")))]
    gm_nodes = gm(nodes_vals) if nodes_vals else 0.0

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
    dry_run: bool,
    debug: bool,
    early_stop_patience: int,
    early_stop_delta: float,
    whitelist_regime: str = "curated",
) -> Dict[str, Any]:
    os.makedirs(outdir, exist_ok=True)
    # Version check is enforced centrally in the shared runner; no duplicate check here

    # If running in batch mode (outdir root is not an instance-named directory),
    # clean per-instance subdirectories to avoid mixing previous artifacts.
    try:
        base = os.path.basename(os.path.normpath(outdir))
        from ..components.features_batch import instance_name as _iname
        for p in (instances or []):
            nm = _iname(p)
            if nm and base != nm:
                per_dir = os.path.join(outdir, nm)
                if os.path.exists(per_dir):
                    import shutil as _sh
                    _sh.rmtree(per_dir, ignore_errors=True)
    except Exception:
        pass

    whitelist = get_whitelist(regime=whitelist_regime)
    defaults = _build_defaults(whitelist)

    # Prepare parameter output folder (clean + recreate)
    param_dir = os.path.join(outdir, "param")
    try:
        if os.path.exists(param_dir):
            import shutil as _sh
            _sh.rmtree(param_dir, ignore_errors=True)
        os.makedirs(param_dir, exist_ok=True)
    except Exception:
        pass

    # Features (lightweight, via CLI). In dry-run, skip to avoid invoking CLI.
    feats = collect_batch_features(instances, dry_run=dry_run)

    # Logging
    gpt_log_path = os.path.join(outdir, "gpt_reasoning.log")
    gpt_log = open(gpt_log_path, "w", encoding="utf-8")
    gpt_log.write("# SolverMind GPT Reasoning Log\n")
    gpt_log.write(f"# Model: {gpt_model}\n")
    import time as time_module
    gpt_log.write(f"# Time: {time_module.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    trials_csv = os.path.join(outdir, "batch_trials.csv")
    per_inst_csv = os.path.join(outdir, "per_instance_metrics.csv")

    # Baseline settings file (.set)
    baseline_param_file = os.path.join(param_dir, "params_trial_0.set")
    with open(baseline_param_file, "w", encoding="utf-8") as f:
        f.write("# SCIP parameter set file - BASELINE (defaults)\n# Trial 0\n\n")
        f.write(f"limits/time = {time_limit}\n")
        try:
            if seed is not None:
                f.write(f"randomization/randomseedshift = {int(seed)}\n")
        except Exception:
            pass

    # Baseline metrics
    if dry_run:
        baseline_metrics = {instance_name(p): {"status": "dry-run", "solve_time": 0.0, "time_limit": time_limit, "primal": float("inf"), "dual": float("inf"), "gap": 1.0, "n_nodes": 0, "log_path": ""} for p in instances}
    else:
        baseline_metrics = run_batch_cli(instances, settings_file=baseline_param_file, time_limit=time_limit, outdir=outdir, seed=seed, trial_id=0)

    j0, sig0 = compute_j_hat(baseline_metrics, tau=time_limit)
    summ0 = _aggregate_summary(baseline_metrics, tau=time_limit)

    # Outputs
    with open(trials_csv, "w", encoding="utf-8", newline="") as f:
        import csv as _csv
        w = _csv.DictWriter(f, fieldnames=["trial", "j_hat", "solved_pct", "gm_time", "gm_gap", "gm_nodes", "applied_param_count", "param_file"])
        w.writeheader()
        w.writerow({"trial": 0, "j_hat": j0, "solved_pct": summ0["solved_pct"], "gm_time": summ0["gm_time"], "gm_gap": summ0["gm_gap"], "gm_nodes": summ0["gm_nodes"], "applied_param_count": 0, "param_file": baseline_param_file})

    with open(per_inst_csv, "w", encoding="utf-8", newline="") as f:
        import csv as _csv
        w = _csv.DictWriter(f, fieldnames=["trial", "instance", "sigma", "status", "solve_time", "primal", "dual", "gap", "n_nodes", "log_path"])
        w.writeheader()
        for name, m in baseline_metrics.items():
            w.writerow({"trial": 0, "instance": name, "sigma": sig0.get(name), "status": m.get("status"), "solve_time": m.get("solve_time"), "primal": m.get("primal"), "dual": m.get("dual"), "gap": m.get("gap"), "n_nodes": m.get("n_nodes"), "log_path": m.get("log_path")})

    incumbent = {"trial": 0, "j_hat": j0, "params": {}, "param_file": baseline_param_file}
    stopper = EarlyStopping(patience=early_stop_patience, min_delta=early_stop_delta)
    last_trial = 0
    llm_stop_count = 0

    validator = ParamValidator(whitelist=whitelist, defaults=defaults, mode="cli")
    history_data: List[Dict[str, Any]] = [
        {"trial": 0, "params": {}, "param_defaults": {}, "j_hat": j0, "summary": summ0}
    ]

    # Attach baseline log digest into history for trial 1, but do not print it here
    if not dry_run:
        name_to_idx0 = {instance_name(p): idx for idx, p in enumerate(instances, start=1)}
        logs0: Dict[str, str] = {}
        logs0_indexed: List[Dict[str, Any]] = []
        for nm, m in baseline_metrics.items():
            try:
                lp = m.get("log_path")
                if lp and os.path.exists(lp):
                    with open(lp, "r", encoding="utf-8", errors="ignore") as fh:
                        snippet = shrink_scip_log_for_gpt(fh.read(), max_length=10000)
                        logs0[nm] = snippet
                        logs0_indexed.append({
                            "index": int(name_to_idx0.get(nm, 0)),
                            "instance": nm,
                            "snippet": snippet,
                        })
            except Exception:
                pass
        combined0_lines = []
        for entry in sorted(logs0_indexed, key=lambda x: x.get("index", 0)):
            combined0_lines.append(f"[Instance {entry.get('index',0)}: {entry.get('instance','?')}]\n{entry.get('snippet','')}\n")
        combined0 = "\n".join(combined0_lines)

        history_data[0]["log_snippets"] = logs0
        history_data[0]["log_snippets_indexed"] = logs0_indexed
        history_data[0]["combined_log_digest"] = combined0
        history_data[0]["param_file"] = baseline_param_file
        history_data[0]["tau"] = time_limit

    for trial_idx in range(1, int(max_trials) + 1):
        # Build and call GPT (use history per Algorithm 1)
        prompt = build_prompt(features_batch=feats, history=history_data, whitelist=whitelist, defaults=defaults, max_changes=max_edits)
        # Write only what LLM sees: the prompt in plain text within separators
        gpt_log.write(f"=== TRIAL {trial_idx} ===\n")
        gpt_log.write('""""""""""""""""\n')
        gpt_log.write("LLM INPUT (Plain Text):\n")
        gpt_log.write(_llm_input_plain(prompt) + "\n")
        gpt_log.write('""""""""""""""""\n\n')

        from ..gpt.call_gpt import call_gpt
        gpt_reply = call_gpt(prompt, model=gpt_model)
        gpt_log.write("LLM RESPONSE SUMMARY:\n")
        gpt_log.write(_human_reply_summary(gpt_reply) + "\n\n")

        params = gpt_reply.get("params", {}) or {}
        # Accept both {params,reasons} and {params,meta,reasons} but ignore meta.
        reasons = gpt_reply.get("reasons", "")

        applied, rejected, run_meta = validator.validate(params=params, meta={}, max_edits=max_edits)

        # Write param file for this trial
        param_file = os.path.join(param_dir, f"params_trial_{trial_idx}.set")
        with open(param_file, "w", encoding="utf-8") as f:
            f.write("# SCIP parameter set file generated by SolverMind\n")
            f.write(f"# Trial {trial_idx}\n\n")
            for k, v in applied.items():
                if not k.startswith("meta:") and k in defaults:
                    f.write(f"{k} = {v}\n")
            f.write(f"\nlimits/time = {time_limit}\n")
            try:
                if seed is not None:
                    f.write(f"randomization/randomseedshift = {int(seed)}\n")
            except Exception:
                pass

        # Run and score
        per_m = run_batch_cli(instances, settings_file=param_file, time_limit=time_limit, outdir=outdir, seed=seed, trial_id=trial_idx)
        jhat, sigmas = compute_j_hat(per_m, tau=time_limit)
        summary = _aggregate_summary(per_m, tau=time_limit)

        with open(per_inst_csv, "a", encoding="utf-8", newline="") as f:
            import csv as _csv
            w = _csv.DictWriter(f, fieldnames=["trial", "instance", "sigma", "status", "solve_time", "primal", "dual", "gap", "n_nodes", "log_path"])
            for name, m in per_m.items():
                w.writerow({"trial": trial_idx, "instance": name, "sigma": sigmas.get(name), "status": m.get("status"), "solve_time": m.get("solve_time"), "primal": m.get("primal"), "dual": m.get("dual"), "gap": m.get("gap"), "n_nodes": m.get("n_nodes"), "log_path": m.get("log_path")})

        with open(trials_csv, "a", encoding="utf-8", newline="") as f:
            import csv as _csv
            w = _csv.DictWriter(f, fieldnames=["trial", "j_hat", "solved_pct", "gm_time", "gm_gap", "gm_nodes", "applied_param_count", "param_file"])
            w.writerow({"trial": trial_idx, "j_hat": jhat, "solved_pct": summary["solved_pct"], "gm_time": summary["gm_time"], "gm_gap": summary["gm_gap"], "gm_nodes": summary["gm_nodes"], "applied_param_count": len([k for k in applied.keys() if not k.startswith("meta:")]), "param_file": param_file})

        # Collect per-instance log snippets and a combined, indexed digest
        name_to_idx = {instance_name(p): idx for idx, p in enumerate(instances, start=1)}
        logs: Dict[str, str] = {}
        logs_indexed: List[Dict[str, Any]] = []
        for name, mtr in per_m.items():
            try:
                lp = mtr.get("log_path")
                if lp and os.path.exists(lp):
                    with open(lp, "r", encoding="utf-8", errors="ignore") as fh:
                        snippet = shrink_scip_log_for_gpt(fh.read(), max_length=10000)
                        logs[name] = snippet
                        logs_indexed.append({
                            "index": int(name_to_idx.get(name, 0)),
                            "instance": name,
                            "snippet": snippet,
                        })
            except Exception:
                pass
        # Build a compact digest that GPT can scan quickly
        combined_digest_lines = []
        for entry in sorted(logs_indexed, key=lambda x: x.get("index", 0)):
            idx = entry.get("index", 0)
            nm = entry.get("instance", "?")
            combined_digest_lines.append(f"[Instance {idx}: {nm}]\n{entry.get('snippet','')}\n")
        combined_digest = "\n".join(combined_digest_lines)

        # Build per-trial defaults for the parameters we changed
        changed_params = {k: v for k, v in applied.items() if not k.startswith("meta:")}
        param_defaults_map = {k: defaults.get(k) for k in changed_params.keys()}

        history_data.append({
            "trial": trial_idx,
            "params": changed_params,
            "param_defaults": param_defaults_map,
            "j_hat": jhat,
            "summary": summary,
            "reasons": reasons,
            "log_snippets": logs,
            "log_snippets_indexed": logs_indexed,
            "combined_log_digest": combined_digest,
            "param_file": param_file,
            "tau": time_limit,
        })

        last_trial = trial_idx
        # LLM-guided saturation check (skip on final trial; no chance to run further)
        if not dry_run and early_stop_patience > 0 and trial_idx < int(max_trials):
            try:
                from ..gpt.build_saturation_prompt import build_saturation_prompt
                from ..gpt.call_gpt import call_gpt_json
                traj = [float(h.get("j_hat", 0.0)) for h in history_data]
                sat_prompt = build_saturation_prompt(jhat_trajectory=traj, patience=early_stop_patience, min_delta=early_stop_delta, trial_index=trial_idx)
                gpt_log.write(f"=== TRIAL {trial_idx} - LLM SATURATION CHECK ===\n")
                # Write only plain-text summary of saturation input within separators
                gpt_log.write('""""""""""""""""\n')
                gpt_log.write("LLM SATURATION INPUT (Plain Text):\n")
                gpt_log.write(_human_saturation_prompt_summary(sat_prompt) + "\n")
                gpt_log.write('""""""""""""""""\n')

                sat_reply = call_gpt_json(sat_prompt, model=gpt_model)
                gpt_log.write(_human_saturation_reply_summary(sat_reply) + "\n\n")
                exp_impr = float(sat_reply.get("expected_improvement", 0.0))
                cont = bool(sat_reply.get("continue", True))
                if (not cont) or (exp_impr < float(early_stop_delta)):
                    llm_stop_count += 1
                else:
                    llm_stop_count = 0
                if llm_stop_count >= int(early_stop_patience):
                    break
            except Exception:
                # Do not fail the run if the LLM saturation check errors
                pass
        # Numeric early stopping as a secondary criterion
        if stopper.update(jhat):
            break

        if jhat + early_stop_delta < incumbent["j_hat"]:
            incumbent = {
                "trial": trial_idx,
                "j_hat": jhat,
                "params": {k: v for k, v in applied.items() if not k.startswith("meta:")},
                "param_file": param_file,
            }

    gpt_log.write("=== TUNING SESSION COMPLETED ===\n")
    gpt_log.close()

    return {
        "trials_csv": trials_csv,
        "per_instance_csv": per_inst_csv,
        "incumbent": incumbent,
        "outdir": outdir,
        "completed_trials": last_trial,
    }
