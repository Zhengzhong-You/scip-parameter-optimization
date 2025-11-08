import json
from typing import Dict, Any, List
from utilities.scoring import CAP_LIMIT as _CAP_LIMIT


def build_tuning_prompt(features: Dict[str, Any], history: List[Dict[str, Any]], whitelist: Dict[str, Any], max_changes: int = 8) -> list:
    system = {
        "role": "system",
        "content": (
            "You are a SCIP parameter tuning assistant.\n"
            "You MUST only propose changes from a small reinforced whitelist of parameters (no meta settings).\n"
            "Prefer mathematically impactful solver knobs (branching, presolving, heuristics, separators, tolerances).\n"
            "Do NOT propose I/O, display, logging/verbosity, output formatting, file, or time measurement parameters.\n"
            "Optimize the distribution-level ratio objective R_hat across the batch, not single instances.\n"
            "Analyze solver log snippets from previous trials (and any provided metadata) to understand solving behavior.\n"
            "Use log information (presolving, progress, timing) to guide parameter choices.\n"
            "ALWAYS return strict JSON with keys: params (dict of name->value), reasons (string).\n\n"
            "Scoring: we minimize a batch-level time ratio R_hat computed from extrapolated time-to-optimality surrogates (T_infty).\n"
            "Treat improvements in solve status, time, gap, and search effort as beneficial signals.\n"
        )
    }

    enriched_history: List[Dict[str, Any]] = []
    for trial in history:
        t = trial.get("trial")
        entry: Dict[str, Any] = {"trial": t}
        if trial.get("reasons"):
            entry["reasons"] = trial.get("reasons")
        if trial.get("params") is not None:
            entry["params"] = trial.get("params")
        if trial.get("r_hat") is not None:
            entry["r_hat"] = trial.get("r_hat")
        if trial.get("summary") is not None:
            entry["summary"] = trial.get("summary")
        if trial.get("param_defaults") is not None:
            entry["param_defaults"] = trial.get("param_defaults")
        if trial.get("log_snippets"):
            entry["scip_log_snippets"] = trial.get("log_snippets")
        if trial.get("scip_logs_indexed"):
            entry["scip_logs_indexed"] = trial.get("scip_logs_indexed")
        if trial.get("combined_log_digest"):
            entry["scip_logs_batch_digest"] = trial.get("combined_log_digest")
        if trial.get("param_file") is not None:
            entry["param_file"] = trial.get("param_file")
        if trial.get("tau") is not None:
            entry["tau"] = trial.get("tau")
        enriched_history.append(entry)

    user_payload = {
        "task": "propose_parameters",
        "features": features,
        "reinforced_whitelist": whitelist,
        "trial_history": enriched_history,
        "constraints": {
            "max_changes": int(max_changes),
            "avoid_unlisted": True
        },
        "guidance": {
            "batch_focus": "Prefer changes likely to reduce R_hat across the batch",
            "avoid_defaults": "Avoid proposing default-equal values unless justified",
            "threads_policy": "Do not propose thread-related changes; keep default single-thread settings",
        },
        "return_format": {"params": "name->value", "reasons": "short rationale"}
    }
    user = {"role": "user", "content": json.dumps(user_payload)}
    return [system, user]
