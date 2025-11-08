
import json
from typing import Dict, Any, List
from solvermind.core.scoring import CAP_LIMIT as _CAP_LIMIT

def build_tuning_prompt(features: Dict[str, Any], history: List[Dict[str, Any]], whitelist: Dict[str, Any], max_changes: int = 8) -> list:
    """Create a Responses API-ready 'input' list with system+user messages.
    We instruct GPT-5 Pro to ONLY choose from the reinforced whitelist and return strict JSON.
    """
    cap_str = f"{_CAP_LIMIT:g}"
    system = {
        "role": "system",
        "content": (
            "You are a SCIP parameter tuning assistant.\n"
            "You MUST only propose changes from a small reinforced whitelist of parameters (no meta settings).\n"
            "Prefer mathematically impactful solver knobs (branching, presolving, heuristics, separators, tolerances).\n"
            "Do NOT propose I/O, display, logging/verbosity, output formatting, file, or time measurement parameters.\n"
            "Optimize the distribution-level objective J_hat across the batch, not single instances.\n"
            "Analyze solver log snippets from previous trials (and any provided metadata) to understand solving behavior.\n"
            "Use log information (presolving, progress, timing) to guide parameter choices.\n"
            "ALWAYS return strict JSON with keys: params (dict of name->value), reasons (string). Do NOT include any meta settings.\n"
            "If any of your suggested values equal SCIP defaults, either justify keeping the default or choose a non-default alternative.\n\n"
            "Problem and scoring definition (for guidance):\n"
            "- Instances i are drawn from an unknown distribution D; we run solver S in environment E with time budget tau.\n"
            "- For configuration p and instance i, a run returns t (seconds), status s in {opt, inf, unb, tl, other}, primal/dual bounds (z_pr, z_du) when defined, and processed nodes n.\n"
            "- Runs exceeding tau are truncated at tau and marked tl.\n"
            "- tier(s) = 0 for conclusive statuses (opt, inf, unb), else 1.\n"
            "- Per-instance scalarization: sigma(p,i) = f1(tier(s)) + f2(t/tau) + f3(g) + f4(n/Nmax), where Nmax is the max nodes across the batch.\n"
            f"  f1(x)=min({cap_str}*x, {cap_str}); f2(x)=min(x, {cap_str}); f3(x)=min(exp(x)-1, {cap_str}); f4(x)=min(1e-6*x, {cap_str}).\n"
            "  We ensure sigma>0 using EPS=1e-9 if needed for geometric means.\n"
            "- The batch objective J_hat is the geometric mean of sigma(p,i) over the batch. Minimize J_hat."
        )
    }
    # Reorder and enrich trial history so each entry starts with the LLM's reasons
    # for that trial (which were formed after reading the previous trial's logs),
    # then parameters, then J_hat and summaries. This is the structured JSON the
    # LLM sees.
    enriched_history: List[Dict[str, Any]] = []
    for trial in history:
        t = trial.get("trial")
        # Build in desired key order: trial -> reasons -> params -> j_hat -> summary -> logs
        entry: Dict[str, Any] = {"trial": t}
        if trial.get("reasons"):
            entry["reasons"] = trial.get("reasons")
        if trial.get("params") is not None:
            entry["params"] = trial.get("params")
        if trial.get("j_hat") is not None:
            entry["j_hat"] = trial.get("j_hat")
        if trial.get("summary") is not None:
            entry["summary"] = trial.get("summary")
        # Include parameter defaults for changed parameters when provided
        if trial.get("param_defaults") is not None:
            entry["param_defaults"] = trial.get("param_defaults")
        # Include log snippets if available (using consistent keys for the model)
        if trial.get("log_snippets"):
            entry["scip_log_snippets"] = trial.get("log_snippets")
            entry["logs_by_instance"] = trial.get("log_snippets")
        elif trial.get("log_snippet"):
            entry["scip_log_snippet"] = trial.get("log_snippet")
        if trial.get("log_snippets_indexed"):
            entry["scip_logs_indexed"] = trial.get("log_snippets_indexed")
        if trial.get("combined_log_digest"):
            entry["scip_logs_batch_digest"] = trial.get("combined_log_digest")
            entry["log_summary"] = trial.get("combined_log_digest")
        # Carry along reproducibility fields when present
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
            "batch_focus": "Prefer changes likely to reduce J_hat across the batch",
            "reference_logs": "Cite instance indices when explaining reasons",
            "avoid_defaults": "Avoid proposing default-equal values unless justified",
            "threads_policy": "Do not propose any thread-related changes; keep all thread counts at default (single-thread) values",
            "math_impact_policy": "Tune algorithmic knobs only; avoid I/O/display/logging/time options",
            "avoid_repeats": "Do not repeat any previously tried parameter set; ensure novelty across trials",
        },
        "return_format": {
            "params": "name->value",
            "reasons": "short rationale"
        }
    }
    user = {"role": "user", "content": json.dumps(user_payload)}
    return [system, user]
