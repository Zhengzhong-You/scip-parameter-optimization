from typing import Dict, Any, List

# Unified runner from SolverMind core
from solvermind.core.runner import run_instance as _run_instance


def run_instance(
    cfg: Dict[str, Any],
    instance_path: str,
    tau: float,
    bin: str = None,
    workdir: str = None,
    preset_params: List[str] = None,
    extra_flags: List[str] = None,
    **_,
) -> Dict[str, Any]:
    """Delegate to shared SolverMind runner to avoid duplication.

    Notes:
    - `bin`, `preset_params`, `extra_flags` are ignored; the shared runner
      applies parameters via a .set file and enforces time/seed inside it.
    - `workdir` (if provided) serves as the output root.
    """
    metrics = _run_instance(cfg=cfg, instance_path=instance_path, tau=float(tau), outdir=workdir)
    # Keep baseline's expected keys
    return {
        "time": float(metrics.get("solve_time", metrics.get("time", 0.0))),
        "status": metrics.get("status"),
        "primal": metrics.get("primal"),
        "dual": metrics.get("dual"),
        "nodes": float(metrics.get("nodes", metrics.get("n_nodes", 0.0)) or 0.0),
    }
