
from typing import Dict, Any

def summarize_run_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Return a flat dict with the key metrics that are essential for subsequent analysis.
    (We assume `metrics` already came from SCIP's API, not log parsing.)
    """
    return {
        "timestamp": metrics.get("timestamp"),
        "status": metrics.get("status"),
        "solve_time": metrics.get("solve_time"),
        "time_limit": metrics.get("time_limit"),
        "primal": metrics.get("primal"),
        "dual": metrics.get("dual"),
        "gap": metrics.get("gap"),
        "n_nodes": metrics.get("n_nodes"),
        "lp_iterations": metrics.get("lp_iterations"),
        "n_solutions": metrics.get("n_solutions"),
        "obj_sense": metrics.get("obj_sense"),
        "log_path": metrics.get("log_path"),
    }
