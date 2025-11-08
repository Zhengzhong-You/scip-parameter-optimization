from __future__ import annotations

from typing import Any, Dict, List

import yaml

# Use the solvermind.tuner scip_cli wrapper to obtain defaults
from solvermind.tuner.utils.scip_cli import get_default_params


def _curated_list() -> List[str]:
    params: List[str] = []
    # Limits (useful controls always included)
    params += [
        "limits/gap",
        "limits/absgap",
        "limits/nodes",
        "limits/memory",
        "limits/restarts",
    ]
    # Branching
    params += [
        "branching/scorefunc",
        "branching/preferbinary",
        "branching/relpscost/minreliable",
        "branching/relpscost/maxreliable",
        "branching/relpscost/sbiterquot",
        "branching/relpscost/sbiterofs",
    ]
    # Node selection
    params += [
        "nodeselection/bestestimate/stdpriority",
        "nodeselection/dfs/stdpriority",
        "nodeselection/bfs/stdpriority",
        "nodeselection/childsel",
    ]
    # Cutting planes (global)
    params += [
        "separating/maxrounds",
        "separating/maxroundsroot",
        "separating/maxcuts",
        "separating/maxcutsroot",
    ]
    # Cutting plane families (freq/round controls)
    for fam in [
        "gomory",
        "mir",
        "cmir",
        "flowcover",
        "clique",
        "knapsackcover",
        "oddcycle",
    ]:
        params += [
            f"separating/{fam}/freq",
            f"separating/{fam}/maxrounds",
            f"separating/{fam}/maxroundsroot",
        ]
    # Presolving (global)
    params += [
        "presolving/maxrounds",
        "presolving/maxrestarts",
        "presolving/abortfac",
    ]
    # Presolver-specific maxrounds
    for pres in [
        "probing",
        "aggregation",
        "boundshift",
        "dualfix",
        "implications",
        "trivial",
    ]:
        params.append(f"presolving/{pres}/maxrounds")
    # Primal heuristics (families)
    params += [
        # FeasPump
        "heuristics/feaspump/freq",
        "heuristics/feaspump/freqofs",
        "heuristics/feaspump/maxdepth",
        "heuristics/feaspump/maxlpiterquot",
        "heuristics/feaspump/maxlpiterofs",
        "heuristics/feaspump/beforecuts",
        # RINS
        "heuristics/rins/nodesofs",
        "heuristics/rins/nodesquot",
        "heuristics/rins/minnodes",
        "heuristics/rins/maxnodes",
        "heuristics/rins/nwaitingnodes",
        "heuristics/rins/minfixingrate",
        # Local branching
        "heuristics/localbranching/neighborhoodsize",
        "heuristics/localbranching/nodesofs",
        "heuristics/localbranching/nodesquot",
        "heuristics/localbranching/lplimfac",
        # RENS
        "heuristics/rens/nodesofs",
        "heuristics/rens/nodesquot",
        "heuristics/rens/minnodes",
        "heuristics/rens/maxnodes",
        "heuristics/rens/minfixingrate",
        "heuristics/rens/startsol",
    ]
    # Tolerances
    params += [
        "numerics/feastol",
        "numerics/epsilon",
        "numerics/dualfeastol",
    ]
    return params


def _minimal_list() -> List[str]:
    # A tight set of high-leverage controls suitable for small edit caps
    params: List[str] = []
    # Branching
    params += ["branching/scorefunc"]
    # Node selection
    params += [
        "nodeselection/bestestimate/stdpriority",
        "nodeselection/dfs/stdpriority",
    ]
    # Cutting planes
    params += [
        "separating/maxroundsroot",
        "separating/maxcutsroot",
    ]
    # Presolve
    params += [
        "presolving/maxrounds",
        "presolving/abortfac",
    ]
    # Heuristics
    params += [
        "heuristics/feaspump/freq",
        "heuristics/feaspump/maxlpiterquot",
    ]
    return params


def get_whitelist(regime: str = "curated") -> Dict[str, Any]:
    """Return the reinforced parameter space (no meta toggles).

    regime:
      - "minimal": small, high-impact subset
      - "curated": balanced default list (recommended)
      - "full": all SCIP parameters discovered via `set save` (can be very large)
    """
    regime = (regime or "curated").lower()
    if regime == "minimal":
        params = _minimal_list()
    elif regime == "full":
        # All parameters discovered from CLI, EXCLUDING limits/* and thread-related params
        all_params = sorted(list(get_default_params().keys()))
        def _allowed(name: str) -> bool:
            n = str(name).lower()
            if n.startswith("limits/"):
                return False
            if "/threads" in n or n.startswith("parallel/"):
                return False
            return True
        params = [k for k in all_params if _allowed(k)]
    else:
        params = _curated_list()
    return {"params": params, "regime": regime}


def load_yaml_whitelist(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or []
    return data
