from __future__ import annotations

from typing import Any, Dict, List

import yaml
from .scip_cli import get_default_params


def _curated_list() -> List[str]:
    params: List[str] = []
    params += [
        "limits/gap",
        "limits/absgap",
        "limits/nodes",
        "limits/memory",
        "limits/restarts",
    ]
    params += [
        "branching/scorefunc",
        "branching/preferbinary",
        "branching/relpscost/minreliable",
        "branching/relpscost/maxreliable",
        "branching/relpscost/sbiterquot",
        "branching/relpscost/sbiterofs",
    ]
    params += [
        "nodeselection/bestestimate/stdpriority",
        "nodeselection/dfs/stdpriority",
        "nodeselection/bfs/stdpriority",
        "nodeselection/childsel",
    ]
    params += [
        "separating/maxrounds",
        "separating/maxroundsroot",
        "separating/maxcuts",
        "separating/maxcutsroot",
    ]
    for fam in ["gomory", "mir", "cmir", "flowcover", "clique", "knapsackcover", "oddcycle"]:
        params += [f"separating/{fam}/freq", f"separating/{fam}/maxrounds", f"separating/{fam}/maxroundsroot"]
    params += [
        "presolving/maxrounds",
        "presolving/maxrestarts",
        "presolving/abortfac",
    ]
    for pres in ["probing", "aggregation", "boundshift", "dualfix", "implications", "trivial"]:
        params.append(f"presolving/{pres}/maxrounds")
    params += [
        "heuristics/feaspump/freq",
        "heuristics/feaspump/freqofs",
        "heuristics/feaspump/maxdepth",
        "heuristics/feaspump/maxlpiterquot",
        "heuristics/feaspump/maxlpiterofs",
        "heuristics/feaspump/beforecuts",
        "heuristics/rins/nodesofs",
        "heuristics/rins/nodesquot",
        "heuristics/rins/minnodes",
        "heuristics/rins/maxnodes",
        "heuristics/rins/nwaitingnodes",
        "heuristics/rins/minfixingrate",
        "heuristics/localbranching/neighborhoodsize",
        "heuristics/localbranching/nodesofs",
        "heuristics/localbranching/nodesquot",
        "heuristics/localbranching/lplimfac",
        "heuristics/rens/nodesofs",
        "heuristics/rens/nodesquot",
        "heuristics/rens/minnodes",
        "heuristics/rens/maxnodes",
        "heuristics/rens/minfixingrate",
        "heuristics/rens/startsol",
    ]
    params += ["numerics/feastol", "numerics/epsilon", "numerics/dualfeastol"]
    return params


def _minimal_list() -> List[str]:
    params: List[str] = []
    params += ["branching/scorefunc"]
    params += ["nodeselection/bestestimate/stdpriority", "nodeselection/dfs/stdpriority"]
    params += ["separating/maxroundsroot", "separating/maxcutsroot"]
    params += ["presolving/maxrounds", "presolving/abortfac"]
    params += ["heuristics/feaspump/freq", "heuristics/feaspump/maxlpiterquot"]
    return params


def get_whitelist(regime: str = "curated") -> Dict[str, Any]:
    regime = (regime or "curated").lower()
    if regime == "minimal":
        params = _minimal_list()
    elif regime == "full":
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

