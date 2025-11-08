import time, random
from typing import Dict, Any

def run_instance(cfg: Dict[str, Any], instance_path: str, tau: float, **kwargs) -> Dict[str, Any]:
    seed = abs(hash((instance_path, tuple(sorted(cfg.items()))))) % (2**32)
    rng = random.Random(seed)

    base = 0.2 + 0.01 * (abs(float(cfg.get("presolving/abortfac", 0.5)) - 0.3) * 100.0)
    base += 0.002 * float(cfg.get("separating/maxroundsroot", 5))
    base += 0.0001 * float(cfg.get("separating/maxcutsroot", 1000))
    t = min(tau, base + rng.random() * 0.05)

    gap = max(0.0, 0.2 - 0.01*float(cfg.get("presolving/maxrounds", 3)) - 0.00005*float(cfg.get("separating/maxcutsroot", 1000)))
    gap += 0.02 * rng.random()

    if gap < 0.02 and t < 0.95 * tau:
        status = "opt"; primal = 1000.0 + rng.random(); dual = primal
    else:
        status = "tl" if t >= tau else "other"
        primal = 1000.0 + rng.random(); dual = primal * (1.0 + gap)

    nodes = int(1e4 * gap + rng.random()*1000)
    time.sleep(0.01)
    return dict(time=t, status=status, primal=primal, dual=dual, nodes=nodes)
