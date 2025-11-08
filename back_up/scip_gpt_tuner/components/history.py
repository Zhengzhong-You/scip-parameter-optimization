from __future__ import annotations

from typing import Dict, Any, List


class History:
    def __init__(self):
        self._h: List[Dict[str, Any]] = []

    def add_baseline(self, j_hat: float, summary: Dict[str, Any]):
        self._h.append({"trial": 0, "params": {}, "j_hat": j_hat, "summary": summary})

    def add_trial(self, trial_idx: int, params: Dict[str, Any], j_hat: float, summary: Dict[str, Any], reasons: str, log_snippets: Dict[str, str]):
        self._h.append({
            "trial": trial_idx,
            "params": params,
            "j_hat": j_hat,
            "summary": summary,
            "reasons": reasons,
            "log_snippets": log_snippets,
        })

    @property
    def as_list(self) -> List[Dict[str, Any]]:
        return list(self._h)
