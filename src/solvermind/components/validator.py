from __future__ import annotations

from typing import Dict, Any, Tuple


class ParamValidator:
    def __init__(self, whitelist: Dict[str, Any], defaults: Dict[str, Any]):
        self.whitelist = whitelist or {}
        self.defaults = defaults or {}

    def validate(self, params: Dict[str, Any], max_edits: int) -> Tuple[Dict[str, Any], Dict[str, str]]:
        applied: Dict[str, Any] = {}
        rejected: Dict[str, str] = {}
        wl_params = set(self.whitelist.get("params", []) or [])
        cnt = 0
        for name, val in (params or {}).items():
            if name not in wl_params:
                rejected[name] = "not in whitelist"; continue
            if name in self.defaults and self.defaults.get(name) == val:
                rejected[name] = "equals default"; continue
            if cnt >= int(max_edits):
                rejected[name] = "exceeds edit cap"; continue
            applied[name] = val; cnt += 1
        return applied, rejected

