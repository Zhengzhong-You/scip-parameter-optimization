from __future__ import annotations

from typing import Dict, Any, Tuple


class ParamValidator:
    def __init__(self, whitelist: Dict[str, Any], defaults: Dict[str, Any], mode: str = "cli"):
        self.whitelist = whitelist or {}
        self.defaults = defaults or {}
        self.mode = "cli"  # CLI-only validation

    def _validate_cli(self, params: Dict[str, Any], max_edits: int) -> Tuple[Dict[str, Any], Dict[str, str], Dict[str, Any]]:
        applied: Dict[str, Any] = {}
        rejected: Dict[str, str] = {}

        wl_params = set(self.whitelist.get("params", []) or [])

        # Parameters only (meta removed)
        cnt = 0
        for name, val in (params or {}).items():
            if name not in wl_params:
                rejected[name] = "not in reinforced whitelist"
                continue
            if name in self.defaults and self.defaults.get(name) == val:
                rejected[name] = "equals default; consider proposing alternative"
                continue
            if cnt >= int(max_edits):
                rejected[name] = "exceeds edit cap"
                continue
            applied[name] = val
            cnt += 1

        # No meta to apply; run_meta stays empty
        run_meta: Dict[str, Any] = {}
        return applied, rejected, run_meta

    def validate(self, params: Dict[str, Any], meta: Dict[str, Any], max_edits: int) -> Tuple[Dict[str, Any], Dict[str, str], Dict[str, Any]]:
        """Returns (applied, rejected, run_meta(empty)) with at most max_edits changes."""
        return self._validate_cli(params, max_edits)
