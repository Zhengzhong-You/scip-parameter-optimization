from __future__ import annotations

from .scip_cli import scip_version


def ensure_scip_version(required: str = "9.2.4") -> None:
    v = scip_version()
    if not v:
        raise RuntimeError("Unable to determine SCIP version; ensure scip is installed and on PATH.")
    if v != required:
        raise RuntimeError(f"SCIP CLI version mismatch: found {v}, required {required}.")
