from __future__ import annotations

import os
import re
import subprocess
import tempfile
from typing import Dict, Any, Tuple, Optional


def _scip_bin() -> str:
    return os.environ.get("SCIP_BIN", "scip")


def run_scip_script(commands: str | list[str], env: Optional[dict[str, str]] = None) -> Tuple[int, str]:
    if isinstance(commands, list):
        script = "\n".join(commands) + "\n"
    else:
        script = str(commands)
        if not script.endswith("\n"):
            script += "\n"
    p = subprocess.run([
        _scip_bin()
    ], input=script.encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    out = p.stdout.decode("utf-8", errors="replace")
    return p.returncode, out


def scip_version() -> Optional[str]:
    try:
        p = subprocess.run([_scip_bin(), "--version"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out = p.stdout.decode("utf-8", errors="replace")
        m = re.search(r"\b(\d+\.\d+\.\d+)\b", out)
        return m.group(1) if m else None
    except FileNotFoundError:
        return None


def ensure_version(required: str = "9.2.4") -> None:
    v = scip_version()
    if not v:
        raise RuntimeError("Unable to determine SCIP version. Ensure SCIP is installed and on PATH (SCIP_BIN or scip).")
    if v != required:
        raise RuntimeError(f"SCIP CLI version mismatch: found {v}, required {required}.")


def get_default_params() -> Dict[str, Any]:
    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, "defaults.set")
        cmds = [
            "set default",
            f"set save {fp}",
            "quit",
        ]
        rc, out = run_scip_script(cmds)
        if rc != 0:
            raise RuntimeError(f"SCIP CLI failed while saving defaults: rc={rc}\n{out[:500]}")
        params: Dict[str, Any] = {}
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                name, val = line.split("=", 1)
                name = name.strip()
                val = val.strip()
                if val.lower() in ("true", "false"):
                    parsed: Any = (val.lower() == "true")
                else:
                    try:
                        if "." in val or "e" in val.lower():
                            parsed = float(val)
                        else:
                            parsed = int(val)
                    except Exception:
                        parsed = val
                params[name] = parsed
    return params

