from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from typing import Dict, Any, Tuple, Optional


def _scip_bin() -> str:
    """Get SCIP binary path, checking common installation locations."""
    # First check SCIP_BIN environment variable
    if "SCIP_BIN" in os.environ:
        return os.environ["SCIP_BIN"]

    # Check if 'scip' is in PATH
    scip_path = shutil.which("scip")
    if scip_path:
        return scip_path

    # Check common installation locations
    common_locations = [
        "/usr/local/bin/scip",
        "/usr/bin/scip",
        os.path.expanduser("~/miniconda3/bin/scip"),
        os.path.expanduser("~/anaconda3/bin/scip"),
    ]

    for location in common_locations:
        if os.path.isfile(location) and os.access(location, os.X_OK):
            return location

    # Fall back to 'scip' and let it fail with a clear error
    return "scip"


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
        scip_bin = _scip_bin()
        raise RuntimeError(
            f"Unable to determine SCIP version. Tried: {scip_bin}\n"
            f"Ensure SCIP {required} is installed. You can:\n"
            f"  - Run: python3 install.py (installs SCIP automatically)\n"
            f"  - Or install via conda: conda install -c conda-forge scip={required}\n"
            f"  - Or set SCIP_BIN environment variable to the SCIP binary path"
        )
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

