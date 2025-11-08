from __future__ import annotations

import os
import re
import shlex
import subprocess
import tempfile
from typing import Dict, Any, Tuple, Optional


def _scip_bin() -> str:
    return os.environ.get("SCIP_BIN", "scip")


def run_scip_script(commands: str | list[str], env: Optional[dict[str, str]] = None) -> Tuple[int, str]:
    """Run SCIP shell with commands provided via stdin, return (rc, stdout+stderr)."""
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
    """Return SCIP version string from `scip --version` output, or None."""
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
    """Fetch all parameters and their default values via CLI.

    Implementation: reset to defaults, then `set save <file>` and parse the resulting .set file.
    """
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
        try:
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
                    # Cast to int/float/bool when obvious
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
        except FileNotFoundError:
            # Some distros write to current working dir instead of absolute tmp
            raise
    return params


_ORIG_LINE_RE = re.compile(
    r"original problem has\s+(?P<nvars>\d+)\s+variables.*?and\s+(?P<nconss>\d+)\s+constraints",
    re.IGNORECASE,
)


def extract_basic_features(instance_path: str, presolve: bool = True) -> Dict[str, Any]:
    """Extract lightweight static features via SCIP CLI without PySCIPOpt.

    Features:
    - nvars, nconss (always when available)
    - nbin, nint, ncont (best effort if present in the original-line format)
    - presolved counts (best effort if a 'transformed problem has ...' line appears)
    """
    with tempfile.TemporaryDirectory() as td:
        stats_fp = os.path.join(td, "stats.txt")
        cmds: list[str] = [
            f"read {instance_path}",
        ]
        if presolve:
            cmds += [
                "set limits nodes 0",
                "optimize",
            ]
        cmds += [
            f"write statistics {stats_fp}",
            "quit",
        ]
        rc, out = run_scip_script(cmds)
        if rc != 0:
            # Return minimal structure on failure
            return {"error": f"scip_cli_failed_rc_{rc}"}

        feats: Dict[str, Any] = {}
        # Parse original problem line
        for line in out.splitlines():
            m = _ORIG_LINE_RE.search(line)
            if m and ("nvars" not in feats or "nconss" not in feats):
                try:
                    feats["nvars"] = int(m.group("nvars"))
                    feats["nconss"] = int(m.group("nconss"))
                except Exception:
                    pass
                # Try to extract types from the same line if present
                try:
                    b = re.search(r"(\d+)\s+bin", line, re.I)
                    if b:
                        feats["nbin"] = int(b.group(1))
                except Exception:
                    pass
                try:
                    ii = re.search(r"(\d+)\s+int", line, re.I)
                    if ii:
                        feats["nint"] = int(ii.group(1))
                except Exception:
                    pass
                try:
                    cc = re.search(r"(\d+)\s+cont", line, re.I)
                    if cc:
                        feats["ncont"] = int(cc.group(1))
                except Exception:
                    pass

        # Attempt to parse transformed/presolved counts
        TRANS_RE = re.compile(
            r"(transformed|presolved) problem has\s+(?P<nvars>\d+)\s+variables.*?and\s+(?P<nconss>\d+)\s+constraints",
            re.I,
        )
        for line in out.splitlines():
            mt = TRANS_RE.search(line)
            if mt:
                try:
                    feats["presolved_nvars"] = int(mt.group("nvars"))
                    feats["presolved_nconss"] = int(mt.group("nconss"))
                except Exception:
                    pass
                break

        # Include a tag that features came from CLI
        feats["source"] = "scip_cli"
        return feats
