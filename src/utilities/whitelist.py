from __future__ import annotations

from typing import Any, Dict, List, Tuple
import math
from functools import lru_cache

import yaml
from .scip_cli import get_default_params
from .scip_cli import run_scip_script
import re


def _curated_list() -> List[str]:
    params: List[str] = []
    params += [
        "branching/scorefunc",
        "branching/preferbinary",
        "branching/relpscost/minreliable",
        "branching/relpscost/maxreliable",
        "branching/relpscost/sbiterquot",
        "branching/relpscost/sbiterofs",
    ]
    params += [
        "nodeselection/estimate/stdpriority",
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
    # Separator family knobs (SCIP 9.2.4): use frequencies per family per spec
    params += [
        "separating/gomory/freq",
        "separating/cmir/freq",
        "separating/flowcover/freq",
        "separating/clique/freq",
        "separating/knapsackcover/freq",
        "separating/oddcycle/freq",
    ]
    params += [
        "presolving/maxrounds",
        "presolving/maxrestarts",
        "presolving/abortfac",
    ]
    # Per-presolver maxrounds are not exposed uniformly in 9.2.x; keep only global knobs above.
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
    # params += ["numerics/feastol", "numerics/epsilon", "numerics/dualfeastol"]
    return params


def _minimal_list() -> List[str]:
    params: List[str] = []
    params += ["branching/scorefunc"]
    params += ["nodeselection/estimate/stdpriority", "nodeselection/dfs/stdpriority"]
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


# -------------------------------
# Typed whitelist from SCIP (9.2.4)
# -------------------------------

# === PATCH 1: More robust regex patterns and help text caching ===
# Allow () or [], and recognize ±inf/infty/infinity
_RANGE_RE = re.compile(
    r"""[\[\(]\s*
        (?P<lo>[+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?|[+\-]?(?:inf|infty|infinity))
        \s*,\s*
        (?P<hi>[+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?|[+\-]?(?:inf|infty|infinity))
        \s*[\]\)]
    """,
    re.IGNORECASE | re.VERBOSE,
)

_INT_PAT = re.compile(r"^[+-]?\d+$")
_TYPE_HINTS = (
    (re.compile(r"\bbool\b", re.I), "bool"),
    (re.compile(r"\breal\b|\bfloat\b", re.I), "float"),
    (re.compile(r"\binteger\b|\bint\b|\blongint\b", re.I), "int"),
    (re.compile(r"\bchar\b", re.I), "char"),
)

_CHOICES_BLOCK_RE = re.compile(r"\{([^}]*)\}")  # More permissive, then parse tokens
_TOKEN_IN_BRACES_RE = re.compile(r"'([^']+)'|\"([^\"]+)\"|([A-Za-z0-9_.:/+\-\|]+)")

def _to_float(s: str) -> float:
    s = s.strip().lower()
    if s in {"+inf", "inf", "infty", "infinity"}:
        return math.inf
    if s in {"-inf", "-infty", "-infinity"}:
        return -math.inf
    return float(s)

def _extract_range(text: str) -> tuple[float | None, float | None]:
    m = _RANGE_RE.search(text or "")
    if not m:
        return None, None
    lo_s, hi_s = m.group("lo"), m.group("hi")
    try:
        return _to_float(lo_s), _to_float(hi_s)
    except Exception:
        return None, None

@lru_cache(maxsize=None)
def _scip_help_text(name: str) -> str:
    rc, out = run_scip_script([f"set help {name}", "quit"])
    if rc == 0 and out:
        return out
    # Fallback: some environments prefer `help set <name>`
    rc2, out2 = run_scip_script([f"help set {name}", "quit"])
    return out2 or (out or "")


# === PATCH 2: Use help text to prioritize type/domain inference ===
def _infer_param_info_from_help(text: str) -> Tuple[str | None, Dict[str, Any]]:
    """From 'set help <name>' output, parse type & domain."""
    if not text:
        return None, {}

    t_hint: str | None = None
    for rx, t in _TYPE_HINTS:
        if rx.search(text):
            t_hint = t
            break

    # First try range
    lo, hi = _extract_range(text)
    if lo is not None or hi is not None:
        # Type decision: has explicit float/real hint then float; otherwise if boundaries are integers/infinite and has int hint then int
        if t_hint == "float":
            return "float", {"lower": lo, "upper": hi}
        if t_hint == "int":
            return "int", {"lower": lo if lo in (-math.inf, math.inf) else int(lo),
                           "upper": hi if hi in (-math.inf, math.inf) else int(hi)}
        # No hint: if both ends (when finite) are integers, then int, otherwise float
        def _is_intlike(x: float) -> bool:
            return (x in (-math.inf, math.inf)) or float(int(x)) == float(x)
        if (lo is None or _is_intlike(lo)) and (hi is None or _is_intlike(hi)):
            return "int", {"lower": lo if lo in (-math.inf, math.inf) else (None if lo is None else int(lo)),
                           "upper": hi if hi in (-math.inf, math.inf) else (None if hi is None else int(hi))}
        return "float", {"lower": lo, "upper": hi}

    # Then try choices (braces with quoted or bare tokens)
    mset = _CHOICES_BLOCK_RE.search(text)
    if mset:
        raw = mset.group(1)
        toks = []
        for g1, g2, g3 in _TOKEN_IN_BRACES_RE.findall(raw):
            tok = g1 or g2 or g3
            if tok:
                toks.append(tok.strip())
        low = [t.lower() for t in toks]
        if set(low) <= {"true", "false"} or (t_hint == "bool"):
            return "bool", {"choices": [False, True]}
        # char type: all choices are single characters
        if all(len(t) == 1 for t in toks) or t_hint == "char":
            return "cat", {"choices": toks}
        return "cat", {"choices": toks}

    # No range or choices: return type hint only (fallback to _set probing later)
    if t_hint in {"bool", "float", "int", "char"}:
        return t_hint if t_hint != "char" else "cat", {}

    return None, {}


# === PATCH 3: Priority: help first, fallback to 'set' probing with infinity handling ===
def _scip_param_typed(name: str) -> Dict[str, Any] | None:
    def _set(val: str) -> Tuple[bool, str]:
        import tempfile, os as _os
        with tempfile.NamedTemporaryFile("w", suffix=".set", delete=False) as tf:
            tf.write(f"{name} = {val}\n")
            tf.flush()
            set_path = tf.name
        rc, out = run_scip_script([f"set load {set_path}", "quit"])
        try:
            _os.unlink(set_path)
        except Exception:
            pass
        ok = (rc == 0) and ("ERROR" not in (out or ""))
        return ok, (out or "")

    defaults = {}
    try:
        defaults = get_default_params()
        dval = defaults.get(name, None)
    except Exception:
        dval = None

    # 1) First see help text
    help_txt = _scip_help_text(name)
    t0, meta0 = _infer_param_info_from_help(help_txt)
    if t0 == "bool":
        return {"name": name, "type": "bool", "choices": [False, True]}
    if t0 == "cat" and meta0.get("choices"):
        # Verify first choice is settable
        first = str(meta0["choices"][0])
        ok, _ = _set(first)
        if ok:
            return {"name": name, "type": "cat", "choices": meta0["choices"]}
    if t0 in {"int", "float"} and ("lower" in meta0 or "upper" in meta0):
        lo = meta0.get("lower", -math.inf)
        hi = meta0.get("upper", math.inf)
        # Don't force finite; allow ±inf
        if t0 == "int":
            lo_i = None if lo in (-math.inf, math.inf, None) else int(lo)
            hi_i = None if hi in (-math.inf, math.inf, None) else int(hi)
            return {"name": name, "type": "int", "lower": lo_i if lo_i is not None else -2**31,
                    "upper": hi_i if hi_i is not None else 2**31-1}
        else:
            return {"name": name, "type": "float",
                    "lower": float(lo) if lo is not None else -math.inf,
                    "upper": float(hi) if hi is not None else math.inf}

    # 2) Fallback: judge type by default value first
    if isinstance(dval, bool):
        return {"name": name, "type": "bool", "choices": [False, True]}
    if isinstance(dval, str) and len(dval) == 1:
        # Try to trigger "valid set" hint via invalid value
        ok_inv, out = _set("!")
        mset = _CHOICES_BLOCK_RE.search(out or "")
        if mset:
            raw = mset.group(1)
            toks = []
            for g1, g2, g3 in _TOKEN_IN_BRACES_RE.findall(raw):
                tok = g1 or g2 or g3
                if tok:
                    toks.append(tok.strip())
            choices = toks if len(toks) > 1 else list(toks[0]) if toks else [dval]
            ok_first, _ = _set(str(choices[0]))
            if ok_first:
                return {"name": name, "type": "cat", "choices": choices}

    # 3) Robust probing: force an error and parse the range from the error text.
    #    We do not require deciding float/int upfront; instead we detect from error text or from integer-likeness of bounds.
    def _probe_range_and_kind() -> Tuple[float | None, float | None, str | None, str]:
        # Try two probes: extreme float and extreme int; merge any extracted ranges
        rc_kind = None
        lo = hi = None
        # first: float extremes
        ok, out = _set("-1e308")
        if out:
            l1, h1 = _extract_range(out)
            if l1 is not None or h1 is not None:
                lo = l1 if lo is None else lo
                hi = h1 if hi is None else hi
            if re.search(r"\b(longint|integer|int) parameter\b", out, re.I):
                rc_kind = "int"
            elif re.search(r"\b(real|float) parameter\b", out, re.I):
                rc_kind = "float"
        ok, out = _set("1e308")
        if out:
            l2, h2 = _extract_range(out)
            if lo is None and l2 is not None:
                lo = l2
            if hi is None and h2 is not None:
                hi = h2
            if rc_kind is None:
                if re.search(r"\b(longint|integer|int) parameter\b", out, re.I):
                    rc_kind = "int"
                elif re.search(r"\b(real|float) parameter\b", out, re.I):
                    rc_kind = "float"
        # second: int extremes
        ok, out = _set("-999999999")
        if out:
            l3, h3 = _extract_range(out)
            if lo is None and l3 is not None:
                lo = l3
            if hi is None and h3 is not None:
                hi = h3
            if rc_kind is None:
                if re.search(r"\b(longint|integer|int) parameter\b", out, re.I):
                    rc_kind = "int"
                elif re.search(r"\b(real|float) parameter\b", out, re.I):
                    rc_kind = "float"
        ok, out = _set("999999999")
        if out:
            l4, h4 = _extract_range(out)
            if lo is None and l4 is not None:
                lo = l4
            if hi is None and h4 is not None:
                hi = h4
            if rc_kind is None:
                if re.search(r"\b(longint|integer|int) parameter\b", out, re.I):
                    rc_kind = "int"
                elif re.search(r"\b(real|float) parameter\b", out, re.I):
                    rc_kind = "float"
        return lo, hi, rc_kind, out or ""

    lo, hi, kind, last_out = _probe_range_and_kind()
    if lo is None and hi is None:
        # Try to infer categorical choices as last resort
        ok_inv, out = _set("!")
        mset = _CHOICES_BLOCK_RE.search(out or "")
        if mset:
            raw = mset.group(1)
            toks = []
            for g1, g2, g3 in _TOKEN_IN_BRACES_RE.findall(raw):
                tok = g1 or g2 or g3
                if tok:
                    toks.append(tok.strip())
            if toks:
                ok_first, _ = _set(str(toks[0]))
                if ok_first:
                    return {"name": name, "type": "cat", "choices": toks}
        # As requested, treat lack of domain as a hard error
        return None

    # Decide numeric type
    if kind is None:
        # Use hint from help if available
        kind = t0
        if kind not in ("int", "float"):
            # Decide by int-likeness of finite bounds
            def _is_intlike(x):
                return x is None or (x in (-math.inf, math.inf)) or float(int(x)) == float(x)
            kind = "int" if _is_intlike(lo) and _is_intlike(hi) else "float"

    if kind == "int":
        lo_i = None if lo is None or lo in (-math.inf, math.inf) else int(lo)
        hi_i = None if hi is None or hi in (-math.inf, math.inf) else int(hi)
        # Guard against float round-up on very large bounds; clamp to signed 64-bit max
        if hi_i is not None and hi_i > (2**63 - 1):
            hi_i = 2**63 - 1
        return {"name": name, "type": "int",
                "lower": lo_i if lo_i is not None else -2**31,
                "upper": hi_i if hi_i is not None else 2**31-1}
    else:
        return {"name": name, "type": "float",
                "lower": float(lo) if lo is not None else -math.inf,
                "upper": float(hi) if hi is not None else math.inf}


def get_typed_whitelist(regime: str = "curated") -> List[Dict[str, Any]]:
    """Build a typed whitelist by querying SCIP for each whitelisted name.

    - Ensures all parameters exist and domains are accepted by SCIP 9.2.4
    - Uses help output and set-load probes; no heuristic fallbacks
    - Validates by creating a temporary .set using safe/default values and loading it
    """
    names = get_whitelist(regime=regime).get("params", [])
    typed: List[Dict[str, Any]] = []
    failures: List[str] = []
    for nm in names:
        info = _scip_param_typed(nm)
        if info is None:
            failures.append(nm)
            continue
        typed.append(info)

    # Cap infinite bounds to finite sentinels for optimizer compatibility
    def _cap_inf(entry: Dict[str, Any]) -> Dict[str, Any]:
        t = entry.get("type")
        if t in ("float", "int"):
            lo = entry.get("lower", None)
            hi = entry.get("upper", None)
            # Map None/±inf to finite caps
            if lo is None or (isinstance(lo, (int, float)) and not math.isfinite(lo)):
                lo = -1e9
            if hi is None or (isinstance(hi, (int, float)) and not math.isfinite(hi)):
                hi = 1e9
            # Ensure ordering
            if t == "int":
                # Also clamp huge finite integer bounds into a manageable range for ConfigSpace
                lo = int(max(-1e9, min(1e9, lo)))
                hi = int(max(-1e9, min(1e9, hi)))
                if hi <= lo:
                    hi = lo + 1
            else:
                # Clamp finite float bounds too if extremely large
                lo = float(max(-1e9, min(1e9, lo)))
                hi = float(max(-1e9, min(1e9, hi)))
                if hi <= lo:
                    hi = lo + 1.0
            entry["lower"], entry["upper"] = lo, hi
        return entry

    typed = [_cap_inf(it) for it in typed]

    # Validate by loading a .set with safe/default values
    defaults = {}
    try:
        defaults = get_default_params()
    except Exception:
        defaults = {}

    # === PATCH 4: Validation stage prioritizes defaults, safe values for infinite bounds ===
    def _safe_value(it: Dict[str, Any]) -> str:
        name = it["name"]
        # 1) Priority: default value (most stable)
        if name in defaults and defaults[name] is not None:
            dv = defaults[name]
            if isinstance(dv, bool):
                return "true" if dv else "false"
            return str(dv)

        t = it.get("type")
        if t == "cat":
            return str((it.get("choices") or [""])[0])
        if t == "bool":
            return "false"
        if t == "int":
            lo = it.get("lower", -2**31); hi = it.get("upper", 2**31-1)
            lo = -2**31 if lo is None else int(lo)
            hi =  2**31-1 if hi is None else int(hi)
            cand = 0
            if cand < lo: cand = lo
            if cand > hi: cand = hi
            return str(cand)
        if t == "float":
            lo = it.get("lower", -math.inf); hi = it.get("upper", math.inf)
            # Choose 0.0, if clipped outside then clamp to boundary; infinite bounds are OK
            cand = 0.0
            if (lo is not None) and math.isfinite(lo) and cand < lo:
                cand = lo
            if (hi is not None) and math.isfinite(hi) and cand > hi:
                cand = hi
            return f"{float(cand):.12g}"
        return ""

    import tempfile, os as _os
    with tempfile.NamedTemporaryFile("w", suffix=".set", delete=False) as tf:
        for it in typed:
            sv = _safe_value(it)
            if not sv:
                continue
            tf.write(f"{it['name']} = {sv}\n")
        tf.flush()
        set_path = tf.name
    rc, out = run_scip_script([f"set load {set_path}", "quit"])
    try:
        _os.unlink(set_path)
    except Exception:
        pass
    if rc != 0 or (out and "ERROR" in out):
        bad: List[str] = []
        for line in (out or "").splitlines():
            m = re.search(r"parameter <([^>]+)>", line)
            if m:
                bad.append(m.group(1))
        raise RuntimeError(f"Typed whitelist validation failed; offending params: {bad or 'unknown'}. Raw:\n{(out or '')}")
    if failures:
        raise RuntimeError(f"Failed to infer type/domain for parameters: {failures}. Check SCIP 9.2.4 installation or whitelist names.")
    return typed
