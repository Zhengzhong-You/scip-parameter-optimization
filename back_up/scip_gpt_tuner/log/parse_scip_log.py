import re
from typing import Dict, Any, List, Optional
from collections import deque

def shrink_scip_log_for_gpt(log_text: str, max_length: int = 1500) -> str:
    """Extract key SCIP log information for GPT analysis.

    Adapted from the log_processor.py to extract:
    - Original problem size
    - Presolving summary
    - Key progress rows
    - Final statistics
    """
    if not log_text:
        return ""

    # Parse the log into structured components
    parsed = parse_scip_log_lines(log_text.splitlines())

    # Build GPT-friendly summary
    out = []

    # Original problem info
    if parsed["original_problem"]:
        out.append("PROBLEM SIZE:")
        out.append(parsed["original_problem"])
        out.append("")

    # Presolving info (first block only to keep concise)
    if parsed["presolve_blocks"]:
        out.append("PRESOLVING:")
        # Take first presolve block and limit lines
        presolve_lines = parsed["presolve_blocks"][0][:8]  # Max 8 lines
        out.extend(presolve_lines)
        out.append("")

    # Progress table (last few rows from final table)
    if parsed["progress_blocks"]:
        last_block = parsed["progress_blocks"][-1]
        out.append("SOLVING PROGRESS:")
        out.append(last_block["header"])
        # Show last few rows
        rows = list(last_block["rows"])[-3:]  # Last 3 rows
        out.extend(rows)
        out.append("")

    # Restarts (if any, show count)
    if parsed["restarts"]:
        out.append(f"RESTARTS: {len(parsed['restarts'])} restart(s)")
        out.append("")

    # Final statistics
    if parsed["summary_lines"]:
        out.append("FINAL RESULTS:")
        out.extend(parsed["summary_lines"])

    result = "\n".join(out)

    # Truncate if too long
    if len(result) > max_length:
        result = result[:max_length-15] + "\n...[truncated]"

    return result


def parse_scip_log_lines(lines: List[str]) -> Dict[str, Any]:
    """Parse SCIP log into structured format."""
    # Regex patterns
    HEADER_RE = re.compile(r'^\s*time\s*\|\s*node\s*\|\s*left\s*\|', re.I)
    TIMEROW_RE = re.compile(r'^\s*(?:\*|d)?\d+(?:\.\d+)?s\|')
    ORIGINAL_RE = re.compile(r'original problem has .*? and \d+\s+constraints', re.I)
    PRESOLVE_SUMMARY_RE = re.compile(r'^\s*presolving\s*\(', re.I)
    PRESOLVE_TIME_RE = re.compile(r'^\s*Presolving Time\s*:', re.I)
    RESTART_RE = re.compile(r'(Restart triggered.*|performing user restart|\(restart\).*)', re.I)

    SUMMARY_KEYS = [
        re.compile(r'^\s*SCIP Status\s*:', re.I),
        re.compile(r'^\s*Solving Time\s*\(sec\)\s*:', re.I),
        re.compile(r'^\s*Solving Nodes\s*:', re.I),
        re.compile(r'^\s*Primal Bound\s*:', re.I),
        re.compile(r'^\s*Dual Bound\s*:', re.I),
        re.compile(r'^\s*Gap\s*:', re.I),
    ]

    res = {
        "original_problem": None,
        "presolve_blocks": [],
        "progress_blocks": [],
        "restarts": [],
        "summary_lines": []
    }

    # States
    in_presolve = False
    curr_presolve = None
    curr_prog = None

    def end_presolve():
        nonlocal in_presolve, curr_presolve
        if in_presolve and curr_presolve:
            res["presolve_blocks"].append(curr_presolve)
        in_presolve = False
        curr_presolve = None

    def end_progress():
        nonlocal curr_prog
        if curr_prog and curr_prog["rows"]:
            res["progress_blocks"].append({
                "header": curr_prog["header"],
                "rows": list(curr_prog["rows"])
            })
        curr_prog = None

    for line in lines:
        line = line.rstrip()

        # Original problem
        if res["original_problem"] is None and ORIGINAL_RE.search(line):
            res["original_problem"] = line.strip()

        # Restarts
        if RESTART_RE.search(line):
            res["restarts"].append(line.strip())

        # Presolve start
        if PRESOLVE_SUMMARY_RE.match(line):
            end_presolve()
            in_presolve = True
            curr_presolve = [line]
            continue

        # Presolve content
        if in_presolve and curr_presolve is not None:
            curr_presolve.append(line)
            if PRESOLVE_TIME_RE.match(line):
                end_presolve()
                continue

        # Progress header
        if HEADER_RE.match(line):
            end_progress()
            curr_prog = {"header": line, "rows": deque(maxlen=5)}
            continue

        # Progress rows
        if curr_prog is not None and TIMEROW_RE.match(line):
            curr_prog["rows"].append(line)

        # Summary lines
        if any(pat.match(line) for pat in SUMMARY_KEYS):
            res["summary_lines"].append(line.strip())

    # Cleanup
    end_presolve()
    end_progress()

    return res