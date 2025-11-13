
import os, json
from typing import Any, Dict, List

# Default API parameters for reproducibility
DEFAULT_API_PARAMS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 1000000000
}

def get_api_params() -> dict:
    """Return the current API parameters used for reproducibility."""
    return DEFAULT_API_PARAMS.copy()

def call_gpt(input_messages: List[Dict[str, Any]], model: str = "gpt-5-nano") -> Dict[str, Any]:
    """Call the OpenAI *Responses API* and return a parsed JSON object with keys:
    { 'params': {...}, 'meta': {...}, 'reasons': '...' }.

    Uses DEFAULT_API_PARAMS for reproducibility.
    Requires env var OPENAI_API_KEY to be set.
    """
    # Lazy import to keep package import-time light
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Ask for structured JSON back (Responses API doesn't use response_format)
    # Add reproducibility parameters for consistent results
    resp = client.responses.create(
        model=model,
        input=input_messages,
        **DEFAULT_API_PARAMS
    )

    # Prefer the convenience accessor if present, otherwise drill into output
    try:
        text = resp.output_text
    except Exception:
        # Fallback path
        items = getattr(resp, "output", []) or []
        if items and hasattr(items[0], "content") and items[0].content:
            text = items[0].content[0].text
        else:
            raise RuntimeError("Unexpected Responses API payload; cannot find text content")

    try:
        data = json.loads(text)
    except Exception as e:
        raise RuntimeError(f"Model did not return valid JSON: {e}\nRaw: {text[:500]}")

    params = data.get("params", {})
    meta = data.get("meta", {})  # optional; ignored by pipeline
    reasons = data.get("reasons", "")
    if not isinstance(params, dict):
        raise RuntimeError("Invalid schema from model (params must be an object)")
    return {"params": params, "meta": meta, "reasons": reasons}


def call_gpt_json(input_messages: List[Dict[str, Any]], model: str = "gpt-5-nano") -> Dict[str, Any]:
    """General JSON caller for the OpenAI Responses API.

    Uses DEFAULT_API_PARAMS for reproducibility.
    Returns the parsed JSON object as-is (no schema enforcement).
    """
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Add reproducibility parameters for consistent results
    resp = client.responses.create(
        model=model,
        input=input_messages,
        **DEFAULT_API_PARAMS
    )

    try:
        text = resp.output_text
    except Exception:
        items = getattr(resp, "output", []) or []
        if items and hasattr(items[0], "content") and items[0].content:
            text = items[0].content[0].text
        else:
            raise RuntimeError("Unexpected Responses API payload; cannot find text content")

    try:
        data = json.loads(text)
    except Exception as e:
        raise RuntimeError(f"Model did not return valid JSON: {e}\nRaw: {text[:500]}")

    if not isinstance(data, dict):
        raise RuntimeError("Expected a JSON object response")
    return data
