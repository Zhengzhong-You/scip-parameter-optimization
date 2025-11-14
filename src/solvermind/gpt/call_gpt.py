
import os, json
from typing import Any, Dict, List

DEFAULT_API_PARAMS = {
    "max_completion_tokens": 20000,
    # "temperature": 0,
}

# Fixed seed for reproducibility
DEFAULT_SEED = 42

def get_api_params() -> dict:
    """Return the current API parameters used for reproducibility."""
    return DEFAULT_API_PARAMS.copy()

def _make_client():
    from openai import OpenAI
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def call_gpt(input_messages: List[Dict[str, Any]], model: str = "gpt-4.1-mini") -> Dict[str, Any]:
    """
    Call the OpenAI Chat Completions API and return a parsed JSON object with keys:
      { 'params': {...}, 'meta': {...}, 'reasons': '...' }.
    """
    client = _make_client()

    resp = client.chat.completions.create(
        model=model,
        messages=input_messages,
        response_format={"type": "json_object"},
        seed=DEFAULT_SEED,           # Use seed for reproducibility
        **DEFAULT_API_PARAMS,
    )

    text = resp.choices[0].message.content

    try:
        data = json.loads(text)
    except Exception as e:
        raise RuntimeError(f"Model did not return valid JSON: {e}\nRaw: {text[:500]}")

    params = data.get("params", {})
    meta = data.get("meta", {})
    reasons = data.get("reasons", "")
    if not isinstance(params, dict):
        raise RuntimeError("Invalid schema from model (params must be an object)")
    return {"params": params, "meta": meta, "reasons": reasons}
