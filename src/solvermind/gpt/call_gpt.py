import os, json
from typing import Any, Dict, List


def call_gpt(input_messages: List[Dict[str, Any]], model: str = "gpt-5") -> Dict[str, Any]:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    resp = client.responses.create(model=model, input=input_messages)
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
    params = data.get("params", {})
    reasons = data.get("reasons", "")
    if not isinstance(params, dict):
        raise RuntimeError("Invalid schema from model (params must be an object)")
    return {"params": params, "reasons": reasons}


def call_gpt_json(input_messages: List[Dict[str, Any]], model: str = "gpt-5") -> Dict[str, Any]:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    resp = client.responses.create(model=model, input=input_messages)
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

