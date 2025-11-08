from __future__ import annotations

from typing import Dict, Any, List
from ..gpt.build_tuning_prompt import build_tuning_prompt as _base_build


def build_prompt(features_batch: Dict[str, Any], history: List[Dict[str, Any]], whitelist: Dict[str, Any], max_changes: int) -> list:
    features_payload = [{"instance": k, "features": v} for k, v in features_batch.items()]
    return _base_build(
        features={"batch": features_payload},
        history=history,
        whitelist=whitelist,
        max_changes=max_changes,
    )

