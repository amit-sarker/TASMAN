from __future__ import annotations

import hashlib
import json

from src.agents.prompt_adapter import SUPPLEMENT_PROMPT_PROTOCOL
from src.prompts import system_template
from src.prompts.feedback_prompt import FEEDBACK_RUBRIC
from src.prompts.prompt_builder import _decision_hint

def brs_prompt_fingerprint() -> str:
    payload = {
        "feedback_rubric": FEEDBACK_RUBRIC,
        "prompt_protocol": SUPPLEMENT_PROMPT_PROTOCOL,
        "system_template": {
            name: value for name, value in vars(system_template).items()
            if name.isupper() and isinstance(value, (str, dict))
        },
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

def build_system_trust_block(
    trust_lines: str,
    *,
    initial_score: float = 0.5,
    hint_level: str = "prescriptive",
    is_malicious: bool = False,
    min_blocking_rejects: int = 2,
) -> str:
    if hint_level not in {"scores_only", "prescriptive"}:
        raise ValueError("The supplement supports only scores_only (S) and prescriptive (S+P)")
    block = system_template.TRUST_VISIBILITY_BLOCK.format(
        trust_lines=trust_lines,
        initial_trust_score=initial_score,
    ).strip()
    if not is_malicious:
        hint = _decision_hint(hint_level).format(
            min_blocking_rejects=min_blocking_rejects,
        ).strip()
        if hint:
            block += "\n\n" + hint
    return block
