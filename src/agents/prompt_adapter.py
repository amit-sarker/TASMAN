from __future__ import annotations

from typing import Any

from src.prompts import prompt_builder as templates

SUPPLEMENT_PROMPT_PROTOCOL = "supplement_s_sp_strategic_v1"
TRUST_DECISION_HINT_LEVELS = frozenset({"scores_only", "prescriptive"})

def _validate_condition(level: str) -> None:
    if level not in TRUST_DECISION_HINT_LEVELS:
        raise ValueError(
            "The supplement supports only scores_only (S) and prescriptive (S+P)"
        )

def build_system_prompt(
    *, trust_decision_hint_level: str = "prescriptive", **kwargs: Any,
) -> str:
    _validate_condition(trust_decision_hint_level)
    return templates.build_system_prompt(
        trust_decision_hint_level=trust_decision_hint_level, **kwargs,
    )

def _round_template_level(level: str, is_malicious: bool) -> str:
    _validate_condition(level)
    return "prescriptive" if level == "prescriptive" and not is_malicious else "visible_only"

def build_proposer_user_message(
    *,
    trust_decision_hint_level: str = "prescriptive",
    is_malicious: bool = False,
    **kwargs: Any,
) -> str:
    return templates.build_proposer_user_message(
        trust_decision_hint_level=_round_template_level(
            trust_decision_hint_level, is_malicious,
        ),
        is_malicious=False,
        **kwargs,
    )

def build_voter_user_message(
    *,
    trust_decision_hint_level: str = "prescriptive",
    is_malicious: bool = False,
    **kwargs: Any,
) -> str:
    return templates.build_voter_user_message(
        trust_decision_hint_level=_round_template_level(
            trust_decision_hint_level, is_malicious,
        ),
        is_malicious=False,
        **kwargs,
    )
