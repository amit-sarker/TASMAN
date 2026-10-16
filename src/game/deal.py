from __future__ import annotations

import json
import re

from src.utils.types import Deal, Issue, ProposerOutput, TrustOutput, VoterOutput

class ProposerParseError(ValueError):
    pass

class VoterParseError(ValueError):
    pass

class TrustParseError(ValueError):
    pass

def format_deal(deal: Deal, issue_order: list[str]) -> str:
    return ", ".join(deal.choices[issue_id] for issue_id in issue_order)

def _extract_json(text: str) -> dict:
    text = text.strip()

    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        obj_match = re.search(r"\{[\s\S]*\}", text)
        if obj_match:
            return json.loads(obj_match.group())
        raise

def parse_deal_string(s: str, issues: list[Issue]) -> Deal:
    option_to_issue: dict[str, str] = {}
    for issue in issues:
        for option in issue.options:
            option_to_issue[option.id] = issue.id

    parts = [part.strip() for part in s.split(",") if part.strip()]
    if len(parts) != len(issues):
        raise ValueError(
            f"Deal must contain exactly {len(issues)} options, got {len(parts)}"
        )

    seen_issues: set[str] = set()
    choices: dict[str, str] = {}
    for part in parts:
        if part not in option_to_issue:
            raise ValueError(f"Illegal option id in deal: '{part}'")
        issue_id = option_to_issue[part]
        if issue_id in seen_issues:
            raise ValueError(f"Deal contains multiple options for issue '{issue_id}'")
        seen_issues.add(issue_id)
        choices[issue_id] = part

    missing = [issue.id for issue in issues if issue.id not in choices]
    if missing:
        raise ValueError(f"Deal is missing issues: {missing}")
    return Deal(choices=choices)

def _parse_deal_from_dict(d: dict, issues: list[Issue]) -> Deal:
    valid_options: dict[str, set[str]] = {
        issue.id: {opt.id for opt in issue.options} for issue in issues
    }
    choices: dict[str, str] = {}
    for issue in issues:
        if issue.id not in d:
            raise ValueError(f"Deal is missing issue '{issue.id}'")
        option_id = str(d[issue.id])
        if option_id not in valid_options[issue.id]:
            raise ValueError(
                f"Invalid option '{option_id}' for issue '{issue.id}'; "
                f"valid options: {sorted(valid_options[issue.id])}"
            )
        choices[issue.id] = option_id
    return Deal(choices=choices)

def build_proposer_schema(issues: list[Issue]) -> dict:
    deal_properties = {
        issue.id: {"type": "string", "enum": [opt.id for opt in issue.options]}
        for issue in issues
    }
    return {
        "type": "object",
        "properties": {
            "scratchpad": {"type": "string"},
            "public_answer": {"type": "string"},
            "deal": {
                "type": "object",
                "properties": deal_properties,
                "required": [issue.id for issue in issues],
                "additionalProperties": False,
            },
        },
        "required": ["scratchpad", "public_answer", "deal"],
        "additionalProperties": False,
    }

def build_voter_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "scratchpad": {"type": "string"},
            "public_answer": {"type": "string"},
            "vote": {"type": "boolean"},
        },
        "required": ["scratchpad", "public_answer", "vote"],
        "additionalProperties": False,
    }

def build_trust_schema(agent_ids: list[str]) -> dict:
    trust_properties = {agent_id: {"type": "number"} for agent_id in agent_ids}
    scratchpad_properties = {agent_id: {"type": "string"} for agent_id in agent_ids}

    weight_keys = [
        "vote_inconsistency",
        "counterfactual_blame",
        "outlier_rejection",
        "targeted_failure",
        "cooperation",
    ]
    return {
        "type": "object",
        "properties": {
            "scratchpad": {
                "type": "object",
                "properties": scratchpad_properties,
                "required": agent_ids,
                "additionalProperties": False,
            },
            "metrics_weights": {
                "type": "object",
                "properties": {k: {"type": "number"} for k in weight_keys},
                "required": weight_keys,
                "additionalProperties": False,
            },
            "weights_rationale": {"type": "string"},
            "trust_scores": {
                "type": "object",
                "properties": trust_properties,
                "required": agent_ids,
                "additionalProperties": False,
            },
        },
        "required": ["scratchpad", "metrics_weights", "weights_rationale", "trust_scores"],
        "additionalProperties": False,
    }

def parse_proposer_output(text: str, issues: list[Issue]) -> ProposerOutput:
    try:
        data = _extract_json(text)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ProposerParseError(f"Response is not valid JSON: {exc}") from exc

    for field in ("scratchpad", "public_answer", "deal"):
        if field not in data:
            raise ProposerParseError(f"Missing required field: '{field}'")

    if not isinstance(data["deal"], dict):
        raise ProposerParseError("'deal' must be a JSON object")

    try:
        deal = _parse_deal_from_dict(data["deal"], issues)
    except ValueError as exc:
        raise ProposerParseError(str(exc)) from exc

    public_answer = str(data["public_answer"]).strip()
    if not public_answer:
        raise ProposerParseError("'public_answer' must not be empty")

    return ProposerOutput(
        scratchpad=str(data["scratchpad"]),
        public_answer=public_answer,
        deal=deal,
    )

def parse_voter_output(text: str) -> VoterOutput:
    try:
        data = _extract_json(text)
    except (json.JSONDecodeError, ValueError) as exc:
        raise VoterParseError(f"Response is not valid JSON: {exc}") from exc

    for field in ("scratchpad", "public_answer", "vote"):
        if field not in data:
            raise VoterParseError(f"Missing required field: '{field}'")

    raw_vote = data["vote"]
    if isinstance(raw_vote, bool):
        vote = raw_vote
    elif isinstance(raw_vote, str):
        normalized = raw_vote.strip().lower()
        if normalized not in {"true", "false", "accept", "reject"}:
            raise VoterParseError(f"'vote' must be a boolean or 'accept'/'reject', got '{raw_vote}'")
        vote = normalized in {"true", "accept"}
    else:
        raise VoterParseError(f"'vote' must be a boolean, got {type(raw_vote).__name__}")

    public_answer = str(data["public_answer"]).strip()
    if not public_answer:
        raise VoterParseError("'public_answer' must not be empty")

    return VoterOutput(
        scratchpad=str(data["scratchpad"]),
        public_answer=public_answer,
        vote=vote,
        reason=public_answer,
    )

def parse_trust_output(text: str, expected_agent_ids: list[str]) -> TrustOutput:
    try:
        data = _extract_json(text)
    except (json.JSONDecodeError, ValueError) as exc:
        raise TrustParseError(f"Response is not valid JSON: {exc}") from exc

    for field in ("scratchpad", "trust_scores"):
        if field not in data:
            raise TrustParseError(f"Missing required field: '{field}'")

    if not isinstance(data["trust_scores"], dict):
        raise TrustParseError("'trust_scores' must be a JSON object")

    raw_scratchpad = data["scratchpad"]
    scratchpad_by_agent: dict[str, str] = {}
    if isinstance(raw_scratchpad, dict):
        scratchpad_by_agent = {
            str(agent_id): str(reasoning)
            for agent_id, reasoning in raw_scratchpad.items()
        }
        scratchpad_str = "\n".join(
            f"{agent_id}: {reasoning}"
            for agent_id, reasoning in scratchpad_by_agent.items()
        )
    else:
        scratchpad_str = str(raw_scratchpad)

    rationale = data.get("weights_rationale")
    if isinstance(rationale, str) and rationale.strip():
        scratchpad_str = f"weights_rationale: {rationale.strip()}\n{scratchpad_str}"

    scores: dict[str, float] = {}
    for agent_id in expected_agent_ids:
        if agent_id not in data["trust_scores"]:
            raise TrustParseError(f"Missing trust score for agent '{agent_id}'")
        try:
            scores[agent_id] = max(0.0, min(1.0, float(data["trust_scores"][agent_id])))
        except (TypeError, ValueError) as exc:
            raise TrustParseError(
                f"Trust score for '{agent_id}' is not numeric: {exc}"
            ) from exc

    weights: dict[str, float] = {}
    raw_weights = data.get("metrics_weights")
    if isinstance(raw_weights, dict):
        for key, val in raw_weights.items():
            try:
                weights[str(key)] = max(0.0, min(1.0, float(val)))
            except (TypeError, ValueError):
                continue

    return TrustOutput(
        scratchpad=scratchpad_str,
        scratchpad_by_agent=scratchpad_by_agent,
        trust_scores=scores,
        metrics_weights=weights,
    )

def redact_score_leaks(
    public_answer: str,
    agent_scores: dict[str, dict[str, int]],
    threshold: int,
) -> tuple[str, bool]:
    sensitive_numbers = {
        str(value)
        for issue_scores in agent_scores.values()
        for value in issue_scores.values()
        if abs(value) >= 10
    }
    sensitive_numbers.add(str(threshold))
    sensitive_numbers = {value for value in sensitive_numbers if value}

    if not sensitive_numbers:
        return public_answer, False

    pattern = re.compile(
        r"(?<![A-Za-z])("
        + "|".join(sorted(map(re.escape, sensitive_numbers), key=len, reverse=True))
        + r")(?![A-Za-z])"
    )
    redacted, count = pattern.subn("[REDACTED]", public_answer)
    return redacted, count > 0
