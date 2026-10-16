from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from src.game.deal import TrustParseError, _extract_json
from src.utils.types import RoundRecord

METRIC_KEYS_V4: tuple[str, ...] = (
    "vote_inconsistency",
    "counterfactual_blame",
    "outlier_rejection",
    "targeted_failure",
    "cooperation",
)

OBSERVABLE_METRIC_KEYS_V4: tuple[str, ...] = (
    "counterfactual_blame",
    "outlier_rejection",
    "targeted_failure",
    "cooperation",
)

V4_TRUST_UPDATE_PROTOCOL = "v4_exact_v3_trust_blocks_v3"

class TrustSignalParseErrorV4(TrustParseError):
    pass

@dataclass(frozen=True)
class TrustSignalAssessmentV4:
    scratchpad: str
    episode_signals: dict[str, float]
    trust_scores: dict[str, float]
    metrics_weights: dict[str, float]
    weights_rationale: str
    metric_values: dict[str, dict[str, float]] = field(default_factory=dict)
    criterion_values: dict[str, dict[str, float]] = field(default_factory=dict)

def _validated_agent_map(expected_agents: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(expected_agents, Mapping):
        raise TypeError("expected_agents must map internal IDs to display names")
    if not expected_agents:
        raise ValueError("expected_agents must not be empty")

    normalized: dict[str, str] = {}
    display_names: set[str] = set()
    for agent_id, display_name in expected_agents.items():
        if not isinstance(agent_id, str) or not agent_id.strip():
            raise ValueError("Every expected agent ID must be a non-empty string")
        if agent_id != agent_id.strip():
            raise ValueError(f"Agent ID must not have surrounding whitespace: {agent_id!r}")
        if not isinstance(display_name, str) or not display_name.strip():
            raise ValueError(
                f"Display name for agent '{agent_id}' must be a non-empty string"
            )
        clean_display_name = display_name.strip()
        if clean_display_name in display_names:
            raise ValueError(f"Duplicate display name: {clean_display_name!r}")
        normalized[agent_id] = clean_display_name
        display_names.add(clean_display_name)
    return normalized

def _per_agent_schema(
    expected_agents: Mapping[str, str],
    value_schema_factory,
) -> dict:
    properties = {
        agent_id: value_schema_factory(agent_id, display_name)
        for agent_id, display_name in expected_agents.items()
    }
    return {
        "type": "object",
        "properties": properties,
        "required": list(expected_agents),
        "additionalProperties": False,
    }

def _scratchpad_schema(expected_agents: Mapping[str, str]) -> dict:
    return _per_agent_schema(
        expected_agents,
        lambda agent_id, display_name: {
            "type": "string",
            "minLength": 1,
            "description": (
                f"Private evidence assessment for {display_name} "
                f"(internal ID: {agent_id})."
            ),
        },
    )

def _unit_interval_per_agent_schema(
    expected_agents: Mapping[str, str],
    description: str,
) -> dict:
    return _per_agent_schema(
        expected_agents,
        lambda agent_id, display_name: {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": description.format(
                agent_id=agent_id,
                display_name=display_name,
            ),
        },
    )

def _metrics_weights_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            key: {"type": "number", "minimum": 0.0, "maximum": 1.0}
            for key in METRIC_KEYS_V4
        },
        "required": list(METRIC_KEYS_V4),
        "additionalProperties": False,
        "description": "Model-chosen shared metric weights; values must sum to 1.0.",
    }

def build_metrics_trust_signal_schema_v4(
    expected_agents: Mapping[str, str],
) -> dict:
    agents = _validated_agent_map(expected_agents)
    return {
        "type": "object",
        "properties": {
            "scratchpad": _scratchpad_schema(agents),
            "metrics_weights": _metrics_weights_schema(),
            "weights_rationale": {"type": "string", "minLength": 1},
            "trust_scores": _unit_interval_per_agent_schema(
                agents,
                "Model-calculated EMA-updated trust score for {display_name} "
                "(internal ID: {agent_id}).",
            ),
        },
        "required": [
            "scratchpad",
            "metrics_weights",
            "weights_rationale",
            "trust_scores",
        ],
        "additionalProperties": False,
    }

def build_freeform_trust_signal_schema_v4(
    expected_agents: Mapping[str, str],
) -> dict:
    agents = _validated_agent_map(expected_agents)
    return {
        "type": "object",
        "properties": {
            "scratchpad": {"type": "string", "minLength": 1},
            "episode_signals": _unit_interval_per_agent_schema(
                agents,
                "Model-calculated current-episode trust signal for {display_name} "
                "(internal ID: {agent_id}).",
            ),
            "metrics_weights": {
                "type": "object",
                "minProperties": 1,
                "additionalProperties": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
            },
            "trust_scores": _unit_interval_per_agent_schema(
                agents,
                "Model-calculated EMA-updated trust score for {display_name} "
                "(internal ID: {agent_id}).",
            ),
        },
        "required": [
            "scratchpad",
            "episode_signals",
            "metrics_weights",
            "trust_scores",
        ],
        "additionalProperties": False,
    }

def _extract_object(text: str) -> dict:
    try:
        data = _extract_json(text)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise TrustSignalParseErrorV4(f"Response is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise TrustSignalParseErrorV4("Response must be a JSON object")
    return data

def _require_exact_keys(
    data: Mapping[str, object],
    expected_keys: set[str],
    location: str,
    expected_agents: Mapping[str, str] | None = None,
) -> None:
    actual_keys = set(data)
    if actual_keys == expected_keys:
        return

    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    details: list[str] = []
    if missing:
        details.append(f"missing {missing}")
    if unexpected:
        details.append(f"unexpected {unexpected}")
    if expected_agents is not None:
        mapping = ", ".join(
            f"{agent_id}={display_name}"
            for agent_id, display_name in expected_agents.items()
        )
        details.append(f"expected internal-ID mapping [{mapping}]")
    raise TrustSignalParseErrorV4(f"{location} has invalid keys: {'; '.join(details)}")

def _require_object(value: object, location: str) -> dict:
    if not isinstance(value, dict):
        raise TrustSignalParseErrorV4(f"'{location}' must be a JSON object")
    return value

def _unit_interval_number(value: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrustSignalParseErrorV4(f"'{location}' must be a JSON number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise TrustSignalParseErrorV4(f"'{location}' must be finite") from exc
    if not math.isfinite(number):
        raise TrustSignalParseErrorV4(f"'{location}' must be finite")
    if not 0.0 <= number <= 1.0:
        raise TrustSignalParseErrorV4(f"'{location}' must be in [0, 1], got {number}")
    return number

def compute_observable_metric_values_v4(
    history: Sequence[RoundRecord],
    target_agent_ids: Sequence[str],
    *,
    min_blocking_rejects: int,
    min_sample_rounds: int,
) -> dict[str, dict[str, float]]:
    if min_blocking_rejects < 1:
        raise ValueError("min_blocking_rejects must be positive")
    if min_sample_rounds < 1:
        raise ValueError("min_sample_rounds must be positive")
    targets = tuple(target_agent_ids)
    if len(set(targets)) != len(targets) or any(
        not isinstance(agent_id, str) or not agent_id.strip()
        for agent_id in targets
    ):
        raise ValueError("target_agent_ids must contain unique non-empty strings")

    result: dict[str, dict[str, float]] = {}
    for agent_id in targets:
        voter_rounds: list[tuple[RoundRecord, bool]] = []
        for round_record in history:
            if round_record.proposer_id == agent_id:
                continue
            vote = next(
                (
                    bool(item.accept)
                    for item in round_record.votes
                    if item.agent_id == agent_id
                ),
                None,
            )
            if vote is not None:
                voter_rounds.append((round_record, vote))

        decisive_voter_rounds = [
            (round_record, vote)
            for round_record, vote in voter_rounds
            if sum(not item.accept for item in round_record.votes)
            == min_blocking_rejects
        ]
        counterfactual_blame = (
            sum(not vote for _, vote in decisive_voter_rounds)
            / len(decisive_voter_rounds)
            if decisive_voter_rounds
            else 0.0
        )

        proposals = [
            round_record
            for round_record in history
            if round_record.proposer_id == agent_id
        ]
        proposal_accept_rates = [
            sum(item.accept for item in round_record.votes)
            / len(round_record.votes)
            for round_record in proposals
            if round_record.votes
        ]
        cooperation = (
            math.fsum(proposal_accept_rates) / len(proposal_accept_rates)
            if proposal_accept_rates
            else 0.5
        )

        outlier_rejection = (
            sum(
                (not vote)
                and sum(
                    not item.accept
                    for item in round_record.votes
                    if item.agent_id != agent_id
                )
                < min_blocking_rejects
                for round_record, vote in voter_rounds
            )
            / len(voter_rounds)
            if len(voter_rounds) >= min_sample_rounds
            else 0.5
        )

        targeted_failure = (
            sum(
                sum(not item.accept for item in round_record.votes)
                == min_blocking_rejects
                for round_record in proposals
            )
            / len(proposals)
            if proposals
            else 0.0
        )

        result[agent_id] = {
            "counterfactual_blame": float(counterfactual_blame),
            "outlier_rejection": float(outlier_rejection),
            "targeted_failure": float(targeted_failure),
            "cooperation": float(cooperation),
        }
    return result

def _parse_scratchpad(
    value: object,
    expected_agents: Mapping[str, str],
) -> dict[str, str]:
    raw = _require_object(value, "scratchpad")
    _require_exact_keys(
        raw,
        set(expected_agents),
        "scratchpad",
        expected_agents,
    )
    parsed: dict[str, str] = {}
    for agent_id in expected_agents:
        reasoning = raw[agent_id]
        if not isinstance(reasoning, str):
            raise TrustSignalParseErrorV4(
                f"'scratchpad.{agent_id}' must be a JSON string"
            )
        if not reasoning.strip():
            raise TrustSignalParseErrorV4(
                f"'scratchpad.{agent_id}' must not be empty"
            )
        parsed[agent_id] = reasoning.strip()
    return parsed

def _parse_metric_weights(value: object) -> dict[str, float]:
    raw = _require_object(value, "metrics_weights")
    _require_exact_keys(raw, set(METRIC_KEYS_V4), "metrics_weights")
    return {
        key: _unit_interval_number(raw[key], f"metrics_weights.{key}")
        for key in METRIC_KEYS_V4
    }

def _parse_rationale(value: object) -> str:
    if not isinstance(value, str):
        raise TrustSignalParseErrorV4("'weights_rationale' must be a JSON string")
    if not value.strip():
        raise TrustSignalParseErrorV4("'weights_rationale' must not be empty")
    return value.strip()

def _parse_freeform_scratchpad(value: object) -> str:
    if not isinstance(value, str):
        raise TrustSignalParseErrorV4("'scratchpad' must be a JSON string")
    if not value.strip():
        raise TrustSignalParseErrorV4("'scratchpad' must not be empty")
    return value.strip()

def _parse_freeform_weights(value: object) -> dict[str, float]:
    raw = _require_object(value, "metrics_weights")
    if not raw:
        raise TrustSignalParseErrorV4("'metrics_weights' must not be empty")
    parsed: dict[str, float] = {}
    for criterion, weight in raw.items():
        if not isinstance(criterion, str) or not criterion.strip():
            raise TrustSignalParseErrorV4(
                "Every metrics_weights criterion must be a non-empty string"
            )
        if criterion != criterion.strip():
            raise TrustSignalParseErrorV4(
                f"Criterion name must not have surrounding whitespace: {criterion!r}"
            )
        parsed[criterion] = _unit_interval_number(
            weight,
            f"metrics_weights.{criterion}",
        )
    return parsed

def _parse_per_agent_numbers(
    value: object,
    expected_agents: Mapping[str, str],
    location: str,
) -> dict[str, float]:
    raw = _require_object(value, location)
    _require_exact_keys(
        raw,
        set(expected_agents),
        location,
        expected_agents,
    )
    return {
        agent_id: _unit_interval_number(
            raw[agent_id],
            f"{location}.{agent_id}",
        )
        for agent_id in expected_agents
    }

def parse_metrics_trust_signal_output_v4(
    text: str,
    expected_agents: Mapping[str, str],
) -> TrustSignalAssessmentV4:
    agents = _validated_agent_map(expected_agents)
    data = _extract_object(text)
    _require_exact_keys(
        data,
        {"scratchpad", "metrics_weights", "weights_rationale", "trust_scores"},
        "top-level response",
    )
    scratchpad_by_agent = _parse_scratchpad(data["scratchpad"], agents)
    return TrustSignalAssessmentV4(
        scratchpad="\n".join(
            f"{agent_id}: {scratchpad_by_agent[agent_id]}"
            for agent_id in agents
        ),
        episode_signals={},
        trust_scores=_parse_per_agent_numbers(
            data["trust_scores"], agents, "trust_scores"
        ),
        metrics_weights=_parse_metric_weights(data["metrics_weights"]),
        weights_rationale=_parse_rationale(data["weights_rationale"]),
    )

def parse_freeform_trust_signal_output_v4(
    text: str,
    expected_agents: Mapping[str, str],
) -> TrustSignalAssessmentV4:
    agents = _validated_agent_map(expected_agents)
    data = _extract_object(text)
    _require_exact_keys(
        data,
        {"scratchpad", "episode_signals", "metrics_weights", "trust_scores"},
        "top-level response",
    )
    return TrustSignalAssessmentV4(
        scratchpad=_parse_freeform_scratchpad(data["scratchpad"]),
        episode_signals=_parse_per_agent_numbers(
            data["episode_signals"], agents, "episode_signals"
        ),
        trust_scores=_parse_per_agent_numbers(
            data["trust_scores"], agents, "trust_scores"
        ),
        metrics_weights=_parse_freeform_weights(data["metrics_weights"]),
        weights_rationale="",
    )

__all__ = [
    "METRIC_KEYS_V4",
    "OBSERVABLE_METRIC_KEYS_V4",
    "V4_TRUST_UPDATE_PROTOCOL",
    "TrustSignalAssessmentV4",
    "TrustSignalParseErrorV4",
    "build_freeform_trust_signal_schema_v4",
    "build_metrics_trust_signal_schema_v4",
    "compute_observable_metric_values_v4",
    "parse_freeform_trust_signal_output_v4",
    "parse_metrics_trust_signal_output_v4",
]
