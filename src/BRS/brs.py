from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .models import AccumulatedEvidence, FeedbackPair, ObservationCounts, finite_number

def validate_agent_ids(agent_ids: Iterable[str]) -> list[str]:
    if isinstance(agent_ids, (str, bytes)):
        raise ValueError("agent_ids must be a collection of IDs")
    ids = list(agent_ids)
    if not ids or any(not isinstance(value, str) or not value.strip() for value in ids):
        raise ValueError("agent IDs must be nonempty strings")
    if len(ids) != len(set(ids)):
        raise ValueError("agent IDs must be unique")
    return ids

def initialize_state(agent_ids: Iterable[str]) -> dict[str, AccumulatedEvidence]:
    return {agent_id: AccumulatedEvidence() for agent_id in validate_agent_ids(agent_ids)}

def normalize_counts(
    counts: Mapping[str, ObservationCounts | dict[str, int]],
) -> dict[str, FeedbackPair]:
    if not isinstance(counts, Mapping):
        raise ValueError("observation counts must be a mapping")
    ids = validate_agent_ids(counts)
    validated = {agent_id: ObservationCounts.model_validate(counts[agent_id]) for agent_id in ids}
    result: dict[str, FeedbackPair] = {}
    for agent_id, pair in validated.items():
        total = pair.P + pair.N
        result[agent_id] = (
            FeedbackPair(r=pair.P / total, s=pair.N / total)
            if total else FeedbackPair(r=0, s=0)
        )
    return result

def _validated_state(state: Mapping[str, Any]) -> dict[str, AccumulatedEvidence]:
    if not isinstance(state, Mapping):
        raise ValueError("state must be a mapping")
    ids = validate_agent_ids(state)
    return {agent_id: AccumulatedEvidence.model_validate(state[agent_id]) for agent_id in ids}

def update_brs(
    state: Mapping[str, AccumulatedEvidence | dict[str, float]],
    feedback: Mapping[str, FeedbackPair | dict[str, float]],
    forgetting_factor: float = 0.9,
) -> dict[str, AccumulatedEvidence]:
    previous = _validated_state(state)
    factor = finite_number(forgetting_factor)
    if not 0 <= factor <= 1:
        raise ValueError("forgetting_factor must be in [0,1]")
    if not isinstance(feedback, Mapping) or set(feedback) != set(previous):
        raise ValueError("feedback must contain exactly the state agent IDs")
    observations = {key: FeedbackPair.model_validate(feedback[key]) for key in previous}
    return {
        key: AccumulatedEvidence(
            R=factor * old.R + observations[key].r,
            S=factor * old.S + observations[key].s,
        )
        for key, old in previous.items()
    }

def scores_from_state(state: Mapping[str, AccumulatedEvidence | dict[str, float]]) -> dict[str, float]:
    return {key: (1 + value.R) / (2 + value.R + value.S) for key, value in _validated_state(state).items()}

def initialize_observer_state(agent_ids: Iterable[str]) -> dict[str, dict[str, AccumulatedEvidence]]:
    ids = validate_agent_ids(agent_ids)
    if len(ids) < 2:
        raise ValueError("observer state requires at least two agents")
    return {observer: initialize_state(target for target in ids if target != observer) for observer in ids}

def deserialize_observer_state(state: Mapping[str, Any]) -> dict[str, dict[str, AccumulatedEvidence]]:
    if not isinstance(state, Mapping):
        raise ValueError("observer state must be a mapping")
    ids = validate_agent_ids(state)
    if len(ids) < 2:
        raise ValueError("observer state requires at least two agents")
    result: dict[str, dict[str, AccumulatedEvidence]] = {}
    for observer in ids:
        row = _validated_state(state[observer])
        if set(row) != set(ids) - {observer}:
            raise ValueError("each observer row must contain exactly its counterparts")
        result[observer] = row
    return result

def serialize_observer_state(state: Mapping[str, Any]) -> dict[str, dict[str, dict[str, float]]]:
    return {
        observer: {target: evidence.model_dump(mode="json") for target, evidence in row.items()}
        for observer, row in deserialize_observer_state(state).items()
    }

def scores_from_observer_state(state: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    return {observer: scores_from_state(row) for observer, row in deserialize_observer_state(state).items()}
