from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

class EMAValidationError(ValueError):
    pass

def _target_order(
    previous_trust: Mapping[str, float],
    expected_agent_ids: Iterable[str] | None,
) -> tuple[str, ...]:
    if expected_agent_ids is None:
        return tuple(previous_trust)
    if isinstance(expected_agent_ids, (str, bytes)):
        raise EMAValidationError("expected_agent_ids must be an iterable of IDs, not a string")

    targets = tuple(expected_agent_ids)
    if any(
        not isinstance(agent_id, str)
        or not agent_id
        or agent_id != agent_id.strip()
        for agent_id in targets
    ):
        raise EMAValidationError("Every expected agent ID must be a non-empty string")
    if len(set(targets)) != len(targets):
        raise EMAValidationError("expected_agent_ids contains duplicate IDs")
    return targets

def _validate_mapping_keys(
    values: Mapping[str, float],
    expected: set[str],
    label: str,
) -> None:
    if not isinstance(values, Mapping):
        raise EMAValidationError(f"{label} must be a mapping")
    if any(not isinstance(key, str) or not key for key in values):
        raise EMAValidationError(f"Every {label} key must be a non-empty string agent ID")

    actual = set(values)
    if actual == expected:
        return
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    details: list[str] = []
    if missing:
        details.append(f"missing {missing}")
    if unexpected:
        details.append(f"unexpected {unexpected}")
    raise EMAValidationError(f"{label} target set mismatch: {'; '.join(details)}")

def _unit_interval_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EMAValidationError(f"{label} must be a number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise EMAValidationError(f"{label} must be finite") from exc
    if not math.isfinite(number):
        raise EMAValidationError(f"{label} must be finite")
    if not 0.0 <= number <= 1.0:
        raise EMAValidationError(f"{label} must be in [0, 1], got {number}")
    return number

def apply_ema(
    previous_trust: Mapping[str, float],
    episode_signals: Mapping[str, float],
    decay: float,
    *,
    expected_agent_ids: Iterable[str] | None = None,
) -> dict[str, float]:
    if not isinstance(previous_trust, Mapping):
        raise EMAValidationError("previous_trust must be a mapping")
    if not isinstance(episode_signals, Mapping):
        raise EMAValidationError("episode_signals must be a mapping")

    targets = _target_order(previous_trust, expected_agent_ids)
    expected = set(targets)
    _validate_mapping_keys(previous_trust, expected, "previous_trust")
    _validate_mapping_keys(episode_signals, expected, "episode_signals")
    decay_value = _unit_interval_number(decay, "decay")

    updated: dict[str, float] = {}
    update_weight = 1.0 - decay_value
    for agent_id in targets:
        old = _unit_interval_number(
            previous_trust[agent_id],
            f"previous_trust[{agent_id!r}]",
        )
        signal = _unit_interval_number(
            episode_signals[agent_id],
            f"episode_signals[{agent_id!r}]",
        )
        value = decay_value * old + update_weight * signal
        updated[agent_id] = max(0.0, min(1.0, value))
    return updated

__all__ = ["EMAValidationError", "apply_ema"]
