from __future__ import annotations

import json
import math

from src.game.deal import (
    TrustParseError,
    _extract_json,
)
from src.utils.types import TrustOutput

class FreeformTrustParseError(TrustParseError):
    pass

def build_freeform_trust_schema(
    agent_ids: list[str],
    *,
    structured_scratchpad: bool = False,
) -> dict:
    per_agent_properties = {agent_id: {"type": "number"} for agent_id in agent_ids}
    scratchpad_schema: dict = {"type": "string"}
    if structured_scratchpad:
        scratchpad_schema = {
            "type": "object",
            "properties": {
                agent_id: {"type": "string", "minLength": 1}
                for agent_id in agent_ids
            },
            "required": agent_ids,
            "additionalProperties": False,
        }
    return {
        "type": "object",
        "properties": {
            "scratchpad": scratchpad_schema,
            "episode_signals": {
                "type": "object",
                "properties": per_agent_properties,
                "required": agent_ids,
                "additionalProperties": False,
            },
            "metrics_weights": {
                "type": "object",
                "additionalProperties": {"type": "number"},
            },
            "trust_scores": {
                "type": "object",
                "properties": per_agent_properties,
                "required": agent_ids,
                "additionalProperties": False,
            },
        },
        "required": [
            "scratchpad",
            "episode_signals",
            "metrics_weights",
            "trust_scores",
        ],
        "additionalProperties": False,
    }

def parse_freeform_trust_output(
    text: str,
    expected_agent_ids: list[str],
    *,
    structured_scratchpad: bool = False,
    strict_contract: bool = False,
) -> TrustOutput:
    if strict_contract:
        def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
            result: dict[str, object] = {}
            for key, value in pairs:
                if key in result:
                    raise FreeformTrustParseError(
                        f"Duplicate JSON key is not allowed: {key!r}"
                    )
                result[key] = value
            return result

        def reject_nonfinite_constant(value: str) -> None:
            raise FreeformTrustParseError(
                f"Non-finite JSON number is not allowed: {value}"
            )

        try:
            data = json.loads(
                text.strip(),
                object_pairs_hook=reject_duplicate_keys,
                parse_constant=reject_nonfinite_constant,
            )
        except FreeformTrustParseError:
            raise
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise FreeformTrustParseError(
                f"Response is not exactly one valid JSON object: {exc}"
            ) from exc
        if not isinstance(data, dict):
            raise FreeformTrustParseError("Response must be a JSON object")
    else:
        try:
            data = _extract_json(text)
        except (json.JSONDecodeError, ValueError) as exc:
            raise FreeformTrustParseError(
                f"Response is not valid JSON: {exc}"
            ) from exc

    required_fields = {
        "scratchpad",
        "episode_signals",
        "metrics_weights",
        "trust_scores",
    }
    if strict_contract and set(data) != required_fields:
        missing = sorted(required_fields - set(data))
        unexpected = sorted(set(data) - required_fields)
        details: list[str] = []
        if missing:
            details.append(f"missing {missing}")
        if unexpected:
            details.append(f"unexpected {unexpected}")
        raise FreeformTrustParseError(
            "Response must contain exactly the four required fields: "
            + "; ".join(details)
        )

    for field in required_fields:
        if field not in data:
            raise FreeformTrustParseError(f"Missing required field: '{field}'")

    if not isinstance(data["episode_signals"], dict):
        raise FreeformTrustParseError("'episode_signals' must be a JSON object")
    if not isinstance(data["metrics_weights"], dict):
        raise FreeformTrustParseError("'metrics_weights' must be a JSON object")
    if not isinstance(data["trust_scores"], dict):
        raise FreeformTrustParseError("'trust_scores' must be a JSON object")

    scratchpad_by_agent: dict[str, str] = {}
    if structured_scratchpad:
        raw_scratchpad = data["scratchpad"]
        if not isinstance(raw_scratchpad, dict):
            raise FreeformTrustParseError("'scratchpad' must be a JSON object")
        expected = set(expected_agent_ids)
        actual = set(raw_scratchpad)
        if actual != expected:
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            details: list[str] = []
            if missing:
                details.append(f"missing {missing}")
            if unexpected:
                details.append(f"unexpected {unexpected}")
            raise FreeformTrustParseError(
                "'scratchpad' must contain exactly the target agent IDs: "
                + "; ".join(details)
            )
        for agent_id in expected_agent_ids:
            reasoning = raw_scratchpad[agent_id]
            if not isinstance(reasoning, str):
                raise FreeformTrustParseError(
                    f"scratchpad for '{agent_id}' must be a string"
                )
            reasoning = reasoning.strip()
            if not reasoning:
                raise FreeformTrustParseError(
                    f"scratchpad for '{agent_id}' must not be empty"
                )
            scratchpad_by_agent[agent_id] = reasoning
        scratchpad = "\n".join(
            f"{agent_id}: {scratchpad_by_agent[agent_id]}"
            for agent_id in expected_agent_ids
        )
    else:
        if strict_contract and not isinstance(data["scratchpad"], str):
            raise FreeformTrustParseError("'scratchpad' must be a string")
        scratchpad = str(data["scratchpad"])

    expected_ids = set(expected_agent_ids)
    if strict_contract:
        for field in ("episode_signals", "trust_scores"):
            actual_ids = set(data[field])
            if actual_ids != expected_ids:
                missing = sorted(expected_ids - actual_ids)
                unexpected = sorted(actual_ids - expected_ids)
                details = []
                if missing:
                    details.append(f"missing {missing}")
                if unexpected:
                    details.append(f"unexpected {unexpected}")
                raise FreeformTrustParseError(
                    f"'{field}' must contain exactly the target agent IDs: "
                    + "; ".join(details)
                )

    def strict_unit_interval(value: object, location: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise FreeformTrustParseError(
                f"{location} must be a JSON number in [0, 1]"
            )
        number = float(value)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise FreeformTrustParseError(
                f"{location} must be a finite number in [0, 1]"
            )
        return number

    signals: dict[str, float] = {}
    for agent_id in expected_agent_ids:
        if agent_id not in data["episode_signals"]:
            raise FreeformTrustParseError(
                f"Missing episode_signal for agent '{agent_id}'"
            )
        if strict_contract:
            signals[agent_id] = strict_unit_interval(
                data["episode_signals"][agent_id],
                f"episode_signal for '{agent_id}'",
            )
        else:
            try:
                signals[agent_id] = max(
                    0.0,
                    min(1.0, float(data["episode_signals"][agent_id])),
                )
            except (TypeError, ValueError) as exc:
                raise FreeformTrustParseError(
                    f"episode_signal for '{agent_id}' is not numeric: {exc}"
                ) from exc

    weights: dict[str, float] = {}
    if strict_contract and not data["metrics_weights"]:
        raise FreeformTrustParseError(
            "'metrics_weights' must contain at least one criterion"
        )
    for key, val in data["metrics_weights"].items():
        if strict_contract:
            if not isinstance(key, str) or not key.strip():
                raise FreeformTrustParseError(
                    "Every metrics_weights criterion must be a non-empty string"
                )
            weights[key] = strict_unit_interval(
                val,
                f"metrics_weight for '{key}'",
            )
        else:
            try:
                weights[str(key)] = max(0.0, min(1.0, float(val)))
            except (TypeError, ValueError) as exc:
                raise FreeformTrustParseError(
                    f"metrics_weight for '{key}' is not numeric: {exc}"
                ) from exc

    scores: dict[str, float] = {}
    for agent_id in expected_agent_ids:
        if agent_id not in data["trust_scores"]:
            raise FreeformTrustParseError(
                f"Missing trust score for agent '{agent_id}'"
            )
        if strict_contract:
            scores[agent_id] = strict_unit_interval(
                data["trust_scores"][agent_id],
                f"Trust score for '{agent_id}'",
            )
        else:
            try:
                scores[agent_id] = max(
                    0.0,
                    min(1.0, float(data["trust_scores"][agent_id])),
                )
            except (TypeError, ValueError) as exc:
                raise FreeformTrustParseError(
                    f"Trust score for '{agent_id}' is not numeric: {exc}"
                ) from exc

    return TrustOutput(
        scratchpad=scratchpad,
        scratchpad_by_agent=scratchpad_by_agent,
        episode_signals=signals,
        metrics_weights=weights,
        trust_scores=scores,
    )
