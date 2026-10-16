from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

EMA_ABSOLUTE_TOLERANCE = 0.005000001

METRIC_ABSOLUTE_TOLERANCE = 0.010000001

METRIC_WEIGHT_KEYS: tuple[str, ...] = (
    "vote_inconsistency",
    "counterfactual_blame",
    "outlier_rejection",
    "targeted_failure",
    "cooperation",
)

_TRACE_COMPONENTS: tuple[str, ...] = ("b", "C", "O", "T", "F", "S")
_NUMBER_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"

def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None

def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, value))

def _different(left: float, right: float, tolerance: float) -> bool:
    return abs(left - right) > tolerance

def _split_flattened_scratchpad(
    scratchpad: str,
    target_agent_ids: Sequence[str],
) -> dict[str, str]:
    target_ids = [str(agent_id) for agent_id in target_agent_ids]
    if not target_ids:
        return {}
    alternatives = "|".join(
        re.escape(agent_id) for agent_id in sorted(target_ids, key=len, reverse=True)
    )
    block_pattern = re.compile(
        rf"(?ms)^({alternatives}):\s*(.*?)"
        rf"(?=^(?:{alternatives}):\s*|\Z)"
    )
    return {
        match.group(1): match.group(2).strip()
        for match in block_pattern.finditer(str(scratchpad))
    }

def parse_metrics_scratchpad(
    scratchpad: Mapping[str, object] | str | None,
    target_agent_ids: Sequence[str],
) -> tuple[dict[str, dict[str, float]], list[str]]:
    target_ids = [str(agent_id) for agent_id in target_agent_ids]
    if isinstance(scratchpad, Mapping):
        per_target = {
            str(agent_id): str(reasoning)
            for agent_id, reasoning in scratchpad.items()
        }
    elif isinstance(scratchpad, str):
        per_target = _split_flattened_scratchpad(scratchpad, target_ids)
    else:
        per_target = {}

    parsed: dict[str, dict[str, float]] = {}
    unparseable: list[str] = []
    for agent_id in target_ids:
        text = per_target.get(agent_id)
        if not text:
            unparseable.append(agent_id)
            continue
        trace: dict[str, float] = {}
        for component in _TRACE_COMPONENTS:
            match = re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(component)}\s*=\s*"
                rf"({_NUMBER_PATTERN})",
                text,
            )
            if match is None:
                break
            try:
                value = float(match.group(1))
            except (OverflowError, ValueError):
                break
            if not math.isfinite(value):
                break
            trace[component] = value
        if len(trace) != len(_TRACE_COMPONENTS):
            unparseable.append(agent_id)
            continue
        parsed[agent_id] = trace
    return parsed, unparseable

def audit_ema_calculation(
    *,
    trust_mode: str,
    trust_scores_prompted: Mapping[str, object],
    trust_scores_after: Mapping[str, object],
    decay: float,
    target_agent_ids: Sequence[str],
    episode_signals: Mapping[str, object] | None = None,
    metrics_scratchpad: Mapping[str, object] | str | None = None,
) -> dict[str, Any] | None:
    decay_value = _number(decay)
    if decay_value is None or not 0.0 <= decay_value <= 1.0:
        raise ValueError("decay must be a finite number in [0, 1]")

    update_weight = round(1.0 - decay_value, 4)
    target_ids = [str(agent_id) for agent_id in target_agent_ids]

    signals: dict[str, tuple[float, str]] = {}
    if trust_mode == "freeform":
        for agent_id in target_ids:
            signal = _number((episode_signals or {}).get(agent_id))
            if signal is not None:
                signals[agent_id] = (signal, "episode_signal_E")
    elif trust_mode == "metrics":
        traces, _ = parse_metrics_scratchpad(metrics_scratchpad, target_ids)
        for agent_id, trace in traces.items():
            signals[agent_id] = (1.0 - trace["S"], "one_minus_reported_S")
    else:
        raise ValueError("trust_mode must be 'metrics' or 'freeform'")

    inaccurate: dict[str, dict[str, Any]] = {}
    for agent_id in target_ids:
        old = _number(trust_scores_prompted.get(agent_id))
        reported = _number(trust_scores_after.get(agent_id))
        signal_and_source = signals.get(agent_id)
        if old is None or reported is None or signal_and_source is None:
            continue
        signal, signal_source = signal_and_source
        expected_unclamped = decay_value * old + update_weight * signal
        expected = _clamp_unit(expected_unclamped)
        absolute_error = abs(reported - expected)
        if absolute_error <= EMA_ABSOLUTE_TOLERANCE:
            continue
        inaccurate[agent_id] = {
            "trust_old_prompted": old,
            "signal": signal,
            "signal_source": signal_source,
            "expected_unclamped": expected_unclamped,
            "expected_clamped": expected,
            "reported_trust": reported,
            "absolute_error": absolute_error,
        }

    if not inaccurate:
        return None
    return {
        "calculation": "ema",
        "trust_mode": trust_mode,
        "decay": decay_value,
        "update_weight": update_weight,
        "absolute_tolerance": EMA_ABSOLUTE_TOLERANCE,
        "inaccurate_targets": inaccurate,
    }

def audit_metrics_calculation(
    *,
    target_agent_ids: Sequence[str],
    metrics_scratchpad: Mapping[str, object] | str | None,
    metrics_weights: Mapping[str, object],
) -> dict[str, Any] | None:
    target_ids = [str(agent_id) for agent_id in target_agent_ids]
    traces, unparseable = parse_metrics_scratchpad(metrics_scratchpad, target_ids)

    parsed_weights: dict[str, float] = {}
    for key in METRIC_WEIGHT_KEYS:
        value = _number(metrics_weights.get(key))
        if value is not None:
            parsed_weights[key] = value
    formula_available = len(parsed_weights) == len(METRIC_WEIGHT_KEYS)

    target_issues: dict[str, dict[str, Any]] = {}
    for agent_id in unparseable:
        target_issues[agent_id] = {
            "issues": [
                {
                    "issue": "unverifiable_calculation_trace",
                    "reason": "missing one or more of b, C, O, T, F, or S",
                }
            ],
        }

    for agent_id, trace in traces.items():
        if not formula_available:
            target_issues[agent_id] = {
                "issues": [
                    {
                        "issue": "unverifiable_calculation_trace",
                        "reason": "one or more formula weights are missing or nonnumeric",
                    }
                ],
            }
            continue

        recomputed_s = (
            parsed_weights["vote_inconsistency"] * trace["F"]
            + parsed_weights["counterfactual_blame"] * trace["b"]
            + parsed_weights["outlier_rejection"] * trace["O"]
            + parsed_weights["targeted_failure"] * trace["T"]
            + parsed_weights["cooperation"] * (1.0 - trace["C"])
        )
        if _different(trace["S"], recomputed_s, METRIC_ABSOLUTE_TOLERANCE):
            target_issues[agent_id] = {
                "issues": [
                    {
                        "issue": "weighted_behavior_score",
                        "reported": trace["S"],
                        "expected": recomputed_s,
                        "absolute_error": abs(trace["S"] - recomputed_s),
                    }
                ],
                "reported_inputs": {
                    component: trace[component]
                    for component in ("b", "C", "O", "T", "F")
                },
                "reported_behavior_score": trace["S"],
                "recomputed_behavior_score": recomputed_s,
            }

    if not target_issues:
        return None
    return {
        "calculation": "metrics",
        "equation": "S=w_F*F+w_b*b+w_O*O+w_T*T+w_C*(1-C)",
        "metric_absolute_tolerance": METRIC_ABSOLUTE_TOLERANCE,
        "reported_weights": {
            str(key): value for key, value in metrics_weights.items()
        },
        "target_issues": target_issues,
        "input_policy": {
            "verification": "reported_values_used_as_given",
            "not_checked": ["b", "C", "O", "T", "F", "weight_sum"],
        },
    }

def audit_freeform_ema(
    *,
    trust_scores_prompted: Mapping[str, object],
    trust_scores_after: Mapping[str, object],
    episode_signals: Mapping[str, object],
    decay: float,
    target_agent_ids: Sequence[str] | None = None,
) -> dict[str, Any] | None:
    targets = list(target_agent_ids or trust_scores_after.keys())
    return audit_ema_calculation(
        trust_mode="freeform",
        trust_scores_prompted=trust_scores_prompted,
        trust_scores_after=trust_scores_after,
        decay=decay,
        target_agent_ids=targets,
        episode_signals=episode_signals,
    )

def audit_metrics_update(
    *,
    target_agent_ids: Sequence[str],
    metrics_scratchpad: Mapping[str, object] | str | None,
    metrics_weights: Mapping[str, object],
    trust_scores_prompted: Mapping[str, object],
    trust_scores_after: Mapping[str, object],
    decay: float,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    targets = list(target_agent_ids)
    metrics_payload = audit_metrics_calculation(
        target_agent_ids=targets,
        metrics_scratchpad=metrics_scratchpad,
        metrics_weights=metrics_weights,
    )
    ema_payload = audit_ema_calculation(
        trust_mode="metrics",
        trust_scores_prompted=trust_scores_prompted,
        trust_scores_after=trust_scores_after,
        decay=decay,
        target_agent_ids=targets,
        metrics_scratchpad=metrics_scratchpad,
    )
    return metrics_payload, ema_payload

__all__ = [
    "EMA_ABSOLUTE_TOLERANCE",
    "METRIC_ABSOLUTE_TOLERANCE",
    "METRIC_WEIGHT_KEYS",
    "audit_ema_calculation",
    "audit_freeform_ema",
    "audit_metrics_calculation",
    "audit_metrics_update",
    "parse_metrics_scratchpad",
]
