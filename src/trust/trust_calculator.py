from __future__ import annotations

from typing import Any, Optional

from src.trust.metrics import (
    cooperation_score,
    counterfactual_blame,
    outlier_rejection_rate,
    targeted_failure_rate,
    vote_consistency_deterministic,
    vote_consistency_on_equivalent_deals,
)

def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))

class TrustCalculator:
    def __init__(
        self,
        trust_config: Any,
        *,
        unavailable_metric_default: float = 0.0,
    ) -> None:
        if isinstance(unavailable_metric_default, bool) or not isinstance(
            unavailable_metric_default, (int, float)
        ):
            raise TypeError("unavailable_metric_default must be a number in [0, 1]")
        if not 0.0 <= float(unavailable_metric_default) <= 1.0:
            raise ValueError("unavailable_metric_default must be in [0, 1]")
        self._decay = trust_config.decay
        self._update = 1.0 - self._decay
        self._min_br = trust_config.min_blocking_rejects
        self._min_sr = trust_config.min_sample_rounds
        self._unavailable_metric_default = float(unavailable_metric_default)

        mw = trust_config.metric_weights
        self._w_f = mw.get("vote_inconsistency", 0.35)
        self._w_b = mw.get("counterfactual_blame", 0.25)
        self._w_o = mw.get("outlier_rejection", 0.20)
        self._w_t = mw.get("targeted_failure", 0.10)
        self._w_c = mw.get("cooperation", 0.10)

    def _compute(
        self,
        other_id: str,
        record: Any,
        cover_profiles: Optional[dict[str, dict]] = None,
    ) -> tuple[float, float, float, float, float, float]:
        b = counterfactual_blame(
            other_id,
            record,
            self._min_br,
            unavailable_default=self._unavailable_metric_default,
        )
        c = cooperation_score(other_id, record)
        o = outlier_rejection_rate(other_id, record, self._min_br, self._min_sr)
        t = targeted_failure_rate(
            other_id,
            record,
            self._min_br,
            unavailable_default=self._unavailable_metric_default,
        )
        profile = (cover_profiles or {}).get(other_id)
        if profile is not None:
            f = vote_consistency_deterministic(
                other_id,
                record,
                profile["scores"],
                profile["threshold"],
                self._min_sr,
            )
        else:
            f = vote_consistency_on_equivalent_deals(other_id, record, self._min_sr)
        suspicion = (
            self._w_f * f
            + self._w_b * b
            + self._w_o * o
            + self._w_t * t
            + self._w_c * (1.0 - c)
        )
        return b, c, o, t, f, suspicion

    def update(
        self,
        prev_trust: dict[str, float],
        record: Any,
        focal_agent_id: str,
        cover_profiles: Optional[dict[str, dict]] = None,
    ) -> dict[str, float]:
        new: dict[str, float] = {}
        for other_id in prev_trust:
            if other_id == focal_agent_id:
                continue
            _, _, _, _, _, suspicion = self._compute(other_id, record, cover_profiles)
            new[other_id] = clamp(
                self._decay * prev_trust[other_id] + self._update * (1.0 - suspicion),
                0.0,
                1.0,
            )
        return new

    def update_with_detail(
        self,
        prev_trust: dict[str, float],
        record: Any,
        focal_agent_id: str,
        cover_profiles: Optional[dict[str, dict]] = None,
    ) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
        new: dict[str, float] = {}
        detail: dict[str, dict[str, float]] = {}
        for other_id in prev_trust:
            if other_id == focal_agent_id:
                continue
            b, c, o, t, f, suspicion = self._compute(other_id, record, cover_profiles)
            trust_signal = 1.0 - suspicion
            new[other_id] = clamp(
                self._decay * prev_trust[other_id] + self._update * trust_signal,
                0.0,
                1.0,
            )
            detail[other_id] = {
                "b": b,
                "C": c,
                "O": o,
                "T": t,
                "F": f,
                "suspicion": suspicion,
                "trust_signal": trust_signal,
                "prev_trust": prev_trust[other_id],
                "new_trust": new[other_id],
            }
        return new, detail
