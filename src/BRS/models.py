from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

FEEDBACK_SUM_TOLERANCE = 1e-6

def finite_number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("must be a number, not a boolean or numeric string")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError("must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError("must be a finite number")
    return result

class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

class ObservationCounts(_StrictModel):
    P: int = Field(ge=0, strict=True)
    N: int = Field(ge=0, strict=True)

class AgentFeedback(_StrictModel):
    scratchpad: dict[str, str]
    feedback: dict[str, ObservationCounts]

    @field_validator("scratchpad")
    @classmethod
    def trim_summaries(cls, value: dict[str, str]) -> dict[str, str]:
        return {agent_id: summary.strip() for agent_id, summary in value.items()}

    @model_validator(mode="after")
    def matching_targets(self) -> "AgentFeedback":
        if not self.feedback or set(self.scratchpad) != set(self.feedback):
            raise ValueError("scratchpad and feedback must cover the same nonempty set of counterparts")
        if any(not agent_id.strip() or agent_id != agent_id.strip() for agent_id in self.feedback):
            raise ValueError("counterpart IDs must be nonempty and trimmed")
        return self

class FeedbackPair(_StrictModel):
    r: float = Field(ge=0, le=1)
    s: float = Field(ge=0, le=1)

    @field_validator("r", "s", mode="before")
    @classmethod
    def valid_number(cls, value: Any) -> float:
        return finite_number(value)

    @model_validator(mode="after")
    def fixed_episode_mass(self) -> "FeedbackPair":
        if self.r == 0 and self.s == 0:
            return self
        if not math.isclose(self.r + self.s, 1.0, rel_tol=0, abs_tol=FEEDBACK_SUM_TOLERANCE):
            raise ValueError("feedback must have r+s=1, or r=s=0 for insufficient evidence")
        return self

class AccumulatedEvidence(_StrictModel):
    R: float = Field(default=0.0, ge=0)
    S: float = Field(default=0.0, ge=0)

    @field_validator("R", "S", mode="before")
    @classmethod
    def valid_number(cls, value: Any) -> float:
        return finite_number(value)

    @model_validator(mode="after")
    def finite_total(self) -> "AccumulatedEvidence":
        if not math.isfinite(self.R + self.S):
            raise ValueError("total accumulated evidence must be finite")
        return self
