from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

class BRSConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    forgetting_factor: float = Field(default=0.9, ge=0, le=1, allow_inf_nan=False, strict=True)
