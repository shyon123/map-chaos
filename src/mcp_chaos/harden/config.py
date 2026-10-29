"""Guard runtime configuration: the ablation toggles plus timing parameters."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from mcp_chaos.schemas.runs import HardeningConfig

DEFAULT_DEADLINE_S = 20.0


class GuardConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    hardening: HardeningConfig = Field(default_factory=HardeningConfig)
    deadline_s: float = Field(default=DEFAULT_DEADLINE_S, gt=0)
    backoff_base_s: float = Field(default=0.25, ge=0)
    backoff_max_s: float = Field(default=4.0, ge=0)
    breaker_threshold: int = Field(default=3, ge=1)
    breaker_cooldown_s: float = Field(default=30.0, ge=0)
    seed: int = 0
