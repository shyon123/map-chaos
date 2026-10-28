"""Run configuration schemas."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from mcp_chaos.schemas.faults import FaultPlan


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ModelConfig(_Frozen):
    provider: Literal["anthropic"] = "anthropic"
    model: str = "claude-haiku-5-5"
    temperature: float = 0.0
    max_tokens: int = 2048


class HardeningConfig(_Frozen):
    schema_validation: bool = False
    retry: bool = False
    circuit_breaker: bool = False
    injection_filter: bool = False
    size_cap: bool = False
    prompt_addendum: bool = False
    max_retries: int = Field(default=2, ge=0)
    size_cap_bytes: int = Field(default=200_000, ge=1)


Framework = Literal["langgraph", "agentsdk"]


class RunConfig(_Frozen):
    schema_version: Literal["1"] = "1"
    run_id: str
    framework: Framework
    task_id: str
    agent_model: ModelConfig = Field(default_factory=ModelConfig)
    fault_plan: FaultPlan
    hardening: HardeningConfig | None = None
    seed: int
    max_turns: int = Field(default=25, ge=1)
    timeout_s: float = Field(default=300, gt=0)
    budget_usd: float = Field(ge=0)

    def compute_id(self) -> str:
        """16-hex sha256 prefix over the canonical JSON of the whole config except ``run_id``."""
        body = self.model_dump(mode="json", exclude={"run_id"})
        payload = json.dumps(body, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    @classmethod
    def with_derived_id(cls, **fields: Any) -> RunConfig:
        """Build a config whose ``run_id`` is derived from every other field."""
        if "run_id" in fields:
            raise TypeError("with_derived_id() derives run_id; do not pass it")
        draft = cls(run_id="", **fields)
        return draft.model_copy(update={"run_id": draft.compute_id()})


def derive_run_id(cfg: RunConfig) -> str:
    return cfg.compute_id()
