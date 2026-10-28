"""Run configuration schemas."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

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


def derive_run_id(
    framework: str,
    task_id: str,
    plan_id: str,
    seed: int,
    hardening: HardeningConfig | None,
) -> str:
    """Stable 16-hex run id; hardening is serialized canonically (None -> "baseline")."""
    hard = (
        "baseline"
        if hardening is None
        else json.dumps(hardening.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    )
    payload = json.dumps([framework, task_id, plan_id, seed, hard], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


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

    @classmethod
    def derive_run_id(
        cls,
        framework: str,
        task_id: str,
        plan_id: str,
        seed: int,
        hardening: HardeningConfig | None,
    ) -> str:
        return derive_run_id(framework, task_id, plan_id, seed, hardening)
