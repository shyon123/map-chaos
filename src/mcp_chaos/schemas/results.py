"""Run result schemas."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from mcp_chaos.schemas.runs import RunConfig
from mcp_chaos.schemas.trace import FaultApplied


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class BlastRadius(StrEnum):
    NONE = "none"
    SELF = "self"
    PEER = "peer"
    SUPERVISOR = "supervisor"
    PERSISTENT = "persistent"


class Labels(_Frozen):
    task_success: bool
    failure_detected: bool
    false_success_claim: bool
    followed_injection: bool | None = None
    blast_radius: BlastRadius
    label_source: dict[str, Literal["deterministic", "llm", "llm+deterministic"]]
    judge_rationale: str | None = Field(default=None, max_length=500)


class Usage(_Frozen):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    cache_hits: int = 0


class SideEffects(_Frozen):
    fs_added: list[str] = Field(default_factory=list)
    fs_removed: list[str] = Field(default_factory=list)
    fs_modified: list[str] = Field(default_factory=list)
    git_new_commits: list[str] = Field(default_factory=list)
    db_changes: dict[str, int] = Field(default_factory=dict)  # table -> net row delta
    snapshot_sha256: str = ""


class RunResult(_Frozen):
    schema_version: Literal["1"] = "1"
    run_id: str
    config: RunConfig
    status: Literal["completed", "agent_error", "timeout", "budget_abort", "harness_error"]
    final_answer: str | None = None
    side_effects: SideEffects = Field(default_factory=SideEffects)
    faults_fired: list[FaultApplied] = Field(default_factory=list[FaultApplied])
    labels: Labels | None = None
    usage: Usage = Field(default_factory=Usage)
    agent_usage: Usage = Field(default_factory=Usage)
    judge_usage: Usage | None = None
    versions: dict[str, str] = Field(default_factory=dict)
    started_at: datetime
    wall_s: float
