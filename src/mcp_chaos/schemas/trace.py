"""Trace event schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, JsonValue

from mcp_chaos.schemas.faults import FaultMode

Direction = Literal["c2s", "s2c"]


class FaultApplied(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    fault_id: str
    mode: FaultMode
    detail: dict[str, JsonValue]
    parseable: bool | None = None


class TraceEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1"] = "1"
    run_id: str
    seq: int
    t_ms: int
    agent: str
    server: str
    direction: Direction
    method: str
    tool: str | None = None
    tool_call_index: int | None = None
    args_sha256: str | None = None
    result_sha256: str | None = None
    result_bytes: int | None = None
    is_error: bool | None = None
    latency_ms: int | None = None
    fault: FaultApplied | None = None
