"""Fault plan schemas (docs/schemas.md, docs/fault-taxonomy.md)."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_CANARY_RE = re.compile(r"^CANARY-[0-9a-f]{8}$")


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class FaultMode(StrEnum):
    TIMEOUT = "timeout"
    LATENCY = "latency"
    SERVER_ERROR = "server_error"
    MALFORMED_JSON = "malformed_json"
    SCHEMA_DRIFT = "schema_drift"
    STALE_DATA = "stale_data"
    OVERSIZED_OUTPUT = "oversized_output"
    PROMPT_INJECTION = "prompt_injection"
    EMPTY_RESULT = "empty_result"
    LYING_STATUS = "lying_status"
    TOOL_VANISH = "tool_vanish"


class Trigger(_Frozen):
    tool: str = "*"
    server: str = "*"
    calls: list[int] | None = None
    probability: float = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator("calls")
    @classmethod
    def _calls_non_negative(cls, v: list[int] | None) -> list[int] | None:
        if v is not None and any(i < 0 for i in v):
            raise ValueError("call indices must be >= 0")
        return v


class TimeoutParams(_Frozen):
    mode: Literal[FaultMode.TIMEOUT] = FaultMode.TIMEOUT
    delay_s: float = Field(default=60.0, ge=0.0, le=600.0)
    hang: bool = False


class LatencyParams(_Frozen):
    mode: Literal[FaultMode.LATENCY] = FaultMode.LATENCY
    delay_s: float = Field(default=1.0, ge=0.0, le=600.0)
    jitter_s: float = Field(default=0.0, ge=0.0, le=600.0)


class ServerErrorParams(_Frozen):
    mode: Literal[FaultMode.SERVER_ERROR] = FaultMode.SERVER_ERROR
    code: int = Field(default=-32000, ge=-32768, le=32767)
    crash: bool = False


class MalformedJsonParams(_Frozen):
    mode: Literal[FaultMode.MALFORMED_JSON] = FaultMode.MALFORMED_JSON
    variant: Literal["truncate", "bad_escape", "trailing_garbage", "wrong_type"] = "truncate"
    layer: Literal["frame", "content"] = "content"


class SchemaDriftParams(_Frozen):
    mode: Literal[FaultMode.SCHEMA_DRIFT] = FaultMode.SCHEMA_DRIFT
    op: Literal["rename", "drop", "retype", "nest"] = "rename"
    path: str = Field(min_length=1)


class StaleDataParams(_Frozen):
    mode: Literal[FaultMode.STALE_DATA] = FaultMode.STALE_DATA
    lag_calls: int = Field(default=1, ge=1, le=1000)
    snapshot_id: str | None = None


class OversizedOutputParams(_Frozen):
    mode: Literal[FaultMode.OVERSIZED_OUTPUT] = FaultMode.OVERSIZED_OUTPUT
    bytes: int = Field(default=1_000_000, ge=1, le=50_000_000)
    filler: Literal["zeros", "lorem", "rows"] = "lorem"


class PromptInjectionParams(_Frozen):
    mode: Literal[FaultMode.PROMPT_INJECTION] = FaultMode.PROMPT_INJECTION
    template_id: Literal["write_file", "echo_in_answer", "call_tool", "delegate"]
    position: Literal["prefix", "suffix", "field"]
    canary: str | None = None  # None = derive from seed at injection time

    @field_validator("canary")
    @classmethod
    def _canary_format(cls, v: str | None) -> str | None:
        if v is not None and not _CANARY_RE.match(v):
            raise ValueError("canary must match ^CANARY-[0-9a-f]{8}$")
        return v


class EmptyResultParams(_Frozen):
    mode: Literal[FaultMode.EMPTY_RESULT] = FaultMode.EMPTY_RESULT


class LyingStatusParams(_Frozen):
    mode: Literal[FaultMode.LYING_STATUS] = FaultMode.LYING_STATUS
    # error_as_ok: error text with isError=false; ok_as_error: success text with isError=true
    direction: Literal["error_as_ok", "ok_as_error"] = "error_as_ok"


class ToolVanishParams(_Frozen):
    mode: Literal[FaultMode.TOOL_VANISH] = FaultMode.TOOL_VANISH
    after_calls: int = Field(default=1, ge=0, le=10_000)


FaultParams = Annotated[
    TimeoutParams
    | LatencyParams
    | ServerErrorParams
    | MalformedJsonParams
    | SchemaDriftParams
    | StaleDataParams
    | OversizedOutputParams
    | PromptInjectionParams
    | EmptyResultParams
    | LyingStatusParams
    | ToolVanishParams,
    Field(discriminator="mode"),
]


class FaultSpec(_Frozen):
    id: str
    mode: FaultMode
    params: FaultParams
    trigger: Trigger = Field(default_factory=Trigger)

    @model_validator(mode="after")
    def _mode_matches_params(self) -> FaultSpec:
        if self.params.mode != self.mode:
            raise ValueError(f"params.mode {self.params.mode!s} does not match mode {self.mode!s}")
        return self


class FaultPlan(_Frozen):
    schema_version: Literal["1"] = "1"
    plan_id: str
    seed: int
    faults: list[FaultSpec]  # first matching fault wins per call
    source: Literal["static", "redteam"] = "static"
    parent_plan_id: str | None = None
