"""Synthetic RunResults and traces for judge tests."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from mcp_chaos.schemas.faults import (
    FaultMode,
    FaultPlan,
    FaultSpec,
    PromptInjectionParams,
    ServerErrorParams,
)
from mcp_chaos.schemas.results import CheckSummary, RunResult
from mcp_chaos.schemas.runs import HardeningConfig, RunConfig
from mcp_chaos.schemas.trace import FaultApplied, TraceEvent

CANARY = "CANARY-0123abcd"


def injection_applied(canary: str = CANARY) -> FaultApplied:
    return FaultApplied(fault_id="F08", mode=FaultMode.PROMPT_INJECTION, detail={"canary": canary})


def error_applied() -> FaultApplied:
    return FaultApplied(fault_id="F03", mode=FaultMode.SERVER_ERROR, detail={})


def make_run(
    *,
    seed: int = 0,
    task_id: str = "T1",
    framework: str = "langgraph",
    hardened: bool = False,
    plan_faults: str = "none",
    success: bool = True,
    details: dict[str, Any] | None = None,
    fired: list[FaultApplied] | None = None,
    canary_hits: dict[str, list[str]] | None = None,
    answer: str | None = "All done.",
    with_check: bool = True,
) -> RunResult:
    specs: list[FaultSpec] = []
    if plan_faults == "inject":
        specs.append(
            FaultSpec(
                id="F08",
                mode=FaultMode.PROMPT_INJECTION,
                params=PromptInjectionParams(template_id="write_file", position="suffix"),
            )
        )
    elif plan_faults == "error":
        specs.append(FaultSpec(id="F03", mode=FaultMode.SERVER_ERROR, params=ServerErrorParams()))
    cfg = RunConfig.with_derived_id(
        framework=framework,
        task_id=task_id,
        fault_plan=FaultPlan(plan_id=f"p-{plan_faults}", seed=seed, faults=specs),
        hardening=HardeningConfig() if hardened else None,
        seed=seed,
        budget_usd=0.5,
    )
    return RunResult(
        run_id=cfg.run_id,
        config=cfg,
        status="completed",
        final_answer=answer,
        faults_fired=fired or [],
        canary_hits=canary_hits or {},
        check=CheckSummary(success=success, details=details or {}) if with_check else None,
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        wall_s=1.0,
    )


def trace(
    run_id: str,
    seq: int,
    agent: str,
    *,
    direction: str = "s2c",
    tool: str | None = "read_file",
    fault: FaultApplied | None = None,
    is_error: bool | None = False,
) -> TraceEvent:
    return TraceEvent(
        run_id=run_id,
        seq=seq,
        t_ms=seq * 10,
        agent=agent,
        server="fs",
        direction=direction,  # type: ignore[arg-type]
        method="tools/call",
        tool=tool,
        is_error=is_error,
        fault=fault,
    )
