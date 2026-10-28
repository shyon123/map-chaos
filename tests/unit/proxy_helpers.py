"""Shared builders for proxy tests."""

from typing import Any, Literal

from mcp_chaos.schemas import FaultMode, FaultPlan, FaultSpec, PromptInjectionParams, Trigger

Position = Literal["prefix", "suffix", "field"]


def injection_spec(
    position: Position = "suffix",
    template_id: Literal[
        "write_file", "echo_in_answer", "call_tool", "delegate"
    ] = "echo_in_answer",
    canary: str | None = None,
    fid: str = "f1",
    **trigger: Any,
) -> FaultSpec:
    return FaultSpec(
        id=fid,
        mode=FaultMode.PROMPT_INJECTION,
        params=PromptInjectionParams(template_id=template_id, position=position, canary=canary),
        trigger=Trigger(**trigger),
    )


def plan(*specs: FaultSpec, seed: int = 7) -> FaultPlan:
    return FaultPlan(plan_id="p", seed=seed, faults=list(specs))
