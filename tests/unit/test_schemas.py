from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from mcp_chaos.schemas import (
    FaultMode,
    FaultPlan,
    FaultSpec,
    HardeningConfig,
    PromptInjectionParams,
    RunConfig,
    RunResult,
    SideEffects,
    TimeoutParams,
    TraceEvent,
    Trigger,
    derive_run_id,
)


def _spec(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "F01",
        "mode": "timeout",
        "params": {"mode": "timeout", "delay_s": 5},
    }
    base.update(over)
    return base


def test_all_modes_parse_with_defaults() -> None:
    for mode in FaultMode:
        params: dict[str, Any] = {"mode": mode.value}
        if mode is FaultMode.SCHEMA_DRIFT:
            params["path"] = "rows[].id"
        if mode is FaultMode.PROMPT_INJECTION:
            params.update(template_id="delegate", position="prefix")
        spec = FaultSpec.model_validate({"id": "x", "mode": mode.value, "params": params})
        assert spec.params.mode == mode
    assert len(FaultMode) == 11


def test_mode_params_mismatch() -> None:
    with pytest.raises(ValidationError, match="does not match"):
        FaultSpec.model_validate(_spec(mode="latency"))


def test_unknown_mode() -> None:
    with pytest.raises(ValidationError):
        FaultSpec.model_validate(_spec(mode="bogus", params={"mode": "bogus"}))


@pytest.mark.parametrize("p", [-0.1, 1.1])
def test_probability_out_of_range(p: float) -> None:
    with pytest.raises(ValidationError):
        Trigger(probability=p)


def test_negative_call_index() -> None:
    with pytest.raises(ValidationError):
        Trigger(calls=[0, -1])


def test_param_bounds() -> None:
    with pytest.raises(ValidationError):
        TimeoutParams(delay_s=601)
    with pytest.raises(ValidationError):
        FaultSpec.model_validate(
            _spec(mode="oversized_output", params={"mode": "oversized_output", "bytes": 0})
        )


@pytest.mark.parametrize(
    "canary", ["CANARY-xyz", "canary-7f3a1b2c", "CANARY-7F3A1B2C", "CANARY-7f3a"]
)
def test_bad_canary(canary: str) -> None:
    with pytest.raises(ValidationError, match="canary"):
        PromptInjectionParams(template_id="write_file", position="field", canary=canary)


def test_good_and_none_canary() -> None:
    assert PromptInjectionParams(template_id="write_file", position="field").canary is None
    p = PromptInjectionParams(template_id="delegate", position="suffix", canary="CANARY-7f3a1b2c")
    assert p.canary == "CANARY-7f3a1b2c"


def test_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        Trigger.model_validate({"tool": "x", "bogus": 1})
    with pytest.raises(ValidationError):
        FaultPlan.model_validate({"plan_id": "p", "seed": 1, "faults": [], "x": 1})
    with pytest.raises(ValidationError):
        TraceEvent.model_validate({"bogus": 1})


def test_frozen() -> None:
    t = Trigger()
    with pytest.raises(ValidationError):
        t.tool = "y"  # pyright: ignore[reportAttributeAccessIssue]


def test_schema_version_literal() -> None:
    assert FaultPlan(plan_id="p", seed=0, faults=[]).schema_version == "1"
    with pytest.raises(ValidationError):
        FaultPlan.model_validate({"plan_id": "p", "seed": 0, "faults": [], "schema_version": "2"})


def test_run_id_stable_and_sensitive() -> None:
    h = HardeningConfig(retry=True)
    a = derive_run_id("langgraph", "t1", "p1", 7, h)
    assert a == derive_run_id("langgraph", "t1", "p1", 7, HardeningConfig(retry=True))
    assert a == RunConfig.derive_run_id("langgraph", "t1", "p1", 7, h)
    assert len(a) == 16
    int(a, 16)
    others = {
        derive_run_id("agentsdk", "t1", "p1", 7, h),
        derive_run_id("langgraph", "t2", "p1", 7, h),
        derive_run_id("langgraph", "t1", "p2", 7, h),
        derive_run_id("langgraph", "t1", "p1", 8, h),
        derive_run_id("langgraph", "t1", "p1", 7, None),
        derive_run_id("langgraph", "t1", "p1", 7, HardeningConfig(retry=True, max_retries=3)),
    }
    assert a not in others
    assert len(others) == 6


def test_hardening_defaults_and_side_effects() -> None:
    h = HardeningConfig()
    assert (h.max_retries, h.size_cap_bytes) == (2, 200_000)
    se = SideEffects()
    assert se.db_changes == {} and se.snapshot_sha256 == ""


def test_run_result_minimal_json_roundtrip() -> None:
    plan = FaultPlan(plan_id="p", seed=1, faults=[])
    cfg = RunConfig(
        run_id=derive_run_id("langgraph", "t", "p", 1, None),
        framework="langgraph",
        task_id="t",
        fault_plan=plan,
        seed=1,
        budget_usd=1.0,
    )
    r = RunResult.model_validate(
        {
            "run_id": cfg.run_id,
            "config": cfg,
            "status": "completed",
            "started_at": "2026-01-01T00:00:00Z",
            "wall_s": 1.5,
        }
    )
    assert RunResult.model_validate_json(r.model_dump_json()) == r
