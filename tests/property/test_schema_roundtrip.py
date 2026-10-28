"""P5: every record survives a JSON round trip unchanged."""

from __future__ import annotations

from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import JsonValue

from mcp_chaos.schemas import (
    BlastRadius,
    CheckSummary,
    EmptyResultParams,
    FaultApplied,
    FaultMode,
    FaultPlan,
    FaultSpec,
    HardeningConfig,
    Labels,
    LatencyParams,
    LyingStatusParams,
    MalformedJsonParams,
    ModelConfig,
    OversizedOutputParams,
    PromptInjectionParams,
    RunConfig,
    RunResult,
    SchemaDriftParams,
    ServerErrorParams,
    SideEffects,
    StaleDataParams,
    TimeoutParams,
    ToolVanishParams,
    TraceEvent,
    Trigger,
    Usage,
)

text = st.text(max_size=20)
i64 = st.integers(min_value=-(2**53), max_value=2**53)
f = st.floats(allow_nan=False, allow_infinity=False)
unit = st.floats(min_value=0.0, max_value=1.0)
secs = st.floats(min_value=0.0, max_value=600.0)

json_values: st.SearchStrategy[JsonValue] = st.recursive(
    st.none() | st.booleans() | i64 | f | text,
    lambda c: st.lists(c, max_size=3) | st.dictionaries(text, c, max_size=3),
    max_leaves=8,
)

canary = st.none() | st.from_regex(r"CANARY-[0-9a-f]{8}", fullmatch=True)

params_by_mode: dict[FaultMode, st.SearchStrategy[Any]] = {
    FaultMode.TIMEOUT: st.builds(TimeoutParams, delay_s=secs, hang=st.booleans()),
    FaultMode.LATENCY: st.builds(LatencyParams, delay_s=secs, jitter_s=secs),
    FaultMode.SERVER_ERROR: st.builds(
        ServerErrorParams, code=st.integers(-32768, 32767), crash=st.booleans()
    ),
    FaultMode.MALFORMED_JSON: st.builds(MalformedJsonParams),
    FaultMode.SCHEMA_DRIFT: st.builds(SchemaDriftParams, path=st.text(min_size=1, max_size=20)),
    FaultMode.STALE_DATA: st.builds(
        StaleDataParams, lag_calls=st.integers(1, 1000), snapshot_id=st.none() | text
    ),
    FaultMode.OVERSIZED_OUTPUT: st.builds(OversizedOutputParams, bytes=st.integers(1, 50_000_000)),
    FaultMode.PROMPT_INJECTION: st.builds(PromptInjectionParams, canary=canary),
    FaultMode.EMPTY_RESULT: st.builds(EmptyResultParams),
    FaultMode.LYING_STATUS: st.builds(LyingStatusParams),
    FaultMode.TOOL_VANISH: st.builds(ToolVanishParams, after_calls=st.integers(0, 10_000)),
}

triggers = st.builds(
    Trigger,
    tool=text,
    server=text,
    calls=st.none() | st.lists(st.integers(0, 1000), max_size=5),
    probability=unit,
)


@st.composite
def fault_specs(draw: st.DrawFn) -> FaultSpec:
    mode = draw(st.sampled_from(list(FaultMode)))
    return FaultSpec(
        id=draw(text), mode=mode, params=draw(params_by_mode[mode]), trigger=draw(triggers)
    )


fault_plans = st.builds(
    FaultPlan,
    plan_id=text,
    seed=i64,
    faults=st.lists(fault_specs(), max_size=4),
    parent_plan_id=st.none() | text,
)

fault_applied = st.builds(
    FaultApplied,
    fault_id=text,
    mode=st.sampled_from(list(FaultMode)),
    detail=st.dictionaries(text, json_values, max_size=3),
    parseable=st.none() | st.booleans(),
)

opt_text = st.none() | text
opt_int = st.none() | st.integers(0, 10**9)

trace_events = st.builds(
    TraceEvent,
    run_id=text,
    seq=st.integers(0, 10**9),
    t_ms=st.integers(0, 10**9),
    agent=text,
    server=text,
    method=text,
    tool=opt_text,
    tool_call_index=opt_int,
    args_sha256=opt_text,
    result_sha256=opt_text,
    result_bytes=opt_int,
    is_error=st.none() | st.booleans(),
    latency_ms=opt_int,
    fault=st.none() | fault_applied,
)

usage = st.builds(
    Usage,
    input_tokens=st.integers(0, 10**9),
    output_tokens=st.integers(0, 10**9),
    cache_read_tokens=st.integers(0, 10**9),
    cache_write_tokens=st.integers(0, 10**9),
    cost_usd=f,
    cache_hits=st.integers(0, 10**6),
)

run_configs = st.builds(
    RunConfig,
    run_id=text,
    framework=st.sampled_from(["langgraph", "agentsdk"]),
    task_id=text,
    agent_model=st.builds(ModelConfig, model=text, temperature=unit),
    fault_plan=fault_plans,
    hardening=st.none() | st.builds(HardeningConfig),
    seed=i64,
    timeout_s=st.floats(min_value=0.001, max_value=1e6),
    budget_usd=st.floats(min_value=0.0, max_value=1e6),
)

labels = st.builds(
    Labels,
    task_success=st.booleans(),
    failure_detected=st.booleans(),
    false_success_claim=st.booleans(),
    followed_injection=st.none() | st.booleans(),
    blast_radius=st.sampled_from(list(BlastRadius)),
    label_source=st.dictionaries(
        text, st.sampled_from(["deterministic", "llm", "llm+deterministic"]), max_size=3
    ),
    judge_rationale=st.none() | st.text(max_size=500),
)

side_effects = st.builds(
    SideEffects,
    fs_added=st.lists(text, max_size=3).map(sorted),
    fs_removed=st.lists(text, max_size=3).map(sorted),
    fs_modified=st.lists(text, max_size=3).map(sorted),
    git_new_commits=st.lists(text, max_size=3),
    db_changes=st.dictionaries(text, st.integers(-1000, 1000), max_size=3),
    snapshot_sha256=text,
)

run_results = st.builds(
    RunResult,
    run_id=text,
    config=run_configs,
    status=st.sampled_from(
        ["completed", "agent_error", "timeout", "budget_abort", "harness_error"]
    ),
    final_answer=opt_text,
    side_effects=side_effects,
    check=st.none() | st.builds(CheckSummary, success=st.booleans(), details=st.just({})),
    faults_fired=st.lists(fault_applied, max_size=3),
    labels=st.none() | labels,
    usage=usage,
    agent_usage=usage,
    judge_usage=st.none() | usage,
    versions=st.dictionaries(text, text, max_size=3),
    started_at=st.datetimes(),
    wall_s=f,
)


@settings(max_examples=100)
@given(fault_plans)
def test_fault_plan_roundtrip(m: FaultPlan) -> None:
    assert FaultPlan.model_validate_json(m.model_dump_json()) == m


@settings(max_examples=100)
@given(trace_events)
def test_trace_event_roundtrip(m: TraceEvent) -> None:
    assert TraceEvent.model_validate_json(m.model_dump_json()) == m


@settings(max_examples=50)
@given(run_results)
def test_run_result_roundtrip(m: RunResult) -> None:
    assert RunResult.model_validate_json(m.model_dump_json()) == m
