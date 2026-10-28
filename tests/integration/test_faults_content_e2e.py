"""Content/frame faults through the real proxy CLI + fake server."""

import json
from pathlib import Path
from typing import Any

import anyio
import mcp_types as types
import pytest

from mcp_chaos.schemas import (
    FaultMode,
    FaultSpec,
    MalformedJsonParams,
    SchemaDriftParams,
    StaleDataParams,
    Trigger,
)
from tests.unit.proxy_helpers import plan

from .test_proxy_e2e import events, via_proxy, write_plan


def spec(params: Any, **trigger: Any) -> FaultSpec:
    return FaultSpec(id="f1", mode=params.mode, params=params, trigger=Trigger(**trigger))


def no_output_schema(src: Path, tmp_path: Path) -> Path:
    script = json.loads(src.read_text())
    del script["tools"][0]["output_schema"]
    out = tmp_path / "script.json"
    out.write_text(json.dumps(script))
    return out


@pytest.mark.anyio
async def test_schema_drift_rename_visible(fake_script_path: Path, tmp_path: Path) -> None:
    trace = tmp_path / "t.jsonl"
    p = plan(spec(SchemaDriftParams(op="rename", path="rows[].id"), tool="read_rows"))
    # same tool without an outputSchema: the SDK client validates structured content against it
    script_path = no_output_schema(fake_script_path, tmp_path)
    with anyio.fail_after(30):
        async with via_proxy(script_path, write_plan(tmp_path, p), trace) as c:
            res = await c.call_tool("read_rows", {})
    assert res.structured_content is not None
    assert res.structured_content["rows"][0] == {"id_v2": 1, "name": "a"}
    text = next(b.text for b in res.content if isinstance(b, types.TextContent))
    assert '"id_v2":1' in text
    fault = next(e.fault for e in events(trace) if e.fault is not None)
    assert fault.mode == FaultMode.SCHEMA_DRIFT and fault.detail["matched"] == 2
    assert fault.parseable is True


@pytest.mark.anyio
async def test_stale_data_replays_first_response(fake_script_path: Path, tmp_path: Path) -> None:
    trace = tmp_path / "t.jsonl"
    # flaky: call 0 -> error "transient failure", call 1 -> "ok"; fault on call 1 replays call 0
    p = plan(spec(StaleDataParams(lag_calls=1), tool="flaky", calls=[1]))
    with anyio.fail_after(30):
        async with via_proxy(fake_script_path, write_plan(tmp_path, p), trace) as c:
            first = await c.call_tool("flaky", {})
            second = await c.call_tool("flaky", {})
    assert first.is_error is True
    assert second.is_error is True
    assert second.content == first.content
    fault = next(e.fault for e in events(trace) if e.fault is not None)
    assert fault.detail == {"lag_calls": 1, "replayed_from_index": 0}


@pytest.mark.anyio
async def test_stale_data_no_history_is_passthrough(fake_script_path: Path, tmp_path: Path) -> None:
    trace = tmp_path / "t.jsonl"
    p = plan(spec(StaleDataParams(lag_calls=1), tool="flaky", calls=[0]))
    with anyio.fail_after(30):
        async with via_proxy(fake_script_path, write_plan(tmp_path, p), trace) as c:
            first = await c.call_tool("flaky", {})
    assert first.is_error is True
    assert all(e.fault is None for e in events(trace))


async def call_with_deadline(
    script: Path, plan_path: Path, trace: Path, seconds: float
) -> tuple[str, bool]:
    """Returns (outcome, hung): the SDK client's outcome for one get_doc call."""
    outcome = "returned"
    with anyio.move_on_after(seconds) as scope:
        try:
            async with via_proxy(script, plan_path, trace) as c:
                await c.call_tool("get_doc", {"id": "1"})
        except Exception as exc:
            outcome = type(exc).__name__
    return outcome, scope.cancelled_caught


@pytest.mark.anyio
@pytest.mark.parametrize("variant", ["truncate", "bad_escape", "trailing_garbage"])
async def test_malformed_frame_hangs_sdk_client(
    fake_script_path: Path, tmp_path: Path, variant: str
) -> None:
    """The SDK silently drops an unparseable line, so the pending request never completes.

    The harness must therefore treat F04-frame like a timeout at the client deadline.
    """
    trace = tmp_path / "t.jsonl"
    params = MalformedJsonParams(variant=variant, layer="frame")  # type: ignore[arg-type]
    p = plan(spec(params, tool="get_doc"))
    outcome, hung = await call_with_deadline(fake_script_path, write_plan(tmp_path, p), trace, 4)
    assert hung and outcome == "returned"
    ev = next(e for e in events(trace) if e.fault is not None)
    assert ev.fault is not None and ev.fault.mode == FaultMode.MALFORMED_JSON
    assert ev.fault.parseable is False
    # the trace describes the emitted (corrupted) bytes, not the original result
    assert ev.result_bytes is not None and ev.result_sha256 is not None
