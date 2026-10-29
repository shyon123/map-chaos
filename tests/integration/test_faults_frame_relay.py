"""Frame-layer faults (F01-F03) through the relay: in-process and via the SDK client."""

import time
from pathlib import Path
from typing import Any

import anyio
import pytest
from mcp import MCPError

from mcp_chaos.schemas import (
    FaultMode,
    FaultSpec,
    LatencyParams,
    ServerErrorParams,
    TimeoutParams,
    Trigger,
)
from tests.unit.proxy_helpers import plan

from .test_proxy_e2e import events, via_proxy, write_plan
from .test_relay_inprocess import CALL, ECHO_SERVER, run


def spec(mode: FaultMode, params: Any, **trigger: Any) -> FaultSpec:
    return FaultSpec(id="f1", mode=mode, params=params, trigger=Trigger(**trigger))


@pytest.mark.anyio
async def test_latency_delivers_after_delay(tmp_path: Path) -> None:
    p = plan(spec(FaultMode.LATENCY, LatencyParams(delay_s=0.3)))
    t0 = time.monotonic()
    _, out, evs = await run(tmp_path, p, ECHO_SERVER, CALL)
    assert time.monotonic() - t0 >= 0.3
    assert b'"ok"' in out
    assert evs[1]["fault"]["mode"] == "latency"
    assert evs[1]["fault"]["detail"] == {"delay_s": 0.3}


@pytest.mark.anyio
async def test_timeout_delay_delivers_intact(tmp_path: Path) -> None:
    p = plan(spec(FaultMode.TIMEOUT, TimeoutParams(delay_s=0.2)))
    t0 = time.monotonic()
    _, out, _ = await run(tmp_path, p, ECHO_SERVER, CALL)
    assert time.monotonic() - t0 >= 0.2
    assert b'"ok"' in out


@pytest.mark.anyio
async def test_timeout_hang_never_delivers(tmp_path: Path) -> None:
    p = plan(spec(FaultMode.TIMEOUT, TimeoutParams(hang=True)))
    _, out, evs = await run(tmp_path, p, ECHO_SERVER, CALL)
    assert out == b""
    assert evs[1]["fault"]["detail"] == {"delay_s": 60.0, "hang": True}


@pytest.mark.anyio
async def test_sdk_hang_client_read_times_out(fake_script_path: Path, tmp_path: Path) -> None:
    p = plan(spec(FaultMode.TIMEOUT, TimeoutParams(hang=True), tool="echo"))
    trace = tmp_path / "t.jsonl"
    with anyio.fail_after(30):
        async with via_proxy(fake_script_path, write_plan(tmp_path, p), trace) as c:
            assert (await c.call_tool("read_rows", {})).structured_content is not None
            with anyio.move_on_after(0.5) as scope:
                await c.call_tool("echo", {})
            assert scope.cancelled_caught
    assert any(e.fault is not None and e.fault.mode == "timeout" for e in events(trace))


@pytest.mark.anyio
async def test_sdk_server_error_raises_mcp_error(fake_script_path: Path, tmp_path: Path) -> None:
    p = plan(spec(FaultMode.SERVER_ERROR, ServerErrorParams(code=-32099), tool="echo"))
    trace = tmp_path / "t.jsonl"
    with anyio.fail_after(30):
        async with via_proxy(fake_script_path, write_plan(tmp_path, p), trace) as c:
            with pytest.raises(MCPError) as ei:
                await c.call_tool("echo", {})
            assert ei.value.error.code == -32099
            assert ei.value.error.message == "Internal server error"
            assert (await c.call_tool("read_rows", {})).structured_content is not None
    evs = [e for e in events(trace) if e.fault is not None]
    assert len(evs) == 1 and evs[0].is_error


@pytest.mark.anyio
async def test_crash_kills_upstream_in_process(tmp_path: Path) -> None:
    p = plan(spec(FaultMode.SERVER_ERROR, ServerErrorParams(crash=True)))
    script = ECHO_SERVER.replace("sys.exit(3)", "")
    t0 = time.monotonic()
    code, out, evs = await run(tmp_path, p, script, CALL, delay_close=1.0)
    assert out == b"" and code != 0
    assert time.monotonic() - t0 < 5
    assert evs[1]["fault"]["detail"] == {"code": -32000, "crash": True}


@pytest.mark.anyio
async def test_sdk_crash_does_not_hang(fake_script_path: Path, tmp_path: Path) -> None:
    p = plan(spec(FaultMode.SERVER_ERROR, ServerErrorParams(crash=True), tool="echo"))
    trace = tmp_path / "t.jsonl"
    t0 = time.monotonic()
    with pytest.raises(Exception) as ei:
        with anyio.fail_after(3):
            async with via_proxy(fake_script_path, write_plan(tmp_path, p), trace) as c:
                await c.call_tool("echo", {})
    assert not isinstance(ei.value, TimeoutError)
    assert time.monotonic() - t0 < 5
