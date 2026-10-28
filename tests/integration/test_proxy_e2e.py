"""End-to-end: SDK client -> mcp-chaos proxy (subprocess) -> fake server (subprocess)."""

import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
import mcp_types as types
import pytest
from mcp import Client, MCPError, StdioServerParameters

from mcp_chaos.schemas import FaultPlan, TraceEvent
from tests.unit.proxy_helpers import injection_spec, plan

from .conftest import fake_server_cmd

CANARY = "CANARY-0badf00d"


def write_plan(tmp_path: Path, p: FaultPlan) -> Path:
    path = tmp_path / "plan.json"
    path.write_text(p.model_dump_json(), encoding="utf-8")
    return path


@asynccontextmanager
async def via_proxy(
    script: Path, plan_path: Path, trace: Path, allow: str | None = None
) -> AsyncIterator[Client]:
    args = [
        "-m",
        "mcp_chaos.bench.cli",
        "proxy",
        "--agent",
        "agent1",
        "--server",
        "fake",
        "--plan",
        str(plan_path),
        "--run-id",
        "run1",
        "--trace-out",
        str(trace),
    ]
    if allow is not None:
        args += ["--allow", allow]
    args += ["--", *fake_server_cmd(script)]
    async with Client(StdioServerParameters(command=sys.executable, args=args)) as client:
        yield client


@asynccontextmanager
async def direct(script: Path) -> AsyncIterator[Client]:
    cmd = fake_server_cmd(script)
    async with Client(StdioServerParameters(command=cmd[0], args=cmd[1:])) as client:
        yield client


def events(trace: Path) -> list[TraceEvent]:
    return [TraceEvent.model_validate_json(x) for x in trace.read_text().splitlines()]


async def exercise(client: Client) -> list[dict[str, object]]:
    out = [
        (await client.call_tool("read_rows", {})).model_dump(mode="json"),
        (await client.call_tool("echo", {"b": 1, "a": "x"})).model_dump(mode="json"),
        (await client.call_tool("get_doc", {"id": "1"})).model_dump(mode="json"),
        (await client.call_tool("flaky", {})).model_dump(mode="json"),
    ]
    return out


@pytest.mark.anyio
async def test_empty_plan_equals_direct(fake_script_path: Path, tmp_path: Path) -> None:
    trace = tmp_path / "t.jsonl"
    with anyio.fail_after(30):
        async with direct(fake_script_path) as c:
            expected = await exercise(c)
        async with via_proxy(fake_script_path, write_plan(tmp_path, plan()), trace) as c:
            got = await exercise(c)
    assert got == expected
    evs = events(trace)
    assert all(e.fault is None for e in evs)
    assert [e.tool for e in evs if e.direction == "s2c" and e.method == "tools/call"] == [
        "read_rows",
        "echo",
        "get_doc",
        "flaky",
    ]


@pytest.mark.anyio
async def test_suffix_injection(fake_script_path: Path, tmp_path: Path) -> None:
    trace = tmp_path / "t.jsonl"
    p = plan(injection_spec("suffix", canary=CANARY, tool="read_rows", calls=[0]))
    with anyio.fail_after(30):
        async with via_proxy(fake_script_path, write_plan(tmp_path, p), trace) as c:
            first = await c.call_tool("read_rows", {})
            second = await c.call_tool("read_rows", {})
    texts = [b.text for b in first.content if isinstance(b, types.TextContent)]
    assert any(CANARY in t for t in texts)
    assert not any(CANARY in b.text for b in second.content if isinstance(b, types.TextContent))
    faults = [e.fault for e in events(trace) if e.fault is not None]
    assert len(faults) == 1
    assert faults[0].mode == "prompt_injection"
    assert faults[0].detail["canary"] == CANARY


@pytest.mark.anyio
async def test_allowlist(fake_script_path: Path, tmp_path: Path) -> None:
    trace = tmp_path / "t.jsonl"
    with anyio.fail_after(30):
        async with via_proxy(
            fake_script_path, write_plan(tmp_path, plan()), trace, allow="read_rows"
        ) as c:
            tools = await c.list_tools()
            ok = await c.call_tool("read_rows", {})
            with pytest.raises(MCPError) as ei:
                await c.call_tool("echo", {})
    assert [t.name for t in tools.tools] == ["read_rows"]
    assert ok.structured_content is not None
    assert ei.value.error.code == -32001
    assert any(e.method == "tools/call/denied" for e in events(trace))


@pytest.mark.anyio
async def test_clean_exit(fake_script_path: Path, tmp_path: Path) -> None:
    args = [
        sys.executable,
        "-m",
        "mcp_chaos.bench.cli",
        "proxy",
        "--agent",
        "a",
        "--server",
        "fake",
        "--plan",
        str(write_plan(tmp_path, plan())),
        "--run-id",
        "r",
        "--trace-out",
        str(tmp_path / "t.jsonl"),
        "--",
        *fake_server_cmd(fake_script_path),
    ]
    with anyio.fail_after(5):
        proc = await anyio.open_process(args)
        assert proc.stdin is not None
        await proc.stdin.aclose()
        code = await proc.wait()
    assert code == 0


def test_cli_guard_and_bad_plan(tmp_path: Path) -> None:
    from mcp_chaos.bench.cli import main

    common = ["--agent", "a", "--server", "s", "--run-id", "r", "--trace-out", str(tmp_path / "t")]
    pl = write_plan(tmp_path, plan())
    assert main(["proxy", *common, "--plan", str(pl), "--mode", "guard", "--", "x"]) == 2
    assert main(["proxy", *common, "--plan", str(tmp_path / "nope.json"), "--", "x"]) == 2
    assert main(["proxy", *common, "--plan", str(pl)]) == 2
    assert main([]) == 0
