"""SDK client -> real proxy -> fake server: tool_vanish and catalog schema_drift."""

import sys
from pathlib import Path
from typing import Any

import anyio
import mcp_types as types
import pytest
from mcp import Client, MCPError, StdioServerParameters

from mcp_chaos.schemas import FaultMode, FaultSpec, SchemaDriftParams, ToolVanishParams, Trigger
from tests.unit.proxy_helpers import plan

from .conftest import fake_server_cmd
from .test_proxy_e2e import events, via_proxy, write_plan


@pytest.mark.anyio
async def test_tool_vanish_notifies_and_relist_omits(
    fake_script_path: Path, tmp_path: Path
) -> None:
    trace = tmp_path / "t.jsonl"
    spec = FaultSpec(
        id="v1",
        mode=FaultMode.TOOL_VANISH,
        params=ToolVanishParams(after_calls=2),
        trigger=Trigger(tool="get_doc"),
    )
    changed = anyio.Event()
    seen: list[Any] = []

    async def handler(message: Any) -> None:
        if isinstance(message, types.ToolListChangedNotification):
            seen.append(message)
            changed.set()

    plan_path = write_plan(tmp_path, plan(spec))
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
        "--",
        *fake_server_cmd(fake_script_path),
    ]
    params = StdioServerParameters(command=sys.executable, args=args)
    async with Client(params, message_handler=handler) as client:
        with anyio.fail_after(30):
            assert "get_doc" in [t.name for t in (await client.list_tools()).tools]
            await client.call_tool("get_doc", {"id": "1"})
            assert not changed.is_set()
            await client.call_tool("get_doc", {"id": "1"})
            await changed.wait()
            names = [t.name for t in (await client.list_tools()).tools]
            assert "get_doc" not in names
            assert "echo" in names
            with pytest.raises(MCPError) as ei:
                await client.call_tool("get_doc", {"id": "1"})
            assert ei.value.error.code == -32602
            assert "Unknown tool" in ei.value.error.message
            await client.call_tool("echo", {})
    assert len(seen) == 1
    evs = events(trace)
    assert [e.method for e in evs if e.method == "notifications/tools/list_changed"] == [
        "notifications/tools/list_changed"
    ]
    rejected = [e for e in evs if e.fault is not None and e.fault.detail.get("rejected")]
    assert len(rejected) == 1


@pytest.mark.anyio
async def test_catalog_drift_e2e(fake_script_path: Path, tmp_path: Path) -> None:
    trace = tmp_path / "t.jsonl"
    spec = FaultSpec(
        id="d1",
        mode=FaultMode.SCHEMA_DRIFT,
        params=SchemaDriftParams(op="rename", path="table", layer="catalog"),
        trigger=Trigger(tool="read_rows"),
    )
    with anyio.fail_after(30):
        async with via_proxy(fake_script_path, write_plan(tmp_path, plan(spec)), trace) as c:
            tools = {t.name: t for t in (await c.list_tools()).tools}
    assert "table_v2" in tools["read_rows"].input_schema["properties"]
    assert (
        "table" in tools["get_doc"].input_schema["properties"]
        or "id" in (tools["get_doc"].input_schema["properties"])
    )
    faulted = [e for e in events(trace) if e.method == "tools/list" and e.fault]
    assert len(faulted) == 1
