"""Tests for the deterministic fake MCP server (subprocess over stdio + in-process)."""

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import anyio
import mcp_types as types
import pytest
from mcp import Client, MCPError, StdioServerParameters
from pydantic import ValidationError

from mcp_chaos.servers.fake_server import (
    FakeScript,
    FakeState,
    build_server,
    load_script,
    main,
)

from .conftest import fake_server_cmd


@asynccontextmanager
async def spawn(script: Path) -> AsyncIterator[Client]:
    cmd = fake_server_cmd(script)
    params = StdioServerParameters(command=cmd[0], args=cmd[1:])
    async with Client(params) as client:
        yield client


def text_of(result: types.CallToolResult) -> str:
    block = result.content[0]
    assert isinstance(block, types.TextContent)
    return block.text


# ---------- subprocess over stdio ----------


@pytest.mark.anyio
async def test_list_tools(fake_script_path: Path) -> None:
    with anyio.fail_after(5):
        async with spawn(fake_script_path) as client:
            res = await client.list_tools()
    by_name = {t.name: t for t in res.tools}
    assert set(by_name) == {"read_rows", "echo", "flaky", "get_doc"}
    assert by_name["read_rows"].output_schema is not None
    assert by_name["read_rows"].input_schema["properties"]["table"] == {"type": "string"}
    assert by_name["echo"].output_schema is None


@pytest.mark.anyio
async def test_read_rows_stable(fake_script_path: Path) -> None:
    expected = {"rows": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}], "count": 2}
    with anyio.fail_after(5):
        async with spawn(fake_script_path) as client:
            r1 = await client.call_tool("read_rows", {})
            r2 = await client.call_tool("read_rows", {})
    assert r1.structured_content == r2.structured_content == expected
    assert json.loads(text_of(r1)) == expected
    assert text_of(r1) == text_of(r2)


@pytest.mark.anyio
async def test_echo_and_doc(fake_script_path: Path) -> None:
    with anyio.fail_after(5):
        async with spawn(fake_script_path) as client:
            echo = await client.call_tool("echo", {"b": 1, "a": "x"})
            doc = await client.call_tool("get_doc", {"id": "1"})
    assert text_of(echo) == '{"a": "x", "b": 1}'
    assert "quick brown fox" in text_of(doc)


@pytest.mark.anyio
async def test_flaky_sequence(fake_script_path: Path) -> None:
    with anyio.fail_after(5):
        async with spawn(fake_script_path) as client:
            rs = [await client.call_tool("flaky", {}) for _ in range(3)]
    assert [r.is_error for r in rs] == [True, False, False]
    assert [text_of(r) for r in rs] == ["transient failure", "ok", "ok"]


@pytest.mark.anyio
async def test_unknown_tool_errors(fake_script_path: Path) -> None:
    with anyio.fail_after(5):
        async with spawn(fake_script_path) as client:
            with pytest.raises(MCPError):
                await client.call_tool("nope", {})


# ---------- in-process (covers handler logic) ----------


@pytest.mark.anyio
async def test_in_process_roundtrip(fake_script_path: Path) -> None:
    server = build_server(load_script(fake_script_path))
    with anyio.fail_after(5):
        async with Client(server) as client:
            tools = await client.list_tools()
            assert len(tools.tools) == 4
            r = await client.call_tool("echo", {"k": 1})
            assert text_of(r) == '{"k": 1}'
            flaky = [await client.call_tool("flaky", {}) for _ in range(2)]
            assert [f.is_error for f in flaky] == [True, False]
            with pytest.raises(MCPError):
                await client.call_tool("nope", {})


def _script(**tool: Any) -> FakeScript:
    base: dict[str, Any] = {"name": "t", "responses": [{"text": "a"}, {"text": "b"}]}
    return FakeScript.model_validate({"server_name": "s", "tools": [base | tool]})


def _texts(state: FakeState, n: int) -> list[str]:
    out: list[str] = []
    for _ in range(n):
        block = state.call("t", None).content[0]
        assert isinstance(block, types.TextContent)
        out.append(block.text)
    return out


def test_state_cycle_and_no_cycle() -> None:
    assert _texts(FakeState(_script()), 3) == ["a", "b", "a"]
    assert _texts(FakeState(_script(cycle=False)), 3) == ["a", "b", "b"]


def test_state_empty_args_and_no_text() -> None:
    st = FakeState(_script(responses=[{"echo_args": True}, {}]))
    assert _texts(FakeState(_script(responses=[{"echo_args": True}])), 1) == ["{}"]
    st.call("t", None)
    assert st.call("t", {}).content == []


def test_state_structured_only_gets_compact_text() -> None:
    st = FakeState(_script(responses=[{"structured": {"b": 1, "a": 2}}]))
    res = st.call("t", None)
    block = res.content[0]
    assert isinstance(block, types.TextContent)
    assert block.text == '{"a":2,"b":1}'
    assert res.structured_content == {"b": 1, "a": 2}


def test_main_rejects_missing_script(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        main(["--script", str(tmp_path / "missing.json")])


# ---------- script model validation ----------


def test_script_validation_ok(fake_script_path: Path) -> None:
    script = load_script(fake_script_path)
    assert script.server_name == "fake"
    assert [t.name for t in script.tools] == ["read_rows", "echo", "flaky", "get_doc"]
    assert script.tools[2].cycle is False
    assert script.tools[0].responses[0].is_error is False


@pytest.mark.parametrize(
    "bad",
    [
        {"tools": []},
        {"server_name": "s", "tools": [{"name": "t", "responses": []}]},
        {"server_name": "s", "tools": [{"name": "t", "responses": [{"bogus": 1}]}]},
        {"server_name": "s", "tools": [{"responses": [{}]}]},
        {"server_name": "s", "tools": [], "extra": 1},
    ],
)
def test_script_validation_rejects(bad: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        FakeScript.model_validate(bad)
