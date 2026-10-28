"""SQLite MCP server over real stdio with the SDK client."""

import json
import sqlite3
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
import mcp_types as types
import pytest
from mcp import Client, StdioServerParameters


@asynccontextmanager
async def spawn(db: Path, *extra: str) -> AsyncIterator[Client]:
    args = ["-m", "mcp_chaos.servers.sqlite_server", "--db", str(db), *extra]
    async with Client(StdioServerParameters(command=sys.executable, args=args)) as client:
        yield client


@pytest.fixture
def db(tmp_path: Path) -> Path:
    path = tmp_path / "s.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t(id INTEGER PRIMARY KEY, v TEXT)")
    conn.commit()
    conn.close()
    return path


def text_of(res: types.CallToolResult) -> str:
    block = res.content[0]
    assert isinstance(block, types.TextContent)
    return block.text


@pytest.mark.anyio
async def test_list_tools_and_roundtrip(db: Path) -> None:
    with anyio.fail_after(15):
        async with spawn(db) as client:
            tools = {t.name: t for t in (await client.list_tools()).tools}
            assert set(tools) == {"list_tables", "describe_table", "read_query", "write_query"}
            assert all(t.output_schema is not None for t in tools.values())
            w = await client.call_tool(
                "write_query", {"query": "INSERT INTO t(v) VALUES (?)", "params": ["hi"]}
            )
            assert w.structured_content == {"rows_affected": 1}
            r = await client.call_tool("read_query", {"query": "SELECT id, v FROM t"})
            assert r.structured_content is not None
            assert r.structured_content["rows"] == [[1, "hi"]]
            assert json.loads(text_of(r)) == r.structured_content


@pytest.mark.anyio
async def test_error_surfaces_as_is_error(db: Path) -> None:
    with anyio.fail_after(15):
        async with spawn(db) as client:
            r = await client.call_tool("read_query", {"query": "DROP TABLE t"})
            assert r.is_error
            r2 = await client.call_tool("describe_table", {"table": "missing"})
            assert r2.is_error


@pytest.mark.anyio
async def test_read_only_hides_write(db: Path) -> None:
    with anyio.fail_after(15):
        async with spawn(db, "--read-only") as client:
            names = [t.name for t in (await client.list_tools()).tools]
            assert "write_query" not in names
            r = await client.call_tool("write_query", {"query": "DELETE FROM t"})
            assert r.is_error
