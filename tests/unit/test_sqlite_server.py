"""Unit tests for the SQLite MCP server handlers (direct calls on a temp DB)."""

import json
import sqlite3
from pathlib import Path
from typing import Any

import anyio
import mcp_types as types
import pytest
from mcp import Client

from mcp_chaos.servers.sqlite_server import (
    MAX_ROWS,
    SqliteTools,
    ToolError,
    build_server,
    main,
    validate_db_path,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def db(tmp_path: Path) -> Path:
    path = tmp_path / "t.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE users(id INTEGER PRIMARY KEY, name TEXT NOT NULL);"
        "CREATE TABLE a_first(x);"
        "INSERT INTO users(name) VALUES ('ann'),('bob');"
    )
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def tools(db: Path) -> SqliteTools:
    return SqliteTools(db, read_only=False)


def text(res: types.CallToolResult) -> str:
    block = res.content[0]
    assert isinstance(block, types.TextContent)
    return block.text


def rows(t: SqliteTools, q: str) -> Any:
    res = t.call("read_query", {"query": q})
    assert not res.is_error, text(res)
    assert res.structured_content is not None
    return res.structured_content["rows"]


def test_list_tables_sorted_excludes_internal(tools: SqliteTools) -> None:
    conn = sqlite3.connect(tools.db_path)
    conn.executescript(
        "CREATE TABLE z(id INTEGER PRIMARY KEY AUTOINCREMENT);INSERT INTO z DEFAULT VALUES;"
    )
    conn.close()
    res = tools.call("list_tables", {})
    assert res.structured_content == {"tables": ["a_first", "users", "z"]}
    assert text(res) == '{"tables":["a_first","users","z"]}'


def test_describe_table(tools: SqliteTools) -> None:
    res = tools.call("describe_table", {"table": "users"})
    assert res.structured_content == {
        "table": "users",
        "columns": [
            {"name": "id", "type": "INTEGER", "notnull": False, "pk": 1},
            {"name": "name", "type": "TEXT", "notnull": True, "pk": 0},
        ],
    }


@pytest.mark.parametrize("args", [{"table": "nope"}, {"table": "sqlite_master"}, {}, {"table": ""}])
def test_describe_table_errors(tools: SqliteTools, args: dict[str, Any]) -> None:
    assert tools.call("describe_table", args).is_error


def test_read_query_basic_and_params(tools: SqliteTools) -> None:
    res = tools.call(
        "read_query",
        {"query": "SELECT id, name FROM users WHERE id > ? ORDER BY id", "params": [0]},
    )
    assert res.structured_content == {
        "columns": ["id", "name"],
        "rows": [[1, "ann"], [2, "bob"]],
        "row_count": 2,
        "truncated": False,
    }
    assert json.loads(text(res)) == res.structured_content


def test_read_query_blob_hex_and_recursive_cte(tools: SqliteTools) -> None:
    assert rows(tools, "SELECT x'ff00'") == [["ff00"]]
    cte = "WITH RECURSIVE c(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM c WHERE n<3) SELECT n FROM c"
    assert rows(tools, cte) == [[1], [2], [3]]


def _series(n: int) -> str:
    return (
        f"WITH RECURSIVE c(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM c WHERE n<{n}) SELECT n FROM c"  # noqa: S608
    )


def test_truncation(tools: SqliteTools) -> None:
    sc = tools.call("read_query", {"query": _series(MAX_ROWS + 5)}).structured_content
    assert sc is not None
    assert sc["truncated"] is True
    assert sc["row_count"] == MAX_ROWS == len(sc["rows"])
    exact = tools.call("read_query", {"query": _series(MAX_ROWS)}).structured_content
    assert exact is not None
    assert exact["truncated"] is False
    assert exact["row_count"] == MAX_ROWS


@pytest.mark.parametrize(
    "query",
    [
        "INSERT INTO users(name) VALUES ('x')",
        "UPDATE users SET name='x'",
        "DELETE FROM users",
        "CREATE TABLE q(a)",
        "DROP TABLE users",
        "PRAGMA table_info(users)",
        "ATTACH DATABASE ':memory:' AS m",
        "SELECT 1; SELECT 2",
        "SELECT 1; DROP TABLE users",
        "BEGIN",
        "VACUUM",
        "SELECT load_extension('x')",
    ],
)
def test_read_query_denies(tools: SqliteTools, query: str) -> None:
    res = tools.call("read_query", {"query": query})
    assert res.is_error, query
    assert "Traceback" not in text(res)
    assert str(tools.db_path) not in text(res)
    assert rows(tools, "SELECT count(*) FROM users") == [[2]]


def test_write_query_roundtrip(tools: SqliteTools) -> None:
    insert = {"query": "INSERT INTO users(name) VALUES (?)", "params": ["cy"]}
    ins = tools.call("write_query", insert)
    assert ins.structured_content == {"rows_affected": 1}
    upd = tools.call("write_query", {"query": "UPDATE users SET name='z'"})
    assert upd.structured_content == {"rows_affected": 3}
    dele = tools.call("write_query", {"query": "DELETE FROM users WHERE id=1"})
    assert dele.structured_content == {"rows_affected": 1}
    assert tools.call("write_query", {"query": "CREATE TABLE n(a)"}).structured_content == {
        "rows_affected": 0
    }
    assert not tools.call("write_query", {"query": "CREATE INDEX i ON n(a)"}).is_error
    assert rows(tools, "SELECT name FROM users ORDER BY id") == [["z"], ["z"]]


@pytest.mark.parametrize(
    "query",
    [
        "DROP TABLE users",
        "ALTER TABLE users ADD COLUMN c",
        "ALTER TABLE users RENAME TO u2",
        "PRAGMA writable_schema=1",
        "PRAGMA journal_mode=DELETE",
        "ATTACH DATABASE ':memory:' AS m",
        "DETACH DATABASE m",
        "CREATE VIEW v AS SELECT 1",
        "CREATE TRIGGER tr AFTER INSERT ON users BEGIN SELECT 1; END",
        "CREATE TEMP TABLE tt(a)",
        "CREATE VIRTUAL TABLE vt USING fts5(a)",
        "INSERT INTO users(name) VALUES ('a'); DROP TABLE users",
        "DROP INDEX nothing",
        "UPDATE sqlite_master SET name='x'",
        "COMMIT",
        "VACUUM",
        "ANALYZE",
        "SELECT load_extension('x')",
    ],
)
def test_write_query_denies(tools: SqliteTools, query: str) -> None:
    res = tools.call("write_query", {"query": query})
    assert res.is_error, query
    assert tools.list_tables({})["tables"] == ["a_first", "users"]
    assert rows(tools, "SELECT count(*) FROM users") == [[2]]


def test_rollback_on_error(tools: SqliteTools) -> None:
    res = tools.call("write_query", {"query": "INSERT INTO users(name) VALUES ('ok'),(NULL)"})
    assert res.is_error
    assert rows(tools, "SELECT count(*) FROM users") == [[2]]


def test_injection_via_params(tools: SqliteTools) -> None:
    evil = "x'); DROP TABLE users; --"
    tools.call("write_query", {"query": "INSERT INTO users(name) VALUES (?)", "params": [evil]})
    res = tools.call(
        "read_query", {"query": "SELECT name FROM users WHERE name=?", "params": [evil]}
    )
    assert res.structured_content is not None
    assert res.structured_content["rows"] == [[evil]]
    assert "users" in tools.list_tables({})["tables"]


def test_read_only_mode(db: Path) -> None:
    ro = SqliteTools(db, read_only=True)
    assert [t.name for t in ro.tool_defs()] == ["list_tables", "describe_table", "read_query"]
    assert ro.call("write_query", {"query": "DELETE FROM users"}).is_error
    assert rows(ro, "SELECT count(*) FROM users") == [[2]]
    with pytest.raises(ToolError, match="read-only"):
        ro.write_query({"query": "DELETE FROM users"})
    rw = SqliteTools(db, read_only=False)
    assert [t.name for t in rw.tool_defs()][-1] == "write_query"


@pytest.mark.parametrize(
    "args",
    [
        {},
        {"query": ""},
        {"query": 1},
        {"query": "SELECT 1", "params": "x"},
        {"query": "SELECT 1", "params": [[1]]},
    ],
)
def test_bad_arguments(tools: SqliteTools, args: dict[str, Any]) -> None:
    assert tools.call("read_query", args).is_error


def test_unknown_tool_and_none_args(tools: SqliteTools) -> None:
    assert tools.call("nope", None).is_error
    assert not tools.call("list_tables", None).is_error


def test_syntax_error_is_short(tools: SqliteTools) -> None:
    res = tools.call("read_query", {"query": "SELEKT"})
    assert res.is_error
    assert text(res).startswith("sqlite error:")


def test_non_database_file_errors(tmp_path: Path) -> None:
    bad = tmp_path / "bad.db"
    bad.write_text("this is not a database" * 100)
    t = SqliteTools(bad, read_only=False)
    assert t.call("list_tables", {}).is_error
    assert t.call("describe_table", {"table": "x"}).is_error
    assert t.call("read_query", {"query": "SELECT 1 FROM x"}).is_error


def test_cannot_open_database(tmp_path: Path) -> None:
    res = SqliteTools(tmp_path / "gone.db", read_only=False).call("list_tables", {})
    assert res.is_error
    assert text(res) == "cannot open database"


def test_validate_db_path(tmp_path: Path, db: Path) -> None:
    assert validate_db_path(str(db)) == db.resolve()
    with pytest.raises(ValueError, match="existing"):
        validate_db_path(str(tmp_path / "missing.db"))
    with pytest.raises(ValueError, match="existing"):
        validate_db_path(str(tmp_path))
    link = tmp_path / "link.db"
    link.symlink_to(db)
    with pytest.raises(ValueError, match="symlink"):
        validate_db_path(str(link))


def test_main_rejects_bad_path(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["--db", str(tmp_path / "missing.db")])


def test_main_serves(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[Path, bool]] = []

    async def fake_serve(p: Path, ro: bool) -> None:
        seen.append((p, ro))

    monkeypatch.setattr("mcp_chaos.servers.sqlite_server.serve", fake_serve)
    main(["--db", str(db), "--read-only"])
    assert seen == [(db.resolve(), True)]


@pytest.mark.anyio
async def test_in_process_roundtrip(db: Path) -> None:
    with anyio.fail_after(5):
        async with Client(build_server(db, read_only=True)) as client:
            names = [t.name for t in (await client.list_tools()).tools]
            assert names == ["list_tables", "describe_table", "read_query"]
            r = await client.call_tool("read_query", {"query": "SELECT 1"})
            assert r.structured_content is not None
            assert r.structured_content["rows"] == [[1]]


@pytest.mark.anyio
async def test_infinite_recursive_cte_is_aborted(tools: SqliteTools) -> None:
    q = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) SELECT count(*) FROM c"
    with anyio.fail_after(8):
        res = await anyio.to_thread.run_sync(tools.call, "read_query", {"query": q})
    assert res.is_error
    assert text(res) == "query exceeded resource limits"


def test_wall_time_limit(tools: SqliteTools, monkeypatch: pytest.MonkeyPatch) -> None:
    import mcp_chaos.servers.sqlite_server as m

    monkeypatch.setattr(m, "MAX_WALL_S", -1.0)
    res = tools.call(
        "read_query",
        {
            "query": "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c WHERE x<100000) "
            "SELECT count(*) FROM c"
        },
    )
    assert res.is_error and text(res) == "query exceeded resource limits"


@pytest.mark.parametrize("fn", ["zeroblob(1000000000)", "randomblob(1000000000)"])
def test_huge_blob_rejected(tools: SqliteTools, fn: str) -> None:
    res = tools.call("read_query", {"query": f"SELECT length({fn})"})
    assert res.is_error
    assert "too big" in text(res)


def test_large_result_truncated_by_size(tools: SqliteTools) -> None:
    res = tools.call(
        "read_query",
        {
            "query": "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c WHERE x<50) "
            "SELECT hex(zeroblob(100000)) FROM c"
        },
    )
    assert not res.is_error, text(res)
    sc = res.structured_content
    assert sc is not None and sc["truncated"] is True
    assert 0 < sc["row_count"] < 50
    assert sum(len(r[0]) for r in sc["rows"]) <= 1_000_000
