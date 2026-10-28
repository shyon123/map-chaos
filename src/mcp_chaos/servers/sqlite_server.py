"""Our own SQLite MCP server (ADR 0004; the official one is archived).

Tools: ``list_tables``, ``describe_table``, ``read_query`` and (unless ``--read-only``)
``write_query``. Statement classes are enforced with the ``sqlite3`` authorizer, never by
string matching, and exactly one statement is allowed per call. Every call opens its own
connection; each write call is its own transaction (commit on success, rollback on error).

Determinism: no time or randomness. Row order is whatever the SQL says; queries without
``ORDER BY`` are NOT order-guaranteed. Logs go to stderr only.
Run: ``python -m mcp_chaos.servers.sqlite_server --db PATH [--read-only]``.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import anyio
import mcp_types as types
from mcp.server.lowlevel.server import Server
from mcp.server.stdio import stdio_server

MAX_ROWS = 1000
MAX_RESULT_BYTES = 1_000_000  # sum of cell sizes returned
MAX_BLOB_BYTES = 1_000_000  # SQLITE_LIMIT_LENGTH: bounds zeroblob/randomblob/replace/printf
MAX_VM_STEPS = 5_000_000
MAX_WALL_S = 5.0  # only bounds a query; never influences its result
_PROGRESS_EVERY = 1000

_READ_ACTIONS = frozenset(
    {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        sqlite3.SQLITE_RECURSIVE,  # recursive CTEs
    }
)
_WRITE_ACTIONS = _READ_ACTIONS | frozenset(
    {
        sqlite3.SQLITE_INSERT,
        sqlite3.SQLITE_UPDATE,
        sqlite3.SQLITE_DELETE,
        sqlite3.SQLITE_CREATE_TABLE,
        sqlite3.SQLITE_CREATE_INDEX,
        sqlite3.SQLITE_REINDEX,  # CREATE INDEX populates via this; bare REINDEX is a harmless no-op
    }
)

_JSON_SCALAR: dict[str, Any] = {"type": ["string", "number", "boolean", "null"]}
_QUERY_INPUT: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Exactly one SQL statement; use ? placeholders.",
        },
        "params": {"type": "array", "items": _JSON_SCALAR, "default": []},
    },
    "required": ["query"],
}


class ToolError(Exception):
    """A short, user-safe error message surfaced as ``isError``."""


def _authorizer(allowed: frozenset[int]) -> Callable[[int, Any, Any, Any, Any], int]:
    def check(action: int, _a: Any, _b: Any, _c: Any, _d: Any) -> int:
        return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY

    return check


def _cell_bytes(v: Any) -> int:
    if isinstance(v, (str, bytes)):
        return len(v)
    return 8


def _limiter() -> tuple[Callable[[], int], list[bool]]:
    """Progress handler aborting after MAX_VM_STEPS or MAX_WALL_S; flag set when it fires."""
    deadline = time.monotonic() + MAX_WALL_S
    steps = 0
    fired = [False]

    def handler() -> int:
        nonlocal steps
        steps += _PROGRESS_EVERY
        if steps > MAX_VM_STEPS or time.monotonic() > deadline:
            fired[0] = True
            return 1
        return 0

    return handler, fired


def _jsonable(value: Any) -> Any:
    return value.hex() if isinstance(value, bytes) else value


def _short(e: sqlite3.Error) -> str:
    lines = str(e).splitlines()
    return f"sqlite error: {lines[0] if lines else type(e).__name__}"


def validate_db_path(raw: str) -> Path:
    p = Path(raw)
    if p.is_symlink():
        raise ValueError("database path must not be a symlink")
    if not p.is_file():
        raise ValueError("database path must be an existing file")
    return p.resolve()


class SqliteTools:
    def __init__(self, db_path: Path, read_only: bool) -> None:
        self.db_path = db_path
        self.read_only = read_only

    # ----- plumbing -----

    def _connect(self, *, write: bool) -> sqlite3.Connection:
        mode = "rw" if write else "ro"
        try:
            conn = sqlite3.connect(
                f"{self.db_path.as_uri()}?mode={mode}", uri=True, isolation_level=None
            )
        except sqlite3.Error as e:
            raise ToolError("cannot open database") from e
        conn.enable_load_extension(False)
        return conn

    @staticmethod
    def _query_args(arguments: dict[str, Any]) -> tuple[str, list[Any]]:
        query = arguments.get("query")
        raw: Any = arguments.get("params", [])
        if not isinstance(query, str) or not query.strip():
            raise ToolError("'query' must be a non-empty string")
        if not isinstance(raw, list):
            raise ToolError("'params' must be a list of scalars")
        params: list[Any] = list(cast("list[Any]", raw))
        if not all(p is None or isinstance(p, (str, int, float, bool)) for p in params):
            raise ToolError("'params' must be a list of scalars")
        return query, params

    def _run(
        self, query: str, params: list[Any], allowed: frozenset[int], *, write: bool
    ) -> tuple[list[str], list[Any], int, bool]:
        """Run exactly one statement under the authorizer in its own transaction.

        Returns (columns, rows, rows_affected, size_truncated).
        """
        conn = self._connect(write=write)
        handler, fired = _limiter()
        try:
            conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, MAX_BLOB_BYTES)
            conn.execute("BEGIN")
            conn.set_authorizer(_authorizer(allowed))
            conn.set_progress_handler(handler, _PROGRESS_EVERY)
            rows: list[Any] = []
            size_truncated = False
            try:
                # Connection.execute rejects multiple statements.
                cur = conn.execute(query, params)
                columns = [d[0] for d in cur.description or []]
                total = 0
                while columns and len(rows) <= MAX_ROWS:
                    row = cur.fetchone()
                    if row is None:
                        break
                    total += sum(_cell_bytes(v) for v in row)
                    if total > MAX_RESULT_BYTES:
                        size_truncated = True
                        break
                    rows.append(row)
                affected = max(cur.rowcount, 0)
            finally:
                conn.set_authorizer(None)  # COMMIT/ROLLBACK are authorizer-checked too
                conn.set_progress_handler(None, 0)
            conn.execute("COMMIT")
            return columns, rows, affected, size_truncated
        except sqlite3.Error as e:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            if fired[0]:
                raise ToolError("query exceeded resource limits") from e
            raise ToolError(_short(e)) from e
        finally:
            conn.close()

    # ----- tools -----

    def list_tables(self, _arguments: dict[str, Any]) -> dict[str, Any]:
        conn = self._connect(write=False)
        try:
            names = [
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name NOT LIKE 'sqlite\\_%' ESCAPE '\\' ORDER BY name"
                )
            ]
        except sqlite3.Error as e:
            raise ToolError(_short(e)) from e
        finally:
            conn.close()
        return {"tables": names}

    def describe_table(self, arguments: dict[str, Any]) -> dict[str, Any]:
        table = arguments.get("table")
        if not isinstance(table, str) or not table:
            raise ToolError("'table' must be a non-empty string")
        conn = self._connect(write=False)
        try:
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? "
                "AND name NOT LIKE 'sqlite\\_%' ESCAPE '\\'",
                (table,),
            ).fetchone()
            if exists is None:
                raise ToolError(f"unknown table: {table}")
            info = conn.execute(
                'SELECT name, type, "notnull", pk FROM pragma_table_info(?)', (table,)
            ).fetchall()
        except sqlite3.Error as e:
            raise ToolError(_short(e)) from e
        finally:
            conn.close()
        cols = [{"name": n, "type": t, "notnull": bool(nn), "pk": int(pk)} for n, t, nn, pk in info]
        return {"table": table, "columns": cols}

    def read_query(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query, params = self._query_args(arguments)
        columns, rows, _, by_size = self._run(query, params, _READ_ACTIONS, write=False)
        truncated = by_size or len(rows) > MAX_ROWS
        rows = rows[:MAX_ROWS]
        return {
            "columns": columns,
            "rows": [[_jsonable(v) for v in r] for r in rows],
            "row_count": len(rows),
            "truncated": truncated,
        }

    def write_query(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.read_only:
            raise ToolError("write_query is unavailable in read-only mode")
        query, params = self._query_args(arguments)
        _, _, affected, _ = self._run(query, params, _WRITE_ACTIONS, write=True)
        return {"rows_affected": affected}

    # ----- MCP surface -----

    def tool_defs(self) -> list[types.Tool]:
        tools = [
            types.Tool(
                name="list_tables",
                description="List user tables, sorted by name.",
                input_schema={"type": "object", "properties": {}},
                output_schema={
                    "type": "object",
                    "properties": {"tables": {"type": "array", "items": {"type": "string"}}},
                    "required": ["tables"],
                },
            ),
            types.Tool(
                name="describe_table",
                description="Describe the columns of a table.",
                input_schema={
                    "type": "object",
                    "properties": {"table": {"type": "string"}},
                    "required": ["table"],
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "table": {"type": "string"},
                        "columns": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "type": {"type": "string"},
                                    "notnull": {"type": "boolean"},
                                    "pk": {"type": "integer"},
                                },
                                "required": ["name", "type", "notnull", "pk"],
                            },
                        },
                    },
                    "required": ["table", "columns"],
                },
            ),
            types.Tool(
                name="read_query",
                description=(
                    f"Run one SELECT statement (max {MAX_ROWS} rows). "
                    "Without ORDER BY, row order is not guaranteed."
                ),
                input_schema=_QUERY_INPUT,
                output_schema={
                    "type": "object",
                    "properties": {
                        "columns": {"type": "array", "items": {"type": "string"}},
                        "rows": {"type": "array", "items": {"type": "array"}},
                        "row_count": {"type": "integer"},
                        "truncated": {"type": "boolean"},
                    },
                    "required": ["columns", "rows", "row_count", "truncated"],
                },
            ),
        ]
        if not self.read_only:
            tools.append(
                types.Tool(
                    name="write_query",
                    description=(
                        "Run one INSERT/UPDATE/DELETE/CREATE TABLE/CREATE INDEX statement "
                        "as its own transaction."
                    ),
                    input_schema=_QUERY_INPUT,
                    output_schema={
                        "type": "object",
                        "properties": {"rows_affected": {"type": "integer"}},
                        "required": ["rows_affected"],
                    },
                )
            )
        return tools

    def call(self, name: str, arguments: dict[str, Any] | None) -> types.CallToolResult:
        handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
            "list_tables": self.list_tables,
            "describe_table": self.describe_table,
            "read_query": self.read_query,
        }
        if not self.read_only:
            handlers["write_query"] = self.write_query
        handler = handlers.get(name)
        try:
            if handler is None:
                raise ToolError(f"unknown tool: {name}")
            result = handler(arguments or {})
        except ToolError as e:
            return types.CallToolResult(content=[types.TextContent(text=str(e))], is_error=True)
        text = json.dumps(result, sort_keys=True, separators=(",", ":"))
        return types.CallToolResult(
            content=[types.TextContent(text=text)], structured_content=result
        )


def build_server(db_path: Path, read_only: bool = False) -> Server[Any]:
    tools = SqliteTools(db_path, read_only)

    async def on_list_tools(
        _ctx: Any, _params: types.PaginatedRequestParams | None
    ) -> types.ListToolsResult:
        return types.ListToolsResult(tools=tools.tool_defs())

    async def on_call_tool(_ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        return tools.call(params.name, params.arguments)

    return Server("sqlite", on_list_tools=on_list_tools, on_call_tool=on_call_tool)


async def serve(db_path: Path, read_only: bool) -> None:
    server = build_server(db_path, read_only)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="SQLite MCP server")
    parser.add_argument("--db", required=True)
    parser.add_argument("--read-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        db = validate_db_path(args.db)
    except ValueError as e:
        parser.error(str(e))
    sys.stderr.write(f"sqlite_server: serving (read_only={args.read_only})\n")
    anyio.run(serve, db, args.read_only)


if __name__ == "__main__":
    main()
