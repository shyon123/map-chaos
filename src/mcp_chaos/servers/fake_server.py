"""Deterministic, scriptable fake MCP server for integration tests.

No time, randomness, or I/O beyond reading the script at startup. Logs only to stderr.
Run: ``python -m mcp_chaos.servers.fake_server --script PATH``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import anyio
import mcp_types as types
from mcp.server.lowlevel.server import Server
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import MCPError
from pydantic import BaseModel, ConfigDict, Field

_INVALID_PARAMS = -32602


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Response(_Model):
    text: str | None = None
    structured: dict[str, Any] | None = None
    is_error: bool = False
    echo_args: bool = False


class ToolScript(_Model):
    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=lambda: {"type": "object"})
    output_schema: dict[str, Any] | None = None
    responses: list[Response] = Field(min_length=1)
    cycle: bool = True


class FakeScript(_Model):
    server_name: str
    tools: list[ToolScript]


def load_script(path: Path) -> FakeScript:
    return FakeScript.model_validate_json(path.read_text(encoding="utf-8"))


class FakeState:
    """Per-tool call counters plus pure response selection."""

    def __init__(self, script: FakeScript) -> None:
        self.script = script
        self._tools = {t.name: t for t in script.tools}
        self._counts: dict[str, int] = dict.fromkeys(self._tools, 0)

    def list_tools(self) -> list[types.Tool]:
        return [
            types.Tool(
                name=t.name,
                description=t.description,
                input_schema=t.input_schema,
                output_schema=t.output_schema,
            )
            for t in self.script.tools
        ]

    def call(self, name: str, arguments: dict[str, Any] | None) -> types.CallToolResult:
        tool = self._tools.get(name)
        if tool is None:
            raise MCPError(_INVALID_PARAMS, f"Unknown tool: {name}")
        n = self._counts[name]
        self._counts[name] = n + 1
        idx = n % len(tool.responses) if tool.cycle else min(n, len(tool.responses) - 1)
        resp = tool.responses[idx]
        text = resp.text
        if resp.echo_args:
            text = json.dumps(arguments or {}, sort_keys=True)
        elif text is None and resp.structured is not None:
            text = json.dumps(resp.structured, sort_keys=True, separators=(",", ":"))
        content: list[types.ContentBlock] = []
        if text is not None:
            content.append(types.TextContent(text=text))
        return types.CallToolResult(
            content=content, structured_content=resp.structured, is_error=resp.is_error
        )


def build_server(script: FakeScript) -> Server[Any]:
    state = FakeState(script)

    async def on_list_tools(
        _ctx: Any, _params: types.PaginatedRequestParams | None
    ) -> types.ListToolsResult:
        return types.ListToolsResult(tools=state.list_tools())

    async def on_call_tool(_ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        return state.call(params.name, params.arguments)

    return Server(script.server_name, on_list_tools=on_list_tools, on_call_tool=on_call_tool)


async def serve(script: FakeScript) -> None:
    server = build_server(script)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Deterministic fake MCP server")
    parser.add_argument("--script", required=True, type=Path)
    args = parser.parse_args(argv)
    script = load_script(args.script)
    sys.stderr.write(f"fake_server: serving {script.server_name!r}\n")
    anyio.run(serve, script)


if __name__ == "__main__":
    main()
