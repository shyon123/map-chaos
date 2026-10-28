"""Thin isolation layer over the beta ``langchain.mcp`` API.

Error behaviour is the framework default on purpose: an MCP ``isError`` result becomes an
error ToolMessage; transport failures propagate as exceptions. No retries, no catching.
"""

from __future__ import annotations

import warnings
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from langchain_core._api import LangChainBetaWarning
from langchain_core.tools import BaseTool

from mcp_chaos.log import get_logger
from mcp_chaos.targets.workers import WorkerSpec

with warnings.catch_warnings():
    warnings.simplefilter("ignore", LangChainBetaWarning)
    from fastmcp.client.transports import StdioTransport
    from langchain.mcp import MCPAdapter

log = get_logger(__name__)


@asynccontextmanager
async def worker_tools(worker: WorkerSpec, cmd: list[str]) -> AsyncGenerator[list[BaseTool]]:
    """Launch one stdio MCP server with ``cmd`` and yield its tools as LangChain tools."""
    transport = StdioTransport(command=cmd[0], args=cmd[1:], keep_alive=False)
    async with MCPAdapter(transport) as adapter:
        tools = await adapter.list_tools()
        log.info("worker_tools_loaded", worker=worker.name, tools=len(tools))
        yield tools
