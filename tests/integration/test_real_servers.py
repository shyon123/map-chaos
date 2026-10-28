"""Allowlists match the real filesystem / VCS servers' tool names.

Live-marked: ``npx`` / ``uvx`` may fetch packages from the network on first use.
"""

import shutil
from pathlib import Path

import anyio
import pytest
from mcp import Client, StdioServerParameters

from mcp_chaos.targets.workers import FILES, VCS, WorkerSpec, upstream_cmd

pytestmark = pytest.mark.live


async def list_tool_names(worker: WorkerSpec, scratch: Path) -> set[str]:
    cmd = upstream_cmd(worker, env="local", scratch=scratch)
    with anyio.fail_after(120):
        async with Client(StdioServerParameters(command=cmd[0], args=cmd[1:])) as client:
            return {t.name for t in (await client.list_tools()).tools}


@pytest.mark.skipif(shutil.which("npx") is None, reason="npx not installed")
@pytest.mark.anyio
async def test_filesystem_server_has_allowlisted_tools(tmp_path: Path) -> None:
    names = await list_tool_names(FILES, tmp_path)
    assert FILES.allow <= names


@pytest.mark.skipif(shutil.which("uvx") is None, reason="uvx not installed")
@pytest.mark.anyio
async def test_vcs_server_has_allowlisted_tools(tmp_path: Path) -> None:
    await anyio.run_process(["git", "init", "-q", str(tmp_path)])
    names = await list_tool_names(VCS, tmp_path)
    assert VCS.allow <= names
    assert {"git_reset", "git_checkout"} <= names - VCS.allow
