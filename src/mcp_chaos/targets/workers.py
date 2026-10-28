"""Framework-neutral worker definitions: allowlists and server/proxy command lines.

Real tool names (discovered by launching the servers, 2026-10-28):
  @modelcontextprotocol/server-filesystem 2.0.1: create_directory, directory_tree, edit_file,
    get_file_info, list_allowed_directories, list_directory, list_directory_with_sizes,
    move_file, read_file, read_media_file, read_multiple_files, read_text_file, search_files,
    write_file
  mcp-server-git 2026.10.19: git_add, git_branch, git_checkout, git_commit, git_create_branch,
    git_diff, git_diff_staged, git_diff_unstaged, git_log, git_reset, git_show, git_status
  sqlite_server (ours): list_tables, describe_table, read_query, write_query
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

FS_SERVER_VERSION = "2.0.1"
VCS_SERVER_VERSION = "2026.10.19"

WorkerName = Literal["files", "data", "vcs"]


class WorkerSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: WorkerName
    server_id: str
    allow: frozenset[str]


FILES = WorkerSpec(
    name="files",
    server_id="fs",
    allow=frozenset(
        {
            "read_file",
            "read_text_file",
            "read_media_file",
            "read_multiple_files",
            "list_directory",
            "list_directory_with_sizes",
            "directory_tree",
            "search_files",
            "get_file_info",
            "list_allowed_directories",
            "write_file",
            "edit_file",
            "create_directory",
            "move_file",
        }
    ),
)
DATA = WorkerSpec(
    name="data",
    server_id="sqlite",
    allow=frozenset({"list_tables", "describe_table", "read_query", "write_query"}),
)
VCS = WorkerSpec(
    name="vcs",
    server_id="git",
    allow=frozenset(
        {
            "git_status",
            "git_log",
            "git_diff",
            "git_diff_staged",
            "git_diff_unstaged",
            "git_show",
            "git_add",
            "git_commit",
        }
    ),
)
WORKERS: dict[WorkerName, WorkerSpec] = {w.name: w for w in (FILES, DATA, VCS)}


def upstream_cmd(
    worker: WorkerSpec, *, env: Literal["docker", "local"], scratch: Path
) -> list[str]:
    """Command that launches the worker's real MCP server rooted at ``scratch``."""
    s = str(scratch)
    if worker.name == "files":
        if env == "docker":
            return [
                "node",
                "/opt/mcp/node_modules/@modelcontextprotocol/server-filesystem/dist/index.js",
                s,
            ]
        return ["npx", "-y", f"@modelcontextprotocol/server-filesystem@{FS_SERVER_VERSION}", s]
    if worker.name == "data":
        return [
            sys.executable,
            "-m",
            "mcp_chaos.servers.sqlite_server",
            "--db",
            str(scratch / "db.sqlite"),
        ]
    if env == "docker":
        return ["mcp-server-git", "--repository", s]
    return ["uvx", f"mcp-server-git=={VCS_SERVER_VERSION}", "--repository", s]


def proxy_cmd(
    worker: WorkerSpec,
    *,
    run_id: str,
    plan_path: Path,
    trace_dir: Path,
    upstream: list[str],
) -> list[str]:
    """Fault-proxy command wrapping ``upstream`` for this worker."""
    return [
        sys.executable,
        "-m",
        "mcp_chaos.bench.cli",
        "proxy",
        "--agent",
        worker.name,
        "--server",
        worker.server_id,
        "--plan",
        str(plan_path),
        "--run-id",
        run_id,
        "--allow",
        ",".join(sorted(worker.allow)),
        "--trace-out",
        str(trace_dir / f"{worker.name}-{worker.server_id}.jsonl"),
        "--",
        *upstream,
    ]
