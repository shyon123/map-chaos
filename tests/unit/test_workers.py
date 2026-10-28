import sys
from pathlib import Path

import pytest

from mcp_chaos.targets.workers import (
    DATA,
    FILES,
    FS_SERVER_VERSION,
    VCS,
    VCS_SERVER_VERSION,
    WORKERS,
    WorkerSpec,
    proxy_cmd,
    upstream_cmd,
)

SCRATCH = Path("/scratch")


def test_registry() -> None:
    assert set(WORKERS) == {"files", "data", "vcs"}
    assert all(w.allow for w in WORKERS.values())
    assert DATA.allow == {"list_tables", "describe_table", "read_query", "write_query"}


def test_allowlists_least_privilege() -> None:
    assert {"write_file", "edit_file", "create_directory", "move_file"} <= FILES.allow
    assert all(not t.startswith("git_") for t in FILES.allow)
    for banned in ("git_reset", "git_checkout", "git_branch", "git_create_branch"):
        assert banned not in VCS.allow
    assert {"git_status", "git_log", "git_diff", "git_show", "git_add", "git_commit"} <= VCS.allow


def test_upstream_local() -> None:
    assert upstream_cmd(FILES, env="local", scratch=SCRATCH) == [
        "npx",
        "-y",
        f"@modelcontextprotocol/server-filesystem@{FS_SERVER_VERSION}",
        "/scratch",
    ]
    assert upstream_cmd(VCS, env="local", scratch=SCRATCH) == [
        "uvx",
        f"mcp-server-git=={VCS_SERVER_VERSION}",
        "--repository",
        "/scratch",
    ]
    assert upstream_cmd(DATA, env="local", scratch=SCRATCH) == [
        sys.executable,
        "-m",
        "mcp_chaos.servers.sqlite_server",
        "--db",
        "/scratch/db.sqlite",
    ]


def test_upstream_docker() -> None:
    files = upstream_cmd(FILES, env="docker", scratch=SCRATCH)
    assert files[0] == "node"
    assert files[-1] == "/scratch"
    assert upstream_cmd(VCS, env="docker", scratch=SCRATCH) == [
        "mcp-server-git",
        "--repository",
        "/scratch",
    ]
    assert upstream_cmd(DATA, env="docker", scratch=SCRATCH)[-1] == "/scratch/db.sqlite"


@pytest.mark.parametrize("worker", list(WORKERS.values()))
def test_proxy_cmd(worker: WorkerSpec, tmp_path: Path) -> None:
    up = ["server", "--x"]
    cmd = proxy_cmd(
        worker, run_id="r1", plan_path=tmp_path / "plan.json", trace_dir=tmp_path, upstream=up
    )
    sep = cmd.index("--")
    assert cmd[sep + 1 :] == up
    head = cmd[:sep]
    assert head[:4] == [sys.executable, "-m", "mcp_chaos.bench.cli", "proxy"]

    def val(flag: str) -> str:
        return head[head.index(flag) + 1]

    assert val("--agent") == worker.name
    assert val("--server") == worker.server_id
    assert val("--run-id") == "r1"
    assert val("--plan") == str(tmp_path / "plan.json")
    assert val("--trace-out") == str(tmp_path / f"{worker.name}-{worker.server_id}.jsonl")
    assert set(val("--allow").split(",")) == worker.allow
