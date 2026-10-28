"""Run one benchmark run end to end (runner-contract.md)."""

from __future__ import annotations

import contextlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from mcp_chaos.bench import sandbox
from mcp_chaos.bench.gateway_host import GatewayHost
from mcp_chaos.bench.snapshot import diff, snapshot
from mcp_chaos.log import get_logger
from mcp_chaos.schemas.results import CheckSummary, RunResult, SideEffects
from mcp_chaos.schemas.runs import RunConfig
from mcp_chaos.schemas.trace import FaultApplied, TraceEvent
from mcp_chaos.targets import tasks, workers
from mcp_chaos.targets.entry import FAKE_MODELS_ENV, UPSTREAM_ENV, EntryResult
from mcp_chaos.targets.fixtures import build_fixture

log = get_logger(__name__)

SandboxKind = Literal["docker", "local"]
DUMMY_KEY = "dummy-not-a-key"
REPO_ROOT = Path(__file__).resolve().parents[3]
RUNS_FILE = "runs.jsonl"
_PACKAGES = ("mcp-chaos", "langgraph", "langchain", "langchain-anthropic", "mcp", "pydantic")


class RunnerError(RuntimeError):
    pass


def load_runs(path: Path) -> dict[str, RunResult]:
    """Read ``runs.jsonl``; the last line per run_id wins (re-runs supersede earlier lines)."""
    out: dict[str, RunResult] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = RunResult.model_validate_json(line)
        except ValidationError:
            continue
        out[r.run_id] = r
    return out


def _existing(results_dir: Path, cfg: RunConfig) -> RunResult | None:
    """Stored result to reuse, or None to (re-)run. Config drift raises."""
    r = load_runs(results_dir / RUNS_FILE).get(cfg.run_id)
    if r is None:
        return None
    if r.config != cfg:
        raise RunnerError(
            f"run_id {cfg.run_id} collision or config drift: stored config differs from requested"
        )
    return None if r.status == "harness_error" else r


def _git_sha() -> str:
    try:
        p = subprocess.run(  # noqa: S603
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short=12", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return p.stdout.strip() if p.returncode == 0 and p.stdout.strip() else "unknown"


def _versions(sandbox_kind: SandboxKind, image_tag: str) -> dict[str, str]:
    v: dict[str, str] = {"sandbox": sandbox_kind, "git_sha": _git_sha()}
    v["python"] = ".".join(map(str, sys.version_info[:3]))
    for pkg in _PACKAGES:
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            v[pkg] = importlib.metadata.version(pkg)
    v["fs_server"] = f"@modelcontextprotocol/server-filesystem@{workers.FS_SERVER_VERSION}"
    v["vcs_server"] = f"mcp-server-git=={workers.VCS_SERVER_VERSION}"
    if sandbox_kind == "docker":
        try:
            v["image_id"] = sandbox.image_id(image_tag)
        except sandbox.SandboxError:
            v["image_id"] = "unknown"
    return v


def _read_traces(out: Path) -> list[FaultApplied]:
    events: list[TraceEvent] = []
    for f in sorted((out / "traces").glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                with contextlib.suppress(ValidationError):
                    events.append(TraceEvent.model_validate_json(line))
    events.sort(key=lambda e: (e.t_ms, e.agent, e.seq))
    return [e.fault for e in events if e.fault is not None]


def _chmod_tree(path: Path, mode: int) -> None:
    for p in [path, *path.rglob("*")]:
        with contextlib.suppress(OSError):
            if not p.is_symlink():
                p.chmod(mode | (0o111 if p.is_dir() else 0))


def _cleanup(path: Path) -> None:
    _chmod_tree(path, 0o777)
    shutil.rmtree(path, ignore_errors=True)


def _run_local(
    run: Path, cfg: RunConfig, gw: GatewayHost
) -> tuple[subprocess.CompletedProcess[str] | None, bool]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("ANTHROPIC_")}
    env.update(
        ANTHROPIC_BASE_URL=gw.base_url(cfg.run_id),
        ANTHROPIC_API_KEY=DUMMY_KEY,
        MCP_CHAOS_SCRATCH=str(run / "scratch"),
        **{UPSTREAM_ENV: "local"},
    )
    argv = [
        sys.executable,
        "-m",
        "mcp_chaos.targets.entry",
        "--config",
        str(run / "in" / "config.json"),
        "--out",
        str(run / "out"),
    ]
    try:
        p = subprocess.run(  # noqa: S603
            argv,
            env=env,
            capture_output=True,
            text=True,
            timeout=cfg.timeout_s + sandbox.KILL_GRACE_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, True
    return p, False


def _run_docker(
    run: Path, cfg: RunConfig, gw: GatewayHost, image_tag: str
) -> tuple[int | None, bool, str]:
    env = {
        "ANTHROPIC_BASE_URL": f"http://{sandbox.RELAY_ALIAS}:{sandbox.RELAY_PORT}/r/{cfg.run_id}",
        "ANTHROPIC_API_KEY": DUMMY_KEY,
        "MCP_CHAOS_SCRATCH": "/scratch",
    }
    # Bind-mounted trees are owned by the host user; the container runs as an unrelated UID.
    _chmod_tree(run / "scratch", 0o777)
    _chmod_tree(run / "out", 0o777)
    r = sandbox.run_container(
        image=image_tag,
        scratch=run / "scratch",
        in_dir=run / "in",
        out_dir=run / "out",
        env=env,
        timeout_s=cfg.timeout_s,
    )
    return r.exit_code, r.timed_out, r.stderr_tail


def run_one(
    cfg: RunConfig,
    *,
    sandbox: SandboxKind,
    gw: GatewayHost,
    results_dir: Path,
    image_tag: str = "mcpchaos-target:dev",
    work_root: Path | None = None,
) -> RunResult:
    """Execute ``cfg`` once and append the ``RunResult`` to ``results_dir/runs.jsonl``.

    Resumable: a stored result with an identical config is returned (``harness_error``
    results are re-run and superseded by appending;
    a differing stored config raises ``RunnerError``).
    """
    done = _existing(results_dir, cfg)
    if done is not None:
        log.info("run_skipped", run_id=cfg.run_id)
        return done
    if sandbox == "docker" and os.environ.get(FAKE_MODELS_ENV):
        raise RunnerError(f"{FAKE_MODELS_ENV} is refused in docker mode")

    root = work_root or (Path.cwd() / ".cache" / "runs")
    root.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=f"{cfg.run_id}-", dir=root))
    scratch, in_dir, out = run / "scratch", run / "in", run / "out"
    started = datetime.now(UTC)
    t0 = time.monotonic()
    try:
        in_dir.mkdir()
        (out / "traces").mkdir(parents=True)
        scratch.mkdir()
        build_fixture(cfg.task_id, scratch)
        (in_dir / "config.json").write_text(cfg.model_dump_json(), encoding="utf-8")
        (in_dir / "plan.json").write_text(cfg.fault_plan.model_dump_json(), encoding="utf-8")
        before = snapshot(scratch)
        gw.register_run(cfg.run_id, cfg.seed, cfg.budget_usd)

        timed_out = False
        detail = ""
        if sandbox == "local":
            proc, timed_out = _run_local(run, cfg, gw)
            detail = proc.stderr[-1000:] if proc is not None else ""
        else:
            _, timed_out, detail = _run_docker(run, cfg, gw, image_tag)

        entry: EntryResult | None = None
        with contextlib.suppress(OSError, ValidationError):
            entry = EntryResult.model_validate_json(
                (out / "result.json").read_text(encoding="utf-8")
            )
        after = snapshot(scratch)
        effects: SideEffects = diff(before, after)
        final_answer = entry.final_answer if entry else None

        status: Literal["completed", "agent_error", "timeout", "budget_abort", "harness_error"]
        if gw.budget_aborted(cfg.run_id):
            status = "budget_abort"
        elif timed_out:
            status = "timeout"
        elif entry is None:
            status = "harness_error"
            log.warning("run_harness_error", run_id=cfg.run_id, stderr=detail[-300:])
        else:
            status = entry.status

        check: CheckSummary | None = None
        if status != "harness_error":
            res = tasks.check(cfg.task_id, scratch, final_answer, effects)
            check = CheckSummary(success=res.success, details=res.details)

        result = RunResult(
            run_id=cfg.run_id,
            config=cfg,
            status=status,
            final_answer=final_answer,
            side_effects=effects,
            faults_fired=_read_traces(out),
            check=check,
            usage=gw.usage(cfg.run_id),
            agent_usage=gw.usage(cfg.run_id),
            versions=_versions(sandbox, image_tag),
            started_at=started,
            wall_s=round(time.monotonic() - t0, 3),
        )
        _persist(results_dir, result, out)
        log.info(
            "run_done",
            run_id=cfg.run_id,
            status=status,
            success=check.success if check else None,
            cost_usd=result.usage.cost_usd,
        )
        return result
    finally:
        _cleanup(run)


def _persist(results_dir: Path, result: RunResult, out: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    dest = results_dir / "traces" / result.run_id
    if (out / "traces").is_dir():
        shutil.copytree(out / "traces", dest, dirs_exist_ok=True)
    line = json.dumps(json.loads(result.model_dump_json()), separators=(",", ":")) + "\n"
    with (results_dir / RUNS_FILE).open("a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())
