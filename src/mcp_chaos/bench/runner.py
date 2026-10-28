"""Run one benchmark run end to end (runner-contract.md)."""

from __future__ import annotations

import contextlib
import importlib.metadata
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from mcp_chaos.bench import sandbox
from mcp_chaos.bench.gateway_host import GatewayHost
from mcp_chaos.bench.safe_repo import scratch_usage
from mcp_chaos.bench.snapshot import diff, snapshot
from mcp_chaos.log import get_logger
from mcp_chaos.schemas.results import CheckSummary, RunResult, SideEffects
from mcp_chaos.schemas.runs import RunConfig, derive_run_id
from mcp_chaos.schemas.trace import FaultApplied, TraceEvent
from mcp_chaos.targets import tasks, workers
from mcp_chaos.targets.entry import FAKE_MODELS_ENV, UPSTREAM_ENV, EntryResult
from mcp_chaos.targets.fixtures import build_fixture

log = get_logger(__name__)

SandboxKind = Literal["docker", "local"]
DUMMY_KEY = "dummy-not-a-key"
REPO_ROOT = Path(__file__).resolve().parents[3]
RUNS_FILE = "runs.jsonl"
DEFAULT_SCRATCH_QUOTA_BYTES = 256 * 1024 * 1024
WATCHDOG_INTERVAL_S = 2.0
TERM_GRACE_S = 5.0
QUOTA_ERROR = "scratch_quota_exceeded"
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
    if cfg.run_id != derive_run_id(cfg):
        raise RunnerError(f"run_id {cfg.run_id} does not match the id derived from its config")
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


class _Watchdog:
    """Measures the scratch tree every ``WATCHDOG_INTERVAL_S`` and fires ``kill`` over quota."""

    def __init__(self, scratch: Path, quota: int, kill: Callable[[], None]) -> None:
        self._scratch, self._quota, self._kill = scratch, quota, kill
        self._stop = threading.Event()
        self.tripped = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="scratch-watchdog", daemon=True)

    def _loop(self) -> None:
        while not self._stop.wait(WATCHDOG_INTERVAL_S):
            used, truncated = scratch_usage(self._scratch)
            if truncated or used > self._quota:
                self.tripped.set()
                log.warning("scratch_quota_exceeded", used=used, truncated=truncated)
                self._kill()
                return

    def __enter__(self) -> _Watchdog:
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=30)


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover - a member we cannot signal still exists
        return True
    return True


def _wait_group_gone(proc: subprocess.Popen[bytes], grace_s: float) -> bool:
    deadline = time.monotonic() + grace_s
    while True:
        proc.poll()  # reap the leader: a zombie still counts as a group member
        if not _group_alive(proc.pid):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def _kill_group(proc: subprocess.Popen[bytes]) -> None:
    """TERM the whole process group, wait ``TERM_GRACE_S``, then KILL; raise if members remain."""
    pgid = proc.pid
    proc.poll()
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if not _group_alive(pgid):
            return
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pgid, sig)
        if _wait_group_gone(proc, TERM_GRACE_S):
            return
    raise RunnerError(f"process group {pgid} survived SIGKILL")


def _local_argv(run: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "mcp_chaos.targets.entry",
        "--config",
        str(run / "in" / "config.json"),
        "--out",
        str(run / "out"),
    ]


def _run_local(run: Path, cfg: RunConfig, gw: GatewayHost, quota: int) -> tuple[str, bool, bool]:
    """Returns ``(stderr_tail, timed_out, quota_exceeded)``; no group member survives."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("ANTHROPIC_")}
    env.update(
        ANTHROPIC_BASE_URL=gw.base_url(cfg.run_id),
        ANTHROPIC_API_KEY=DUMMY_KEY,
        MCP_CHAOS_SCRATCH=str(run / "scratch"),
        **{UPSTREAM_ENV: "local"},
    )
    log_path = run / "entry.log"  # a file, not a pipe: stray grandchildren cannot block us
    timed_out = False
    with log_path.open("wb") as logf:
        proc = subprocess.Popen(  # noqa: S603
            _local_argv(run),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=logf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        pgid = proc.pid

        def kill() -> None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pgid, signal.SIGKILL)

        try:
            with _Watchdog(run / "scratch", quota, kill) as wd:
                try:
                    proc.wait(timeout=cfg.timeout_s + sandbox.KILL_GRACE_S)
                except subprocess.TimeoutExpired:
                    timed_out = True
                quota_hit = wd.tripped.is_set()
        finally:
            _kill_group(proc)
    tail = log_path.read_bytes()[-1000:].decode(errors="replace")
    return tail, timed_out and not quota_hit, quota_hit


def _confirm_container_gone(name: str) -> None:
    """Return only once ``docker inspect`` shows the container gone or stopped."""

    def state() -> str:
        try:
            p = sandbox.docker("inspect", "-f", "{{.State.Status}}", name, check=False, timeout=30)
        except subprocess.TimeoutExpired:
            return "unknown"
        return "gone" if p.returncode != 0 else p.stdout.strip()

    for attempt in range(2):
        st = state()
        if st in {"gone", "exited", "dead", "created"}:
            return
        if attempt == 0:
            for cmd in (("kill", name), ("rm", "-f", name)):
                with contextlib.suppress(subprocess.TimeoutExpired):
                    sandbox.docker(*cmd, check=False, timeout=30)
    raise RunnerError(f"container {name} could not be confirmed stopped")


def _run_docker(
    run: Path, cfg: RunConfig, gw: GatewayHost, image_tag: str, quota: int
) -> tuple[bool, bool, str]:
    """Returns ``(timed_out, quota_exceeded, stderr_tail)``; the container is gone on return."""
    env = {
        "ANTHROPIC_BASE_URL": f"http://{sandbox.RELAY_ALIAS}:{sandbox.RELAY_PORT}/r/{cfg.run_id}",
        "ANTHROPIC_API_KEY": DUMMY_KEY,
        "MCP_CHAOS_SCRATCH": "/scratch",
    }
    # Bind-mounted trees are owned by the host user; the container runs as an unrelated UID.
    _chmod_tree(run / "scratch", 0o777)
    _chmod_tree(run / "out", 0o777)
    name = f"mcpchaos-run-{uuid.uuid4().hex[:12]}"

    def kill() -> None:
        with contextlib.suppress(subprocess.TimeoutExpired):
            sandbox.docker("kill", name, check=False, timeout=30)

    try:
        with _Watchdog(run / "scratch", quota, kill) as wd:
            r = sandbox.run_container(
                image=image_tag,
                name=name,
                scratch=run / "scratch",
                in_dir=run / "in",
                out_dir=run / "out",
                env=env,
                timeout_s=cfg.timeout_s,
            )
            quota_hit = wd.tripped.is_set()
    finally:
        _confirm_container_gone(name)
    return r.timed_out and not quota_hit, quota_hit, r.stderr_tail


def run_one(
    cfg: RunConfig,
    *,
    sandbox: SandboxKind,
    gw: GatewayHost,
    results_dir: Path,
    image_tag: str = "mcpchaos-target:dev",
    work_root: Path | None = None,
    scratch_quota_bytes: int = DEFAULT_SCRATCH_QUOTA_BYTES,
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

        if sandbox == "local":
            detail, timed_out, quota_hit = _run_local(run, cfg, gw, scratch_quota_bytes)
        else:
            timed_out, quota_hit, detail = _run_docker(run, cfg, gw, image_tag, scratch_quota_bytes)

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
        elif quota_hit:
            status = "agent_error"
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
            details = dict(res.details)
            if quota_hit:
                details["error"] = QUOTA_ERROR
            if after.snapshot_error is not None:
                details["snapshot_error"] = after.snapshot_error
                res = res.model_copy(update={"success": False})
            check = CheckSummary(success=res.success and not quota_hit, details=details)

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
