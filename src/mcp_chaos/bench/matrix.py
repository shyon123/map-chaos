"""Matrix benchmark runner (SPEC section 5): expand a ``BenchConfig`` and execute every run."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import threading
from collections import Counter
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, Literal, cast

import anyio
import anyio.to_thread
import yaml  # pyright: ignore[reportMissingModuleSource]
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from mcp_chaos.bench.gateway_host import GatewayHost
from mcp_chaos.bench.runner import (
    REPO_ROOT,
    RUNS_FILE,
    RunnerError,
    SandboxKind,
    load_runs,
    run_one,
)
from mcp_chaos.llm.gateway import GatewayConfig
from mcp_chaos.llm.prices import PRICES, ModelPrice, Rates
from mcp_chaos.log import get_logger
from mcp_chaos.schemas.faults import FaultPlan, FaultSpec
from mcp_chaos.schemas.results import RunResult, Usage
from mcp_chaos.schemas.runs import Framework, HardeningConfig, ModelConfig, RunConfig
from mcp_chaos.targets.fixtures import TASK_IDS

log = get_logger(__name__)

Mode = Literal["live", "record", "replay"]
MANIFEST_FILE = "manifest.json"

# Per-run token assumption for cost estimates (to be re-measured from the baseline): a run is
# roughly 5-15 agent turns; the prompt grows each turn, so a whole run sends 30k-120k input
# tokens in total and generates 2k-6k output tokens. Prompt caching and the response cache are
# ignored (conservative); the long-context tier applies when an estimate exceeds the threshold.
TOKENS_LOW: tuple[int, int] = (30_000, 2_000)  # (input, output)
TOKENS_HIGH: tuple[int, int] = (120_000, 6_000)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Cell(_Frozen):
    """One fault condition. ``faults == []`` is the clean control."""

    name: str = Field(min_length=1)
    faults: list[FaultSpec] = Field(default_factory=list[FaultSpec])

    @model_validator(mode="before")
    @classmethod
    def _fill_param_modes(cls, data: Any) -> Any:
        """Let YAML omit ``params.mode`` (the discriminator): copy it from ``mode``."""
        if not isinstance(data, dict):
            return data
        d = cast("dict[str, Any]", data)
        fixed: list[Any] = []
        for f in cast("list[Any]", d.get("faults") or []):
            if isinstance(f, dict):
                fd = dict(cast("dict[str, Any]", f))
                p = fd.get("params")
                if isinstance(p, dict) and "mode" not in p and "mode" in fd:
                    fd["params"] = {"mode": fd["mode"], **cast("dict[str, Any]", p)}
                fixed.append(fd)
            else:
                fixed.append(f)
        return {**d, "faults": fixed}


class Arm(_Frozen):
    name: str = Field(min_length=1)
    hardening: HardeningConfig | None = None


class BenchConfig(_Frozen):
    bench_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    frameworks: list[Framework]
    tasks: list[str] | Literal["all"] = "all"
    cells: list[Cell]
    arms: list[Arm]
    seeds: list[int]
    agent_model: str = "claude-haiku-5-5"
    max_turns: int = Field(default=25, ge=1)
    timeout_s: float = Field(default=300, gt=0)
    per_run_budget_usd: float = Field(default=0.5, ge=0)
    concurrency: int = Field(default=4, ge=1)

    @model_validator(mode="after")
    def _check(self) -> BenchConfig:
        for label, names in (
            ("cell", [c.name for c in self.cells]),
            ("arm", [a.name for a in self.arms]),
        ):
            if len(set(names)) != len(names):
                raise ValueError(f"duplicate {label} names")
        if not (self.frameworks and self.cells and self.arms and self.seeds):
            raise ValueError("frameworks, cells, arms and seeds must be non-empty")
        if isinstance(self.tasks, list):
            unknown = [t for t in self.tasks if t not in TASK_IDS]
            if unknown:
                raise ValueError(f"unknown task(s): {','.join(unknown)}")
        return self

    def task_ids(self) -> list[str]:
        return list(TASK_IDS) if self.tasks == "all" else list(self.tasks)

    def sha256(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()


class ConfigError(ValueError):
    pass


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        raise ConfigError(f"cannot read {path}: {e}") from e
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return cast("dict[str, Any]", raw)


def load_cells(path: Path) -> list[Cell]:
    raw = _read_yaml(path)
    try:
        return [Cell.model_validate(c) for c in raw.get("cells", [])]
    except ValidationError as e:
        raise ConfigError(f"{path}: {e}") from e


def load_config(path: Path) -> BenchConfig:
    """Load a bench YAML. ``cells_file`` (relative to ``path``) plus ``cells`` (names or
    ``all``) selects from a shared cell library; otherwise ``cells`` is an inline list."""
    raw = _read_yaml(path)
    cells_file = raw.pop("cells_file", None)
    if cells_file is not None:
        lib = {c.name: c for c in load_cells(path.parent / str(cells_file))}
        sel: Any = raw.get("cells", "all")
        names = list(lib) if sel == "all" else cast("list[str]", sel)
        missing = [n for n in names if n not in lib]
        if missing:
            raise ConfigError(f"{path}: unknown cell(s): {','.join(missing)}")
        raw["cells"] = [lib[n].model_dump(mode="json") for n in names]
    try:
        return BenchConfig.model_validate(raw)
    except ValidationError as e:
        raise ConfigError(f"{path}: {e}") from e


def expand(cfg: BenchConfig) -> list[RunConfig]:
    """All runs in deterministic order: framework, task, cell, arm, seed."""
    runs: list[RunConfig] = []
    model = ModelConfig(model=cfg.agent_model)
    for fw in cfg.frameworks:
        for task in cfg.task_ids():
            for cell in cfg.cells:
                for arm in cfg.arms:
                    for seed in cfg.seeds:
                        runs.append(
                            RunConfig.with_derived_id(
                                framework=fw,
                                task_id=task,
                                agent_model=model,
                                fault_plan=FaultPlan(
                                    plan_id=cell.name, seed=seed, faults=list(cell.faults)
                                ),
                                hardening=arm.hardening,
                                seed=seed,
                                max_turns=cfg.max_turns,
                                timeout_s=cfg.timeout_s,
                                budget_usd=cfg.per_run_budget_usd,
                            )
                        )
    return runs


def _cost(price: ModelPrice, tokens: tuple[int, int]) -> float:
    inp, out = tokens
    rates: Rates = price.rates
    if (
        price.long_context_threshold is not None
        and price.long_context_rates is not None
        and inp > price.long_context_threshold
    ):
        rates = price.long_context_rates
    return (inp * rates.input + out * rates.output) / 1_000_000


def estimate_cost(
    runs: list[RunConfig], prices: Mapping[str, ModelPrice] | None = None
) -> tuple[float, float]:
    """(low, high) USD for ``runs`` under the TOKENS_LOW / TOKENS_HIGH assumption."""
    table = PRICES if prices is None else prices
    low = high = 0.0
    for r in runs:
        p = table[r.agent_model.model]
        low += _cost(p, TOKENS_LOW)
        high += _cost(p, TOKENS_HIGH)
    return low, high


def summarize(cfg: BenchConfig, runs: list[RunConfig]) -> dict[str, dict[str, int]]:
    """Run counts grouped by framework, cell (plan id) and arm."""
    arm_by_hardening = {
        (a.hardening.model_dump_json() if a.hardening else ""): a.name for a in cfg.arms
    }

    def arm(r: RunConfig) -> str:
        return arm_by_hardening[r.hardening.model_dump_json() if r.hardening else ""]

    return {
        "framework": dict(Counter(r.framework for r in runs)),
        "cell": dict(Counter(r.fault_plan.plan_id for r in runs)),
        "arm": dict(Counter(arm(r) for r in runs)),
    }


class Manifest(_Frozen):
    bench_id: str
    config_sha256: str
    git_sha: str
    versions: dict[str, str]
    started_at: datetime
    finished_at: datetime
    n_planned: int
    n_completed: int  # executed during this invocation
    n_skipped: int  # already stored (resume)
    status_counts: dict[str, int]  # over executed + skipped runs
    total_usage: Usage
    total_cost_usd: float
    aborted: bool
    abort_reason: str | None = None


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


def _package_versions() -> dict[str, str]:
    v = {"python": ".".join(map(str, sys.version_info[:3]))}
    for pkg in ("mcp-chaos", "langgraph", "langchain", "langchain-anthropic", "mcp", "pydantic"):
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            v[pkg] = importlib.metadata.version(pkg)
    return v


def _sum_usage(results: list[RunResult]) -> Usage:
    return Usage(
        input_tokens=sum(r.usage.input_tokens for r in results),
        output_tokens=sum(r.usage.output_tokens for r in results),
        cache_read_tokens=sum(r.usage.cache_read_tokens for r in results),
        cache_write_tokens=sum(r.usage.cache_write_tokens for r in results),
        cost_usd=sum(r.usage.cost_usd for r in results),
        cache_hits=sum(r.usage.cache_hits for r in results),
    )


class _State:
    """Bookkeeping shared by the worker tasks (a lock guards it: results arrive from tasks)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.executed: list[RunResult] = []
        self.errors = 0
        self.aborted = False
        self.reason: str | None = None


def _run_all(
    runs: list[RunConfig],
    *,
    cfg: BenchConfig,
    sandbox: SandboxKind,
    gw: GatewayHost,
    results_dir: Path,
    global_budget_usd: float,
    state: _State,
    fn: Callable[..., RunResult],
) -> None:
    limiter = anyio.CapacityLimiter(cfg.concurrency)

    def global_exhausted(r: RunResult) -> bool:
        spend = gw.global_spend_usd
        if spend >= global_budget_usd:
            return True
        # budget_abort with less headroom than one run may spend: the global cap was the cause
        return r.status == "budget_abort" and global_budget_usd - spend < cfg.per_run_budget_usd

    async def one(rc: RunConfig) -> None:
        async with limiter:
            if state.aborted:
                return
            try:
                r = await anyio.to_thread.run_sync(
                    partial(fn, rc, sandbox=sandbox, gw=gw, results_dir=results_dir)
                )
            except RunnerError as e:
                log.error("run_failed", run_id=rc.run_id, error=str(e)[:200])
                with state.lock:
                    state.errors += 1
                return
            with state.lock:
                state.executed.append(r)
                if not state.aborted and global_exhausted(r):
                    state.aborted = True
                    state.reason = f"global budget ${global_budget_usd:g} exhausted"

    async def main() -> None:
        async with anyio.create_task_group() as tg:
            for rc in runs:
                tg.start_soon(one, rc)

    anyio.run(main)


def run_matrix(
    cfg: BenchConfig,
    *,
    mode: Mode,
    sandbox: SandboxKind,
    global_budget_usd: float,
    results_root: Path,
    limit: int | None = None,
    cassette_root: Path,
    cache_dir: Path,
) -> Manifest:
    """Execute (or resume) the matrix and write ``<results_root>/<bench_id>/manifest.json``."""
    started = datetime.now(UTC)
    runs = expand(cfg)
    if limit is not None:
        runs = runs[: max(limit, 0)]
    results_dir = results_root / cfg.bench_id
    results_dir.mkdir(parents=True, exist_ok=True)

    stored = load_runs(results_dir / RUNS_FILE)
    skipped: list[RunResult] = []
    todo: list[RunConfig] = []
    for rc in runs:
        prev = stored.get(rc.run_id)
        if prev is not None and prev.status != "harness_error" and prev.config == rc:
            skipped.append(prev)
        else:
            todo.append(rc)

    state = _State()
    if todo:
        gcfg = GatewayConfig(
            mode=mode,
            cache_dir=cache_dir,
            cassette_dir=cassette_root / cfg.bench_id,
            global_budget_usd=global_budget_usd,
        )
        with GatewayHost(gcfg) as host:
            if sandbox == "docker":
                from mcp_chaos.bench import sandbox as sbx

                sbx.build_image(sbx.DEFAULT_IMAGE)
                sbx.ensure_gw_relay(host.port)
            _run_all(
                todo,
                cfg=cfg,
                sandbox=sandbox,
                gw=host,
                results_dir=results_dir,
                global_budget_usd=global_budget_usd,
                state=state,
                fn=run_one,
            )

    everything = [*skipped, *state.executed]
    counts = Counter(r.status for r in everything)
    if state.errors:
        counts["runner_error"] = state.errors
    usage = _sum_usage(everything)
    manifest = Manifest(
        bench_id=cfg.bench_id,
        config_sha256=cfg.sha256(),
        git_sha=_git_sha(),
        versions=_package_versions(),
        started_at=started,
        finished_at=datetime.now(UTC),
        n_planned=len(runs),
        n_completed=len(state.executed),
        n_skipped=len(skipped),
        status_counts=dict(sorted(counts.items())),
        total_usage=usage,
        total_cost_usd=usage.cost_usd,
        aborted=state.aborted,
        abort_reason=state.reason,
    )
    tmp = results_dir / (MANIFEST_FILE + ".tmp")
    tmp.write_text(json.dumps(json.loads(manifest.model_dump_json()), indent=2) + "\n")
    tmp.replace(results_dir / MANIFEST_FILE)
    log.info(
        "matrix_done",
        bench_id=cfg.bench_id,
        planned=manifest.n_planned,
        executed=manifest.n_completed,
        skipped=manifest.n_skipped,
        aborted=manifest.aborted,
    )
    return manifest
