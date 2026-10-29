"""Matrix expansion, config loading, cost estimate, CLI dry-run and scheduling (fake run_one)."""

from __future__ import annotations

import json
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from mcp_chaos.bench import matrix
from mcp_chaos.bench.cli import main
from mcp_chaos.bench.gateway_host import GatewayHost
from mcp_chaos.bench.matrix import (
    BenchConfig,
    ConfigError,
    Manifest,
    estimate_cost,
    expand,
    load_cells,
    load_config,
    run_matrix,
    summarize,
)
from mcp_chaos.bench.runner import RUNS_FILE, RunnerError
from mcp_chaos.llm.budget import TokenCounts
from mcp_chaos.schemas import FaultMode, RunConfig, RunResult
from mcp_chaos.schemas.results import Usage

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def small_cfg(**over: Any) -> BenchConfig:
    raw: dict[str, Any] = {
        "bench_id": "t",
        "frameworks": ["langgraph"],
        "tasks": ["T1", "T2"],
        "cells": [{"name": "clean"}],
        "arms": [{"name": "baseline"}],
        "seeds": [0, 1],
        "concurrency": 3,
    }
    raw.update(over)
    return BenchConfig.model_validate(raw)


def test_default_matrix_is_864_and_deterministic() -> None:
    cfg = load_config(CONFIGS / "bench.yaml")
    runs = expand(cfg)
    assert len(runs) == 864  # SPEC section 5
    assert [r.run_id for r in runs] == [
        r.run_id for r in expand(load_config(CONFIGS / "bench.yaml"))
    ]
    assert len({r.run_id for r in runs}) == 864
    assert all(r.run_id == r.compute_id() for r in runs)
    assert [(r.framework, r.task_id) for r in runs[:2]] == [("langgraph", "T1")] * 2
    assert runs[0].fault_plan.plan_id == "clean" and runs[0].hardening is None
    assert runs[3].hardening is not None and runs[3].seed == 0  # seed varies fastest, then arm
    assert [r.seed for r in runs[:3]] == [0, 1, 2]
    assert all(r.fault_plan.seed == r.seed for r in runs)


def test_summarize_and_smoke() -> None:
    cfg = load_config(CONFIGS / "bench.yaml")
    s = summarize(cfg, expand(cfg))
    assert s["framework"] == {"langgraph": 432, "agentsdk": 432}
    assert s["arm"] == {"baseline": 432, "guard_full": 432}
    assert len(s["cell"]) == 12 and set(s["cell"].values()) == {72}
    assert len(expand(load_config(CONFIGS / "smoke.yaml"))) == 2


def test_cells_yaml_validates_and_covers_every_mode() -> None:
    cells = load_cells(CONFIGS / "cells.yaml")
    assert cells[0].name == "clean" and cells[0].faults == []
    assert {f.mode for c in cells for f in c.faults} == set(FaultMode)
    assert len({c.name for c in cells}) == len(cells)
    for c in cells[1:]:
        assert len(c.faults) == 1
        t = c.faults[0].trigger
        assert (t.server, t.tool, t.probability) == ("*", "*", 1.0)
        assert t.calls == (None if c.faults[0].mode is FaultMode.STALE_DATA else [0])
    templates = {
        getattr(f.params, "template_id", None)
        for c in cells
        for f in c.faults
        if f.mode is FaultMode.PROMPT_INJECTION
    }
    assert templates == {"write_file", "echo_in_answer", "call_tool", "delegate"}


def test_config_errors(tmp_path: Path) -> None:
    (tmp_path / "cells.yaml").write_text("cells: [{name: a, faults: []}]\n")
    f = tmp_path / "b.yaml"
    f.write_text("bench_id: x\nfrequencies: 1\n")
    with pytest.raises(ConfigError):
        load_config(f)
    f.write_text("bench_id: x\ncells_file: cells.yaml\ncells: [nope]\n")
    with pytest.raises(ConfigError, match="unknown cell"):
        load_config(f)
    f.write_text("- 1\n")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(f)
    with pytest.raises(ConfigError, match="cannot read"):
        load_config(tmp_path / "missing.yaml")
    (tmp_path / "bad.yaml").write_text("cells: [{name: a, faults: [{id: f, mode: nope}]}]\n")
    with pytest.raises(ConfigError):
        load_cells(tmp_path / "bad.yaml")
    f.write_text(
        "bench_id: x\ncells_file: cells.yaml\nframeworks: [langgraph]\n"
        "arms: [{name: a}]\nseeds: [0]\n"
    )
    loaded = load_config(f)
    assert len(loaded.cells) == 1  # cells default: everything in the library
    assert loaded.tasks == "all"


def test_validators() -> None:
    with pytest.raises(ValueError, match="duplicate cell"):
        small_cfg(cells=[{"name": "a"}, {"name": "a"}])
    with pytest.raises(ValueError, match="duplicate arm"):
        small_cfg(arms=[{"name": "a"}, {"name": "a"}])
    with pytest.raises(ValueError, match="non-empty"):
        small_cfg(seeds=[])
    with pytest.raises(ValueError, match="unknown task"):
        small_cfg(tasks=["T9"])
    assert small_cfg().sha256() == small_cfg().sha256() != small_cfg(seeds=[5]).sha256()


def test_estimate_cost() -> None:
    runs = expand(small_cfg())
    low, high = estimate_cost(runs)
    assert 0 < low < high
    assert low == pytest.approx(len(runs) * (30_000 * 0.10 + 2_000 * 0.50) / 1e6)
    # high input (120k) is above the 100k long-context threshold for haiku
    assert high == pytest.approx(len(runs) * (120_000 * 0.50 + 6_000 * 2.50) / 1e6)
    sonnet = expand(small_cfg(agent_model="claude-sonnet-5-5"))
    assert estimate_cost(sonnet)[0] == pytest.approx(len(sonnet) * (30_000 * 2 + 2_000 * 10) / 1e6)
    assert estimate_cost([]) == (0.0, 0.0)


def test_cli_dry_run_output_and_no_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = str(CONFIGS / "bench.yaml")
    assert main(["run", "--config", cfg, "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "864 runs" in out and "langgraph=432" in out and "F11-tool_vanish=72" in out
    assert "baseline=432" in out and "estimated cost: $3.46 - $64.80" in out
    assert not (tmp_path / "results").exists()
    assert main(["run", "--config", cfg, "--dry-run", "--limit", "10"]) == 0
    assert "10 runs" in capsys.readouterr().out


def test_cli_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    smoke = str(CONFIGS / "smoke.yaml")
    assert main(["run", "--config", str(tmp_path / "nope.yaml"), "--dry-run"]) == 2
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert main(["run", "--config", smoke, "--mode", "live", "--yes"]) == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert main(["run", "--config", smoke, "--mode", "record"]) == 2
    err = capsys.readouterr()
    assert "--yes" in err.err and "estimated cost" in err.out
    assert not (tmp_path / "results").exists()


# ---- scheduling with a fake run_one --------------------------------------------------------


class Fake:
    def __init__(self, delay: float = 0.0, status: str = "completed", cost: float = 0.0) -> None:
        self.delay, self.status, self.cost = delay, status, cost
        self.calls: list[str] = []
        self.active = self.peak = 0
        self.lock = threading.Lock()

    def __call__(
        self, cfg: RunConfig, *, sandbox: str, gw: GatewayHost, results_dir: Path
    ) -> RunResult:
        with self.lock:
            self.calls.append(cfg.run_id)
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(self.delay)
        if self.cost:
            gw.gateway.budget.record(cfg.run_id, TokenCounts(), self.cost)
        r = RunResult.model_validate(
            {
                "run_id": cfg.run_id,
                "config": cfg,
                "status": self.status,
                "usage": Usage(cost_usd=self.cost, input_tokens=10),
                "started_at": datetime.now(UTC),
                "wall_s": 0.0,
            }
        )
        with self.lock:
            self.active -= 1
            results_dir.mkdir(parents=True, exist_ok=True)
            with (results_dir / RUNS_FILE).open("a") as f:
                f.write(r.model_dump_json() + "\n")
        return r


def go(
    cfg: BenchConfig, tmp_path: Path, budget: float = 10.0, limit: int | None = None
) -> Manifest:
    return run_matrix(
        cfg,
        mode="replay",
        sandbox="local",
        global_budget_usd=budget,
        results_root=tmp_path / "res",
        limit=limit,
        cassette_root=tmp_path / "cass",
        cache_dir=tmp_path / "cache",
    )


def test_run_matrix_manifest_resume_and_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = Fake(cost=0.01)
    monkeypatch.setattr(matrix, "run_one", fake)
    cfg = small_cfg()  # 4 runs
    m = go(cfg, tmp_path, limit=3)
    assert (m.n_planned, m.n_completed, m.n_skipped, m.aborted) == (3, 3, 0, False)
    assert m.status_counts == {"completed": 3} and m.total_cost_usd == pytest.approx(0.03)
    assert m.config_sha256 == cfg.sha256() and m.versions["python"] and m.git_sha
    assert m.total_usage.input_tokens == 30

    m2 = go(cfg, tmp_path)  # resume: only the 4th run executes
    assert (m2.n_planned, m2.n_completed, m2.n_skipped) == (4, 1, 3)
    assert len(fake.calls) == 4 and len(set(fake.calls)) == 4
    assert m2.total_cost_usd == pytest.approx(0.04)

    m3 = go(cfg, tmp_path)  # nothing left: no gateway, no calls
    assert (m3.n_completed, m3.n_skipped) == (0, 4) and len(fake.calls) == 4
    on_disk = json.loads((tmp_path / "res" / "t" / "manifest.json").read_text())
    assert on_disk["n_skipped"] == 4 and on_disk["bench_id"] == "t"


def test_harness_error_results_are_rerun(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = Fake(status="harness_error")
    monkeypatch.setattr(matrix, "run_one", fake)
    cfg = small_cfg(tasks=["T1"], seeds=[0])
    assert go(cfg, tmp_path).status_counts == {"harness_error": 1}
    assert go(cfg, tmp_path).n_completed == 1 and len(fake.calls) == 2


def test_concurrency_is_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = Fake(delay=0.05)
    monkeypatch.setattr(matrix, "run_one", fake)
    m = go(small_cfg(tasks="all", seeds=[0, 1, 2], concurrency=3), tmp_path)  # 18 runs
    assert m.n_completed == 18
    assert 1 < fake.peak <= 3


def test_global_budget_abort_stops_scheduling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = Fake(delay=0.02, cost=1.0)
    monkeypatch.setattr(matrix, "run_one", fake)
    m = go(small_cfg(tasks="all", seeds=[0, 1, 2], concurrency=2), tmp_path, budget=3.0)
    assert m.aborted and m.abort_reason and "budget" in m.abort_reason
    assert 3 <= len(fake.calls) <= 4  # at most the in-flight run finishes after the cap
    assert m.n_completed == len(fake.calls) and m.n_planned == 18


def test_run_level_budget_abort_with_no_global_headroom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(matrix, "run_one", Fake(status="budget_abort"))
    m = go(small_cfg(per_run_budget_usd=0.5, concurrency=1), tmp_path, budget=0.1)
    assert m.aborted and m.status_counts == {"budget_abort": 1}


def test_run_level_budget_abort_alone_does_not_abort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(matrix, "run_one", Fake(status="budget_abort"))
    m = go(small_cfg(per_run_budget_usd=0.5), tmp_path, budget=10.0)
    assert not m.aborted and m.n_completed == 4


def test_runner_error_is_counted_and_run_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ok = Fake()

    def flaky(cfg: RunConfig, **kw: Any) -> RunResult:
        if cfg.seed == 0:
            raise RunnerError("drift")
        return ok(cfg, **kw)

    monkeypatch.setattr(matrix, "run_one", flaky)
    m = go(small_cfg(), tmp_path)
    assert m.status_counts == {"completed": 2, "runner_error": 2}


def test_cli_run_replay_and_abort_exit_codes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(matrix, "run_one", Fake(cost=1.0))
    smoke = str(CONFIGS / "smoke.yaml")
    argv = ["run", "--config", smoke, "--sandbox", "local", "--budget-usd"]
    assert main([*argv, "10"]) == 0
    assert "planned=2 executed=2" in capsys.readouterr().out
    assert main([*argv, "10"]) == 0  # resumed
    assert "skipped=2" in capsys.readouterr().out
    (tmp_path / "results" / "smoke" / "runs.jsonl").unlink()
    assert main([*argv, "0.5", "--limit", "1"]) == 3
    assert "aborted" in capsys.readouterr().err
    assert (tmp_path / "results" / "smoke" / "manifest.json").is_file()
