"""The $0 self-test pipeline (run -> judge offline -> scorecard) with the naive echo fake model."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mcp_chaos.bench.cli import main
from mcp_chaos.bench.matrix import Arm, BenchConfig, load_cells, run_matrix
from mcp_chaos.schemas.runs import HardeningConfig

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "tests" / "fixtures" / "fake_models" / "selftest_echo.json"
BENCH = "selftest-reduced"


def _cfg() -> BenchConfig:
    cells = [
        c
        for c in load_cells(ROOT / "configs" / "selftest_cells.yaml")
        if c.name in {"clean", "F08-echo_in_answer"}
    ]
    return BenchConfig(
        bench_id=BENCH,
        frameworks=["langgraph"],
        tasks=["T5"],
        cells=cells,
        arms=[
            Arm(name="baseline"),
            Arm(
                name="guard_full",
                hardening=HardeningConfig(
                    schema_validation=True,
                    retry=True,
                    circuit_breaker=True,
                    injection_filter=True,
                    size_cap=True,
                    prompt_addendum=True,
                ),
            ),
        ],
        seeds=[0],
        max_turns=10,
        timeout_s=30,
        concurrency=4,
    )


def _pipeline(workdir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    monkeypatch.setenv("MCP_CHAOS_FAKE_MODELS", str(MODEL))
    m = run_matrix(
        _cfg(),
        mode="replay",
        sandbox="local",
        global_budget_usd=1.0,
        results_root=workdir / "results",
        cassette_root=workdir / "cassettes",
        cache_dir=workdir / ".cache" / "llm",
    )
    assert m.n_completed == 4 and not m.aborted and m.total_cost_usd == 0.0
    assert main(["judge", "--bench-id", BENCH, "--mode", "offline"]) == 0
    assert main(["scorecard", "--bench-id", BENCH, "--deterministic"]) == 0
    return workdir / "results" / BENCH


def _followed(bench_dir: Path) -> dict[tuple[str, bool], bool | None]:
    runs = {
        r["run_id"]: r
        for r in (json.loads(line) for line in (bench_dir / "runs.jsonl").read_text().splitlines())
    }
    out: dict[tuple[str, bool], bool | None] = {}
    for line in (bench_dir / "labels.jsonl").read_text().splitlines():
        lab = json.loads(line)
        cfg = runs[lab["run_id"]]["config"]
        out[(cfg["fault_plan"]["plan_id"], cfg["hardening"] is not None)] = lab["deterministic"][
            "followed_injection"
        ]
    return out


def test_naive_agent_follows_injection_only_without_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    d = _pipeline(tmp_path / "a", monkeypatch)
    f = _followed(d)
    assert f[("F08-echo_in_answer", False)] is True
    assert f[("F08-echo_in_answer", True)] is False
    assert f[("clean", False)] is None and f[("clean", True)] is None
    md = (d / "scorecard.md").read_text()
    assert "SYNTHETIC DATA WARNING" in md


def test_selftest_scorecard_is_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    a = _pipeline(tmp_path / "a", monkeypatch)
    b = _pipeline(tmp_path / "b", monkeypatch)
    for name in ("scorecard.md", "scorecard.json"):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
