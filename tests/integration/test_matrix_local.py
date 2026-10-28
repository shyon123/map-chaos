"""Matrix runner end to end in local mode with fake models ($0)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from mcp_chaos.bench.cli import main
from mcp_chaos.bench.matrix import BenchConfig, Manifest, run_matrix
from mcp_chaos.bench.runner import load_runs

MODELS = Path(__file__).resolve().parent.parent / "fixtures" / "fake_models"


def smoke_cfg() -> BenchConfig:
    return BenchConfig.model_validate(
        {
            "bench_id": "smoke-local",
            "frameworks": ["langgraph"],
            "tasks": ["T5"],
            "cells": [
                {"name": "clean"},
                {
                    "name": "F08-echo_in_answer",
                    "faults": [
                        {
                            "id": "F08",
                            "mode": "prompt_injection",
                            "params": {"template_id": "echo_in_answer", "position": "suffix"},
                            "trigger": {"tool": "read_query", "calls": [0]},
                        }
                    ],
                },
            ],
            "arms": [{"name": "baseline"}],
            "seeds": [0],
            "max_turns": 10,
            "timeout_s": 60,
            "concurrency": 2,
        }
    )


def go(tmp_path: Path) -> Manifest:
    return run_matrix(
        smoke_cfg(),
        mode="replay",
        sandbox="local",
        global_budget_usd=1.0,
        results_root=tmp_path / "results",
        cassette_root=tmp_path / "cassettes",
        cache_dir=tmp_path / "cache",
    )


def test_smoke_matrix_local(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MCP_CHAOS_FAKE_MODELS", str(MODELS / "t5_data_only.json"))
    m = go(tmp_path)
    assert (m.n_planned, m.n_completed, m.n_skipped, m.aborted) == (2, 2, 0, False)
    assert m.status_counts == {"completed": 2} and m.total_cost_usd == 0.0
    stored = load_runs(tmp_path / "results" / "smoke-local" / "runs.jsonl")
    assert len(stored) == 2
    assert {r.config.fault_plan.plan_id for r in stored.values()} == {"clean", "F08-echo_in_answer"}
    assert all(r.check is not None and r.check.success for r in stored.values())
    manifest = json.loads((tmp_path / "results" / "smoke-local" / "manifest.json").read_text())
    assert manifest["n_completed"] == 2

    again = go(tmp_path)
    assert (again.n_completed, again.n_skipped) == (0, 2)
    shutil.rmtree(tmp_path / ".cache", ignore_errors=True)


def test_cli_smoke_config_local(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MCP_CHAOS_FAKE_MODELS", str(MODELS / "t5_data_only.json"))
    cfg = str(Path(__file__).resolve().parents[2] / "configs" / "smoke.yaml")
    assert main(["run", "--config", cfg, "--sandbox", "local", "--budget-usd", "1"]) == 0
    assert "planned=2 executed=2" in capsys.readouterr().out
    assert (tmp_path / "results" / "smoke" / "manifest.json").is_file()
    shutil.rmtree(tmp_path / ".cache", ignore_errors=True)
