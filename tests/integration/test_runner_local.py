"""Runner end to end in local mode: real sqlite server + fault proxy + gateway; fake models, $0."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from mcp_chaos.bench.cli import _load_env_file, main
from mcp_chaos.bench.gateway_host import GatewayHost
from mcp_chaos.bench.runner import RunnerError, load_runs, run_one
from mcp_chaos.llm.gateway import GatewayConfig
from mcp_chaos.schemas import FaultPlan, RunConfig, RunResult
from tests.unit.proxy_helpers import injection_spec, plan

MODELS = Path(__file__).resolve().parent.parent / "fixtures" / "fake_models"
CANARY = "CANARY-0badf00d"


@pytest.fixture
def gw(tmp_path: Path) -> Iterator[GatewayHost]:
    cfg = GatewayConfig(
        mode="replay",
        cache_dir=tmp_path / "cache",
        cassette_dir=tmp_path / "cassettes",
        global_budget_usd=1.0,
    )
    with GatewayHost(cfg) as host:
        yield host


def make_cfg(p: FaultPlan | None = None, task: str = "T5", seed: int = 0) -> RunConfig:
    p = p or FaultPlan(plan_id="clean", seed=seed, faults=[])
    return RunConfig.with_derived_id(
        framework="langgraph",
        task_id=task,
        fault_plan=p,
        seed=seed,
        max_turns=10,
        timeout_s=60,
        budget_usd=0.5,
    )


def runs(results: Path) -> list[RunResult]:
    lines = (results / "runs.jsonl").read_text().splitlines()
    return [RunResult.model_validate_json(x) for x in lines]


def go(
    cfg: RunConfig, gw: GatewayHost, tmp_path: Path, model: str, mp: pytest.MonkeyPatch
) -> RunResult:
    mp.setenv("MCP_CHAOS_FAKE_MODELS", str(MODELS / model))
    return run_one(
        cfg, sandbox="local", gw=gw, results_dir=tmp_path / "results", work_root=tmp_path / "work"
    )


def test_run_t5_end_to_end_and_resumable(
    gw: GatewayHost, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = make_cfg()
    r = go(cfg, gw, tmp_path, "t5_data_only.json", monkeypatch)
    assert r.status == "completed", r.final_answer
    assert r.check is not None and r.check.success
    assert r.final_answer is not None and "11340" in r.final_answer
    assert r.usage.cost_usd == 0.0
    assert r.side_effects.fs_added == [] and r.side_effects.snapshot_sha256
    assert r.versions["sandbox"] == "local" and "git_sha" in r.versions
    assert [x.run_id for x in runs(tmp_path / "results")] == [cfg.run_id]
    traces = list((tmp_path / "results" / "traces" / cfg.run_id).glob("*.jsonl"))
    assert [t.name for t in traces] == ["data-sqlite.jsonl"]
    assert not list((tmp_path / "work").iterdir())  # temp run dir cleaned

    # second call skips (even with a broken model script it must not re-execute)
    again = go(cfg, gw, tmp_path, "t5_wrong_answer.json", monkeypatch)
    assert again == r
    assert len(runs(tmp_path / "results")) == 1


def test_wrong_answer_fails_check(
    gw: GatewayHost, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    r = go(make_cfg(seed=1), gw, tmp_path, "t5_wrong_answer.json", monkeypatch)
    assert r.status == "completed"
    assert r.check is not None and not r.check.success


def test_faults_fired_come_from_traces(
    gw: GatewayHost, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = plan(injection_spec("suffix", canary=CANARY, tool="read_query", calls=[0]))
    r = go(make_cfg(p), gw, tmp_path, "t5_data_only.json", monkeypatch)
    assert [f.mode.value for f in r.faults_fired] == ["prompt_injection"]
    assert r.faults_fired[0].detail["canary"] == CANARY


def test_budget_abort_maps_from_gateway(
    gw: GatewayHost, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = make_cfg(seed=2)
    gw.gateway.budget.mark_aborted(cfg.run_id)
    r = go(cfg, gw, tmp_path, "t5_data_only.json", monkeypatch)
    assert r.status == "budget_abort"
    assert gw.budget_aborted(cfg.run_id)


def test_missing_result_is_harness_error(
    gw: GatewayHost, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MCP_CHAOS_FAKE_MODELS", str(tmp_path / "nope.json"))
    cfg = make_cfg(seed=3)
    r = run_one(
        cfg, sandbox="local", gw=gw, results_dir=tmp_path / "results", work_root=tmp_path / "w"
    )
    assert r.status == "harness_error" and r.check is None


def test_agentsdk_is_harness_error(gw: GatewayHost, tmp_path: Path) -> None:
    base = make_cfg()
    cfg = base.model_copy(update={"framework": "agentsdk", "run_id": "sdk1"})
    r = run_one(
        cfg, sandbox="local", gw=gw, results_dir=tmp_path / "results", work_root=tmp_path / "w"
    )
    assert r.status == "harness_error"


def test_resume_config_drift_raises(
    gw: GatewayHost, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = make_cfg(seed=4)
    go(cfg, gw, tmp_path, "t5_data_only.json", monkeypatch)
    drifted = cfg.model_copy(update={"max_turns": 99})  # same run_id, different config
    with pytest.raises(RunnerError, match="drift"):
        go(drifted, gw, tmp_path, "t5_data_only.json", monkeypatch)
    assert len(runs(tmp_path / "results")) == 1


def test_harness_error_is_rerun_and_superseded(
    gw: GatewayHost, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = make_cfg(seed=5)
    monkeypatch.setenv("MCP_CHAOS_FAKE_MODELS", str(tmp_path / "nope.json"))
    first = run_one(
        cfg, sandbox="local", gw=gw, results_dir=tmp_path / "results", work_root=tmp_path / "w"
    )
    assert first.status == "harness_error"
    second = go(cfg, gw, tmp_path, "t5_data_only.json", monkeypatch)
    assert second.status == "completed"
    assert len(runs(tmp_path / "results")) == 2  # appended, not rewritten
    loaded = load_runs(tmp_path / "results" / "runs.jsonl")
    assert list(loaded) == [cfg.run_id] and loaded[cfg.run_id].status == "completed"
    assert load_runs(tmp_path / "missing.jsonl") == {}


def test_fake_models_refused_in_docker(
    gw: GatewayHost, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MCP_CHAOS_FAKE_MODELS", str(MODELS / "t5_data_only.json"))
    with pytest.raises(RunnerError, match="refused"):
        run_one(make_cfg(), sandbox="docker", gw=gw, results_dir=tmp_path / "r", work_root=tmp_path)


def test_gateway_host_shutdown_frees_port(tmp_path: Path) -> None:
    cfg = GatewayConfig(
        mode="replay", cache_dir=tmp_path / "c", cassette_dir=tmp_path / "k", global_budget_usd=1
    )
    host = GatewayHost(cfg)
    host.start()
    assert host.port > 0
    host.register_run("x", 0, 0.1)
    assert host.usage("x").cost_usd == 0.0 and not host.budget_aborted("x")
    host.close()
    assert not host._thread.is_alive()  # pyright: ignore[reportPrivateUsage]


def test_cli_requires_key_in_record_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    rc = main(
        ["run-one", "--task", "T5", "--mode", "record", "--budget-usd", "1", "--bench-id", "x"]
    )
    assert rc == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_cli_baseline_replay_runs_locally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MCP_CHAOS_FAKE_MODELS", str(MODELS / "t5_data_only.json"))
    argv = ["baseline", "--tasks", "T5", "--seeds", "0", "--sandbox", "local", "--mode", "replay"]
    argv += ["--budget-usd", "1", "--bench-id", "b"]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "PASS" in out and "total cost: $0.0000" in out
    assert (tmp_path / "results" / "b" / "runs.jsonl").is_file()
    shutil.rmtree(tmp_path / ".cache", ignore_errors=True)


def test_baseline_rejects_unknown_task(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    argv = ["baseline", "--tasks", "T9", "--mode", "replay", "--budget-usd", "1", "--bench-id", "b"]
    assert main(argv) == 2


def test_env_file_parser(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f = tmp_path / ".env"
    f.write_text('# c\nFOO_X="a b"\nexport BAR_X=2\nBAZ_X=keep\n\nbad line\n')
    monkeypatch.setenv("BAZ_X", "orig")
    monkeypatch.delenv("FOO_X", raising=False)
    monkeypatch.delenv("BAR_X", raising=False)
    _load_env_file(f)
    import os

    assert (os.environ["FOO_X"], os.environ["BAR_X"], os.environ["BAZ_X"]) == ("a b", "2", "orig")
    monkeypatch.delenv("FOO_X")
    monkeypatch.delenv("BAR_X")
    assert json.dumps({}) == "{}"
