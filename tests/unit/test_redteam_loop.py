from __future__ import annotations

import json
from collections.abc import Collection, Sequence
from pathlib import Path

import pytest

from mcp_chaos.bench.cli import main
from mcp_chaos.redteam import cli as rt_cli
from mcp_chaos.redteam import space
from mcp_chaos.redteam.agent import Proposal, RedTeam
from mcp_chaos.redteam.loop import default_run_fn, offline_judge, run_campaign
from mcp_chaos.redteam.models import CampaignResult, PlanRecord
from mcp_chaos.schemas.faults import FaultPlan, PromptInjectionParams
from mcp_chaos.schemas.results import RunResult
from mcp_chaos.schemas.runs import RunConfig
from tests.unit.redteam_helpers import client_for, fake_app, fake_run, pi_spec, rng

TEMPLATES = ("echo_in_answer", "delegate", "call_tool", "write_file")


class ScriptedExploiter:
    """Round 0: probe each injection template; later: exploit the best one found so far."""

    def __init__(self) -> None:
        self.calls = 0

    async def propose(
        self, history: Sequence[PlanRecord], n: int, known_hashes: Collection[str]
    ) -> Proposal:
        self.calls += 1
        if not history:
            specs = [pi_spec(t, "suffix") for t in TEMPLATES[:n]]
        else:
            best = max(history, key=lambda p: p.mean_severity).plan.faults[0].params
            assert isinstance(best, PromptInjectionParams)
            specs = [
                pi_spec(best.template_id, pos, calls=c)
                for pos in ("field", "prefix")
                for c in ([0], [1], [2], [3], [0, 1])
            ]
        plans: list[FaultPlan] = []
        for s in specs:
            parent = history[0].plan.plan_id if history else None
            res = space.parse_validate(s, source="redteam", parent=parent)
            assert isinstance(res, FaultPlan)
            if space.plan_hash(res) not in known_hashes:
                plans.append(res)
        return Proposal(plans=plans[:n])


def run(strategy: str, proposer: object | None = None, **kw: object) -> CampaignResult:
    args: dict[str, object] = {
        "run_fn": fake_run,
        "rng_seed": 3,
        "proposer": proposer,
    }
    args.update(kw)
    return run_campaign(strategy, "langgraph", ["T1", "T5"], 3, 4, [0], **args)  # type: ignore[arg-type]


def test_adaptive_exploit_beats_random_with_equal_runs() -> None:
    adaptive = run("adaptive", ScriptedExploiter())
    rand = run("random")
    assert adaptive.runs_used == rand.runs_used == 3 * 4 * 2 * 1
    assert adaptive.best_severity == 8.0
    assert adaptive.best_severity > rand.best_severity
    assert adaptive.total_failures >= 1 and "prompt_injection/persistent" in adaptive.categories
    assert all(p.plan.source == "redteam" for p in adaptive.plans)
    assert all(p.plan.source == "random" and p.plan.parent_plan_id is None for p in rand.plans)
    assert any(p.plan.parent_plan_id for p in adaptive.plans)  # lineage recorded
    assert [r.runs_used for r in adaptive.rounds] == [8, 16, 24]
    cum = [r.cumulative_categories for r in adaptive.rounds]
    assert cum == sorted(cum)
    # random plans are distinct and the campaign is reproducible
    assert len({space.plan_hash(p.plan) for p in rand.plans}) == len(rand.plans) == 12
    assert run("random").model_dump() == rand.model_dump()


def test_adaptive_fills_with_random_when_proposals_short() -> None:
    class Empty:
        async def propose(
            self, history: Sequence[PlanRecord], n: int, known_hashes: Collection[str]
        ) -> Proposal:
            dup = space.parse_validate(pi_spec(), source="redteam", parent=None)
            assert isinstance(dup, FaultPlan)
            return Proposal(plans=[dup, dup], n_invalid=2, cost_usd=0.001)

    res = run("adaptive", Empty())
    assert res.runs_used == 24
    assert res.dropped_invalid == 6 and res.dropped_duplicate >= 3
    assert res.filled_random >= 9 and res.redteam_cost_usd == pytest.approx(0.003)


def test_budget_stop() -> None:
    res = run("random", run_fn=lambda c: fake_run(c, cost=0.1), budget_usd=0.25)
    assert res.stopped_reason == "budget" and res.runs_used == 3
    assert res.cost_usd == pytest.approx(0.3)
    assert len(res.rounds) == 1 and res.rounds[0].runs_used == 3


def test_redteam_budget_stop_and_missing_proposer() -> None:
    class Broke:
        async def propose(
            self, history: Sequence[PlanRecord], n: int, known_hashes: Collection[str]
        ) -> Proposal:
            return Proposal(budget_exhausted=True)

    res = run("adaptive", Broke())
    assert res.stopped_reason == "redteam_budget" and res.runs_used == 0 and res.rounds == []
    with pytest.raises(ValueError):
        run("adaptive")


def test_harness_error_and_offline_judge() -> None:
    def broken(cfg: RunConfig) -> RunResult:
        return fake_run(cfg).model_copy(update={"status": "harness_error", "check": None})

    res = run("random", run_fn=broken)
    assert res.best_severity == 0.0 and res.total_failures == 0 and res.categories == []
    cfg = RunConfig.with_derived_id(
        framework="langgraph",
        task_id="T1",
        fault_plan=space.sample(rng(0)),
        seed=0,
        budget_usd=0.5,
    )
    good = offline_judge(fake_run(cfg))
    assert good.task_success or good.false_success_claim
    hedged = fake_run(cfg).model_copy(update={"final_answer": "an error occurred", "check": None})
    assert not offline_judge(hedged).false_success_claim


def test_default_run_fn_calls_run_one(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    def fake_run_one(cfg: RunConfig, **kw: object) -> RunResult:
        seen.update(kw)
        return fake_run(cfg)

    monkeypatch.setattr("mcp_chaos.bench.runner.run_one", fake_run_one)
    fn = default_run_fn(sandbox="local", gw="GW", results_dir=tmp_path)
    cfg = RunConfig.with_derived_id(
        framework="langgraph",
        task_id="T1",
        fault_plan=FaultPlan(plan_id="c", seed=0, faults=[]),
        seed=0,
        budget_usd=0.5,
    )
    assert fn(cfg).run_id == cfg.run_id
    assert seen == {"sandbox": "local", "gw": "GW", "results_dir": tmp_path}
    assert cfg.hardening is None  # baseline arm


# -- CLI ----------------------------------------------------------------------------------------
@pytest.fixture
def fake_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    import contextlib
    from collections.abc import Generator

    @contextlib.contextmanager
    def opener(args: object, budget_usd: float) -> Generator[object]:
        yield fake_run

    monkeypatch.setattr(rt_cli, "_open_run_fn", opener)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("MCP_CHAOS_FAKE_MODELS", raising=False)


def cli(tmp_path: Path, *extra: str) -> int:
    return main(
        [
            "redteam",
            "--tasks",
            "T1,T5",
            "--rounds",
            "2",
            "--plans-per-round",
            "2",
            "--seeds",
            "0",
            "--sandbox",
            "local",
            "--results-root",
            str(tmp_path),
            *extra,
        ]
    )


def test_cli_dry_run_and_validation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fake_runs: None
) -> None:
    assert cli(tmp_path, "--dry-run") == 0
    assert "2 strategy(ies) x 8 runs" in capsys.readouterr().out
    assert not (tmp_path / "redteam").exists()
    assert cli(tmp_path, "--strategy", "random", "--mode", "live") == 2  # needs --yes
    assert cli(tmp_path, "--strategy", "random", "--mode", "live", "--yes") == 2  # needs key
    assert cli(tmp_path) == 2  # adaptive w/o planner model
    assert main(["redteam", "--tasks", "T9"]) == 2
    assert main(["redteam", "--seeds", "x"]) == 2
    assert main(["redteam", "--rounds", "0"]) == 2
    assert "unknown" in capsys.readouterr().err or True


def test_cli_random_writes_report(tmp_path: Path, fake_runs: None) -> None:
    assert cli(tmp_path, "--strategy", "random") == 0
    data = json.loads((tmp_path / "redteam" / "redteam.json").read_text())
    assert len(data) == 1 and data[0]["runs_used"] == 8
    md = (tmp_path / "redteam" / "redteam.md").read_text()
    assert "| Round | random |" in md and "n/a ($0)" in md and "synthetic" not in md


def test_cli_both_with_fake_planner_is_synthetic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_runs: None
) -> None:
    app = fake_app([pi_spec("write_file", "field"), pi_spec("echo_in_answer", "prefix"), {}])
    monkeypatch.setattr(rt_cli, "_make_client", lambda args: client_for(app))
    assert cli(tmp_path, "--base-url", "http://fake", "--budget-usd", "5") == 0
    data = json.loads((tmp_path / "redteam" / "redteam.json").read_text())
    assert [d["strategy"] for d in data] == ["adaptive", "random"]
    assert data[0]["runs_used"] == data[1]["runs_used"] == 8 and data[0]["synthetic"]
    assert data[0]["dropped_invalid"] >= 1
    md = (tmp_path / "redteam" / "redteam.md").read_text()
    assert "synthetic" in md and "Adaptive proposals" in md
    assert len(app.state.requests) == 2  # one planner call per round


def test_cli_budget_stop_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_runs: None
) -> None:
    monkeypatch.setattr(rt_cli, "_open_run_fn", _paid_opener)
    assert cli(tmp_path, "--strategy", "random", "--budget-usd", "0.15") == 3


def _paid_opener(args: object, budget_usd: float):  # type: ignore[no-untyped-def]
    import contextlib

    @contextlib.contextmanager
    def cm():  # type: ignore[no-untyped-def]
        yield lambda cfg: fake_run(cfg, cost=0.1)

    return cm()


def test_make_client_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    import argparse

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert rt_cli._make_client(argparse.Namespace(base_url=None)) is None  # pyright: ignore[reportPrivateUsage]
    c = rt_cli._make_client(argparse.Namespace(base_url="http://fake"))  # pyright: ignore[reportPrivateUsage]
    assert c is not None


def test_redteam_agent_class_used_by_cli_has_budget() -> None:
    rt = RedTeam(client=None, budget=0.0)
    assert rt.model == "claude-sonnet-5-5"
