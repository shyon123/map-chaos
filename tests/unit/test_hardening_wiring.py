"""Wiring of hardening into configs, prompts and command construction."""

import json
from pathlib import Path

import pytest

from mcp_chaos.bench.matrix import expand, load_config
from mcp_chaos.schemas.faults import FaultPlan
from mcp_chaos.schemas.runs import HardeningConfig, RunConfig
from mcp_chaos.targets.entry import build_cmds
from mcp_chaos.targets.langgraph_target.graph import _load_prompt  # pyright: ignore[reportPrivateUsage]
from mcp_chaos.targets.workers import PROMPTS_DIR, needs_guard, system_prompt

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
FULL = HardeningConfig(
    schema_validation=True,
    retry=True,
    circuit_breaker=True,
    injection_filter=True,
    size_cap=True,
    prompt_addendum=True,
)


def arms(name: str) -> dict[str, HardeningConfig | None]:
    cfg = load_config(CONFIGS / name)
    return {a.name: a.hardening for a in cfg.arms}


def test_bench_arms() -> None:
    a = arms("bench.yaml")
    assert set(a) == {"baseline", "guard_full"}
    assert a["baseline"] is None and a["guard_full"] == FULL


def test_ablation_arms_and_cells() -> None:
    cfg = load_config(CONFIGS / "ablation.yaml")
    a = arms("ablation.yaml")
    assert set(a) == {
        "baseline",
        "guard_full",
        "guard_retry",
        "guard_validate",
        "guard_filter",
        "guard_sizecap",
        "prompt_only",
    }
    assert a["guard_full"] == FULL
    assert a["prompt_only"] == HardeningConfig(prompt_addendum=True)
    assert not needs_guard(a["prompt_only"])
    for name in ("guard_retry", "guard_validate", "guard_filter", "guard_sizecap"):
        h = a[name]
        assert h is not None and needs_guard(h) and not h.prompt_addendum
    assert [c.name for c in cfg.cells] == [
        "F01-timeout",
        "F04-malformed_json",
        "F05-content",
        "F08-echo_in_answer",
    ]
    assert len(expand(cfg)) == 2 * 6 * 4 * 7 * 3


def test_needs_guard() -> None:
    assert not needs_guard(None)
    assert not needs_guard(HardeningConfig())
    assert needs_guard(HardeningConfig(size_cap=True))


@pytest.mark.parametrize("role", ["supervisor", "files", "data", "vcs"])
def test_system_prompt_addendum(role: str) -> None:
    base = system_prompt(role)
    assert system_prompt(role, HardeningConfig(retry=True)) == base
    with_add = system_prompt(role, HardeningConfig(prompt_addendum=True))
    assert with_add.startswith(base.rstrip("\n"))
    assert "untrusted" in with_add and "Report tool failures explicitly" in with_add
    assert "{scratch}" not in with_add
    assert "/scratch" in system_prompt("files", scratch="/scratch")
    assert system_prompt("files", scratch="/x").count("/x") >= 1
    assert (PROMPTS_DIR / "hardening_addendum.md").read_text().strip() in with_add


def test_langgraph_prompt_gets_addendum() -> None:
    plain = _load_prompt("files", "/scratch", None).content[0]
    hard = _load_prompt("files", "/scratch", FULL).content[0]
    assert isinstance(plain, dict) and isinstance(hard, dict)
    assert "Handling tool output" not in plain["text"]
    assert "Handling tool output" in hard["text"]


def test_entry_builds_guard_chain(tmp_path: Path) -> None:
    plan = FaultPlan(plan_id="clean", seed=0, faults=[])
    cfg = RunConfig.with_derived_id(
        framework="langgraph", task_id="T1", fault_plan=plan, hardening=FULL, seed=5, budget_usd=0
    )
    cmds = build_cmds(cfg, tmp_path / "plan.json", tmp_path, ["data"])
    cmd = cmds["data"]
    assert cmd[cmd.index("--mode") + 1] == "guard"
    assert json.loads(cmd[cmd.index("--guard-config") + 1])["seed"] == 5
    plain = RunConfig.with_derived_id(
        framework="langgraph", task_id="T1", fault_plan=plan, seed=5, budget_usd=0
    )
    assert "--mode" not in build_cmds(plain, tmp_path / "plan.json", tmp_path, ["data"])["data"]
