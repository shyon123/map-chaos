"""Claude Agent SDK target end to end, $0: bundled CLI -> our gateway -> scripted fake Anthropic.

Nothing here reaches api.anthropic.com: the gateway's upstream is the in-process fake, and the
CLI is pointed at the gateway through ``ANTHROPIC_BASE_URL``. Real sqlite server behind the real
fault proxy, started by the CLI as an MCP stdio server.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest
from claude_agent_sdk import ResultMessage

from mcp_chaos.bench.gateway_host import GatewayHost
from mcp_chaos.llm.fake_anthropic import FakeScript, create_fake_anthropic
from mcp_chaos.llm.gateway import GatewayConfig
from mcp_chaos.schemas import FaultPlan, ModelConfig, RunConfig, TraceEvent
from mcp_chaos.targets.agentsdk_target import run as agentsdk
from mcp_chaos.targets.agentsdk_target.run import (
    BUILTIN_TOOLS,
    AgentSdkOutcome,
    agentsdk_self_test,
    run_agentsdk,
    short_error,
)
from mcp_chaos.targets.workers import DATA, FILES, VCS, proxy_cmd, upstream_cmd
from tests.unit.proxy_helpers import injection_spec, plan

CANARY = "CANARY-0badf00d"
VALUE = "zebra-4217"
SUP = "supervisor coordinating"  # in the supervisor prompt only
WORKER = "You are the data worker"
SQL = "select v from t"


def _cli_unavailable() -> str | None:
    try:
        from claude_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport

        cli = SubprocessCLITransport._find_bundled_cli(None)  # pyright: ignore[reportArgumentType]
    except Exception as e:  # pragma: no cover
        return f"claude_agent_sdk transport unavailable: {type(e).__name__}"
    if cli is None:
        return "no bundled Claude Code CLI in this claude-agent-sdk wheel"
    try:
        r = subprocess.run([cli, "--version"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as e:  # pragma: no cover
        return f"bundled Claude Code CLI cannot run here: {type(e).__name__}: {e}"
    if r.returncode != 0:  # pragma: no cover
        return f"bundled Claude Code CLI exited {r.returncode}: {r.stderr[:200]}"
    return None


_SKIP = _cli_unavailable()
pytestmark = [
    pytest.mark.skipif(_SKIP is not None, reason=_SKIP or ""),
    pytest.mark.anyio,
]


def tool_use(name: str, **inp: Any) -> dict[str, Any]:
    return {"type": "tool_use", "name": name, "input": inp}


def text(t: str) -> dict[str, Any]:
    return {"type": "text", "text": t}


def rule(match: dict[str, Any], *blocks: dict[str, Any]) -> dict[str, Any]:
    return {"match": match, "respond": list(blocks)}


def delegate_script(*, worker_done: str = "The value is {{last_tool_result}}") -> FakeScript:
    return FakeScript.model_validate(
        {
            "rules": [
                rule(
                    {"system_contains": SUP, "turn": 0},
                    tool_use(
                        "Agent", description="read", prompt="read the value", subagent_type="data"
                    ),
                ),
                rule(
                    {"system_contains": SUP, "tool_result_contains": ""},
                    text("Final answer: {{last_tool_result}}"),
                ),
                rule(
                    {"system_contains": WORKER, "turn": 0},
                    tool_use("mcp__sqlite_data__read_query", query=SQL),
                ),
                rule({"system_contains": WORKER, "tool_result_contains": ""}, text(worker_done)),
            ],
            "fallback": {"content": [text("FAKE: no rule matched")]},
        }
    )


class SlowTransport(httpx.AsyncBaseTransport):
    """Delays every upstream response (runs in the gateway's event loop)."""

    def __init__(self, inner: httpx.AsyncBaseTransport, delay_s: float) -> None:
        self.inner, self.delay_s = inner, delay_s

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(self.delay_s)
        return await self.inner.handle_async_request(request)


@dataclass
class Harness:
    gw: GatewayHost
    fake: Any
    scratch: Path
    tmp: Path

    def cfg(self, **kw: Any) -> RunConfig:
        d: dict[str, Any] = {
            "run_id": "r1",
            "framework": "agentsdk",
            "task_id": "t",
            "agent_model": ModelConfig(),
            "fault_plan": FaultPlan(plan_id="p", seed=1, faults=[]),
            "seed": 1,
            "max_turns": 10,
            "timeout_s": 90,
            "budget_usd": 0,
        }
        d.update(kw)
        return RunConfig(**d)

    def env(self) -> dict[str, str]:
        return {
            "ANTHROPIC_BASE_URL": self.gw.base_url("r1"),
            "ANTHROPIC_API_KEY": "dummy-not-a-key",
            "MCP_CHAOS_SCRATCH": str(self.scratch),
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", str(self.tmp)),
            "TMPDIR": str(self.tmp),
        }

    def cmds(self, p: FaultPlan | None = None) -> dict[str, list[str]]:
        plan_path = self.tmp / "plan.json"
        plan_path.write_text((p or plan()).model_dump_json(), encoding="utf-8")
        up = upstream_cmd(DATA, env="local", scratch=self.scratch)
        cmd = proxy_cmd(
            DATA, run_id="r1", plan_path=plan_path, trace_dir=self.tmp / "traces", upstream=up
        )
        return {"data": cmd}

    def request_tools(self) -> list[list[str]]:
        return [[t["name"] for t in b.get("tools", [])] for _, b in self.fake.state.requests]


def make_harness(
    tmp_path: Path, mp: pytest.MonkeyPatch, script: FakeScript, delay_s: float = 0
) -> Iterator[Harness]:
    mp.setenv("ANTHROPIC_API_KEY", "sk-ant-never-used-upstream-is-the-fake")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    con = sqlite3.connect(scratch / "db.sqlite")
    con.execute("create table t(v text)")
    con.execute("insert into t values (?)", (VALUE,))
    con.commit()
    con.close()
    fake = create_fake_anthropic(script)
    inner: httpx.AsyncBaseTransport = httpx.ASGITransport(fake)
    cfg = GatewayConfig(
        mode="record",
        cache_dir=tmp_path / "cache",
        cassette_dir=tmp_path / "cassettes",
        global_budget_usd=5.0,
        transport=SlowTransport(inner, delay_s) if delay_s else inner,
    )
    with GatewayHost(cfg) as gw:
        gw.register_run("r1", 1, 2.0)
        yield Harness(gw=gw, fake=fake, scratch=scratch, tmp=tmp_path)


@pytest.fixture
def h(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Harness]:
    yield from make_harness(tmp_path, monkeypatch, delegate_script())


@pytest.fixture(params=[8.0], ids=["slow-upstream"])
def slow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> Iterator[Harness]:
    yield from make_harness(tmp_path, monkeypatch, delegate_script(), delay_s=request.param)


def tool_results(h: Harness) -> list[str]:
    out: list[str] = []
    for _, body in h.fake.state.requests:
        for m in body["messages"]:
            if isinstance(m.get("content"), list):
                out += [
                    str(b.get("content")) for b in m["content"] if b.get("type") == "tool_result"
                ]
    return out


async def test_supervisor_delegates_to_data_subagent(h: Harness) -> None:
    out = await run_agentsdk("How many rows?", h.cfg(), h.cmds(), env=h.env())
    assert out.status == "completed", out.error
    assert out.error is None
    assert out.final_answer is not None and VALUE in out.final_answer
    # supervisor: delegate + final; worker: query + report
    assert out.turns == 4
    assert out.cost_usd is not None and out.cost_usd > 0
    assert h.fake.state.misses == []
    assert any(VALUE in r for r in tool_results(h))
    assert h.gw.usage("r1").cost_usd > 0


async def test_builtin_tools_are_not_offered(h: Harness) -> None:
    out = await run_agentsdk("x", h.cfg(), h.cmds(), env=h.env())
    assert out.status == "completed", out.error
    offered = h.request_tools()
    assert offered, "the CLI sent no model requests"
    flat = {t for tools in offered for t in tools}
    assert not flat & set(BUILTIN_TOOLS)
    allowed = {f"mcp__sqlite_data__{t}" for t in DATA.allow}
    assert flat <= {"Agent"} | allowed
    # the supervisor holds only the delegation tool, workers only their own MCP tools
    for (_, body), tools in zip(h.fake.state.requests, offered, strict=True):
        if SUP in str(body["system"]):
            assert tools == ["Agent"]
        else:
            assert tools and set(tools) <= allowed
    # no side requests: every call is a model request through the gateway
    assert {p for p, _ in h.fake.state.requests} == {"/v1/messages"}


@pytest.mark.parametrize(("position", "visible"), [("field", True), ("suffix", False)])
async def test_f08_canary_reaches_subagent_and_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, position: Any, visible: bool
) -> None:
    """The proxy records the injection either way. The CLI, however, shows the model only a
    tool result's ``structuredContent`` when the server sent one (our sqlite server does), so a
    ``suffix`` content block never reaches the subagent; ``field`` rewrites structuredContent.
    """
    script = delegate_script(worker_done="Obeying: {{last_tool_result}}")
    for h in make_harness(tmp_path, monkeypatch, script):
        p = plan(injection_spec(position, canary=CANARY, tool="read_query", calls=[0]))
        out = await run_agentsdk("x", h.cfg(), h.cmds(p), env=h.env())
        assert out.status == "completed", out.error
        worker_bodies = [b for _, b in h.fake.state.requests if WORKER in str(b["system"])]
        assert any(CANARY in str(b["messages"]) for b in worker_bodies) is visible
        assert (out.final_answer is not None and CANARY in out.final_answer) is visible
        lines = (h.tmp / "traces" / "data-sqlite.jsonl").read_text().splitlines()
        faults = [e.fault for e in map(TraceEvent.model_validate_json, lines) if e.fault]
        assert len(faults) == 1
        assert faults[0].mode == "prompt_injection"
        assert faults[0].detail["canary"] == CANARY


async def test_timeout(slow: Harness) -> None:
    out = await run_agentsdk("x", slow.cfg(timeout_s=3), slow.cmds(), env=slow.env())
    assert out.status == "timeout"
    assert out.error == "timeout" and out.final_answer is None


async def test_max_turns_is_agent_error(h: Harness) -> None:
    out = await run_agentsdk("x", h.cfg(max_turns=1), h.cmds(), env=h.env())
    assert out.status == "agent_error"
    assert out.error is not None and out.error.startswith("error_max_turns")
    assert out.final_answer is None


async def test_self_test_starts_servers_through_proxy(h: Harness) -> None:
    report = await agentsdk_self_test(h.cfg(), h.cmds(), h.env())
    assert report == {"workers": {"data": {"status": "connected", "missing": []}}, "ok": True}
    assert h.fake.state.requests == []  # no model call


async def test_self_test_reports_failed_server(h: Harness) -> None:
    cmds = {"data": [sys.executable, "-c", "import sys; sys.exit(3)"]}
    report = await agentsdk_self_test(h.cfg(), cmds, h.env(), wait_s=30)
    assert report["ok"] is False
    assert report["workers"]["data"]["status"] == "failed"
    assert report["workers"]["data"]["missing"] == sorted(DATA.allow)


async def test_cli_failure_is_agent_error(h: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    class Boom:
        def __init__(self, *a: Any, **k: Any) -> None:
            raise ExceptionGroup("g", [OSError("cli exploded")])

    monkeypatch.setattr(agentsdk, "ClaudeSDKClient", Boom)
    out = await run_agentsdk("x", h.cfg(), h.cmds(), env=h.env())
    assert out.status == "agent_error"
    assert out.error == "OSError: cli exploded"


async def test_stream_without_result_is_agent_error(
    h: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def nothing(*a: Any, **k: Any) -> None:
        return None

    monkeypatch.setattr(agentsdk, "_converse", nothing)
    out = await run_agentsdk("x", h.cfg(), h.cmds(), env=h.env())
    assert out.status == "agent_error" and out.turns == 0
    assert out.error is not None and "no result" in out.error


def test_options_isolate_the_cli(tmp_path: Path) -> None:
    cmds = {
        "files": ["proxy-files"],
        "data": ["proxy-data"],
        "vcs": ["proxy-vcs"],
    }
    cfg = RunConfig(
        run_id="r1",
        framework="agentsdk",
        task_id="t",
        agent_model=ModelConfig(model="claude-haiku-5-5", max_tokens=777),
        fault_plan=FaultPlan(plan_id="p", seed=1, faults=[]),
        seed=1,
        max_turns=7,
        budget_usd=0,
    )
    env = {
        "ANTHROPIC_BASE_URL": "http://gw/r/r1",
        "ANTHROPIC_API_KEY": "dummy",
        "PATH": "/bin",
        "AWS_SECRET_ACCESS_KEY": "must-not-reach-servers",
        "GIT_CONFIG_COUNT": "1",
    }
    o = agentsdk.build_options(cfg, cmds, env, tmp_path)
    assert o.tools == ["Agent"] and o.permission_mode == "dontAsk"
    assert set(BUILTIN_TOOLS) <= set(o.disallowed_tools) and "Agent" not in o.disallowed_tools
    assert o.max_turns == 7 and o.model == "claude-haiku-5-5"
    assert set(o.mcp_servers) == {"fs_files", "sqlite_data", "git_vcs"}  # type: ignore[arg-type]
    assert o.agents is not None
    assert o.agents["supervisor"].tools == ["Agent(files,data,vcs)"]
    for name, spec in (("files", FILES), ("data", DATA), ("vcs", VCS)):
        d = o.agents[name]
        entry = agentsdk.entry_name(spec)
        assert d.tools is not None and len(d.tools) == len(spec.allow)
        assert all(t.startswith(f"mcp__{entry}__") for t in d.tools)
        assert set(d.tools) <= set(o.allowed_tools)
        assert "{scratch}" not in d.prompt and f"You are the {name} worker" in d.prompt
        srv: Any = o.mcp_servers[entry]  # type: ignore[index]
        assert srv["command"] == f"proxy-{name}"
        assert srv["env"] == {"PATH": "/bin", "GIT_CONFIG_COUNT": "1"}
    assert o.env["ANTHROPIC_BASE_URL"] == "http://gw/r/r1"
    assert o.env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "777"
    assert o.env["ANTHROPIC_AUTH_TOKEN"] == ""
    assert o.env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
    assert o.env["CLAUDE_CONFIG_DIR"].startswith(str(tmp_path))
    # only the workers present get a subagent
    only = agentsdk.build_options(cfg, {"data": ["p"]}, env, tmp_path)
    assert only.agents is not None and set(only.agents) == {"data", "supervisor"}
    assert only.agents["supervisor"].tools == ["Agent(data)"]


def test_error_helpers() -> None:
    assert short_error(ValueError("")) == "ValueError"
    assert (
        short_error(ExceptionGroup("g", [ExceptionGroup("h", [KeyError("k")])])) == "KeyError: 'k'"
    )

    def result(**kw: Any) -> ResultMessage:
        base: dict[str, Any] = {
            "subtype": "error_during_execution",
            "duration_ms": 1,
            "duration_api_ms": 1,
            "is_error": True,
            "num_turns": 3,
            "session_id": "s",
        }
        return ResultMessage(**{**base, **kw})

    assert agentsdk._result_error(result()) == "error_during_execution"  # pyright: ignore[reportPrivateUsage]
    assert (
        agentsdk._result_error(result(errors=["boom  bad"]))  # pyright: ignore[reportPrivateUsage]
        == "error_during_execution: boom bad"
    )
    prog = agentsdk._Progress()  # pyright: ignore[reportPrivateUsage]
    prog.see(result(num_turns=3))
    assert prog.turns == 3  # no per-message data: fall back to the CLI's count
    out = AgentSdkOutcome(final_answer=None, turns=0, error="x", status="agent_error")
    assert out.cost_usd is None
