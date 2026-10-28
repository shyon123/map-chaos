"""LangGraph target end-to-end with scripted fake models (no LLM) through the real proxy."""

import itertools
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

from mcp_chaos.schemas import FaultPlan, ModelConfig, RunConfig, TraceEvent
from mcp_chaos.targets.langgraph_target.graph import (
    LangGraphOutcome,
    default_model_factory,
    message_text,
    run_langgraph,
    short_error,
)
from mcp_chaos.targets.workers import DATA, proxy_cmd
from tests.unit.proxy_helpers import injection_spec, plan

from .conftest import fake_server_cmd

CANARY = "CANARY-0badf00d"
Step = AIMessage | Callable[[list[BaseMessage]], AIMessage]


class ScriptedModel(BaseChatModel):
    """Emits predetermined messages (or message-builders over the history), in order."""

    _steps: list[Step] = PrivateAttr(default_factory=list)
    _i: int = PrivateAttr(default=0)

    def __init__(self, steps: list[Step]) -> None:
        super().__init__()
        self._steps = steps

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedModel":
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Any = None,
        run_manager: Any = None,
        **kw: Any,
    ) -> ChatResult:
        step = self._steps[min(self._i, len(self._steps) - 1)]
        self._i += 1
        msg = step(messages) if callable(step) else step
        return ChatResult(generations=[ChatGeneration(message=msg)])


_ids = itertools.count()


def call(name: str, args: dict[str, Any], cid: str) -> AIMessage:
    """Tool-call message; ids must be unique per call or langgraph's router breaks."""
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": f"{cid}-{next(_ids)}"}]
    )


def last_tool_text(msgs: list[BaseMessage]) -> str:
    tm = [m for m in msgs if isinstance(m, ToolMessage)][-1]
    return message_text(tm)


def cfg(timeout_s: float = 60, max_turns: int = 10) -> RunConfig:
    return RunConfig(
        run_id="r1",
        framework="langgraph",
        task_id="t",
        agent_model=ModelConfig(),
        fault_plan=FaultPlan(plan_id="p", seed=1, faults=[]),
        seed=1,
        max_turns=max_turns,
        timeout_s=timeout_s,
        budget_usd=0,
    )


def write_plan(tmp_path: Path, p: FaultPlan) -> Path:
    path = tmp_path / "plan.json"
    path.write_text(p.model_dump_json(), encoding="utf-8")
    return path


def data_cmd(
    script: Path, p: FaultPlan, tmp_path: Path, upstream: list[str] | None = None
) -> list[str]:
    worker = DATA.model_copy(update={"allow": DATA.allow | {"read_rows", "flaky"}})
    return proxy_cmd(
        worker,
        run_id="r1",
        plan_path=write_plan(tmp_path, p),
        trace_dir=tmp_path,
        upstream=upstream or fake_server_cmd(script),
    )


def factory(seen: list[str], worker_steps: list[Step]) -> Callable[[str], BaseChatModel]:
    sup: list[Step] = [
        call("ask_data", {"instruction": "count rows in the table"}, "s1"),
        lambda m: AIMessage(content=f"Done. {last_tool_text(m)}"),
    ]

    def make(role: str) -> BaseChatModel:
        seen.append(role)
        return ScriptedModel(sup if role == "supervisor" else worker_steps)

    return make


@pytest.fixture(autouse=True)
def _scratch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MCP_CHAOS_SCRATCH", str(tmp_path))


def trace_faults(tmp_path: Path) -> list[Any]:
    lines = (tmp_path / "data-sqlite.jsonl").read_text().splitlines()
    return [e.fault for e in map(TraceEvent.model_validate_json, lines) if e.fault is not None]


@pytest.mark.anyio
async def test_supervisor_delegates_to_data_worker(fake_script_path: Path, tmp_path: Path) -> None:
    seen: list[str] = []
    worker: list[Step] = [
        call("read_rows", {"table": "t"}, "w1"),
        lambda m: AIMessage(content=f"The table has {'2'} rows: {last_tool_text(m)}"),
    ]
    out = await run_langgraph(
        "How many rows?",
        cfg(),
        {"data": data_cmd(fake_script_path, plan(), tmp_path)},
        model_factory=factory(seen, worker),
    )
    assert out.status == "completed", out.error
    assert out.error is None
    assert out.final_answer is not None
    assert "The table has 2 rows" in out.final_answer
    assert '"count": 2' in out.final_answer or '"count":2' in out.final_answer
    assert out.turns == 4
    assert seen == ["data", "supervisor"]


@pytest.mark.anyio
async def test_f08_canary_reaches_worker_and_trace(fake_script_path: Path, tmp_path: Path) -> None:
    p = plan(injection_spec("suffix", canary=CANARY, tool="read_rows", calls=[0]))
    worker_saw: list[str] = []

    def finish(m: list[BaseMessage]) -> AIMessage:
        worker_saw.append(last_tool_text(m))
        return AIMessage(content="done")

    out = await run_langgraph(
        "How many rows?",
        cfg(),
        {"data": data_cmd(fake_script_path, p, tmp_path)},
        model_factory=factory([], [call("read_rows", {}, "w1"), finish]),
    )
    assert out.status == "completed", out.error
    assert any(CANARY in t for t in worker_saw)
    faults = trace_faults(tmp_path)
    assert len(faults) == 1
    assert faults[0].mode == "prompt_injection"
    assert faults[0].detail["canary"] == CANARY


@pytest.mark.anyio
async def test_tool_error_is_error_toolmessage(fake_script_path: Path, tmp_path: Path) -> None:
    saw: list[ToolMessage] = []

    def finish(m: list[BaseMessage]) -> AIMessage:
        saw.extend(x for x in m if isinstance(x, ToolMessage))
        return AIMessage(content="failed")

    out = await run_langgraph(
        "x",
        cfg(),
        {"data": data_cmd(fake_script_path, plan(), tmp_path)},
        model_factory=factory([], [call("flaky", {}, "w1"), finish]),
    )
    assert out.status == "completed", out.error
    assert saw and saw[0].status == "error"


@pytest.mark.anyio
async def test_transport_failure_is_agent_error(tmp_path: Path, fake_script_path: Path) -> None:
    cmd = data_cmd(fake_script_path, plan(), tmp_path, upstream=[sys.executable, "-c", "pass"])
    out = await run_langgraph(
        "x", cfg(), {"data": cmd}, model_factory=factory([], [AIMessage(content="never")])
    )
    assert out.status == "agent_error"
    assert out.final_answer is None
    assert out.error
    assert "Traceback" not in out.error
    assert len(out.error) < 400


@pytest.mark.anyio
async def test_timeout(fake_script_path: Path, tmp_path: Path) -> None:
    out = await run_langgraph(
        "x",
        cfg(timeout_s=0.05),
        {"data": data_cmd(fake_script_path, plan(), tmp_path)},
        model_factory=factory([], [call("read_rows", {}, "w1")]),
    )
    assert out.status == "timeout"
    assert out.error == "timeout"


@pytest.mark.anyio
async def test_recursion_limit_is_agent_error(fake_script_path: Path, tmp_path: Path) -> None:
    out = await run_langgraph(
        "x",
        cfg(max_turns=1),
        {"data": data_cmd(fake_script_path, plan(), tmp_path)},
        model_factory=lambda role: ScriptedModel(
            [lambda _m: call("ask_data", {"instruction": "i"}, "c")]
            if role == "supervisor"
            else [lambda _m: call("read_rows", {}, "w")]
        ),
    )
    assert out.status == "agent_error"
    assert out.error is not None
    assert "Recursion" in out.error


def test_helpers() -> None:
    assert message_text(AIMessage(content=[{"type": "text", "text": "a"}, "b"])) == "ab"
    assert short_error(ValueError()) == "ValueError"
    eg = ExceptionGroup("g", [RuntimeError("boom\n  now")])
    assert short_error(eg) == "RuntimeError: boom now"


def test_outcome_model() -> None:
    o = LangGraphOutcome(final_answer="a", turns=1, error=None, status="completed")
    assert o.turns == 1


def test_default_model_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://gw.invalid:8080/r/r1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-not-a-key")
    model = default_model_factory(cfg())("supervisor")
    assert isinstance(model, ChatAnthropic)
    assert model.model == "claude-haiku-5-5"
    assert model.temperature == 0.0
    assert model.max_tokens == 2048
    assert model.anthropic_api_url == "http://gw.invalid:8080/r/r1"
