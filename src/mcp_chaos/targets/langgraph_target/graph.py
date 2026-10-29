"""LangGraph target: supervisor agent delegating to files/data/vcs worker agents (ADR 0005)."""

from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import AsyncExitStack
from typing import Any, Literal, cast

import anyio
from langchain.agents import create_agent  # pyright: ignore[reportUnknownVariableType]
from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ConfigDict

from mcp_chaos.log import get_logger
from mcp_chaos.schemas.runs import HardeningConfig, RunConfig
from mcp_chaos.targets.langgraph_target.mcp_tools import worker_tools
from mcp_chaos.targets.workers import WORKERS, WorkerName, system_prompt

log = get_logger(__name__)

ModelFactory = Callable[[str], BaseChatModel]


class LangGraphOutcome(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    final_answer: str | None
    turns: int
    error: str | None
    status: Literal["completed", "agent_error", "timeout"]


class _TurnCounter(BaseCallbackHandler):
    """Counts chat-model calls across the supervisor and all workers."""

    def __init__(self) -> None:
        self.turns = 0

    def on_chat_model_start(self, *args: Any, **kwargs: Any) -> None:
        self.turns += 1


def _load_prompt(role: str, scratch: str, hardening: HardeningConfig | None) -> SystemMessage:
    text = system_prompt(role, hardening, scratch=scratch)
    return SystemMessage(
        content=[{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]
    )


def default_model_factory(cfg: RunConfig) -> ModelFactory:
    def build(_role: str) -> BaseChatModel:
        m = cfg.agent_model
        return ChatAnthropic(  # pyright: ignore[reportCallIssue]
            model_name=m.model,
            base_url=os.environ["ANTHROPIC_BASE_URL"],
            api_key=os.environ["ANTHROPIC_API_KEY"],  # pyright: ignore[reportArgumentType]
            temperature=m.temperature,
            max_tokens_to_sample=m.max_tokens,
            timeout=None,
            stop=None,
        )

    return build


def message_text(msg: BaseMessage) -> str:
    """Plain text of a message (content blocks flattened)."""
    content = msg.content
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)


def _first_leaf(exc: BaseException) -> BaseException:
    cur: BaseException = exc
    while True:
        nested: Any = getattr(cur, "exceptions", None)
        if not isinstance(cur, BaseExceptionGroup) or not nested:
            return cast("BaseException", cur)
        cur = nested[0]


def short_error(exc: BaseException) -> str:
    leaf = _first_leaf(exc)
    msg = " ".join(str(leaf).split())[:300]
    return f"{type(leaf).__name__}: {msg}" if msg else type(leaf).__name__


def _delegate_tool(
    name: WorkerName, agent: Any, recursion_limit: int, description: str
) -> BaseTool:
    async def ask(instruction: str, config: RunnableConfig) -> str:
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=instruction)]},
            config={"callbacks": config.get("callbacks"), "recursion_limit": recursion_limit},
        )
        return message_text(result["messages"][-1])

    return StructuredTool.from_function(coroutine=ask, name=f"ask_{name}", description=description)


async def _run(
    task_prompt: str,
    cfg: RunConfig,
    cmds: dict[str, list[str]],
    factory: ModelFactory,
    counter: _TurnCounter,
) -> str | None:
    scratch = os.environ.get("MCP_CHAOS_SCRATCH", "/scratch")
    limit = cfg.max_turns * 2 + 1
    async with AsyncExitStack() as stack:
        delegates: list[BaseTool] = []
        for name, spec in WORKERS.items():
            if name not in cmds:
                continue
            tools = await stack.enter_async_context(worker_tools(spec, cmds[name]))
            agent: Any = create_agent(  # pyright: ignore[reportUnknownVariableType]
                factory(name),
                tools,
                system_prompt=_load_prompt(name, scratch, cfg.hardening),
                name=name,
            )
            delegates.append(
                _delegate_tool(name, agent, limit, f"Delegate an instruction to the {name} worker.")
            )
        supervisor: Any = create_agent(  # pyright: ignore[reportUnknownVariableType]
            factory("supervisor"),
            delegates,
            system_prompt=_load_prompt("supervisor", scratch, cfg.hardening),
            name="supervisor",
        )
        result = await supervisor.ainvoke(
            {"messages": [HumanMessage(content=task_prompt)]},
            config={"recursion_limit": limit, "callbacks": [counter]},
        )
    messages: list[BaseMessage] = result["messages"]
    return message_text(messages[-1]) if messages else None


async def run_langgraph(
    task_prompt: str,
    cfg: RunConfig,
    cmds: dict[str, list[str]],
    *,
    model_factory: ModelFactory | None = None,
) -> LangGraphOutcome:
    factory = model_factory or default_model_factory(cfg)
    counter = _TurnCounter()
    try:
        with anyio.fail_after(cfg.timeout_s):
            answer = await _run(task_prompt, cfg, cmds, factory, counter)
    except TimeoutError:
        log.warning("langgraph_timeout", run_id=cfg.run_id)
        return LangGraphOutcome(
            final_answer=None, turns=counter.turns, error="timeout", status="timeout"
        )
    except Exception as e:
        err = short_error(e)
        log.warning("langgraph_error", run_id=cfg.run_id, error=err)
        return LangGraphOutcome(
            final_answer=None, turns=counter.turns, error=err, status="agent_error"
        )
    return LangGraphOutcome(
        final_answer=answer, turns=counter.turns, error=None, status="completed"
    )
