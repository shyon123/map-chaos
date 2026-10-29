"""In-sandbox entrypoint (runner-contract): `python -m mcp_chaos.targets.entry --config C --out O`.

Reads ``RunConfig`` (``config.json``) and the fault plan (``plan.json`` next to it), builds one
proxy-wrapped MCP server command per worker, runs the framework and writes ``result.json``.
Exit code 0 means ``result.json`` was written, whatever the agent outcome; non-zero is a harness
error. ``--self-test`` only checks MCP server startup and gateway connectivity.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import anyio
import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel, ConfigDict, PrivateAttr, ValidationError

from mcp_chaos.log import configure_logging, get_logger
from mcp_chaos.schemas.faults import FaultPlan
from mcp_chaos.schemas.runs import RunConfig
from mcp_chaos.targets.langgraph_target.graph import ModelFactory, run_langgraph
from mcp_chaos.targets.langgraph_target.mcp_tools import worker_tools
from mcp_chaos.targets.tasks import TASKS
from mcp_chaos.targets.workers import WORKERS, WorkerName, proxy_cmd, upstream_cmd

log = get_logger(__name__)

FAKE_MODELS_ENV = "MCP_CHAOS_FAKE_MODELS"
UPSTREAM_ENV = "MCP_CHAOS_UPSTREAM_ENV"
SCRATCH_ENV = "MCP_CHAOS_SCRATCH"


class EntryResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1"] = "1"
    run_id: str
    status: Literal["completed", "agent_error", "timeout"]
    final_answer: str | None = None
    error: str | None = None
    agent_turns: int = 0
    wall_s: float = 0.0


class HarnessError(RuntimeError):
    """Entry could not run at all (exit code 1, no result.json)."""


# -- scripted fake models (tests only) ---------------------------------------------------------


class _ScriptedModel(BaseChatModel):
    _messages: list[AIMessage] = PrivateAttr(default_factory=list[AIMessage])
    _i: int = PrivateAttr(default=0)

    def __init__(self, messages: list[AIMessage]) -> None:
        super().__init__()
        self._messages = messages

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> _ScriptedModel:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Any = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self._i < len(self._messages):
            msg = self._messages[self._i]
        else:
            msg = AIMessage(content="(script exhausted)")
        self._i += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])


def fake_model_factory(script: dict[str, Any]) -> ModelFactory:
    """Build a model factory from ``{"roles": {role: [{"content": str, "tool_calls": [...]}]}}``.

    Tool-call ids are generated unique per (role, step, index).
    """
    roles: dict[str, Any] = script.get("roles", {})

    def build(role: str) -> BaseChatModel:
        steps: list[dict[str, Any]] = roles.get(role, [])
        msgs: list[AIMessage] = []
        for n, step in enumerate(steps):
            calls: list[dict[str, Any]] = [
                {"name": c["name"], "args": c.get("args", {}), "id": f"{role}-{n}-{k}"}
                for k, c in enumerate(step.get("tool_calls", []))
            ]
            msgs.append(AIMessage(content=step.get("content", ""), tool_calls=calls))
        return _ScriptedModel(msgs)

    return build


def _load_fake(path: Path) -> tuple[ModelFactory, list[WorkerName] | None]:
    script: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    workers: list[WorkerName] | None = script.get("workers")
    return fake_model_factory(script), workers


# -- command construction ----------------------------------------------------------------------


def _upstream_env() -> Literal["docker", "local"]:
    return "local" if os.environ.get(UPSTREAM_ENV) == "local" else "docker"


def build_cmds(
    cfg: RunConfig, plan_path: Path, out: Path, workers: Sequence[WorkerName]
) -> dict[str, list[str]]:
    scratch = Path(os.environ.get(SCRATCH_ENV, "/scratch"))
    cmds: dict[str, list[str]] = {}
    for name in workers:
        spec = WORKERS[name]
        up = upstream_cmd(spec, env=_upstream_env(), scratch=scratch)
        cmds[name] = proxy_cmd(
            spec,
            run_id=cfg.run_id,
            plan_path=plan_path,
            trace_dir=out / "traces",
            upstream=up,
            hardening=cfg.hardening,
            seed=cfg.seed,
        )
    return cmds


def _load_config(config_path: Path) -> tuple[RunConfig, Path]:
    try:
        cfg = RunConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
        plan_path = config_path.parent / "plan.json"
        FaultPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as e:
        raise HarnessError(f"bad config/plan: {type(e).__name__}") from e
    return cfg, plan_path


# -- run ---------------------------------------------------------------------------------------


async def run_entry(config_path: Path, out: Path) -> EntryResult:
    cfg, plan_path = _load_config(config_path)
    if cfg.framework != "langgraph":
        raise HarnessError(f"framework {cfg.framework!r} is not implemented")
    task = TASKS.get(cfg.task_id)
    if task is None:
        raise HarnessError(f"unknown task {cfg.task_id!r}")
    workers: list[WorkerName] = [w for w in WORKERS if w in task.required_workers]
    factory: ModelFactory | None = None
    fake = os.environ.get(FAKE_MODELS_ENV)
    if fake:
        factory, only = _load_fake(Path(fake))
        if only is not None:
            workers = only
    cmds = build_cmds(cfg, plan_path, out, workers)
    t0 = time.monotonic()
    outcome = await run_langgraph(task.prompt, cfg, cmds, model_factory=factory)
    return EntryResult(
        run_id=cfg.run_id,
        status=outcome.status,
        final_answer=outcome.final_answer,
        error=outcome.error,
        agent_turns=outcome.turns,
        wall_s=round(time.monotonic() - t0, 3),
    )


async def run_self_test(out: Path) -> dict[str, Any]:
    """Start every MCP server through the proxy and probe the gateway (no LLM call)."""
    plan = FaultPlan(plan_id="selftest", seed=0, faults=[])
    plan_path = out / "selftest-plan.json"
    _write_atomic(plan_path, plan.model_dump_json())
    cfg = RunConfig(
        run_id="selftest",
        framework="langgraph",
        task_id="T1",
        fault_plan=plan,
        seed=0,
        budget_usd=0,
    )
    names: list[WorkerName] = list(WORKERS)
    cmds = build_cmds(cfg, plan_path, out, names)
    report: dict[str, Any] = {"workers": {}, "gateway_status": None, "ok": True}
    for name in names:
        try:
            with anyio.fail_after(60):
                async with worker_tools(WORKERS[name], cmds[name]) as tools:
                    have = {t.name for t in tools}
            missing = sorted(WORKERS[name].allow - have)
            report["workers"][name] = {"missing": missing}
            report["ok"] = report["ok"] and not missing
        except Exception as e:
            report["workers"][name] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            report["ok"] = False
    base = os.environ.get("ANTHROPIC_BASE_URL", "")
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(f"{base}/v1/messages", json={})
        report["gateway_status"] = r.status_code
        report["ok"] = report["ok"] and r.status_code == 404 and "mcp-chaos gateway" in r.text
    except httpx.HTTPError as e:
        report["gateway_status"] = f"error:{type(e).__name__}"
        report["ok"] = False
    return report


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mcp_chaos.targets.entry")
    p.add_argument("--config", type=Path, default=Path("/run/in/config.json"))
    p.add_argument("--out", type=Path, default=Path("/run/out"))
    p.add_argument("--self-test", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    configure_logging()
    out: Path = args.out
    try:
        if args.self_test:
            report = anyio.run(run_self_test, out)
            _write_atomic(out / "selftest.json", json.dumps(report))
            return 0 if report["ok"] else 1
        result = anyio.run(run_entry, args.config, out)
    except HarnessError as e:
        sys.stderr.write(f"entry: harness error: {e}\n")
        return 1
    except Exception as e:
        sys.stderr.write(f"entry: harness error: {type(e).__name__}: {str(e)[:300]}\n")
        return 1
    _write_atomic(out / "result.json", result.model_dump_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
