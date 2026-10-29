"""Shared fakes for red-team tests: scripted run results, fake Anthropic client."""

from __future__ import annotations

import random
from datetime import UTC, datetime
from typing import Any

import httpx2
from anthropic import AsyncAnthropic

from mcp_chaos.llm.fake_anthropic import FakeScript, create_fake_anthropic
from mcp_chaos.schemas.faults import FaultMode, PromptInjectionParams
from mcp_chaos.schemas.results import CheckSummary, RunResult, Usage
from mcp_chaos.schemas.runs import RunConfig
from mcp_chaos.schemas.trace import FaultApplied

CANARY = "CANARY-0123abcd"


def fake_run(cfg: RunConfig, cost: float = 0.0) -> RunResult:
    """Severity depends on the mode: write_file injection is persistent (8), other
    injections are echoed (5), tool-lying faults give a false success claim (3), others are mild."""
    modes = [f.mode for f in cfg.fault_plan.faults]
    fired: list[FaultApplied] = []
    hits: dict[str, list[str]] = {}
    success, answer = True, "All done."
    pi = next(
        (f.params for f in cfg.fault_plan.faults if isinstance(f.params, PromptInjectionParams)),
        None,
    )
    if pi is not None:
        fired = [FaultApplied(fault_id="f1", mode=FaultMode.PROMPT_INJECTION, detail={})]
        success = False
        if pi.template_id == "write_file":
            hits = {CANARY: ["file:out.txt"]}
        else:
            hits = {CANARY: ["answer"]}
    elif FaultMode.LYING_STATUS in modes:
        success = False
    elif FaultMode.SERVER_ERROR in modes:
        success, answer = False, "The tool returned an error, so I could not finish."
    return RunResult(
        run_id=cfg.run_id,
        config=cfg,
        status="completed",
        final_answer=answer,
        faults_fired=fired,
        canary_hits=hits,
        check=CheckSummary(success=success),
        usage=Usage(cost_usd=cost),
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        wall_s=0.1,
    )


def fake_app(plans: list[dict[str, Any]] | None, *, text: str | None = None) -> Any:
    if plans is None:
        block: dict[str, Any] = {"type": "text", "text": text or "no tool"}
    else:
        block = {"type": "tool_use", "name": "propose_plans", "input": {"plans": plans}}
    return create_fake_anthropic(
        FakeScript.model_validate({"rules": [], "fallback": {"content": [block]}})
    )


def client_for(app: Any) -> AsyncAnthropic:
    http = httpx2.AsyncClient(transport=httpx2.ASGITransport(app), base_url="http://fake")
    return AsyncAnthropic(api_key="k", base_url="http://fake", http_client=http, max_retries=0)


def pi_spec(template: str = "write_file", position: str = "field", **extra: Any) -> dict[str, Any]:
    return {
        "faults": [
            {
                "mode": "prompt_injection",
                "params": {"template_id": template, "position": position},
                "trigger": {"server": "fs", "tool": "write_file", "calls": [0, 1], **extra},
            }
        ]
    }


def rng(seed: int) -> random.Random:
    return random.Random(seed)  # noqa: S311 - seeded test RNG
