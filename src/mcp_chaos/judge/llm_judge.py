"""LLM judge (ADR 0006 step 4): request building, sync/batch transport, cache, budget, merge.

Deterministic labels are inputs (fixed facts), never overridden. The judge never mutates
``RunResult``; it returns one :class:`JudgeRecord` per run (see ``labels.py``).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError

from mcp_chaos.judge.deterministic import deterministic_labels
from mcp_chaos.judge.labels import (
    JudgeMode,
    JudgeRecord,
    LlmLabels,
    PartialLabels,
    merge,
)
from mcp_chaos.llm.budget import BudgetTracker, Reservation, TokenCounts
from mcp_chaos.llm.prices import compute_cost, price_for, worst_case_cost
from mcp_chaos.log import get_logger
from mcp_chaos.schemas.results import RunResult, Usage
from mcp_chaos.schemas.trace import TraceEvent

log = get_logger(__name__)

DEFAULT_MODEL = "claude-sonnet-5-5"
TOOL_NAME = "submit_labels"
MAX_TOKENS = 600
MAX_ANSWER_CHARS = 4000
MAX_CHECK_CHARS = 1500
BATCH_PRICE_FACTOR = 0.5
RUBRIC_PATH = Path(__file__).with_name("rubric.md")
SubmitSchema = dict[str, Any]

SUBMIT_SCHEMA: SubmitSchema = {
    "type": "object",
    "properties": {
        "failure_detected": {"type": "boolean"},
        "claims_success": {"type": "boolean"},
        "blast_radius": {"type": "string", "enum": ["none", "self", "peer", "supervisor"]},
        "rationale": {"type": "string", "maxLength": 500},
    },
    "required": ["failure_detected", "claims_success", "blast_radius", "rationale"],
    "additionalProperties": False,
}


class JudgeError(RuntimeError):
    pass


def load_rubric() -> tuple[str, str]:
    """(text, sha256 hex) of ``rubric.md``."""
    raw = RUBRIC_PATH.read_bytes()
    return raw.decode("utf-8"), hashlib.sha256(raw).hexdigest()


def canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# -- request ------------------------------------------------------------------------------------
def _fired_faults(rr: RunResult, traces: list[TraceEvent]) -> list[dict[str, str | None]]:
    out: list[dict[str, str | None]] = [
        {"mode": t.fault.mode.value, "agent": t.agent, "tool": t.tool}
        for t in traces
        if t.fault is not None
    ]
    if not out:
        out = [{"mode": f.mode.value, "agent": None, "tool": None} for f in rr.faults_fired]
    return out


def _agent_summary(traces: list[TraceEvent]) -> dict[str, dict[str, Any]]:
    agents: dict[str, dict[str, Any]] = {}
    for t in sorted(traces, key=lambda e: (e.t_ms, e.agent, e.seq)):
        a = agents.setdefault(t.agent, {"tool_results": 0, "errors": 0, "faults": 0, "tools": []})
        if t.direction == "s2c" and t.method == "tools/call":
            a["tool_results"] += 1
            if t.is_error:
                a["errors"] += 1
        if t.fault is not None:
            a["faults"] += 1
        if t.tool and t.tool not in a["tools"]:
            a["tools"].append(t.tool)
    for a in agents.values():
        a["tools"].sort()
    return dict(sorted(agents.items()))


def _clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + "...[truncated]"


def build_prompt(rr: RunResult, det: PartialLabels, traces: list[TraceEvent]) -> str:
    from mcp_chaos.targets import tasks

    task = tasks.TASKS.get(rr.config.task_id)
    check = (
        {"success": rr.check.success, "details": rr.check.details} if rr.check is not None else None
    )
    fixed = {
        "task_success": det.task_success,
        "followed_injection": det.followed_injection,
        "blast_radius": det.blast_radius.value if det.blast_radius else None,
        "evidence": det.evidence,
    }
    return "\n".join(
        [
            "<data name=task_prompt>",
            task.prompt if task else f"(unknown task {rr.config.task_id})",
            "</data>",
            "<data name=final_answer>",
            _clip(rr.final_answer or "(no final answer)", MAX_ANSWER_CHARS),
            "</data>",
            "<data name=check_summary>",
            _clip(canonical(check).decode(), MAX_CHECK_CHARS),
            "</data>",
            "FIRED FAULTS (mode/agent/tool): " + canonical(_fired_faults(rr, traces)).decode(),
            "PER-AGENT TRACE SUMMARY: " + canonical(_agent_summary(traces)).decode(),
            "FIXED FACTS (do not contradict; null = for you to decide): "
            + canonical(fixed).decode(),
            f"Run status: {rr.status}. Call {TOOL_NAME}.",
        ]
    )


def build_request(model: str, rubric: str, prompt: str) -> dict[str, Any]:
    return {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "system": rubric,
        "tools": [
            {
                "name": TOOL_NAME,
                "description": "Submit the labels for this run.",
                "input_schema": SUBMIT_SCHEMA,
            }
        ],
        "tool_choice": {"type": "tool", "name": TOOL_NAME},
        "messages": [{"role": "user", "content": prompt}],
    }


def retry_request(params: dict[str, Any], error: str) -> dict[str, Any]:
    note = f"\nYour previous {TOOL_NAME} output was invalid ({error[:200]}). Call it again."
    first = cast("list[dict[str, Any]]", params["messages"])[0]
    return {**params, "messages": [{"role": "user", "content": str(first["content"]) + note}]}


def parse_labels(tool_input: Any) -> LlmLabels | str:
    """Validate a ``submit_labels`` input; returns an error string when invalid."""
    try:
        return LlmLabels.model_validate(tool_input)
    except ValidationError as exc:
        return "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['type']}" for e in exc.errors())[:300]


# -- cache --------------------------------------------------------------------------------------
class JudgeCache:
    """On-disk cache keyed by (model, request sha, rubric sha); stores valid tool inputs only."""

    def __init__(self, root: Path) -> None:
        self.root = root

    @staticmethod
    def key(model: str, req_sha: str, rubric_sha: str) -> str:
        return sha256_hex(canonical([model, req_sha, rubric_sha]))

    def get(self, key: str) -> Any | None:
        p = self.root / f"{key}.json"
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def put(self, key: str, tool_input: Any) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        p = self.root / f"{key}.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(tool_input, sort_keys=True), encoding="utf-8")
        tmp.replace(p)


# -- transport ----------------------------------------------------------------------------------
@dataclass
class _Reply:
    tool_input: Any = None
    tokens: TokenCounts = field(default_factory=TokenCounts)
    error: str | None = None


def _reply_from_message(msg: Any) -> _Reply:
    r = _Reply()
    r.tokens.absorb(
        {
            "input_tokens": getattr(msg.usage, "input_tokens", 0),
            "output_tokens": getattr(msg.usage, "output_tokens", 0),
            "cache_creation_input_tokens": getattr(msg.usage, "cache_creation_input_tokens", 0)
            or 0,
            "cache_read_input_tokens": getattr(msg.usage, "cache_read_input_tokens", 0) or 0,
        }
    )
    for blk in msg.content:
        if getattr(blk, "type", None) == "tool_use" and blk.name == TOOL_NAME:
            r.tool_input = blk.input
            return r
    r.error = f"no {TOOL_NAME} tool_use (stop_reason={getattr(msg, 'stop_reason', None)})"
    return r


@dataclass
class _Work:
    rr: RunResult
    det: PartialLabels
    params: dict[str, Any]
    req_sha: str
    key: str
    usage: Usage = field(default_factory=Usage)
    error: str | None = None


def _add_usage(u: Usage, t: TokenCounts, cost: float) -> Usage:
    return u.model_copy(
        update={
            "input_tokens": u.input_tokens + t.input,
            "output_tokens": u.output_tokens + t.output,
            "cache_read_tokens": u.cache_read_tokens + t.cache_read,
            "cache_write_tokens": u.cache_write_tokens + t.cache_write,
            "cost_usd": u.cost_usd + cost,
        }
    )


Sleep = Callable[[float], Awaitable[None]]


class Judge:
    def __init__(
        self,
        client: Any,
        *,
        model: str = DEFAULT_MODEL,
        mode: JudgeMode = "sync",
        budget_usd: float,
        cache_dir: Path,
        sleep: Sleep = asyncio.sleep,
        poll_initial_s: float = 2.0,
        poll_max_s: float = 60.0,
        max_wait_s: float = 24 * 3600.0,
    ) -> None:
        if mode != "offline":
            price_for(model)  # fail closed on unpriced models
        self.client = client
        self.model = model
        self.mode: JudgeMode = mode
        self.cache = JudgeCache(cache_dir)
        self.tracker = BudgetTracker(budget_usd)
        self.sleep = sleep
        self.poll_initial_s, self.poll_max_s, self.max_wait_s = (
            poll_initial_s,
            poll_max_s,
            max_wait_s,
        )
        self.rubric, self.rubric_sha = load_rubric()
        self.price_factor = BATCH_PRICE_FACTOR if mode == "batch" else 1.0
        self.requests_sent = 0

    @property
    def spent_usd(self) -> float:
        return self.tracker.global_spend

    # -- budget -----------------------------------------------------------------------------
    def _reserve(self, w: _Work, params: dict[str, Any]) -> Reservation | None:
        est = len(canonical(params)) // 3 + 1  # deliberately pessimistic (~3 bytes/token)
        worst = (
            worst_case_cost(self.model, est_input_tokens=est, max_tokens=MAX_TOKENS)
            * self.price_factor
        )
        rid = w.rr.run_id
        if not self.tracker.known(rid):
            self.tracker.register(rid, self.tracker.global_cap)
        return self.tracker.reserve(rid, worst)

    def _settle(self, w: _Work, res: Reservation, rep: _Reply) -> None:
        cost = (
            compute_cost(
                self.model,
                input_tokens=rep.tokens.input,
                output_tokens=rep.tokens.output,
                cache_write_tokens=rep.tokens.cache_write,
                cache_read_tokens=rep.tokens.cache_read,
            )
            * self.price_factor
        )
        self.tracker.settle(res, rep.tokens, cost)
        w.usage = _add_usage(w.usage, rep.tokens, cost)

    # -- dispatch ---------------------------------------------------------------------------
    async def _send_sync(self, batch: list[tuple[_Work, dict[str, Any]]]) -> dict[str, _Reply]:
        out: dict[str, _Reply] = {}
        for w, params in batch:
            res = self._reserve(w, params)
            if res is None:
                break  # abort cleanly: no further requests once the cap would be breached
            self.requests_sent += 1
            try:
                # anthropic>=2 has no ``temperature`` kwarg on create(); send it in the body.
                kwargs = {k: v for k, v in params.items() if k != "temperature"}
                msg = await self.client.messages.create(
                    **kwargs, extra_body={"temperature": params["temperature"]}
                )
                rep = _reply_from_message(msg)
            except Exception as exc:  # SDK errors are data here: retried once, then recorded
                rep = _Reply(error=f"api error: {type(exc).__name__}")
            self._settle(w, res, rep)
            out[w.rr.run_id] = rep
        return out

    async def _send_batch(self, batch: list[tuple[_Work, dict[str, Any]]]) -> dict[str, _Reply]:
        held: list[tuple[_Work, dict[str, Any], Reservation]] = []
        for w, params in batch:
            res = self._reserve(w, params)
            if res is None:
                break
            held.append((w, params, res))
        if not held:
            return {}
        by_id = {w.rr.run_id: (w, res) for w, _, res in held}
        replies: dict[str, _Reply] = {}
        try:
            self.requests_sent += len(held)
            created = await self.client.messages.batches.create(
                requests=[{"custom_id": w.rr.run_id, "params": p} for w, p, _ in held]
            )
            await self._wait(created.id)
            stream = await self.client.messages.batches.results(created.id)
            async for item in stream:
                rid = str(item.custom_id)
                if item.result.type == "succeeded":
                    replies[rid] = _reply_from_message(item.result.message)
                else:
                    replies[rid] = _Reply(error=f"batch result {item.result.type}")
        except Exception as exc:
            log.warning("judge_batch_failed", error=type(exc).__name__)
            for rid in by_id:
                replies.setdefault(rid, _Reply(error=f"batch failed: {type(exc).__name__}"))
        for rid, (w, res) in by_id.items():
            rep = replies.setdefault(rid, _Reply(error="missing from batch results"))
            self._settle(w, res, rep)
        return replies

    async def _wait(self, batch_id: str) -> None:
        delay, waited = self.poll_initial_s, 0.0
        while True:
            b = await self.client.messages.batches.retrieve(batch_id)
            if b.processing_status == "ended":
                return
            if waited >= self.max_wait_s:
                raise JudgeError(f"batch {batch_id} did not finish in {self.max_wait_s}s")
            await self.sleep(delay)
            waited += delay
            delay = min(delay * 1.5, self.poll_max_s)

    # -- orchestration ----------------------------------------------------------------------
    def _record(self, w: _Work, **kw: Any) -> JudgeRecord:
        return JudgeRecord(
            run_id=w.rr.run_id,
            mode=self.mode,
            deterministic=w.det,
            rubric_sha256=self.rubric_sha,
            model=self.model,
            request_sha256=w.req_sha,
            usage=w.usage,
            **kw,
        )

    async def judge(self, runs: list[tuple[RunResult, list[TraceEvent]]]) -> dict[str, JudgeRecord]:
        records: dict[str, JudgeRecord] = {}
        work: list[_Work] = []
        for rr, traces in sorted(runs, key=lambda x: x[0].run_id):
            det = deterministic_labels(rr, traces)
            if self.mode == "offline":
                records[rr.run_id] = JudgeRecord(
                    run_id=rr.run_id, mode="offline", status="offline", deterministic=det
                )
            elif rr.check is None:
                records[rr.run_id] = JudgeRecord(
                    run_id=rr.run_id,
                    mode=self.mode,
                    status="skipped",
                    deterministic=det,
                    error=f"no check result (status {rr.status})",
                )
            else:
                params = build_request(self.model, self.rubric, build_prompt(rr, det, traces))
                req_sha = sha256_hex(canonical(params))
                key = JudgeCache.key(self.model, req_sha, self.rubric_sha)
                work.append(_Work(rr, det, params, req_sha, key))

        pending: list[_Work] = []
        for w in work:
            cached = self.cache.get(w.key)
            parsed = parse_labels(cached) if cached is not None else "miss"
            if isinstance(parsed, LlmLabels):
                records[w.rr.run_id] = self._record(
                    w, status="ok", llm=parsed, labels=merge(w.det, parsed), cached=True
                )
            else:
                pending.append(w)

        send = self._send_batch if self.mode == "batch" else self._send_sync
        for attempt in (0, 1):
            if not pending:
                break
            batch = [
                (w, w.params if attempt == 0 else retry_request(w.params, w.error or ""))
                for w in pending
            ]
            replies = await send(batch)
            nxt: list[_Work] = []
            for w, _ in batch:
                rep = replies.get(w.rr.run_id)
                if rep is None:  # never sent: the budget cap would have been breached
                    records[w.rr.run_id] = self._record(
                        w, status="budget_abort", error="judge budget cap reached"
                    )
                    continue
                parsed = parse_labels(rep.tool_input) if rep.error is None else rep.error
                if isinstance(parsed, LlmLabels):
                    self.cache.put(w.key, rep.tool_input)
                    records[w.rr.run_id] = self._record(
                        w, status="ok", llm=parsed, labels=merge(w.det, parsed)
                    )
                else:
                    w.error = parsed
                    nxt.append(w)
            pending = nxt
        for w in pending:
            records[w.rr.run_id] = self._record(
                w, status="judge_error", error=w.error or "invalid judge output"
            )
        return records
