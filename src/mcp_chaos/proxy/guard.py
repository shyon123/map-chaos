"""Guard runtime (ADR 0007): ``agent -> guard -> fault proxy -> upstream``.

The guard reads the agent's JSON-RPC lines, forwards them to the upstream command (normally the
fault proxy), and hardens ``tools/call`` responses with the components enabled in
``GuardConfig.hardening``:

- every ``tools/call`` is re-issued under a guard-allocated id (``mcpchaos-guard-N``); the
  client's id is restored on the reply, so retries and stale replies cannot confuse the agent;
- deadline / JSON-RPC error / unparseable line / validation failure -> bounded retry with
  seeded jittered backoff (``retry``), per-tool circuit breaker (``circuit_breaker``);
- only tools in ``GuardConfig.retry_tools`` (read-only by default) are ever re-issued: a
  timeout or malformed reply happens after the upstream may have executed, so a write is
  reported as "outcome unknown" instead of being repeated;
- a line from upstream that is not valid JSON-RPC and has no matching id is attributed to the
  attempt when exactly one call is in flight and no earlier (superseded) attempt is still
  outstanding (so it is retried immediately instead of waiting for the deadline); otherwise it
  is dropped and counted; late replies from superseded attempts are dropped and counted;
- successful results go through size cap, then injection filter.

Components are independent: with ``retry`` off there is no deadline either (a hang stays a
hang), because the deadline is what makes a retry or a breaker decision possible.
Guard actions are written to a separate trace file as ``GuardTraceEvent`` lines.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import sys
import time
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import IO, Any, cast

import anyio
from anyio.abc import Process, TaskGroup
from anyio.streams.memory import MemoryObjectSendStream
from pydantic import Field, JsonValue

from mcp_chaos.harden.config import GuardConfig
from mcp_chaos.harden.pipeline import Detail, transform_response, validation_problems
from mcp_chaos.harden.retry import BreakerBank, FailureKind, RetryPolicy, is_retryable_rpc_code
from mcp_chaos.log import get_logger
from mcp_chaos.proxy.framing import FramingError, LineSplitter
from mcp_chaos.schemas.trace import TraceEvent

log = get_logger(__name__)

UID_PREFIX = "mcpchaos-guard-"
CIRCUIT_OPEN_TEXT = "tool temporarily unavailable (circuit open)"
UNKNOWN_OUTCOME_TEXT = (
    "tool call outcome unknown: it may have executed; verify state before retrying"
)
_GRACE_S = 5.0
_CHUNK = 65536

Send = Callable[[bytes], Awaitable[None]]


class GuardTraceEvent(TraceEvent):
    """A ``TraceEvent`` plus guard details (``detail["guard_action"]`` names the action)."""

    detail: dict[str, JsonValue] = Field(default_factory=dict[str, JsonValue])


Emit = Callable[[GuardTraceEvent], Awaitable[None]]


def _dumps(obj: Any) -> bytes:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _id_key(msg_id: Any) -> tuple[str, int | str] | None:
    if isinstance(msg_id, bool):
        return None
    if isinstance(msg_id, int):
        return ("i", msg_id)
    if isinstance(msg_id, str):
        return ("s", msg_id)
    return None


def _parse(line: bytes) -> Any:
    try:
        return json.loads(line)
    except (ValueError, RecursionError):
        return None


def tool_error(orig_id: Any, text: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": orig_id,
        "result": {
            "resultType": "complete",
            "content": [{"type": "text", "text": text}],
            "isError": True,
        },
    }


class _Inflight:
    def __init__(self, tool: str, orig_id: Any) -> None:
        self.tool = tool
        self.orig_id = orig_id
        self.uid: str | None = None


class Guard:
    def __init__(
        self,
        cfg: GuardConfig,
        *,
        run_id: str,
        agent: str,
        server: str,
        send_up: Send,
        send_down: Send,
        emit: Emit,
        tg: TaskGroup,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cfg = cfg
        self._h = cfg.hardening
        self._run_id, self._agent, self._server = run_id, agent, server
        self._send_up, self._send_down, self._emit_cb = send_up, send_down, emit
        self._tg = tg
        self._now = now
        self._t0 = now()
        self._seq = itertools.count()
        self._uids = itertools.count(1)
        self._policy = RetryPolicy(
            cfg.hardening.max_retries, cfg.backoff_base_s, cfg.backoff_max_s, cfg.seed
        )
        self._rng = self._policy.rng()
        self._breakers = BreakerBank(cfg.breaker_threshold, cfg.breaker_cooldown_s)
        self._routes: dict[str, MemoryObjectSendStream[bytes]] = {}  # attempt uid -> reply stream
        self._superseded: set[str] = set()  # attempts abandoned at the deadline, reply may come
        self._inflight: dict[tuple[str, int | str], _Inflight] = {}
        self._list_pending: set[tuple[str, int | str]] = set()
        self._schemas: dict[str, dict[str, Any]] = {}
        self.dropped = 0
        self.stale = 0
        self._use_deadline = self._h.retry or self._h.circuit_breaker

    # -------------------------------------------------------------- trace

    async def _emit(
        self,
        tool: str | None,
        detail: Detail,
        *,
        is_error: bool | None = None,
        latency_ms: int | None = None,
    ) -> None:
        await self._emit_cb(
            GuardTraceEvent(
                run_id=self._run_id,
                seq=next(self._seq),
                t_ms=int((self._now() - self._t0) * 1000),
                agent=self._agent,
                server=self._server,
                direction="s2c",
                method="tools/call",
                tool=tool,
                is_error=is_error,
                latency_ms=latency_ms,
                detail=detail,
            )
        )

    # -------------------------------------------------------------- client side

    async def on_client_line(self, line: bytes) -> None:
        obj: Any = _parse(line)
        if not isinstance(obj, dict):
            await self._send_up(line)
            return
        o = cast("dict[str, Any]", obj)
        method = o.get("method")
        key = _id_key(o.get("id"))
        params: Any = o.get("params")
        if method == "tools/call" and key is not None and isinstance(params, dict):
            name: Any = cast("dict[str, Any]", params).get("name")
            if isinstance(name, str):
                self._tg.start_soon(self._call, o, name, key)
                return
        elif method == "tools/list" and key is not None:
            self._list_pending.add(key)
        elif method == "notifications/cancelled" and isinstance(params, dict):
            p = cast("dict[str, Any]", params)
            rkey = _id_key(p.get("requestId"))
            fl = self._inflight.get(rkey) if rkey is not None else None
            if fl is not None and fl.uid is not None:
                await self._send_up(_dumps({**o, "params": {**p, "requestId": fl.uid}}))
                return
        await self._send_up(line)

    # -------------------------------------------------------------- server side

    async def on_server_line(self, line: bytes) -> None:
        obj: Any = _parse(line)
        if isinstance(obj, dict):
            o = cast("dict[str, Any]", obj)
            if "method" in o:
                await self._send_down(line)
                return
            raw_id: Any = o.get("id")
            if isinstance(raw_id, str) and raw_id.startswith(UID_PREFIX):
                route = self._routes.get(raw_id)
                if route is not None:
                    route.send_nowait(line)
                else:
                    self.stale += 1
                    self._superseded.discard(raw_id)
                    log.info("guard_stale_reply", uid=raw_id)
                return
            key = _id_key(raw_id)
            if key is not None:
                if key in self._list_pending:
                    self._list_pending.discard(key)
                    self._cache_schemas(o)
                await self._send_down(line)
                return
        # garbage, a non-object, or a reply without a usable id
        if len(self._routes) == 1 and not self._superseded:
            next(iter(self._routes.values())).send_nowait(line)
            return
        self.dropped += 1
        await self._emit(
            None, {"guard_action": "dropped_line", "bytes": len(line), "dropped": self.dropped}
        )

    def _cache_schemas(self, resp: dict[str, Any]) -> None:
        result: Any = resp.get("result")
        tools: Any = (
            cast("dict[str, Any]", result).get("tools") if isinstance(result, dict) else None
        )
        if not isinstance(tools, list):
            return
        fresh: dict[str, dict[str, Any]] = {}
        for t in cast("list[Any]", tools):
            if not isinstance(t, dict):
                continue
            td = cast("dict[str, Any]", t)
            schema: Any = td.get("outputSchema")
            if isinstance(td.get("name"), str) and isinstance(schema, dict):
                fresh[cast("str", td["name"])] = cast("dict[str, Any]", schema)
        self._schemas = fresh

    # -------------------------------------------------------------- tools/call supervision

    async def _call(self, req: dict[str, Any], tool: str, key: tuple[str, int | str]) -> None:
        fl = _Inflight(tool, req.get("id"))
        self._inflight[key] = fl
        try:
            resp = await self._supervise(req, fl)
        finally:
            self._inflight.pop(key, None)
        await self._send_down(_dumps(resp))

    def _classify(
        self, line: bytes | None, tool: str
    ) -> tuple[FailureKind | None, dict[str, Any] | None, list[str]]:
        """-> (failure kind or None, parsed response, validation problems)."""
        if line is None:
            return "deadline", None, []
        obj: Any = _parse(line)
        if not isinstance(obj, dict) or "id" not in obj:
            return "unparseable", None, []
        o = cast("dict[str, Any]", obj)
        err: Any = o.get("error")
        if isinstance(err, dict):
            code: Any = cast("dict[str, Any]", err).get("code")
            if is_retryable_rpc_code(code):
                return "rpc_error", o, []
            return None, o, []  # permanent error: forward as is
        problems = validation_problems(self._h, o, self._schemas.get(tool))
        if problems:
            return "validation", o, problems
        return None, o, []

    async def _supervise(self, req: dict[str, Any], fl: _Inflight) -> dict[str, Any]:
        tool, orig_id = fl.tool, fl.orig_id
        breaker = self._breakers.get(tool)
        use_breaker = self._h.circuit_breaker
        t_start = self._now()
        retries = 0
        attempts = 0
        while True:
            if use_breaker and not breaker.allow(self._now()):
                await self._emit(tool, {"guard_action": "breaker_open"}, is_error=True)
                return tool_error(orig_id, CIRCUIT_OPEN_TEXT)
            uid = f"{UID_PREFIX}{next(self._uids)}"
            fl.uid = uid
            send, recv = anyio.create_memory_object_stream[bytes](math.inf)
            self._routes[uid] = send
            attempts += 1
            line: bytes | None = None
            try:
                await self._send_up(_dumps({**req, "id": uid}))
                with anyio.move_on_after(self.cfg.deadline_s if self._use_deadline else math.inf):
                    line = await recv.receive()
            finally:
                self._routes.pop(uid, None)
                send.close()
                recv.close()
            if line is None:
                self._superseded.add(uid)
            kind, obj, problems = self._classify(line, tool)
            if kind is None:
                return await self._succeed(_need(obj), tool, orig_id, breaker, t_start)
            if use_breaker:
                breaker.record_failure(self._now())
            if kind == "validation":
                await self._emit(
                    tool,
                    {"guard_action": "validated_fail", "problems": cast("JsonValue", problems)},
                    is_error=True,
                )
            if self._h.retry and tool in self.cfg.retry_tools and self._policy.may_retry(retries):
                delay = self._policy.backoff_s(retries, self._rng)
                retries += 1
                await self._emit(
                    tool,
                    {
                        "guard_action": "retried",
                        "n": retries,
                        "kind": kind,
                        "backoff_s": round(delay, 4),
                    },
                )
                await anyio.sleep(delay)
                continue
            await self._emit(
                tool,
                {"guard_action": "gave_up", "kind": kind, "attempts": attempts},
                is_error=True,
                latency_ms=int((self._now() - t_start) * 1000),
            )
            return self._final_failure(kind, obj, problems, orig_id, tool, attempts)

    async def _succeed(
        self,
        obj: dict[str, Any],
        tool: str,
        orig_id: Any,
        breaker: Any,
        t_start: float,
    ) -> dict[str, Any]:
        if self._h.circuit_breaker:
            if "error" in obj:  # permanent JSON-RPC error: neither a success nor a failure
                breaker.release_probe()
            else:
                breaker.record_success()
        notes: list[Detail] = []
        if "result" in obj:
            obj, notes = transform_response(self._h, obj)
        for n in notes:
            await self._emit(tool, n, latency_ms=int((self._now() - t_start) * 1000))
        return {"jsonrpc": "2.0", **obj, "id": orig_id}

    def _final_failure(
        self,
        kind: FailureKind,
        obj: dict[str, Any] | None,
        problems: list[str],
        orig_id: Any,
        tool: str,
        attempts: int,
    ) -> dict[str, Any]:
        n = f"{attempts} attempt{'s' if attempts != 1 else ''}"
        if kind == "rpc_error" and obj is not None:
            return {"jsonrpc": "2.0", **obj, "id": orig_id}
        if tool not in self.cfg.retry_tools:
            return tool_error(orig_id, self._unknown_outcome(kind, tool, problems))
        if kind == "deadline":
            text = (
                f"tool '{tool}' timed out after {self.cfg.deadline_s:g}s ({n}); "
                "no result was received. Do not assume the call succeeded."
            )
        elif kind == "unparseable":
            text = (
                f"tool '{tool}' returned an unparseable response ({n}). "
                "Do not assume the call succeeded."
            )
        else:
            body = json.dumps(
                {"error": "validation_failed", "tool": tool, "problems": problems},
                ensure_ascii=False,
            )
            text = f"tool '{tool}' returned a malformed result ({n}): {body}"
        return tool_error(orig_id, text)

    def _unknown_outcome(self, kind: FailureKind, tool: str, problems: list[str]) -> str:
        why = {
            "deadline": f"timed out after {self.cfg.deadline_s:g}s",
            "unparseable": "returned an unparseable response",
            "validation": "returned a malformed result: " + "; ".join(problems[:3]),
            "rpc_error": "returned an error",
        }[kind]
        return f"{UNKNOWN_OUTCOME_TEXT} (tool '{tool}' {why}; not retried: not read-only)."


# ------------------------------------------------------------------ process shell


def _need[T](x: T | None) -> T:
    if x is None:
        raise RuntimeError("missing stream")
    return x


async def _read_fd(fd: int) -> bytes:
    return await anyio.to_thread.run_sync(os.read, fd, _CHUNK, abandon_on_cancel=True)


async def _terminate(proc: Process) -> None:
    with anyio.move_on_after(_GRACE_S):
        await proc.wait()
        return
    proc.terminate()
    with anyio.move_on_after(2.0):
        await proc.wait()
        return
    proc.kill()
    await proc.wait()


async def run_guard(
    cfg: GuardConfig,
    cmd: Sequence[str],
    trace_path: Path,
    *,
    run_id: str,
    agent: str,
    server: str,
    stdin_fd: int = 0,
    out: IO[bytes] | None = None,
) -> int:
    await anyio.Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
    async with await anyio.open_file(trace_path, "a", encoding="utf-8") as trace:
        proc = await anyio.open_process(list(cmd), stdin=-1, stdout=-1, stderr=-1)
        try:
            return await _guard_loop(cfg, proc, trace, run_id, agent, server, stdin_fd, out)
        finally:
            with anyio.CancelScope(shield=True):
                if proc.returncode is None:
                    proc.kill()
                await proc.wait()


async def _guard_loop(
    cfg: GuardConfig,
    proc: Process,
    trace: anyio.AsyncFile[str],
    run_id: str,
    agent: str,
    server: str,
    stdin_fd: int,
    out: IO[bytes] | None,
) -> int:
    out_f = anyio.wrap_file(out if out is not None else sys.stdout.buffer)
    err_f = anyio.wrap_file(sys.stderr.buffer)
    out_lock, trace_lock = anyio.Lock(), anyio.Lock()
    stderr_done = anyio.Event()

    async def send_up(line: bytes) -> None:
        if proc.stdin is None:
            return
        try:
            await proc.stdin.send(line + b"\n")
        except (anyio.BrokenResourceError, anyio.ClosedResourceError):
            log.warning("guard_upstream_stdin_closed", server=server)

    async def send_down(line: bytes) -> None:
        async with out_lock:
            await out_f.write(line + b"\n")
            await out_f.flush()

    async def emit(ev: GuardTraceEvent) -> None:
        async with trace_lock:
            await trace.write(ev.model_dump_json() + "\n")
            await trace.flush()

    async def pump_client(guard: Guard) -> None:
        splitter = LineSplitter()
        try:
            while True:
                chunk = await _read_fd(stdin_fd)
                if not chunk:
                    break
                for line in splitter.feed(chunk):
                    await guard.on_client_line(line)
            rest = splitter.flush()
            if rest is not None:
                await guard.on_client_line(rest)
        except FramingError as e:
            log.error("framing_error", side="client", error=str(e))
        finally:
            if proc.stdin is not None:
                await proc.stdin.aclose()
        await _terminate(proc)

    async def pump_server(guard: Guard, tg: TaskGroup) -> None:
        stdout = _need(proc.stdout)
        splitter = LineSplitter()
        try:
            async for chunk in stdout:
                for line in splitter.feed(chunk):
                    await guard.on_server_line(line)
            rest = splitter.flush()
            if rest is not None:
                await guard.on_server_line(rest)
        except FramingError as e:
            log.error("framing_error", side="server", error=str(e))
            proc.kill()
        await proc.wait()
        await stderr_done.wait()
        tg.cancel_scope.cancel()

    async def pump_stderr() -> None:
        stderr = _need(proc.stderr)
        prefix = f"[guard-upstream:{server}] ".encode()
        splitter = LineSplitter()
        async for chunk in stderr:
            for line in splitter.feed(chunk):
                await err_f.write(prefix + line + b"\n")
                await err_f.flush()
        rest = splitter.flush()
        if rest is not None:
            await err_f.write(prefix + rest + b"\n")
            await err_f.flush()
        stderr_done.set()

    async with anyio.create_task_group() as tg:
        guard = Guard(
            cfg,
            run_id=run_id,
            agent=agent,
            server=server,
            send_up=send_up,
            send_down=send_down,
            emit=emit,
            tg=tg,
        )
        tg.start_soon(pump_client, guard)
        tg.start_soon(pump_server, guard, tg)
        tg.start_soon(pump_stderr)
    await proc.wait()
    code = proc.returncode or 0
    return code if code >= 0 else 128 - code
