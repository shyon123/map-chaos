"""Sans-IO proxy core: lines in, Actions out. No clocks, sockets, or processes here."""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from typing import Any, Protocol, cast

from mcp_chaos.proxy.engine import FaultEngine
from mcp_chaos.proxy.faults.base import REGISTRY, FaultContext
from mcp_chaos.schemas import TraceEvent

DENY_CODE = -32001
DENY_MESSAGE = "tool not allowed by mcp-chaos policy"


class Clock(Protocol):
    def now_ms(self) -> int: ...


@dataclass(frozen=True)
class SendToServer:
    line: bytes


@dataclass(frozen=True)
class SendToClient:
    line: bytes


@dataclass(frozen=True)
class Hang:
    pass


@dataclass(frozen=True)
class Crash:
    pass


@dataclass(frozen=True)
class Trace:
    event: TraceEvent


@dataclass(frozen=True)
class Delay:
    seconds: float
    then: Action


Action = SendToServer | SendToClient | Delay | Hang | Crash | Trace


@dataclass(frozen=True)
class _Pending:
    method: str
    tool: str | None
    args_sha256: str | None
    arguments: dict[str, Any] | None
    t0_ms: int
    call_index: int | None


def _dumps(obj: Any) -> bytes:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _canon(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


class _DuplicateKeyError(ValueError):
    pass


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in pairs:
        if k in out:
            raise _DuplicateKeyError(k)
        out[k] = v
    return out


def _has_duplicate_keys(line: bytes) -> bool:
    try:
        json.loads(line, object_pairs_hook=_unique_pairs)
    except _DuplicateKeyError:
        return True
    except (ValueError, RecursionError):
        return False
    return False


def _parse_obj(line: bytes) -> dict[str, Any] | None:
    try:
        obj: Any = json.loads(line)
    except (ValueError, RecursionError):
        return None
    return cast("dict[str, Any]", obj) if isinstance(obj, dict) else None


def _id_key(msg_id: Any) -> tuple[str, int | str] | None:
    if isinstance(msg_id, bool):
        return None
    if isinstance(msg_id, int):
        return ("i", msg_id)
    if isinstance(msg_id, str):
        return ("s", msg_id)
    return None


class ProxyCore:
    def __init__(
        self,
        engine: FaultEngine,
        *,
        run_id: str,
        agent: str,
        server: str,
        clock: Clock,
        allow: frozenset[str] | None = None,
    ) -> None:
        self._engine = engine
        self._run_id = run_id
        self._agent = agent
        self._server = server
        self._clock = clock
        self._allow = allow
        self._seq = itertools.count()
        self._pending: dict[tuple[str, int | str], _Pending] = {}

    def _event(self, **kw: Any) -> Trace:
        return Trace(
            TraceEvent(
                run_id=self._run_id,
                seq=next(self._seq),
                t_ms=self._clock.now_ms(),
                agent=self._agent,
                server=self._server,
                **kw,
            )
        )

    # ---------------- client -> server ----------------

    def on_client_line(self, line: bytes) -> list[Action]:
        try:
            parsed: Any = json.loads(line)
        except (ValueError, RecursionError):
            return [SendToServer(line)]
        if self._allow is not None and _has_duplicate_keys(line):
            # parsers disagree on duplicate keys (first vs last wins): never forward ambiguity
            return self._reject_invalid("duplicate_keys/denied", "duplicate JSON keys")
        if isinstance(parsed, list):
            if self._allow is None:
                return [SendToServer(line)]
            return self._reject_batch()
        if not isinstance(parsed, dict):
            return [SendToServer(line)]
        obj = cast("dict[str, Any]", parsed)
        method = obj.get("method")
        params: Any = obj.get("params")
        key = _id_key(obj.get("id"))
        if method == "tools/call":
            p = cast("dict[str, Any]", params) if isinstance(params, dict) else {}
            name = p.get("name")
            raw_args: Any = p.get("arguments")
            args = cast("dict[str, Any]", raw_args) if isinstance(raw_args, dict) else None
            if isinstance(name, str):
                args_hash: str | None = hashlib.sha256(
                    _canon(args if args is not None else {})
                ).hexdigest()
                allowed = self._allow is None or name in self._allow
            else:
                args_hash = None
                allowed = self._allow is None
            if not allowed:
                return self._deny(obj, name if isinstance(name, str) else None, args_hash)
            if isinstance(name, str) and key is not None:
                self._pending[key] = _Pending(
                    "tools/call",
                    name,
                    args_hash,
                    args,
                    self._clock.now_ms(),
                    self._engine.reserve_index(name),
                )
                return [
                    self._event(
                        direction="c2s", method="tools/call", tool=name, args_sha256=args_hash
                    ),
                    SendToServer(line),
                ]
        elif method == "tools/list" and key is not None:
            self._pending[key] = _Pending(
                "tools/list", None, None, None, self._clock.now_ms(), None
            )
        return [SendToServer(line)]

    def _reject_batch(self) -> list[Action]:
        return self._reject_invalid("batch/denied", "batch requests")

    def _reject_invalid(self, trace_method: str, what: str) -> list[Action]:
        reply = {
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32600, "message": f"{what} not allowed by mcp-chaos policy"},
        }
        return [
            self._event(direction="c2s", method=trace_method, is_error=True),
            SendToClient(_dumps(reply)),
        ]

    def _deny(self, obj: dict[str, Any], name: str | None, args_hash: str | None) -> list[Action]:
        actions: list[Action] = [
            self._event(
                direction="c2s",
                method="tools/call/denied",
                tool=name,
                args_sha256=args_hash,
                is_error=True,
            )
        ]
        if "id" in obj:
            reply = {
                "jsonrpc": "2.0",
                "id": obj["id"],
                "error": {"code": DENY_CODE, "message": DENY_MESSAGE},
            }
            actions.append(SendToClient(_dumps(reply)))
        return actions

    # ---------------- server -> client ----------------

    def on_server_line(self, line: bytes) -> list[Action]:
        if not self._pending:
            return [SendToClient(line)]
        obj = _parse_obj(line)
        if obj is None or "method" in obj:
            return [SendToClient(line)]
        key = _id_key(obj.get("id"))
        pending = self._pending.pop(key, None) if key is not None else None
        if pending is None:
            return [SendToClient(line)]
        if pending.method == "tools/list":
            return self._on_tools_list(line, obj, pending)
        return self._on_tools_call(line, obj, pending)

    def _on_tools_list(self, line: bytes, obj: dict[str, Any], p: _Pending) -> list[Action]:
        out_line, out_obj = line, obj
        result: Any = obj.get("result")
        if self._allow is not None and isinstance(result, dict):
            tools: Any = cast("dict[str, Any]", result).get("tools")
            if isinstance(tools, list):
                items = cast("list[Any]", tools)
                kept: list[Any] = [
                    t
                    for t in items
                    if isinstance(t, dict) and cast("dict[str, Any]", t).get("name") in self._allow
                ]
                if len(kept) != len(items):
                    new_result: dict[str, Any] = {**cast("dict[str, Any]", result), "tools": kept}
                    out_obj = {**obj, "result": new_result}
                    out_line = _dumps(out_obj)
        return [
            self._response_event("tools/list", out_obj, p, None, None, None),
            SendToClient(out_line),
        ]

    def _response_event(
        self,
        method: str,
        obj: dict[str, Any],
        p: _Pending,
        call_index: int | None,
        fault: Any,
        is_error_override: bool | None,
    ) -> Trace:
        payload: Any = obj.get("result", obj.get("error"))
        blob = _canon(payload)
        is_err = "error" in obj
        res: Any = obj.get("result")
        if isinstance(res, dict) and cast("dict[str, Any]", res).get("isError") is True:
            is_err = True
        return self._event(
            direction="s2c",
            method=method,
            tool=p.tool,
            tool_call_index=call_index,
            args_sha256=p.args_sha256,
            result_sha256=hashlib.sha256(blob).hexdigest(),
            result_bytes=len(blob),
            is_error=is_err if is_error_override is None else is_error_override,
            latency_ms=self._clock.now_ms() - p.t0_ms,
            fault=fault,
        )

    def _on_tools_call(self, line: bytes, obj: dict[str, Any], p: _Pending) -> list[Action]:
        if p.tool is None or p.call_index is None:
            return [SendToClient(line)]
        decision = self._engine.decide(p.tool, p.call_index)
        out_line, out_obj = line, obj
        fault_applied = None
        delay = 0.0
        hang = crash = False
        if decision.spec is not None:
            ctx = FaultContext(
                decision.spec, decision.rng, p.tool, decision.call_index, p.arguments
            )
            outcome = REGISTRY[decision.spec.mode].apply(ctx, obj)
            fault_applied = outcome.applied
            delay, hang, crash = outcome.delay_s, outcome.hang, outcome.crash
            if outcome.response_obj is not None:
                out_obj = outcome.response_obj
                out_line = _dumps(out_obj)
        actions: list[Action] = [
            self._response_event("tools/call", out_obj, p, decision.call_index, fault_applied, None)
        ]
        if hang:
            actions.append(Hang())
        elif crash:
            actions.append(Crash())
        else:
            send: Action = SendToClient(out_line)
            actions.append(Delay(delay, send) if delay > 0 else send)
        return actions
