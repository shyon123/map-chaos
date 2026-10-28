"""Deterministic, scripted fake of the Anthropic API (tests only; costs $0).

Usable as the gateway's injected upstream (``httpx.ASGITransport``) or standalone::

    python -m mcp_chaos.llm.fake_anthropic --script s.json --port 8099

Every request is answered by one rule engine: the first rule whose ``match`` fields all hold wins;
otherwise ``fallback`` is used and the request body is appended to ``app.state.misses``.
Tool-use ids and token counts are pure functions of the request, so runs are reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

StopReason = Literal["end_turn", "tool_use", "max_tokens", "stop_sequence", "pause_turn", "refusal"]
_FIXED_TS = "2026-01-01T00:00:00Z"
_FIXED_EXPIRY = "2026-01-02T00:00:00Z"
_TEMPLATE = re.compile(r"\{\{(?:last_tool_result|tool_result:([^}]+))\}\}")


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class TextBlock(_Frozen):
    type: Literal["text"] = "text"
    text: str


class ToolUseBlock(_Frozen):
    type: Literal["tool_use"] = "tool_use"
    name: str
    input: dict[str, Any] = Field(default_factory=lambda: {})


Block = Annotated[TextBlock | ToolUseBlock, Field(discriminator="type")]


class Respond(_Frozen):
    content: list[Block]
    stop_reason: StopReason | None = None

    @field_validator("content", mode="before")
    @classmethod
    def _single_block(cls, v: Any) -> Any:
        return [v] if isinstance(v, dict) else v

    def resolved_stop_reason(self) -> StopReason:
        if self.stop_reason is not None:
            return self.stop_reason
        return "tool_use" if any(isinstance(b, ToolUseBlock) for b in self.content) else "end_turn"


class Match(_Frozen):
    model: str | None = None
    system_contains: str | None = None
    last_user_contains: str | None = None
    tool_result_contains: str | None = None
    has_tool: str | None = None
    turn: int | None = None
    regex: str | None = None


class Rule(_Frozen):
    match: Match = Field(default_factory=Match)
    respond: Respond

    @field_validator("respond", mode="before")
    @classmethod
    def _list_form(cls, v: Any) -> Any:
        return {"content": v} if isinstance(v, list) else v  # pyright: ignore[reportUnknownVariableType]


def _default_fallback() -> Respond:
    return Respond(content=[TextBlock(text="FAKE: no rule matched")], stop_reason="end_turn")


class FakeScript(_Frozen):
    rules: list[Rule] = Field(default_factory=lambda: [])
    fallback: Respond = Field(default_factory=_default_fallback)

    @classmethod
    def load(cls, path: Path) -> FakeScript:
        return cls.model_validate_json(path.read_text())


# -- request helpers ---------------------------------------------------------------------------
def canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _digest(*parts: bytes) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
        h.update(b"\x00")
    return h.hexdigest()


def _blocks(content: Any) -> list[dict[str, Any]]:
    if not isinstance(content, list):
        return []
    items: list[Any] = content  # pyright: ignore[reportUnknownVariableType]
    return [b for b in items if isinstance(b, dict)]  # pyright: ignore[reportUnknownVariableType]


def _text_of(content: Any) -> str:
    """Concatenate text from a message ``content`` (str or blocks), incl. tool_result content."""
    if isinstance(content, str):
        return content
    out: list[str] = []
    for blk in _blocks(content):
        if blk.get("type") == "text":
            out.append(str(blk.get("text", "")))
        elif blk.get("type") == "tool_result":
            out.append(_text_of(blk.get("content", "")))
    return "\n".join(out)


def _messages(body: dict[str, Any]) -> list[dict[str, Any]]:
    return _blocks(body.get("messages", []))


def _tool_results(body: dict[str, Any]) -> list[tuple[str, str]]:
    """All (tool name, result text) pairs in conversation order."""
    names: dict[str, str] = {}
    out: list[tuple[str, str]] = []
    for m in _messages(body):
        for blk in _blocks(m.get("content")):
            if blk.get("type") == "tool_use":
                names[str(blk.get("id"))] = str(blk.get("name"))
            elif blk.get("type") == "tool_result":
                out.append(
                    (names.get(str(blk.get("tool_use_id")), ""), _text_of(blk.get("content", "")))
                )
    return out


def _last_user_text(body: dict[str, Any]) -> str:
    for m in reversed(_messages(body)):
        if m.get("role") == "user":
            return _text_of(m.get("content"))
    return ""


def _matches(m: Match, body: dict[str, Any]) -> bool:
    if m.model is not None and body.get("model") != m.model:
        return False
    if m.system_contains is not None and m.system_contains not in _text_of(body.get("system", "")):
        return False
    if m.last_user_contains is not None and m.last_user_contains not in _last_user_text(body):
        return False
    if m.tool_result_contains is not None and not any(
        m.tool_result_contains in t for _, t in _tool_results(body)
    ):
        return False
    if m.has_tool is not None and not any(
        t.get("name") == m.has_tool for t in _blocks(body.get("tools", []))
    ):
        return False
    turns = sum(1 for x in _messages(body) if x.get("role") == "assistant")
    if m.turn is not None and turns != m.turn:
        return False
    if m.regex is None:
        return True
    return re.search(m.regex, canonical(body.get("messages", [])).decode()) is not None


def _template(value: Any, results: list[tuple[str, str]]) -> Any:
    if isinstance(value, str):

        def sub(mo: re.Match[str]) -> str:
            name = mo.group(1)
            for n, t in reversed(results):
                if name is None or n == name:
                    return t
            return ""

        return _TEMPLATE.sub(sub, value)
    if isinstance(value, dict):
        d: dict[str, Any] = value  # pyright: ignore[reportUnknownVariableType]
        return {k: _template(v, results) for k, v in d.items()}
    if isinstance(value, list):
        return [_template(v, results) for v in value]  # pyright: ignore[reportUnknownVariableType]
    return value


def _validate(body: Any, *, need_max_tokens: bool) -> str | None:
    if not isinstance(body, dict):
        return "Request body must be a JSON object"
    b: dict[str, Any] = body  # pyright: ignore[reportUnknownVariableType]
    if not isinstance(b.get("model"), str) or not b["model"]:
        return "model: Field required"
    if not isinstance(b.get("messages"), list) or not b["messages"]:
        return "messages: Field required"
    if need_max_tokens:
        mt = b.get("max_tokens")
        if not isinstance(mt, int) or isinstance(mt, bool) or mt < 1:
            return "max_tokens: Field required"
    return None


def _error(status: int, etype: str, message: str) -> JSONResponse:
    return JSONResponse(
        {"type": "error", "error": {"type": etype, "message": message}}, status_code=status
    )


def _bad(message: str) -> JSONResponse:
    return _error(400, "invalid_request_error", message)


# -- the rule engine ---------------------------------------------------------------------------
def build_message(
    script: FakeScript, body: dict[str, Any], misses: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Answer one validated Messages request with a complete Message object."""
    respond = next((r.respond for r in script.rules if _matches(r.match, body)), None)
    if respond is None:
        respond = script.fallback
        if misses is not None:
            misses.append(body)
    results = _tool_results(body)
    req = canonical(body)
    content: list[dict[str, Any]] = []
    for i, blk in enumerate(respond.content):
        if isinstance(blk, TextBlock):
            content.append({"type": "text", "text": _template(blk.text, results)})
        else:
            tid = "toolu_" + _digest(req, str(i).encode())[:24]
            content.append(
                {
                    "type": "tool_use",
                    "id": tid,
                    "name": blk.name,
                    "input": _template(blk.input, results),
                }
            )
    return {
        "id": "msg_" + _digest(req)[:24],
        "type": "message",
        "role": "assistant",
        "model": body["model"],
        "content": content,
        "stop_reason": respond.resolved_stop_reason(),
        "stop_sequence": None,
        "usage": {
            "input_tokens": len(req) // 4,
            "output_tokens": len(canonical(content)) // 4,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }


def _sse(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n".encode()


def _chunks(s: str, n: int = 24) -> list[str]:
    return [s[i : i + n] for i in range(0, len(s), n)] or [""]


def sse_events(msg: dict[str, Any]) -> list[bytes]:
    usage: dict[str, Any] = msg["usage"]
    start: dict[str, Any] = {**msg, "content": [], "stop_reason": None}
    start["usage"] = {**usage, "output_tokens": 1}
    out = [_sse("message_start", {"type": "message_start", "message": start})]
    out.append(_sse("ping", {"type": "ping"}))
    for i, blk in enumerate(msg["content"]):
        if blk["type"] == "text":
            head: dict[str, Any] = {"type": "text", "text": ""}
            deltas = [{"type": "text_delta", "text": c} for c in _chunks(blk["text"])]
        else:
            head = {"type": "tool_use", "id": blk["id"], "name": blk["name"], "input": {}}
            raw = json.dumps(blk["input"], separators=(",", ":"))
            deltas = [{"type": "input_json_delta", "partial_json": c} for c in _chunks(raw)]
        out.append(
            _sse(
                "content_block_start",
                {"type": "content_block_start", "index": i, "content_block": head},
            )
        )
        for d in deltas:
            out.append(
                _sse("content_block_delta", {"type": "content_block_delta", "index": i, "delta": d})
            )
        out.append(_sse("content_block_stop", {"type": "content_block_stop", "index": i}))
    out.append(
        _sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": msg["stop_reason"], "stop_sequence": None},
                "usage": {
                    "input_tokens": usage["input_tokens"],
                    "output_tokens": usage["output_tokens"],
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
            },
        )
    )
    out.append(_sse("message_stop", {"type": "message_stop"}))
    return out


# -- app ---------------------------------------------------------------------------------------
def create_fake_anthropic(script: FakeScript) -> Starlette:
    requests: list[tuple[str, dict[str, Any]]] = []
    misses: list[dict[str, Any]] = []
    batches: dict[str, dict[str, Any]] = {}

    async def read_json(request: Request) -> Any:
        try:
            return json.loads(await request.body())
        except ValueError:
            return None

    async def messages(request: Request) -> Response:
        body = await read_json(request)
        err = _validate(body, need_max_tokens=True)
        if err:
            return _bad(err)
        requests.append(("/v1/messages", body))
        msg = build_message(script, body, misses)
        if body.get("stream") is not True:
            return JSONResponse(msg)

        async def gen() -> AsyncIterator[bytes]:
            for ev in sse_events(msg):
                yield ev

        return StreamingResponse(gen(), media_type="text/event-stream")

    async def count_tokens(request: Request) -> Response:
        body = await read_json(request)
        err = _validate(body, need_max_tokens=False)
        if err:
            return _bad(err)
        requests.append(("/v1/messages/count_tokens", body))
        return JSONResponse({"input_tokens": len(canonical(body)) // 4})

    def batch_obj(bid: str) -> dict[str, Any]:
        n = len(batches[bid]["results"])
        failed = sum(1 for r in batches[bid]["results"] if r["result"]["type"] == "errored")
        return {
            "id": bid,
            "type": "message_batch",
            "processing_status": "ended",
            "request_counts": {
                "processing": 0,
                "succeeded": n - failed,
                "errored": failed,
                "canceled": 0,
                "expired": 0,
            },
            "ended_at": _FIXED_TS,
            "created_at": _FIXED_TS,
            "expires_at": _FIXED_EXPIRY,
            "cancel_initiated_at": None,
            "archived_at": None,
            "results_url": batches[bid]["results_url"],
        }

    async def create_batch(request: Request) -> Response:
        body = await read_json(request)
        reqs = _blocks(body.get("requests")) if isinstance(body, dict) else []  # pyright: ignore[reportUnknownMemberType]
        if not reqs:
            return _bad("requests: Field required")
        results: list[dict[str, Any]] = []
        for item in reqs:
            cid = item.get("custom_id")
            params = item.get("params")
            if not isinstance(cid, str) or not cid:
                return _bad("requests.custom_id: Field required")
            err = _validate(params, need_max_tokens=True)
            if err or not isinstance(params, dict):
                result: dict[str, Any] = {
                    "type": "errored",
                    "error": {
                        "type": "error",
                        "error": {"type": "invalid_request_error", "message": err or "invalid"},
                    },
                }
            else:
                result = {"type": "succeeded", "message": build_message(script, params, misses)}  # pyright: ignore[reportUnknownArgumentType]
            results.append({"custom_id": cid, "result": result})
        bid = "msgbatch_" + _digest(canonical(body))[:24]
        base = str(request.base_url).rstrip("/")
        batches[bid] = {
            "results": results,
            "results_url": f"{base}/v1/messages/batches/{bid}/results",
        }
        requests.append(("/v1/messages/batches", {"requests": reqs}))
        return JSONResponse(batch_obj(bid))

    async def get_batch(request: Request) -> Response:
        bid: str = request.path_params["batch_id"]
        if bid not in batches:
            return _error(404, "not_found_error", f"batch {bid} not found")
        return JSONResponse(batch_obj(bid))

    async def batch_results(request: Request) -> Response:
        bid: str = request.path_params["batch_id"]
        if bid not in batches:
            return _error(404, "not_found_error", f"batch {bid} not found")
        text = "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in batches[bid]["results"])
        return Response(text, media_type="application/binary")

    async def not_found(_request: Request, _exc: Exception) -> Response:
        return _error(404, "not_found_error", "fake anthropic: no such route")

    handlers: dict[Any, Any] = {404: not_found}
    app = Starlette(
        routes=[
            Route("/v1/messages", messages, methods=["POST"]),
            Route("/v1/messages/count_tokens", count_tokens, methods=["POST"]),
            Route("/v1/messages/batches", create_batch, methods=["POST"]),
            Route("/v1/messages/batches/{batch_id}", get_batch, methods=["GET"]),
            Route("/v1/messages/batches/{batch_id}/results", batch_results, methods=["GET"]),
        ],
        exception_handlers=handlers,
    )
    app.state.requests = requests
    app.state.misses = misses
    app.state.script = script
    return app


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    p = argparse.ArgumentParser(description="Scripted fake Anthropic API")
    p.add_argument("--script", type=Path, required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8099)
    a = p.parse_args(argv)
    uvicorn.run(create_fake_anthropic(FakeScript.load(a.script)), host=a.host, port=a.port)


if __name__ == "__main__":  # pragma: no cover
    main()
