"""Local Anthropic Messages pass-through gateway (ADR 0003).

Clients use base URL ``http://host:port/r/<run_id>`` with a dummy API key. The real key is read
from ``ANTHROPIC_API_KEY`` at start and never leaves this process except in the upstream request.
Logs carry only cache-key prefixes, model, token counts, cost and hit/miss -- never bodies/headers.
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from mcp_chaos.llm.budget import BudgetTracker, TokenCounts, usage_from_body, usage_from_sse
from mcp_chaos.llm.cache import DiskCache, StoredResponse
from mcp_chaos.llm.cassette import CassetteEntry, CassetteReader, CassetteWriter
from mcp_chaos.llm.normalize import (
    DEFAULT_RULES,
    BodyError,
    Rule,
    compute_key,
    normalize,
    parse_body,
)
from mcp_chaos.llm.prices import UnknownModelError, compute_cost, price_for
from mcp_chaos.log import get_logger
from mcp_chaos.schemas.results import Usage

log = get_logger(__name__)

Kind = Literal["messages", "count_tokens"]
_FORWARD_HEADERS = ("content-type", "anthropic-version", "anthropic-beta")
_UPSTREAM_PATH: dict[Kind, str] = {
    "messages": "/v1/messages",
    "count_tokens": "/v1/messages/count_tokens",
}


class GatewayConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    mode: Literal["live", "record", "replay"]
    cache_dir: Path
    cassette_dir: Path
    upstream_base: str = "https://api.anthropic.com"
    global_budget_usd: float = 25.0
    normalize_rules: list[Rule] = Field(default_factory=lambda: list(DEFAULT_RULES))
    seq_fallback: bool = True
    transport: httpx.AsyncBaseTransport | None = None


class GatewayConfigError(RuntimeError):
    pass


class _Run:
    def __init__(self, seed: int, budget_usd: float) -> None:
        self.seed = seed
        self.budget_usd = budget_usd
        self.seq = 0
        self.reader: CassetteReader | None = None


def _error(status: int, etype: str, message: str) -> JSONResponse:
    return JSONResponse(
        {"type": "error", "error": {"type": etype, "message": message}}, status_code=status
    )


async def _json_404(_request: Request, _exc: Exception) -> Response:
    return _error(404, "not_found_error", "mcp-chaos gateway: no such route")


async def _json_405(_request: Request, _exc: Exception) -> Response:
    return _error(405, "invalid_request_error", "mcp-chaos gateway: method not allowed")


def _replay(stored: StoredResponse, source: str) -> Response:
    headers = {**stored.headers, "x-mcp-chaos-source": source}
    chunks = stored.chunks
    if chunks is None:
        return Response(stored.body, status_code=stored.status, headers=headers)

    async def gen() -> AsyncIterator[bytes]:
        for c in chunks:
            yield c

    return StreamingResponse(gen(), status_code=stored.status, headers=headers)


class Gateway:
    def __init__(self, cfg: GatewayConfig) -> None:
        self.cfg = cfg
        self._api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if cfg.mode != "replay" and not self._api_key:
            raise GatewayConfigError("ANTHROPIC_API_KEY is required in live/record mode")
        self.budget = BudgetTracker(cfg.global_budget_usd)
        self.cache = DiskCache(cfg.cache_dir)
        self.cassette = CassetteWriter(cfg.cassette_dir)
        self._runs: dict[str, _Run] = {}
        self._client = httpx.AsyncClient(
            base_url=cfg.upstream_base,
            transport=cfg.transport,
            timeout=httpx.Timeout(600.0, connect=15.0),
        )

    # -- public API -------------------------------------------------------------------------
    def register_run(self, run_id: str, seed: int, budget_usd: float) -> None:
        self._runs[run_id] = _Run(seed, budget_usd)
        self.budget.register(run_id, budget_usd)

    def get_usage(self, run_id: str) -> Usage:
        return self.budget.get_usage(run_id)

    def is_budget_aborted(self, run_id: str) -> bool:
        return self.budget.is_aborted(run_id)

    async def aclose(self) -> None:
        await self._client.aclose()

    # -- request handling -------------------------------------------------------------------
    async def handle(self, request: Request, kind: Kind) -> Response:
        run_id: str = request.path_params["run_id"]
        run = self._runs.get(run_id)
        if run is None:
            return _error(404, "not_found_error", "mcp-chaos gateway: unknown run")
        raw = await request.body()
        try:
            body = parse_body(raw)
        except BodyError as e:
            return _error(400, "invalid_request_error", str(e))
        model = body.get("model")
        if not isinstance(model, str) or not model:
            return _error(400, "invalid_request_error", "missing model")

        stream = kind == "messages" and body.get("stream") is True
        normalized = normalize(body, self.cfg.normalize_rules)
        key_model = model if kind == "messages" else f"count_tokens:{model}"
        key = compute_key(key_model, normalized, run.seed)
        run.seq += 1
        seq = run.seq

        if self.cfg.mode == "replay":
            if run.reader is None:
                run.reader = CassetteReader(self.cfg.cassette_dir, run_id, self.cfg.seq_fallback)
            stored = run.reader.lookup(key, seq)
            if stored is None:
                log.warning("llm_replay_miss", run_id=run_id, key=key[:12], model=model, seq=seq)
                return _error(400, "invalid_request_error", f"mcp-chaos replay miss: {key[:12]}")
            self.budget.record_cache_hit(run_id)
            log.info("llm_replay_hit", run_id=run_id, key=key[:12], model=model, seq=seq)
            return _replay(stored, "cassette")

        cached = self.cache.get(key)
        if cached is not None:
            self.budget.record_cache_hit(run_id)
            if self.cfg.mode == "record":
                self.cassette.append(run_id, CassetteEntry.from_response(seq, key, cached))
            log.info("llm_cache_hit", run_id=run_id, key=key[:12], model=model, cost_usd=0.0)
            return _replay(cached, "cache")

        if kind == "messages":
            try:
                price_for(model)
            except UnknownModelError:
                return _error(400, "invalid_request_error", "mcp-chaos unknown model (no price)")
        if self.budget.exceeded(run_id):
            self.budget.mark_aborted(run_id)
            log.warning("llm_budget_abort", run_id=run_id, key=key[:12], model=model)
            return _error(400, "invalid_request_error", "mcp-chaos budget exceeded")

        return await self._forward(request, raw, run_id, key, seq, model, kind, stream)

    async def _forward(
        self,
        request: Request,
        raw: bytes,
        run_id: str,
        key: str,
        seq: int,
        model: str,
        kind: Kind,
        stream: bool,
    ) -> Response:
        headers: dict[str, str] = {"content-type": "application/json"}
        for h in _FORWARD_HEADERS:
            v = request.headers.get(h)
            if v is not None:
                headers[h] = v
        headers["x-api-key"] = self._api_key
        headers["accept-encoding"] = "identity"
        path = _UPSTREAM_PATH[kind]

        if stream:
            req = self._client.build_request("POST", path, content=raw, headers=headers)
            up = await self._client.send(req, stream=True)
            if up.status_code == 200:
                return StreamingResponse(
                    self._relay(up, run_id, key, seq, model),
                    status_code=200,
                    headers=self._out_headers(up),
                )
            body = await up.aread()
            await up.aclose()
            status = up.status_code
            out_headers = self._out_headers(up)
        else:
            up_resp = await self._client.post(path, content=raw, headers=headers)
            body, status, out_headers = (
                up_resp.content,
                up_resp.status_code,
                self._out_headers(up_resp),
            )

        stored = StoredResponse.from_body(status, out_headers, body)
        tokens = usage_from_body(body) if status == 200 and kind == "messages" else TokenCounts()
        self._finalize(run_id, key, seq, model, kind, stored, tokens)
        return Response(
            body, status_code=status, headers={**out_headers, "x-mcp-chaos-source": "upstream"}
        )

    @staticmethod
    def _out_headers(resp: httpx.Response) -> dict[str, str]:
        ct = resp.headers.get("content-type")
        return {"content-type": ct} if ct else {}

    async def _relay(
        self, up: httpx.Response, run_id: str, key: str, seq: int, model: str
    ) -> AsyncIterator[bytes]:
        chunks: list[bytes] = []
        complete = False
        try:
            async for chunk in up.aiter_bytes():
                if not chunk:
                    continue
                chunks.append(chunk)
                yield chunk
            complete = True
        finally:
            await up.aclose()
        if complete:
            stored = StoredResponse.from_sse(up.status_code, self._out_headers(up), chunks)
            self._finalize(run_id, key, seq, model, "messages", stored, usage_from_sse(chunks))

    def _finalize(
        self,
        run_id: str,
        key: str,
        seq: int,
        model: str,
        kind: Kind,
        stored: StoredResponse,
        tokens: TokenCounts,
    ) -> None:
        cost = 0.0
        if kind == "messages" and stored.status == 200:
            cost = compute_cost(
                model,
                input_tokens=tokens.input,
                output_tokens=tokens.output,
                cache_write_tokens=tokens.cache_write,
                cache_read_tokens=tokens.cache_read,
            )
        self.budget.record(run_id, tokens, cost)
        if stored.status == 200:
            self.cache.put(key, stored)
        self.cassette.append(run_id, CassetteEntry.from_response(seq, key, stored))
        log.info(
            "llm_upstream",
            run_id=run_id,
            key=key[:12],
            model=model,
            status=stored.status,
            input_tokens=tokens.input,
            output_tokens=tokens.output,
            cache_read_tokens=tokens.cache_read,
            cache_write_tokens=tokens.cache_write,
            cost_usd=cost,
        )


def create_app(cfg: GatewayConfig) -> Starlette:
    gateway = Gateway(cfg)

    async def messages(request: Request) -> Response:
        return await gateway.handle(request, "messages")

    async def count_tokens(request: Request) -> Response:
        return await gateway.handle(request, "count_tokens")

    @asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncGenerator[None]:
        try:
            yield
        finally:
            await gateway.aclose()

    handlers: dict[Any, Any] = {404: _json_404, 405: _json_405}
    app = Starlette(
        routes=[
            Route("/r/{run_id}/v1/messages", messages, methods=["POST"]),
            Route("/r/{run_id}/v1/messages/count_tokens", count_tokens, methods=["POST"]),
        ],
        exception_handlers=handlers,
        lifespan=lifespan,
    )
    app.state.gateway = gateway
    return app
