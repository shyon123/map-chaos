from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import anyio.to_thread
import httpx
import pytest

from mcp_chaos.llm.gateway import Gateway, GatewayConfig, GatewayConfigError, create_app

FIXTURES = Path(__file__).parent.parent / "fixtures" / "llm"
REAL_KEY = "sk-ant-REAL-SECRET-KEY-123"
DUMMY_KEY = "sk-ant-dummy-in-sandbox"
SSE = (FIXTURES / "sse.txt").read_bytes()
RESPONSE = (FIXTURES / "response.json").read_bytes()


def request_body(**over: Any) -> dict[str, Any]:
    body = json.loads((FIXTURES / "request.json").read_text())
    body.update(over)
    return body


class Upstream:
    """Fake api.anthropic.com recording what it was sent."""

    def __init__(self, status: int = 200, body: bytes = RESPONSE, sse: bool = False) -> None:
        self.status = status
        self.body = body
        self.sse = sse
        self.calls: list[httpx.Request] = []

    def usage_body(self, input_tokens: int) -> None:
        d = json.loads(RESPONSE)
        d["usage"]["input_tokens"] = input_tokens
        self.body = json.dumps(d).encode()

    def __call__(self, req: httpx.Request) -> httpx.Response:
        self.calls.append(req)
        if self.sse and self.status == 200:

            async def gen() -> AsyncIterator[bytes]:
                for i in range(0, len(SSE), 41):
                    yield SSE[i : i + 41]

            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=gen())
        return httpx.Response(
            self.status,
            headers={"content-type": "application/json", "request-id": "req_x"},
            content=self.body,
        )


def make_cfg(tmp_path: Path, mode: str, upstream: Upstream | None, **kw: Any) -> GatewayConfig:
    return GatewayConfig(
        mode=mode,  # type: ignore[arg-type]
        cache_dir=tmp_path / "cache",
        cassette_dir=tmp_path / "cassettes",
        global_budget_usd=kw.pop("global_budget_usd", 100.0),
        transport=httpx.MockTransport(upstream) if upstream else None,
        **kw,
    )


def make(tmp_path: Path, mode: str, upstream: Upstream | None, **kw: Any):
    app = create_app(make_cfg(tmp_path, mode, upstream, **kw))
    gw: Gateway = app.state.gateway
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://gw")
    return gw, client


async def post(
    client: httpx.AsyncClient, body: dict[str, Any], run: str = "r1", path: str = "v1/messages"
) -> httpx.Response:
    return await client.post(
        f"/r/{run}/{path}",
        json=body,
        headers={
            "x-api-key": DUMMY_KEY,
            "authorization": f"Bearer {DUMMY_KEY}",
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "tools-2024",
            "x-other": "leak",
        },
    )


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", REAL_KEY)


@pytest.mark.anyio
async def test_miss_forward_then_hit(tmp_path: Path) -> None:
    up = Upstream()
    gw, client = make(tmp_path, "live", up)
    gw.register_run("r1", seed=1, budget_usd=10)
    r1 = await post(client, request_body())
    assert r1.status_code == 200 and r1.content == RESPONSE
    assert r1.headers["x-mcp-chaos-source"] == "upstream"
    r2 = await post(client, request_body())
    assert r2.content == RESPONSE and r2.headers["x-mcp-chaos-source"] == "cache"
    assert len(up.calls) == 1
    u = gw.get_usage("r1")
    assert u.cache_hits == 1
    assert u.cost_usd == pytest.approx(1000 * 0.10 / 1e6 + 200 * 0.50 / 1e6)
    assert (u.input_tokens, u.output_tokens) == (1000, 200)
    await client.aclose()
    await gw.aclose()


@pytest.mark.anyio
async def test_headers_and_key_isolation(tmp_path: Path) -> None:
    up = Upstream(sse=False)
    gw, client = make(tmp_path, "record", up)
    gw.register_run("r1", seed=1, budget_usd=10)
    await post(client, request_body())
    sent = up.calls[0]
    assert sent.headers["x-api-key"] == REAL_KEY
    assert "authorization" not in sent.headers
    assert "x-other" not in sent.headers
    assert sent.headers["anthropic-version"] == "2023-06-01"
    assert sent.headers["anthropic-beta"] == "tools-2024"
    assert sent.url.path == "/v1/messages"
    assert json.loads(sent.content) == request_body()  # ORIGINAL body forwarded
    for f in await anyio.to_thread.run_sync(lambda: list(tmp_path.rglob("*"))):
        if f.is_file():
            data = f.read_bytes()
            assert REAL_KEY.encode() not in data and DUMMY_KEY.encode() not in data
            assert b"x-api-key" not in data.lower() and b"authorization" not in data.lower()
    await client.aclose()
    await gw.aclose()


@pytest.mark.anyio
async def test_normalization_same_key_and_seed_differs(tmp_path: Path) -> None:
    up = Upstream()
    gw, client = make(tmp_path, "live", up)
    gw.register_run("r1", seed=1, budget_usd=10)
    gw.register_run("r2", seed=2, budget_usd=10)
    await post(client, request_body())
    other = request_body(
        system="You are a test agent. Today's date is 2030-02-03. Session "
        "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee."
    )
    other["metadata"] = {"user_id": "zzz"}
    r = await post(client, other)
    assert r.headers["x-mcp-chaos-source"] == "cache" and len(up.calls) == 1
    diff_tool = request_body()
    diff_tool["messages"][0]["content"][0]["text"] += " 2030-02-03"
    r = await post(client, diff_tool)  # message text is hashed verbatim
    assert r.headers["x-mcp-chaos-source"] == "upstream" and len(up.calls) == 2
    up.calls.pop()
    r = await post(client, request_body(), run="r2")
    assert r.headers["x-mcp-chaos-source"] == "upstream" and len(up.calls) == 2
    await client.aclose()
    await gw.aclose()


def replay_gateway(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **kw: Any):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = GatewayConfig(
        mode="replay",
        cache_dir=tmp_path / "nocache",
        cassette_dir=tmp_path / "cassettes",
        **kw,
    )
    app = create_app(cfg)
    gw: Gateway = app.state.gateway
    return gw, httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://gw")


@pytest.mark.anyio
@pytest.mark.parametrize("stream", [False, True])
async def test_record_then_replay_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stream: bool
) -> None:
    up = Upstream(sse=stream)
    gw, client = make(tmp_path, "record", up)
    gw.register_run("r1", seed=1, budget_usd=10)
    body = request_body(stream=stream)
    live = await post(client, body)
    assert live.content == (SSE if stream else RESPONSE)
    assert live.headers["content-type"].startswith(
        "text/event-stream" if stream else "application/json"
    )
    await client.aclose()
    await gw.aclose()

    lines = (tmp_path / "cassettes" / "r1.jsonl").read_text().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["seq"] == 1 and len(entry["key"]) == 64
    if stream:
        assert len(entry["sse_chunks_b64"]) > 1
        assert b"".join(base64.b64decode(c) for c in entry["sse_chunks_b64"]) == SSE
    else:
        assert "sse_chunks_b64" in entry and entry["sse_chunks_b64"] is None

    # replay: no upstream, no key, empty cache dir
    gw2, client2 = replay_gateway(tmp_path, monkeypatch)
    gw2.register_run("r1", seed=1, budget_usd=10)
    rep = await post(client2, body)
    assert rep.status_code == 200 and rep.content == live.content
    assert rep.headers["x-mcp-chaos-source"] == "cassette"
    assert gw2.get_usage("r1").cache_hits == 1
    await client2.aclose()
    await gw2.aclose()


@pytest.mark.anyio
async def test_replay_miss_and_seq_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    up = Upstream()
    gw, client = make(tmp_path, "record", up)
    gw.register_run("r1", seed=1, budget_usd=10)
    await post(client, request_body(max_tokens=1))
    await client.aclose()
    await gw.aclose()

    gw2, c2 = replay_gateway(tmp_path, monkeypatch)
    gw2.register_run("r1", seed=1, budget_usd=10)
    # different body, seq 1 -> fallback hit
    r = await post(c2, request_body(max_tokens=2))
    assert r.status_code == 200 and r.content == RESPONSE
    # seq 2 has no entry and unknown key -> miss
    r = await post(c2, request_body(max_tokens=3))
    assert r.status_code == 400
    err = r.json()
    assert err["type"] == "error" and err["error"]["type"] == "invalid_request_error"
    assert err["error"]["message"].startswith("mcp-chaos replay miss: ")
    assert len(err["error"]["message"].split(": ")[1]) == 12
    await c2.aclose()
    await gw2.aclose()

    gw3, c3 = replay_gateway(tmp_path, monkeypatch, seq_fallback=False)
    gw3.register_run("r1", seed=1, budget_usd=10)
    r = await post(c3, request_body(max_tokens=2))
    assert r.status_code == 400
    await c3.aclose()
    await gw3.aclose()


@pytest.mark.anyio
async def test_budget_abort_per_run(tmp_path: Path) -> None:
    up = Upstream()
    up.usage_body(90_000)  # haiku: ~$0.0091 per call
    gw, client = make(tmp_path, "live", up)
    gw.register_run("r1", seed=1, budget_usd=0.005)
    gw.register_run("r2", seed=1, budget_usd=5.0)
    r = await post(client, request_body(max_tokens=1))
    assert r.status_code == 200
    assert not gw.is_budget_aborted("r1")
    r = await post(client, request_body(max_tokens=2))
    assert r.status_code == 400
    assert r.json()["error"] == {
        "type": "invalid_request_error",
        "message": "mcp-chaos budget exceeded",
    }
    assert gw.is_budget_aborted("r1")
    assert len(up.calls) == 1  # no upstream call after abort
    # cached request is still served; other run unaffected
    assert (await post(client, request_body(max_tokens=1))).status_code == 200
    assert (await post(client, request_body(max_tokens=9), run="r2")).status_code == 200
    assert not gw.is_budget_aborted("r2")
    await client.aclose()
    await gw.aclose()


@pytest.mark.anyio
async def test_budget_abort_global(tmp_path: Path) -> None:
    up = Upstream()
    up.usage_body(90_000)
    gw, client = make(tmp_path, "live", up, global_budget_usd=0.012)
    gw.register_run("a", seed=1, budget_usd=5)
    gw.register_run("b", seed=1, budget_usd=5)
    assert (await post(client, request_body(max_tokens=1), run="a")).status_code == 200
    assert (await post(client, request_body(max_tokens=2), run="b")).status_code == 200
    assert len(up.calls) == 2  # ~$0.018 spent now >= $0.012
    r = await post(client, request_body(max_tokens=3), run="a")
    assert r.status_code == 400 and r.json()["error"]["message"] == "mcp-chaos budget exceeded"
    assert len(up.calls) == 2
    assert gw.is_budget_aborted("a")
    await client.aclose()
    await gw.aclose()


@pytest.mark.anyio
async def test_streaming_budget_accounting(tmp_path: Path) -> None:
    up = Upstream(sse=True)
    gw, client = make(tmp_path, "live", up)
    gw.register_run("r1", seed=1, budget_usd=10)
    r = await post(client, request_body(stream=True))
    assert r.content == SSE
    u = gw.get_usage("r1")
    assert (u.input_tokens, u.output_tokens) == (1000, 200)
    r = await post(client, request_body(stream=True))
    assert r.content == SSE and len(up.calls) == 1
    await client.aclose()
    await gw.aclose()


@pytest.mark.anyio
async def test_routes_and_errors(tmp_path: Path) -> None:
    up = Upstream()
    gw, client = make(tmp_path, "live", up)
    gw.register_run("r1", seed=1, budget_usd=10)
    r = await post(client, request_body(), run="nope")
    assert r.status_code == 404 and r.json()["type"] == "error"
    for path in ("/v1/messages", "/r/r1/v1/models", "/r/r1/api/event_logging/batch"):
        r = await client.post(path, json={})
        assert r.status_code == 404 and r.json()["type"] == "error"
    r = await client.get("/r/r1/v1/messages")
    assert r.status_code == 405
    r = await client.post("/r/r1/v1/messages", content=b"not json")
    assert r.status_code == 400
    r = await client.post("/r/r1/v1/messages", json={"messages": []})
    assert r.status_code == 400
    r = await post(client, request_body(model="gpt-unknown"))
    assert r.status_code == 400 and "unknown model" in r.json()["error"]["message"]
    assert up.calls == []
    await client.aclose()
    await gw.aclose()


@pytest.mark.anyio
async def test_count_tokens_separate_key_and_no_cost(tmp_path: Path) -> None:
    up = Upstream(body=b'{"input_tokens": 12}')
    gw, client = make(tmp_path, "live", up)
    gw.register_run("r1", seed=1, budget_usd=10)
    body = request_body()
    ct = await post(client, body, path="v1/messages/count_tokens")
    assert ct.json() == {"input_tokens": 12}
    assert up.calls[0].url.path == "/v1/messages/count_tokens"
    assert gw.get_usage("r1").cost_usd == 0.0
    up.body = RESPONSE
    msg = await post(client, body)  # same body, different endpoint -> not a cache hit
    assert msg.content == RESPONSE and len(up.calls) == 2
    await client.aclose()
    await gw.aclose()


@pytest.mark.anyio
async def test_upstream_error_not_cached(tmp_path: Path) -> None:
    up = Upstream(status=529, body=b'{"type":"error","error":{"type":"overloaded_error"}}')
    gw, client = make(tmp_path, "live", up)
    gw.register_run("r1", seed=1, budget_usd=10)
    assert (await post(client, request_body())).status_code == 529
    assert (await post(client, request_body())).status_code == 529
    assert len(up.calls) == 2
    assert gw.get_usage("r1").cost_usd == 0.0
    await client.aclose()
    await gw.aclose()


@pytest.mark.anyio
async def test_streaming_upstream_error(tmp_path: Path) -> None:
    up = Upstream(status=429, body=b'{"type":"error"}', sse=True)
    gw, client = make(tmp_path, "live", up)
    gw.register_run("r1", seed=1, budget_usd=10)
    r = await post(client, request_body(stream=True))
    assert r.status_code == 429 and r.content == b'{"type":"error"}'
    await client.aclose()
    await gw.aclose()


@pytest.mark.anyio
async def test_record_mode_cache_hit_still_recorded(tmp_path: Path) -> None:
    up = Upstream()
    gw, client = make(tmp_path, "live", up)
    gw.register_run("r1", seed=1, budget_usd=10)
    await post(client, request_body())
    await client.aclose()
    await gw.aclose()
    gw2, c2 = make(tmp_path, "record", up)
    gw2.register_run("r2", seed=1, budget_usd=10)
    r = await post(c2, request_body(), run="r2")
    assert r.headers["x-mcp-chaos-source"] == "cache" and len(up.calls) == 1
    assert len((tmp_path / "cassettes" / "r2.jsonl").read_text().splitlines()) == 1
    await c2.aclose()
    await gw2.aclose()


def test_key_required_outside_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    for mode in ("live", "record"):
        with pytest.raises(GatewayConfigError):
            create_app(make_cfg(tmp_path, mode, None))
    create_app(make_cfg(tmp_path, "replay", None))


class SlowUpstream(Upstream):
    """Async upstream that holds each call open so requests overlap."""

    async def __call__(self, req: httpx.Request) -> httpx.Response:  # type: ignore[override]
        await anyio.sleep(0.05)
        return super().__call__(req)


def make_with(tmp_path: Path, fn: Any, **kw: Any):
    cfg = GatewayConfig(
        mode="live",
        cache_dir=tmp_path / "cache",
        cassette_dir=tmp_path / "cassettes",
        transport=httpx.MockTransport(fn),
        **kw,
    )
    app = create_app(cfg)
    gw: Gateway = app.state.gateway
    return gw, httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://gw")


PER_REQ = 100_000 * 0.50 / 1e6 + 0.0002  # haiku, max_tokens=100_000, small input (upper bound)


@pytest.mark.anyio
async def test_concurrent_requests_reserve_worst_case(tmp_path: Path) -> None:
    up = SlowUpstream()
    gw, client = make_with(tmp_path, up)
    gw.register_run("r1", seed=1, budget_usd=PER_REQ * 3.5)
    codes: list[int] = []

    async def one(i: int) -> None:
        codes.append((await post(client, request_body(max_tokens=100_000, n=i))).status_code)

    async with anyio.create_task_group() as tg:
        for i in range(8):
            tg.start_soon(one, i)
    assert codes.count(200) == len(up.calls) == 3
    assert codes.count(400) == 5
    assert gw.budget.get_usage("r1").cost_usd <= PER_REQ * 3.5
    assert gw.budget.reserved("r1") == 0.0 and gw.budget.global_reserved == 0.0
    await client.aclose()
    await gw.aclose()


@pytest.mark.anyio
async def test_concurrent_global_cap(tmp_path: Path) -> None:
    up = SlowUpstream()
    gw, client = make_with(tmp_path, up, global_budget_usd=PER_REQ * 2.5)
    for rid in ("a", "b", "c"):
        gw.register_run(rid, seed=1, budget_usd=10)
    codes: list[int] = []

    async def one(rid: str) -> None:
        codes.append((await post(client, request_body(max_tokens=100_000), run=rid)).status_code)

    async with anyio.create_task_group() as tg:
        for rid in ("a", "b", "c"):
            tg.start_soon(one, rid)
    assert sorted(codes) == [200, 200, 400]
    assert len(up.calls) == 2
    await client.aclose()
    await gw.aclose()


@pytest.mark.anyio
async def test_upstream_exception_settles(tmp_path: Path) -> None:
    def timeout(req: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=req)

    def refused(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no", request=req)

    gw, client = make_with(tmp_path / "a", timeout)
    gw.register_run("r1", seed=1, budget_usd=10)
    with pytest.raises(httpx.ReadTimeout):
        await post(client, request_body(max_tokens=1000))
    assert gw.budget.reserved("r1") == 0.0
    assert gw.budget.get_usage("r1").cost_usd > 0  # failed closed

    gw2, client2 = make_with(tmp_path / "b", refused)
    gw2.register_run("r1", seed=1, budget_usd=10)
    with pytest.raises(httpx.ConnectError):
        await post(client2, request_body(max_tokens=1000))
    assert gw2.budget.get_usage("r1").cost_usd == 0.0
    assert gw2.budget.reserved("r1") == 0.0
    for c in (client, client2):
        await c.aclose()
    await gw.aclose()
    await gw2.aclose()


async def _drive_relay(gw: Gateway, chunks: list[bytes], stop_after: int | None) -> None:
    async def body() -> AsyncIterator[bytes]:
        for c in chunks:
            yield c

    up = httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body())
    res = gw.budget.reserve("r1", 0.123)
    assert res is not None
    gen = gw._relay(up, "r1", "k" * 64, 1, "claude-haiku-5-5", res)  # type: ignore[reportPrivateUsage]
    n = 0
    async for _ in gen:
        n += 1
        if stop_after is not None and n >= stop_after:
            break
    await gen.aclose()  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.anyio
async def test_stream_disconnect_after_usage_seen_records_spend(tmp_path: Path) -> None:
    gw, client = make(tmp_path, "live", Upstream())
    gw.register_run("r1", seed=1, budget_usd=10)
    start_only = SSE.split(b"\n\n")[0] + b"\n\n"
    assert b"message_start" in start_only
    await _drive_relay(gw, [start_only, SSE[len(start_only) :]], stop_after=1)
    u = gw.budget.get_usage("r1")
    assert u.input_tokens == 1000 and u.cost_usd > 0
    assert gw.budget.reserved("r1") == 0.0 and gw.budget.global_reserved == 0.0
    await client.aclose()
    await gw.aclose()


@pytest.mark.anyio
async def test_stream_disconnect_before_usage_settles_worst_case(tmp_path: Path) -> None:
    gw, client = make(tmp_path, "live", Upstream())
    gw.register_run("r1", seed=1, budget_usd=10)
    await _drive_relay(gw, [b"event: ping\ndata: {}\n\n", b"more"], stop_after=1)
    assert gw.budget.get_usage("r1").cost_usd == pytest.approx(0.123)
    assert gw.budget.reserved("r1") == 0.0 and gw.budget.global_reserved == 0.0
    await client.aclose()
    await gw.aclose()
