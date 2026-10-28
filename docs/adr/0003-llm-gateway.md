# ADR 0003 — A local LLM gateway for caching, cassettes, budget, and key isolation

**Status:** accepted · 2026-10-28

## Context
Target agents call the Anthropic API through their own clients: `langchain-anthropic`, and the Claude Code CLI bundled by `claude-agent-sdk`. We need an on-disk cache keyed by (model, prompt, seed), recorded cassettes so CI costs $0, a hard `--budget-usd` cap, and secrets kept out of the sandbox.

## Decision
The harness runs `mcp_chaos.llm.gateway`, a small local HTTP server implementing the Messages endpoint as a pass-through. Both frameworks are pointed at it via `ANTHROPIC_BASE_URL`, and the sandbox gets a **dummy** API key. The gateway:
1. normalizes the request body (drops volatile fields such as request/session ids and metadata timestamps) and computes `key = sha256(model ‖ normalized_body ‖ seed)`;
2. serves from the on-disk cache or a cassette (`replay` mode) or forwards to the API with the real key (`record`/`live` modes);
3. reads `usage` from each response, prices it from a pinned price table, and refuses further calls once the per-run or global budget is hit (HTTP 529-style error → run `status=budget_abort`);
4. logs only body hashes and token counts.
The judge and red team use the same gateway through our own `llm/client.py` (anthropic SDK). The judge can also use the Message Batches API (50% discount) in `batch` mode.

## Rationale
One choke point gives identical caching, replay, and cost accounting for both frameworks, plus a security win: the real key never enters the container. Container egress can be restricted to the gateway alone.

## Risks
- Volatile fields in CLI request bodies could break cache hits. Mitigation: a normalization allowlist, tested against recorded bodies, plus a cassette fallback that matches by (run_id, call sequence).
- Streaming (SSE) responses must be recorded and replayed verbatim.
- The CLI's own retries/telemetry endpoints: the gateway returns 404 for non-Messages paths, and the CLI is configured with telemetry disabled.

## Rejected alternatives
- Per-framework caches (LangChain cache, SDK-side): there's no hook in the CLI, and accounting would be inconsistent.
- VCR.py-style in-process patching: it doesn't reach the CLI subprocess.
