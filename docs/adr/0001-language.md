# ADR 0001 — Python 3.12 for everything

**Status:** accepted · 2026-10-28

## Context
Components: a stdio MCP proxy, two target multi-agent systems (LangGraph, Claude Agent SDK), an LLM gateway, a judge, a red-team agent, and analysis/scorecards.

## Decision
Python 3.12 for every component, managed with uv (lockfile with hashes), ruff, pyright strict, and pydantic v2.

## Rationale
- The MCP Python SDK (`mcp` 2.x), LangGraph/LangChain (`langgraph` 1.2, `langchain-anthropic` 1.7), and the Claude Agent SDK (`claude-agent-sdk` 0.2.x) are all first-class in Python. Both target frameworks *must* be Python or TypeScript.
- The proxy is I/O-bound. One agent turn costs 0.5–10 s of LLM latency; JSON-RPC relay overhead in asyncio is ~0.1 ms/message. A faster language wouldn't change any measured outcome.
- The analysis (Wilson CIs, McNemar tests, plots) is native to the Python data stack, which also suits the author's background.
- 3.12 rather than 3.13: every dependency has 3.12 wheels; the MCP SDK README states 3.11+.

## Rejected alternatives
- **TypeScript:** first-class MCP SDK and an Agent SDK exist, and the reference filesystem server is TS. But the analysis stack is weaker, and LangGraph.js lags the Python version. A split-language repo doubles the tooling (lint, types, CI, lockfiles).
- **Go:** good for a single-binary proxy, but has no LangGraph or Agent SDK, so it would only cover the proxy and add a second toolchain for no measurable gain.
- **Rust:** same as Go, plus slower iteration. Its memory-safety benefit doesn't apply to a test harness that runs trusted code.

## Consequences
The proxy's throughput ceiling is irrelevant at benchmark scale (<10 msg/s). If the proxy is ever used in production traffic, revisit.
