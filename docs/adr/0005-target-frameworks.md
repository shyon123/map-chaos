# ADR 0005 — Target system shape in each framework

**Status:** accepted · 2026-10-28

Both targets implement the same topology: a **supervisor** with no MCP tools, which delegates to three workers — `files` (filesystem), `data` (SQLite), `vcs` (git) — each with its own tool allowlist. The prompts are shared text files in `src/mcp_chaos/targets/prompts/`.

- **LangGraph:** `langchain.agents.create_agent` for each worker. Workers are exposed to the supervisor as tools (the "subagents" pattern; `langgraph-supervisor` is no longer maintained). MCP tools are loaded through a thin interface `targets/langgraph_target/mcp_tools.py`. **Resolved (2026-10-28):** use `langchain[mcp]` → `langchain.mcp.MCPAdapter`. `langchain-mcp-adapters` 0.3.2 pins `mcp<2.0.0`, which conflicts with our `mcp>=2.3`; it's also needed by `mcp-server-git` 2026.10.19. `langchain[mcp]` is built on `fastmcp` 4.x, which has no dependency on the `mcp` SDK. It's a beta API, so it's pinned exactly and isolated in that one file. `mcp-server-git` is installed as a separate `uv tool` in the image (and run via `uvx` in local mode), so its dependencies never couple to ours.
- **Agent SDK:** `ClaudeSDKClient` with `agents={files, data, vcs}` as `AgentDefinition`s. Built-in tools are disabled via `disallowed_tools`; `tools=[]` is unverified and checked in slice 5. The Agent SDK shares MCP connections across subagents, so to keep per-agent fault attribution we register **one server entry per worker** (`fs_files`, `sqlite_data`, `git_vcs`), each launched through its own proxy instance tagged `--agent`, and allowlist each subagent to only its own `mcp__<entry>__*` tools.

Both run in the same container image, with the same model, temperature 0, `max_turns`, and timeout. A known non-equivalence, reported in the README: the frameworks' internal delegation prompts differ and are not under our control.
