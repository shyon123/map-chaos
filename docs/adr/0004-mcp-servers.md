# ADR 0004 — Upstream MCP servers

**Status:** accepted · 2026-10-28

| Domain | Server | Why |
|---|---|---|
| git | `mcp-server-git` (official, Python, `uvx`), pinned to a release | Real, maintained reference server |
| filesystem | `@modelcontextprotocol/server-filesystem` (official, Node), pinned, rooted at `/scratch` | Real, maintained reference server; Node is present in the sandbox image anyway (the Agent SDK bundles the CLI) |
| SQLite | **our own** `mcp_chaos.servers.sqlite_server` (MCP SDK, ~150 LOC), tools: `list_tables`, `describe_table`, `read_query` (SELECT only, enforced with `sqlite3` authorizer), `write_query` | The official `mcp-server-sqlite` was archived in Jul 2025 |
| tests | `mcp_chaos.servers.fake_server`: deterministic, scriptable responses, no side effects | Integration tests with zero LLM calls and no subprocess flakiness |

Using real servers for two of the three domains keeps the benchmark realistic; the owned SQLite server is documented as a threat to validity.
