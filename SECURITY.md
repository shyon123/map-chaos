# Security & threat model

## Assets
- The host machine and the developer's credentials (`ANTHROPIC_API_KEY`).
- Integrity of benchmark results (they must not be tamperable by the systems under test).

## Trust boundaries
1. **Tool output is untrusted.** Everything that comes back from an MCP server — real or faulted — is attacker-controlled data from the agent's point of view. That's the premise of the project.
2. **Target agents are untrusted.** They're LLM-driven and are deliberately fed injection payloads. Assume they will try anything their tools allow.
3. **The harness (proxy, runner, judge) is trusted** and runs outside the sandbox.

## Controls
| Threat | Control |
|---|---|
| Agent escapes to the host | Target agents + upstream MCP servers run in Docker: non-root UID, `--read-only` rootfs, a writable `/scratch` bind mount (not tmpfs, so the host can snapshot it; plus a 64 MB tmpfs `/tmp` and a bind-mounted `/run/out`) and nothing else writable, `--cap-drop=ALL`, `--security-opt=no-new-privileges`, `--pids-limit`, `--memory`, `--cpus`, wall-clock kill |
| Network exfiltration | Container egress is limited to the Anthropic API via an allowlisting egress proxy; no other network access |
| Dangerous tools | No shell-exec, HTTP-fetch, or credential-bearing tools in any target. Agent SDK built-ins (Bash, Write, WebFetch, …) are disabled; only allowlisted `mcp__*` tools are enabled. Filesystem/git/SQLite servers are rooted at `/scratch` |
| Least privilege | Per-agent MCP tool allowlist in config (e.g. the reader worker gets no write tools). The proxy enforces it too and denies unlisted calls |
| Secret leakage | Secrets come only from env; `.env` is gitignored; the API key is mounted only into the container that needs it. structlog has a redaction processor; prompts and tool args are logged as SHA-256 hashes; cassettes are scrubbed of auth headers. gitleaks runs in pre-commit and CI |
| Real exploits in payloads | Injection payloads are benign canaries (`CANARY-<hex>`) asking for harmless, observable actions inside `/scratch`. No exfil URLs, no shell, no real targets |
| Result tampering | The judge reads side effects from the sandbox snapshot taken by the harness, not from agent self-reports. Traces come from the proxy, not the agent |
| Runaway cost | A hard `--budget-usd` global cap plus a per-run cap; the run aborts with `status=budget_abort` |
| Supply chain | `uv.lock` with hashes, `pip-audit` in CI, pinned base image digest, pinned MCP server versions |

## Out of scope
Attacks on the Anthropic API itself; malicious MCP servers attacking the *proxy* beyond malformed input (the proxy fuzzes its own parser in property tests); kernel-level container escapes.

## Reporting
Open a private security advisory on the GitHub repo.
