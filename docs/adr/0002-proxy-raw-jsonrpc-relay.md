# ADR 0002 — Fault injection as a raw JSON-RPC stdio relay

**Status:** accepted · 2026-10-28

## Context
Faults must be identical across frameworks and must cover both frame-level corruption (malformed JSON-RPC, hangs) and content-level lies (schema drift, injection). The MCP Python SDK moved to v2 in Jul 2026 with breaking API changes. Spec 2026-07-28 removed the initialize handshake and session ids, and reference servers in the wild speak different protocol revisions.

## Decision
`mcp-chaos proxy` is a **protocol-version-agnostic, newline-delimited JSON-RPC relay**. It spawns the upstream server as a subprocess and pipes stdin and stdout through. It parses only the messages it needs: `tools/call` requests and responses, `tools/list` responses, and request ids for correlation. Everything else is forwarded **byte-for-byte**. The MCP SDK is used only to build our own servers (the fake test server and the SQLite server), never in the relay path.

## Rationale
- Frame faults (F01, F03, F04-frame) are impossible to express through an SDK bridge, which always emits well-formed messages.
- Byte-identical passthrough is trivially property-testable: there is no re-serialization drift.
- It's immune to SDK v1/v2 churn and spec-revision differences between clients and servers.
- It sits at the same point for LangGraph and the Agent SDK: both launch stdio servers from a command, and that command is the proxy.

## Rejected alternatives
- **SDK bridge (Client → Server):** re-serializes everything, can't emit malformed frames, and is tied to one SDK major version.
- **Framework hooks** (Agent SDK `PostToolUse` → `updatedToolOutput`, LangChain tool wrappers): each framework would get a different, non-identical injection path, which breaks the cross-framework comparison. They're also unavailable for frame-level faults. We keep hooks in mind as a possible *hardening* point (ADR 0007).
- **Toxiproxy-style TCP chaos:** stdio isn't TCP, and it can't understand JSON or schemas.

## Consequences
We own JSON-RPC framing edge cases (partial lines, large lines, stderr passthrough), which are covered by property tests. Streamable-HTTP transport is out of scope for v1.
