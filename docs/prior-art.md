# Prior art (surveyed 2026-10-28)

GitHub/arXiv existence was spot-checked by hand. Feature descriptions come from READMEs and abstracts and haven't been audited.

| Project | What it does | Gap vs. mcp-chaos |
|---|---|---|
| [MCP-Scan](https://invariantlabs.ai/blog/introducing-mcp-scan), [Cisco MCP Scanner](https://pypi.org/project/cisco-ai-mcp-scanner/), MCP-Shield variants | Static scans of tool descriptions (poisoning, rug pulls) | No runtime faults, no behavioral measurement |
| garak, PyRIT, promptfoo | Prompt/model red-teaming, eval pipelines | Don't corrupt tool transports |
| AgentDojo (arXiv 2406.13352), InjecAgent (2403.02691), ASB | Injection-in-tool-output benchmarks | Injection only; single agent; no reliability faults |
| Toxiproxy, Chaos Mesh, LitmusChaos | Network/infra chaos | Not protocol-aware; can't alter JSON/schemas |
| **[dimhold/fault-mcp](https://github.com/dimhold/fault-mcp)** | MCP proxy, six fault types, per-tool, seeded replay (0★, active Oct 2026) | **Closest tool overlap.** No multi-agent targets, judge, red team, or hardening study (per README) |
| **[yashkhou/mcp-chaos](https://github.com/yashkhou/mcp-chaos)** | Deterministic JSON-RPC/MCP fault injection (0★) | **Same name.** Library-level; no benchmark |
| mcp-chaos-monkey (PyPI) | Transport faults for testing retries/breakers | No agent eval |
| [RicoKomenda/agentic-chaos](https://github.com/RicoKomenda/agentic-chaos) | Security chaos for tool/MCP calls | Not yet assessed in depth |
| **arXiv [2610.32555](https://arxiv.org/abs/2610.32555)**, "Tool-Calling LLM Agents Under Realistic MCP Faults: Scale Does Not Purchase Fault Tolerance" | Protocol-aware MCP fault middleware; large single-agent study | **Closest academic overlap.** Must be cited; we position against it |
| arXiv [2608.06790](https://arxiv.org/abs/2608.06790) AgentChaos, AgentCheck (2607.11098) | Faults at the LLM-API layer; reproduce-and-intervene workbench | Different injection layer |

## Positioning
A seeded MCP fault proxy alone is **not novel**. mcp-chaos's contributions are:
1. **Multi-agent propagation:** blast-radius labels showing whether a fault in one worker corrupts peers, the supervisor's claims, or persistent state.
2. **Cross-framework comparison** under byte-identical faults (LangGraph vs. Claude Agent SDK).
3. **Adaptive red team vs. random search** at equal budget.
4. **Hardening before/after with ablations** and paired significance tests.
5. **Engineering rigor:** sandboxing, cassette-replayed $0 CI, and a validated judge (κ vs. human labels).
