# Architecture

```mermaid
flowchart LR
  subgraph Harness["Harness (trusted, host)"]
    CLI["mcp-chaos CLI<br/>bench runner"]
    GW["LLM gateway<br/>cache · cassettes · budget<br/>holds real API key"]
    RT["Red-team agent<br/>traces → next FaultPlan"]
    J["Judge<br/>deterministic → LLM (Batch)"]
    SC["Scorecard + taxonomy<br/>Wilson CI · McNemar"]
    RES[("results/*.jsonl<br/>traces/*.jsonl")]
  end

  subgraph Sandbox["Docker sandbox (untrusted): non-root, ro rootfs, /scratch tmpfs, egress→gateway only"]
    SUP["Supervisor agent<br/>(LangGraph | Agent SDK)"]
    WF["files worker"]
    WD["data worker"]
    WV["vcs worker"]
    subgraph Chain["per worker × server: proxy chain"]
      G1["guard<br/>(hardening, optional)"] --> P1["fault proxy<br/>seeded FaultPlan"]
    end
    FS["filesystem MCP<br/>(official, Node)"]
    SQ["SQLite MCP<br/>(ours)"]
    GIT["git MCP<br/>(official, Python)"]
    SCR[("/scratch<br/>fixture repo, db, files")]
  end

  CLI -- RunConfig --> SUP
  SUP --> WF & WD & WV
  WF & WD & WV -- stdio JSON-RPC --> G1
  P1 --> FS & SQ & GIT
  FS & SQ & GIT --- SCR
  SUP & WF & WD & WV -. Messages API .-> GW
  P1 -- TraceEvents --> RES
  CLI -- snapshot diff, RunResult --> RES
  RES --> J --> RES
  RES --> RT -- FaultPlan --> CLI
  RES --> SC
  J & RT -.-> GW
```

**Data flow of one run:** the runner materializes a fixture into `/scratch`, writes the `FaultPlan`, and starts the container. Each worker's MCP server command is `mcp-chaos proxy --agent <w> --server <s> [--guard cfg] --plan plan.json -- <upstream cmd>`. The agents run the task, the proxy emits `TraceEvent`s, the runner snapshots `/scratch` and diffs it against the fixture, and writes a `RunResult`. The judge labels results in a separate pass. The red team reads labeled results and emits the next `FaultPlan`s.
