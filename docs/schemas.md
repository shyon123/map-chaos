# Run / result schemas

All models are pydantic v2, `frozen=True, extra="forbid"`, in `src/mcp_chaos/schemas/`. Each record carries `schema_version: Literal["1"]`. JSONL files hold one model per line. This doc is the contract; the code must match it.

```python
# faults.py
class Trigger(BaseModel):
    tool: str = "*"                    # fnmatch glob on tool name
    server: str = "*"                  # upstream server id
    calls: list[int] | None = None     # per-tool call indices (0-based); None = any
    probability: float = 1.0           # in [0,1], drawn from the per-call RNG

class FaultSpec(BaseModel):
    id: str                            # e.g. "F08"
    mode: FaultMode                    # StrEnum: timeout, latency, server_error, malformed_json,
                                       # schema_drift, stale_data, oversized_output,
                                       # prompt_injection, empty_result, lying_status, tool_vanish
    params: FaultParams                # discriminated union on `mode`, one model per mode
    trigger: Trigger = Trigger()

class FaultPlan(BaseModel):
    plan_id: str
    seed: int
    faults: list[FaultSpec]            # first matching fault wins per call
    source: Literal["static", "redteam"] = "static"
    parent_plan_id: str | None = None  # red-team lineage

# runs.py
class ModelConfig(BaseModel):
    provider: Literal["anthropic"] = "anthropic"
    model: str = "claude-haiku-5-5"
    temperature: float = 0.0
    max_tokens: int = 2048

class RunConfig(BaseModel):
    run_id: str                        # deterministic: hash(framework, task_id, plan_id, seed, hardening)
    framework: Literal["langgraph", "agentsdk"]
    task_id: str
    agent_model: ModelConfig
    fault_plan: FaultPlan
    hardening: HardeningConfig | None  # None = baseline
    seed: int
    max_turns: int = 25
    timeout_s: float = 300
    budget_usd: float                  # per-run cap; the benchmark also enforces a global cap

# trace.py
class FaultApplied(BaseModel):
    fault_id: str
    mode: FaultMode
    detail: dict[str, JsonValue]       # e.g. {"op":"rename","path":"rows[].id"}
    parseable: bool | None = None

class TraceEvent(BaseModel):
    run_id: str
    seq: int                           # monotonic per run
    t_ms: int                          # from injected Clock, relative to run start
    agent: str                         # "supervisor" | worker name (from proxy instance tag)
    server: str
    direction: Literal["c2s", "s2c"]
    method: str                        # "tools/call", "tools/list", ...
    tool: str | None
    tool_call_index: int | None
    args_sha256: str | None            # args are hashed, never logged raw
    result_sha256: str | None
    result_bytes: int | None
    is_error: bool | None
    latency_ms: int | None
    fault: FaultApplied | None

# results.py
class BlastRadius(StrEnum): none, self, peer, supervisor, persistent

class Labels(BaseModel):
    task_success: bool
    failure_detected: bool             # did the agent's output acknowledge a tool problem?
    false_success_claim: bool          # claimed success while task_success is False
    followed_injection: bool | None    # None when no F08 fault fired
    blast_radius: BlastRadius
    label_source: dict[str, Literal["deterministic", "llm", "llm+deterministic"]]
    judge_rationale: str | None        # LLM judge rationale (<=500 chars)

class Usage(BaseModel):
    input_tokens: int; output_tokens: int
    cache_read_tokens: int; cache_write_tokens: int
    cost_usd: float; cache_hits: int   # on-disk response cache hits

class RunResult(BaseModel):
    run_id: str
    config: RunConfig
    status: Literal["completed", "agent_error", "timeout", "budget_abort", "harness_error"]
    final_answer: str | None
    side_effects: SideEffects          # FS diff, git log diff, SQLite row diff of the scratch sandbox
    faults_fired: list[FaultApplied]
    labels: Labels | None              # filled by the judge in a second pass
    usage: Usage
    agent_usage: Usage                 # target agents only
    judge_usage: Usage | None
    versions: dict[str, str]           # package versions + git sha + cassette id
    started_at: datetime; wall_s: float   # wall time is informational, never used in decisions
```

**Files**: `results/<bench_id>/runs.jsonl` (RunResult), `results/<bench_id>/traces/<run_id>.jsonl` (TraceEvent), `results/<bench_id>/scorecard.json`, and `results/<bench_id>/manifest.json` (configs, seeds, versions, total cost). Results are committed; large traces are gzipped.
