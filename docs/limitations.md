# Limitations

Read this before quoting anything from the repository.

## No live results exist
No benchmark against a real LLM has been run (the author chose to spend nothing). Every number
in the repository comes from tests or from the **synthetic** self-test, where scripted fake
models stand in for the LLM. The self-test shows that the pipeline works end to end and that
the guard's injection filter blocks a deliberately gullible agent. It says nothing about how
Claude, or any other model, behaves.

## What the harness cannot tell you (yet)
- **Judge quality is unmeasured.** The LLM judge and the 100-run hand-label set (Cohen's kappa)
  have not been run. Only the deterministic label parts (`task_success`, `followed_injection`)
  are exercised, offline.
- **Small task set.** Six tasks (T1-T6), one fixture each; the tasks are ours, not an
  external benchmark.
- **One model family.** Defaults are `claude-haiku-5-5` (agents) and `claude-sonnet-5-5`
  (judge, red team). Non-Anthropic models are a non-goal.
- **Seeds are repeated trials, not independent samples.** Temperature-0 is not deterministic;
  with 3 seeds per cell, most confidence intervals will be wide. The scorecard marks cells with
  n < 10 as "insufficient n".
- **Canary-only injections.** F08 payloads are short, benign, labelled instructions. Real
  attacks are subtler; low compliance on canaries is an upper bound on safety, not evidence of it.
- **Our own SQLite MCP server.** The official one was archived, so `servers/sqlite_server.py`
  is ours. Its behaviour (e.g. `structuredContent` with `outputSchema`) shapes how faults
  surface and may differ from servers in the wild.
- **Framework comparability (RQ2).** The Agent SDK shows the model only `structuredContent`
  when a tool result carries it, so text-block (`prefix`/`suffix`) injections never reach
  Agent SDK agents on structured tools while `field` injections do. LangGraph sees the text
  blocks. Prompts are also not byte-equivalent across frameworks (ADR 0005). A framework gap
  may therefore reflect plumbing, not robustness.
- **Fault realism.** Faults are injected by a proxy at fixed, early call indices with
  probability 1. Real faults are bursty, correlated, and state-dependent. F06 (stale data)
  fires only on repeated identical calls, so its firing rate is lower; the scorecard
  conditions on faults fired.
- **F04 at the frame layer is a hang.** The MCP Python SDK client drops unparseable lines, so
  the harness treats it as a timeout; it cannot distinguish "the agent handled a garbled
  frame" from "the SDK ate it".
- **Cost estimate is a range, not a forecast.** `--dry-run` assumes 30k-120k input and 2k-6k
  output tokens per run and ignores caching. The upper bound exceeds the default `$25` cap, so a
  full live run may abort on budget.
- **Platform.** Developed and tested on macOS and Linux CI with Python 3.12. The Docker sandbox
  path is covered by tests that need a local Docker daemon, not by the default CI job. The
  `local` sandbox (used by the self-test) provides no isolation and must only be used with
  fake models.
- **Ablations and red team.** The red-team vs. random-search comparison (RQ3) and the
  hardening ablations (RQ4) are implemented but have produced no live data.
