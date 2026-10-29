# CLAUDE.md — mcp-chaos

Chaos-testing harness for MCP-based multi-agent systems. Full design: `SPEC.md`. Threat model: `SECURITY.md`. Decisions: `docs/adr/`. Faults: `docs/fault-taxonomy.md`. Schemas: `docs/schemas.md`.

## Layout (src/ layout, package `mcp_chaos`)
```
src/mcp_chaos/
  schemas/      pydantic v2 models: faults.py, runs.py, results.py, trace.py  (single source of truth)
  proxy/        raw JSON-RPC stdio relay (ADR 0002): relay.py, core.py, framing.py, engine.py (seeded),
                faults/<mode>.py (F01-F11, discover.py registry), canaries.py, guard.py (hardening proxy, ADR 0007)
  harden/       guard components: config.py, pipeline.py, validate.py, retry.py (retry + breaker),
                inject_filter.py, size_cap.py
  servers/      fake_server.py (deterministic, tests only), sqlite_server.py (ADR 0004)
  targets/      tasks.py, fixtures.py, workers.py, entry.py (in-sandbox entrypoint, scripted fake models),
                prompts/*.md, langgraph_target/, agentsdk_target/
  llm/          gateway.py (ADR 0003), cache.py, cassette.py, budget.py, prices.py, normalize.py,
                fake_anthropic.py (scripted fake Anthropic API for tests)
  judge/        deterministic.py, llm_judge.py, labels.py, kappa.py, cli.py
  redteam/      agent.py, loop.py (adaptive vs random search), space.py, sanitize.py, models.py, cli.py
  bench/        cli.py (`mcp-chaos` entrypoint), matrix.py, runner.py, sandbox.py, gateway_host.py,
                safe_repo.py, snapshot.py, stats.py, scorecard.py
  log.py        structlog config + secret redaction
tests/unit  tests/property  tests/integration  tests/fixtures (fake_models/, fake_scripts/)  tests/live (@live, opt-in)
configs/     bench.yaml, cells.yaml, ablation.yaml, smoke.yaml, selftest.yaml, selftest_cells.yaml
results/     committed run outputs; results/selftest-synthetic/ is the SYNTHETIC $0 demo (make selftest)
cassettes/   recorded LLM traffic (none yet)      docker/   sandbox image      docs/   adr/, architecture, taxonomy, schemas, limitations
```

## Commands
```
uv sync --frozen                         # install from lockfile
uv run ruff check . && uv run ruff format --check .
uv run pyright src                       # strict
uv run pytest -q                         # unit+property+integration, replays cassettes, $0
uv run pytest -q -m live                 # opt-in live LLM tests (needs ANTHROPIC_API_KEY)
uv run pytest --cov=mcp_chaos --cov-fail-under=85
make check                               # all of the above except live
make selftest                            # $0 end-to-end demo with scripted fake models (SYNTHETIC)
uv run mcp-chaos run --config configs/<x>.yaml --budget-usd 2
```

## Conventions
- Python 3.12, full type hints, pyright strict on `src/`. Pydantic v2 for every schema crossing a module boundary; `model_config = ConfigDict(frozen=True, extra="forbid")` by default.
- Async I/O (`anyio`). No `time.time()`/`random` directly: use injected `Clock` and `numpy.random.Generator`/`random.Random(seed)` passed in.
- Logging: `mcp_chaos.log.get_logger(__name__)`; structured key=value; never log secrets or full prompts (log prompt hashes).
- Tool output is UNTRUSTED data. Injection payloads are benign canaries only (`CANARY-<hex>`).
- Tests with or before code. Unit test per fault mode; property tests for determinism/passthrough/parseability; integration tests use `servers/fake_server.py`; no network in default test run. Mark live tests `@pytest.mark.live`.
- Flaky test = bug: fix or `@pytest.mark.quarantine` with an issue link.
- Commits: one per slice, conventional prefix (`feat(proxy): ...`). Nothing is done until `make check` is green.

## Subagent protocol
Agents in `.claude/agents/`: implementer (sonnet), mechanic (haiku), reviewer (sonnet), researcher (haiku). Each task gives exact paths + acceptance criteria; reports <=150 words (researcher <=200). Read files; don't expect pasted context.
