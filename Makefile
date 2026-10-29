sync:
	uv sync --frozen

lint:
	uv run ruff check .
	uv run ruff format --check .

types:
	uv run pyright

test:
	uv run pytest -q

cov:
	uv run pytest -q --cov=mcp_chaos.proxy --cov=mcp_chaos.schemas --cov=mcp_chaos.llm --cov=mcp_chaos.judge --cov=mcp_chaos.harden --cov-report=term-missing --cov-fail-under=85

audit:
	mkdir -p .cache
	uv export --frozen --no-dev --no-emit-project --format requirements-txt > .cache/req.txt
	uv run pip-audit -r .cache/req.txt --require-hashes --disable-pip

SELFTEST_ID = selftest-synthetic

# $0 end-to-end pipeline demo: fake scripted models, local sandbox, offline judge. SYNTHETIC output.
selftest:
	rm -rf results/$(SELFTEST_ID) cassettes/$(SELFTEST_ID)
	MCP_CHAOS_FAKE_MODELS=tests/fixtures/fake_models/selftest_echo.json \
		uv run mcp-chaos run --config configs/selftest.yaml --sandbox local --budget-usd 1
	uv run mcp-chaos judge --bench-id $(SELFTEST_ID) --mode offline
	uv run mcp-chaos scorecard --bench-id $(SELFTEST_ID) --deterministic

check: lint types cov audit

.PHONY: sync lint types test cov audit check selftest
