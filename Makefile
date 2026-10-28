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

check: lint types cov audit

.PHONY: sync lint types test cov audit check
