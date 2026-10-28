from __future__ import annotations

import pytest

from mcp_chaos.llm.prices import UnknownModelError, compute_cost, price_for


def test_known_models_cost() -> None:
    assert compute_cost("claude-haiku-5-5", input_tokens=50_000, output_tokens=50_000) == (
        pytest.approx(50_000 * 0.60 / 1e6)
    )
    assert compute_cost(
        "claude-sonnet-5-5",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_write_tokens=1_000_000,
        cache_read_tokens=1_000_000,
    ) == pytest.approx(2 + 10 + 2.5 + 0.10)
    assert compute_cost("claude-opus-5-5", output_tokens=1_000_000) == pytest.approx(20.0)


def test_haiku_long_context_tier() -> None:
    assert compute_cost("claude-haiku-5-5", input_tokens=100_001) == pytest.approx(
        100_001 * 0.50 / 1e6
    )
    assert compute_cost("claude-haiku-5-5", input_tokens=100_000) == pytest.approx(
        100_000 * 0.10 / 1e6
    )


def test_dated_snapshot_resolves() -> None:
    assert price_for("claude-haiku-5-5-20261001") is price_for("claude-haiku-5-5")


@pytest.mark.parametrize("model", ["gpt-4", "claude-haiku-9-9", "claude-x-5-5-20261001", ""])
def test_unknown_fails_closed(model: str) -> None:
    with pytest.raises(UnknownModelError):
        compute_cost(model, input_tokens=1)
