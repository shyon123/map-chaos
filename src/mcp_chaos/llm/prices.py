"""Pinned Anthropic price table (USD per million tokens). Fails closed on unknown models.

Source: https://platform.claude.com/docs/en/about-claude/pricing  (verified 2026-10-28).
Haiku 5.5 is tiered: prompts over 100,000 tokens (input + cache read + cache write) use the
long-context rates. Cache-write rates are the 5-minute TTL rates.
"""

from __future__ import annotations

import re
from typing import Final

from pydantic import BaseModel, ConfigDict

PRICES_AS_OF: Final = "2026-10-28"
PRICES_SOURCE: Final = "https://platform.claude.com/docs/en/about-claude/pricing"


class Rates(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    input: float
    output: float
    cache_write_5m: float
    cache_read: float


class ModelPrice(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    rates: Rates
    long_context_threshold: int | None = None
    long_context_rates: Rates | None = None


class UnknownModelError(KeyError):
    """Raised for a model absent from the price table (fail closed)."""


PRICES: Final[dict[str, ModelPrice]] = {
    "claude-haiku-5-5": ModelPrice(
        rates=Rates(input=0.10, output=0.50, cache_write_5m=0.125, cache_read=0.01),
        long_context_threshold=100_000,
        long_context_rates=Rates(input=0.50, output=2.50, cache_write_5m=0.625, cache_read=0.05),
    ),
    "claude-sonnet-5-5": ModelPrice(
        rates=Rates(input=2.0, output=10.0, cache_write_5m=2.5, cache_read=0.10)
    ),
    "claude-opus-5-5": ModelPrice(
        rates=Rates(input=4.0, output=20.0, cache_write_5m=5.0, cache_read=0.20)
    ),
}

_DATED = re.compile(r"^(?P<base>.+?)-\d{8}$")


def price_for(model: str) -> ModelPrice:
    """Look up a model (an optional ``-YYYYMMDD`` snapshot suffix is ignored)."""
    if model in PRICES:
        return PRICES[model]
    m = _DATED.match(model)
    if m and m["base"] in PRICES:
        return PRICES[m["base"]]
    raise UnknownModelError(f"no price for model {model!r}")


def compute_cost(
    model: str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    p = price_for(model)
    rates = p.rates
    prompt = input_tokens + cache_write_tokens + cache_read_tokens
    if (
        p.long_context_threshold is not None
        and p.long_context_rates is not None
        and prompt > p.long_context_threshold
    ):
        rates = p.long_context_rates
    return (
        input_tokens * rates.input
        + output_tokens * rates.output
        + cache_write_tokens * rates.cache_write_5m
        + cache_read_tokens * rates.cache_read
    ) / 1_000_000


def worst_case_cost(model: str, *, est_input_tokens: int, max_tokens: int) -> float:
    """Upper-bound cost of one request: every prompt token at the dearest input-side rate
    (cache-write) and ``max_tokens`` output tokens, at long-context rates when they could apply."""
    p = price_for(model)
    rates = p.rates
    if (
        p.long_context_threshold is not None
        and p.long_context_rates is not None
        and est_input_tokens > p.long_context_threshold
    ):
        rates = p.long_context_rates
    in_rate = max(rates.input, rates.cache_write_5m, rates.cache_read)
    return (max(est_input_tokens, 0) * in_rate + max(max_tokens, 0) * rates.output) / 1_000_000
