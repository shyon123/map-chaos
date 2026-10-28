"""structlog configuration with secret redaction. Logs go to stderr (stdout is JSON-RPC)."""

from __future__ import annotations

import logging
import re
import sys
from typing import Any, TextIO, cast

import structlog
from structlog.typing import EventDict, FilteringBoundLogger, Processor, WrappedLogger

REDACTED = "[REDACTED]"
# "tokens" (plural, e.g. input_tokens/max_tokens) are counts, not secrets: not matched.
_SECRET_KEY_RE = re.compile(r"api_key|authorization|token(?!s)|secret|password|x-api-key", re.I)
_SECRET_VALUE_RE = re.compile(r"sk-ant-[A-Za-z0-9_-]+")


def _scrub(value: Any) -> Any:
    if isinstance(value, str):
        return _SECRET_VALUE_RE.sub(REDACTED, value)
    if isinstance(value, dict):
        d = cast("dict[Any, Any]", value)
        return {
            k: REDACTED if isinstance(k, str) and _SECRET_KEY_RE.search(k) else _scrub(v)
            for k, v in d.items()
        }
    if isinstance(value, list | tuple):
        seq = cast("list[Any] | tuple[Any, ...]", value)
        return type(seq)(_scrub(v) for v in seq)
    return value


def redact_processor(_logger: WrappedLogger, _method: str, event_dict: EventDict) -> EventDict:
    """Replace secret-keyed values and scrub `sk-ant-...` strings anywhere in the event."""
    return cast("EventDict", _scrub(dict(event_dict)))


def configure_logging(level: str = "INFO", json: bool = False) -> None:
    renderer: Processor = (
        structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer(colors=False)
    )
    numeric = logging.getLevelNamesMapping()[level.upper()]
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            redact_processor,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric),
        logger_factory=structlog.PrintLoggerFactory(file=cast("TextIO", _CurrentStderr())),
        cache_logger_on_first_use=False,
    )


class _CurrentStderr:
    """Resolves sys.stderr at write time, so a swapped/closed stream is never captured."""

    def write(self, s: str) -> int:
        return sys.stderr.write(s)

    def flush(self) -> None:
        sys.stderr.flush()


def get_logger(name: str) -> FilteringBoundLogger:
    return cast("FilteringBoundLogger", structlog.get_logger(name))
