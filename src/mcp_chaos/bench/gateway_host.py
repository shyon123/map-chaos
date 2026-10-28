"""Run the LLM gateway (ADR 0003) in a background thread on 127.0.0.1 for the runner."""

from __future__ import annotations

import threading
import time
from types import TracebackType
from typing import Self

import uvicorn

from mcp_chaos.llm.gateway import Gateway, GatewayConfig, create_app
from mcp_chaos.log import get_logger
from mcp_chaos.schemas.results import Usage

log = get_logger(__name__)


class GatewayStartError(RuntimeError):
    pass


class GatewayHost:
    """Owns a uvicorn server (own event loop, own thread) bound to loopback on a free port."""

    def __init__(self, cfg: GatewayConfig, *, start_timeout_s: float = 15.0) -> None:
        app = create_app(cfg)
        self.gateway: Gateway = app.state.gateway
        self._server = uvicorn.Server(
            uvicorn.Config(
                app, host="127.0.0.1", port=0, log_level="warning", lifespan="on", access_log=False
            )
        )
        self._thread = threading.Thread(target=self._server.run, name="gateway-host", daemon=True)
        self._start_timeout_s = start_timeout_s
        self.port = 0

    def start(self) -> None:
        self._thread.start()
        deadline = time.monotonic() + self._start_timeout_s
        while not self._server.started:
            if not self._thread.is_alive() or time.monotonic() > deadline:
                raise GatewayStartError("gateway failed to start")
            time.sleep(0.02)
        self.port = int(self._server.servers[0].sockets[0].getsockname()[1])
        log.info("gateway_started", port=self.port)

    def close(self) -> None:
        self._server.should_exit = True
        if self._thread.is_alive():
            self._thread.join(timeout=15)

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def base_url(self, run_id: str, *, host: str = "127.0.0.1", port: int | None = None) -> str:
        return f"http://{host}:{port or self.port}/r/{run_id}"

    def register_run(self, run_id: str, seed: int, budget_usd: float) -> None:
        self.gateway.register_run(run_id, seed, budget_usd)

    def usage(self, run_id: str) -> Usage:
        return self.gateway.get_usage(run_id)

    def budget_aborted(self, run_id: str) -> bool:
        return self.gateway.is_budget_aborted(run_id)

    @property
    def global_spend_usd(self) -> float:
        return self.gateway.budget.global_spend
