"""anyio I/O shell around ProxyCore: spawns the upstream server and relays stdio."""

from __future__ import annotations

import contextlib
import os
import stat
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import IO

import anyio
from anyio.abc import Process, TaskGroup

from mcp_chaos.log import get_logger
from mcp_chaos.proxy.core import (
    Action,
    Crash,
    Delay,
    Hang,
    ProxyCore,
    SendToClient,
    SendToServer,
    Trace,
)
from mcp_chaos.proxy.framing import FramingError, LineSplitter

log = get_logger(__name__)

_GRACE_S = 5.0
_CHUNK = 65536


class SystemClock:
    """Monotonic milliseconds since construction."""

    def __init__(self) -> None:
        self._t0 = time.monotonic_ns()

    def now_ms(self) -> int:
        return (time.monotonic_ns() - self._t0) // 1_000_000


class _Shell:
    def __init__(
        self,
        core: ProxyCore,
        proc: Process,
        trace: anyio.AsyncFile[str],
        server: str,
        out: anyio.AsyncFile[bytes],
    ) -> None:
        self.core = core
        self.proc = proc
        self.trace = trace
        self.server = server
        self.out = out
        self.tg: TaskGroup | None = None
        self._out_lock = anyio.Lock()
        self._trace_lock = anyio.Lock()
        self.inflight = 0
        self._idle = anyio.Event()
        self._idle.set()

    async def execute(self, actions: list[Action]) -> None:
        for a in actions:
            await self._exec(a)

    async def _exec(self, a: Action) -> None:
        match a:
            case SendToServer(line):
                if self.proc.stdin is not None:
                    try:
                        await self.proc.stdin.send(line + b"\n")
                    except (anyio.BrokenResourceError, anyio.ClosedResourceError):
                        log.warning("upstream_stdin_closed", server=self.server)
            case SendToClient(line):
                async with self._out_lock:
                    await self.out.write(line + b"\n")
                    await self.out.flush()
            case Trace(event):
                async with self._trace_lock:
                    await self.trace.write(event.model_dump_json() + "\n")
                    await self.trace.flush()
            case Delay(seconds, then):
                if self.tg is None:
                    raise RuntimeError("task group not started")
                if self.inflight == 0:
                    self._idle = anyio.Event()
                self.inflight += 1
                self.tg.start_soon(self._delayed, seconds, then)
            case Hang():
                log.info("fault_hang", server=self.server)
            case Crash():
                log.info("fault_crash", server=self.server)
                if self.proc.returncode is None:
                    self.proc.kill()

    async def _delayed(self, seconds: float, then: Action) -> None:
        try:
            await anyio.sleep(seconds)
            await self._exec(then)
        finally:
            self.inflight -= 1
            if self.inflight == 0:
                self._idle.set()

    async def drain_delayed(self) -> None:
        await self._idle.wait()


def _need[T](x: T | None) -> T:
    if x is None:
        raise RuntimeError("missing stream")
    return x


def _pollable(fd: int) -> bool:
    try:
        mode = os.fstat(fd).st_mode
    except OSError:
        return False
    return stat.S_ISFIFO(mode) or stat.S_ISSOCK(mode) or stat.S_ISCHR(mode)


async def _read_fd(fd: int) -> bytes:
    if not _pollable(fd):
        # regular files never block: a worker thread is fine and cannot hang shutdown
        return await anyio.to_thread.run_sync(os.read, fd, _CHUNK, abandon_on_cancel=True)
    # pipes/sockets/ttys: non-blocking read, cancellable with no worker thread to join
    os.set_blocking(fd, False)
    while True:
        try:
            return os.read(fd, _CHUNK)
        except BlockingIOError:
            await anyio.wait_readable(fd)


async def _terminate(proc: Process) -> None:
    with anyio.move_on_after(_GRACE_S):
        await proc.wait()
        return
    proc.terminate()
    with anyio.move_on_after(2.0):
        await proc.wait()
        return
    proc.kill()
    await proc.wait()


async def run_relay(
    core: ProxyCore,
    cmd: Sequence[str],
    trace_path: Path,
    server: str,
    *,
    stdin_fd: int = 0,
    out: IO[bytes] | None = None,
) -> int:
    await anyio.Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
    was_blocking = True
    with contextlib.suppress(OSError):
        was_blocking = os.get_blocking(stdin_fd)
    async with await anyio.open_file(trace_path, "a", encoding="utf-8") as trace:
        proc = await anyio.open_process(list(cmd), stdin=-1, stdout=-1, stderr=-1)
        try:
            return await _relay(core, proc, trace, server, stdin_fd, out)
        finally:
            with contextlib.suppress(OSError):
                os.set_blocking(stdin_fd, was_blocking)
            with anyio.CancelScope(shield=True):
                if proc.returncode is None:
                    proc.kill()
                await proc.wait()


async def _relay(
    core: ProxyCore,
    proc: Process,
    trace: anyio.AsyncFile[str],
    server: str,
    stdin_fd: int,
    out: IO[bytes] | None,
) -> int:
    out_f = anyio.wrap_file(out if out is not None else sys.stdout.buffer)
    err_f = anyio.wrap_file(sys.stderr.buffer)
    shell = _Shell(core, proc, trace, server, out_f)
    stderr_done = anyio.Event()
    client_closed = anyio.Event()

    async def pump_client() -> None:
        splitter = LineSplitter()
        try:
            while True:
                chunk = await _read_fd(stdin_fd)
                if not chunk:
                    break
                for line in splitter.feed(chunk):
                    await shell.execute(core.on_client_line(line))
            rest = splitter.flush()
            if rest is not None:
                await shell.execute(core.on_client_line(rest))
        except FramingError as e:
            log.error("framing_error", side="client", error=str(e))
        finally:
            client_closed.set()
            if proc.stdin is not None:
                await proc.stdin.aclose()
        await _terminate(proc)

    async def pump_server() -> None:
        stdout = _need(proc.stdout)
        splitter = LineSplitter()
        try:
            async for chunk in stdout:
                for line in splitter.feed(chunk):
                    await shell.execute(core.on_server_line(line))
            rest = splitter.flush()
            if rest is not None:
                await shell.execute(core.on_server_line(rest))
        except FramingError as e:
            log.error("framing_error", side="server", error=str(e))
            proc.kill()
        await proc.wait()
        await stderr_done.wait()
        # deliver in-flight delayed responses (bounded by their delay; 5 s once client is gone)
        if client_closed.is_set():
            with anyio.move_on_after(_GRACE_S) as scope:
                await shell.drain_delayed()
            if scope.cancelled_caught:
                log.warning("delayed_responses_dropped", server=server, grace_s=_GRACE_S)
        else:
            await shell.drain_delayed()
        _need(shell.tg).cancel_scope.cancel()

    async def pump_stderr() -> None:
        stderr = _need(proc.stderr)
        prefix = f"[upstream:{server}] ".encode()
        splitter = LineSplitter()
        async for chunk in stderr:
            for line in splitter.feed(chunk):
                await err_f.write(prefix + line + b"\n")
                await err_f.flush()
        rest = splitter.flush()
        if rest is not None:
            await err_f.write(prefix + rest + b"\n")
            await err_f.flush()
        stderr_done.set()

    async with anyio.create_task_group() as tg:
        shell.tg = tg
        tg.start_soon(pump_client)
        tg.start_soon(pump_server)
        tg.start_soon(pump_stderr)
    await proc.wait()
    code = proc.returncode or 0
    return code if code >= 0 else 128 - code
