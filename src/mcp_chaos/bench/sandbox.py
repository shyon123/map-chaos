"""Docker sandbox for target runs (SPEC section 10, runner-contract).

The target container sits only on the ``--internal`` network ``mcpchaos-internal``. A socat
sidecar ``mcpchaos-gw`` (default bridge + internal network, alias ``gw``) forwards ``gw:8080`` to
the loopback-bound gateway on the host, reached through the VM's host alias
(``host.lima.internal`` on Colima, ``host.docker.internal`` on Docker Desktop).
"""

from __future__ import annotations

import subprocess
import uuid
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from mcp_chaos.log import get_logger

log = get_logger(__name__)

NETWORK = "mcpchaos-internal"
RELAY_NAME = "mcpchaos-gw"
RELAY_ALIAS = "gw"
RELAY_PORT = 8080
SOCAT_IMAGE = "alpine/socat@sha256:beb4a68d9e4fe6b0f21ea774a0fde6c31f580dde6368939ed70100c5385b015e"
HOST_ALIASES = ("host.lima.internal", "host.docker.internal")
DEFAULT_IMAGE = "mcpchaos-target:dev"
KILL_GRACE_S = 30.0
UID = "10001"
_LABEL = "mcpchaos.host_target"


class SandboxError(RuntimeError):
    pass


class ContainerResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    exit_code: int | None
    timed_out: bool
    stdout_tail: str = ""
    stderr_tail: str = ""


def docker(
    *args: str, check: bool = True, timeout: float | None = None
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(  # noqa: S603
        ["docker", *args],  # noqa: S607
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if check and proc.returncode != 0:
        raise SandboxError(f"docker {args[0]} failed: {proc.stderr.strip()[-500:]}")
    return proc


def ensure_network() -> None:
    if docker("network", "inspect", NETWORK, check=False).returncode == 0:
        return
    docker("network", "create", "--internal", NETWORK)
    log.info("sandbox_network_created", name=NETWORK)


def _relay_target() -> str | None:
    p = docker("inspect", "-f", f'{{{{index .Config.Labels "{_LABEL}"}}}}', RELAY_NAME, check=False)
    if p.returncode != 0:
        return None
    return p.stdout.strip()


def _relay_running() -> bool:
    p = docker("inspect", "-f", "{{.State.Running}}", RELAY_NAME, check=False)
    return p.returncode == 0 and p.stdout.strip() == "true"


def _pick_host_alias() -> str:
    for name in HOST_ALIASES:
        p = docker(
            "run",
            "--rm",
            "--entrypoint",
            "getent",
            SOCAT_IMAGE,
            "hosts",
            name,
            check=False,
            timeout=60,
        )
        if p.returncode == 0:
            return name
    raise SandboxError(f"no host alias resolves in containers (tried {', '.join(HOST_ALIASES)})")


def ensure_gw_relay(host_port: int) -> None:
    """(Re)create the socat sidecar so ``gw:8080`` reaches ``127.0.0.1:host_port`` on the host."""
    ensure_network()
    want = str(host_port)
    if _relay_running() and _relay_target() == want:
        return
    docker("rm", "-f", RELAY_NAME, check=False)
    host = _pick_host_alias()
    docker(
        "run",
        "-d",
        "--name",
        RELAY_NAME,
        "--label",
        f"{_LABEL}={want}",
        "--network",
        "bridge",
        "--user",
        UID,
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=64",
        "--memory=64m",
        "--restart",
        "unless-stopped",
        SOCAT_IMAGE,
        f"TCP-LISTEN:{RELAY_PORT},fork,reuseaddr",
        f"TCP:{host}:{host_port}",
    )
    docker("network", "connect", "--alias", RELAY_ALIAS, NETWORK, RELAY_NAME)
    probe = docker("exec", RELAY_NAME, "nc", "-z", "-w", "3", host, want, check=False, timeout=30)
    if probe.returncode != 0:
        raise SandboxError(f"relay cannot reach host gateway at {host}:{host_port}")
    log.info("sandbox_relay_ready", host=host, port=host_port)


def build_image(tag: str = DEFAULT_IMAGE, *, context: Path | None = None) -> str:
    """Build the target image from ``docker/target.Dockerfile``; return its image id."""
    ctx = context or Path(__file__).resolve().parents[3]
    docker("build", "-q", "-f", str(ctx / "docker" / "target.Dockerfile"), "-t", tag, str(ctx))
    return image_id(tag)


def image_id(tag: str) -> str:
    return docker("image", "inspect", "-f", "{{.Id}}", tag).stdout.strip()


def _mount(src: Path, dst: str, *, ro: bool) -> list[str]:
    spec = f"type=bind,src={src},dst={dst}" + (",readonly" if ro else "")
    return ["--mount", spec]


def container_args(
    *,
    image: str,
    name: str,
    scratch: Path,
    in_dir: Path,
    out_dir: Path,
    env: dict[str, str],
    entry_args: list[str],
    entrypoint: str | None = None,
) -> list[str]:
    """The exact ``docker run`` argv (SPEC section 10 flags)."""
    args = [
        "run",
        "--rm",
        "--name",
        name,
        "--user",
        f"{UID}:{UID}",
        "--read-only",
        "--tmpfs",
        "/tmp:size=64m",  # noqa: S108
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=256",
        "--memory=1g",
        "--cpus=1",
        "--network",
        NETWORK,
        *_mount(scratch, "/scratch", ro=False),
        *_mount(in_dir, "/run/in", ro=True),
        *_mount(out_dir, "/run/out", ro=False),
    ]
    for k, v in env.items():
        args += ["-e", f"{k}={v}"]
    if entrypoint is not None:
        args += ["--entrypoint", entrypoint]
    return [*args, image, *entry_args]


def run_container(
    *,
    image: str,
    name: str | None = None,
    scratch: Path,
    in_dir: Path,
    out_dir: Path,
    env: dict[str, str],
    timeout_s: float,
    entry_args: list[str] | None = None,
    entrypoint: str | None = None,
) -> ContainerResult:
    """Run one target container; kill it after ``timeout_s`` + grace (status: timeout)."""
    cname = name or f"mcpchaos-run-{uuid.uuid4().hex[:12]}"
    argv = container_args(
        image=image,
        name=cname,
        scratch=scratch,
        in_dir=in_dir,
        out_dir=out_dir,
        env=env,
        entrypoint=entrypoint,
        entry_args=entry_args or ["--config", "/run/in/config.json", "--out", "/run/out"],
    )
    proc = subprocess.Popen(  # noqa: S603
        ["docker", *argv],  # noqa: S607
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        outs, err = proc.communicate(timeout=timeout_s + KILL_GRACE_S)
    except subprocess.TimeoutExpired:
        docker("kill", cname, check=False, timeout=30)
        try:
            outs, err = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            outs, err = proc.communicate()
        docker("rm", "-f", cname, check=False, timeout=30)
        log.warning("container_timeout", name=cname)
        return ContainerResult(
            exit_code=None,
            timed_out=True,
            stdout_tail=(outs or "")[-2000:],
            stderr_tail=(err or "")[-1000:],
        )
    return ContainerResult(
        exit_code=proc.returncode,
        timed_out=False,
        stdout_tail=(outs or "")[-2000:],
        stderr_tail=(err or "")[-1000:],
    )
