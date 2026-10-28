"""Safe handling of agent-writable git repositories on the trusted host.

The scratch repo is written by an untrusted agent, so running ``git`` inside it is a code
execution vector (``core.fsmonitor``, filter drivers, hooks, alternates, ...). Post-run
inspection therefore never touches the original: :func:`sanitized_copy` copies it without
following symlinks, forces a pristine ``.git/config`` and strips every other exec-capable file,
and :func:`run_git` runs git hermetically.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

GIT_TIMEOUT_S = 30
_DEV_NULL = "/dev/null"

# Resolved once, on the host side, so later PATH tampering cannot redirect it.
_which = shutil.which("git")
GIT_BIN: str | None = str(Path(_which).absolute()) if _which else None

_HERMETIC_CONFIG: tuple[str, ...] = (
    "core.hooksPath=/dev/null",
    "core.fsmonitor=false",
    "core.pager=cat",
    "safe.directory=*",
    "init.defaultBranch=main",
    "commit.gpgsign=false",
    "core.quotepath=off",
)

_STRIP_GIT_FILES: tuple[str, ...] = (
    "info/attributes",
    "objects/info/alternates",
    "objects/info/http-alternates",
    "config.worktree",
    "commondir",
)
_STRIP_TREE_NAMES = frozenset({".gitattributes", ".gitmodules"})


class UnsafeRepoError(Exception):
    """The scratch directory cannot be inspected safely and was refused."""


@dataclass
class CopyReport:
    """What the sanitizing copy skipped or removed (paths relative to the scratch root)."""

    symlinks: dict[str, str] = field(default_factory=dict[str, str])  # path -> link target
    special: dict[str, str] = field(default_factory=dict[str, str])  # path -> file kind
    stripped: dict[str, bytes] = field(default_factory=dict[str, bytes])  # removed tree files


def git_env(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """Minimal environment: no inherited ``GIT_*``, no user or system config, no prompts."""
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": _DEV_NULL,
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": _DEV_NULL,
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_OPTIONAL_LOCKS": "0",
    }
    if "DEVELOPER_DIR" in os.environ:  # lets the macOS /usr/bin/git shim find its toolchain
        env["DEVELOPER_DIR"] = os.environ["DEVELOPER_DIR"]
    if extra:
        env.update(extra)
    return env


def run_git(
    repo: Path,
    *args: str,
    date: str | None = None,
    check: bool = True,
    extra_env: Mapping[str, str] | None = None,
) -> str:
    """Run ``git`` in ``repo`` hermetically (absolute binary, no shell); return stdout."""
    if GIT_BIN is None:
        raise FileNotFoundError("git executable not found")
    extra = dict(extra_env or {})
    if date is not None:
        extra["GIT_AUTHOR_DATE"] = date
        extra["GIT_COMMITTER_DATE"] = date
    cmd = [GIT_BIN]
    for item in _HERMETIC_CONFIG:
        cmd += ["-c", item]
    proc = subprocess.run(  # noqa: S603
        [*cmd, *args],
        cwd=repo,
        env=git_env(extra),
        capture_output=True,
        text=True,
        errors="replace",
        check=check,
        timeout=GIT_TIMEOUT_S,
        stdin=subprocess.DEVNULL,
    )
    return proc.stdout


def _copy_regular(src: Path, dst: Path) -> bool:
    """Copy ``src`` only if it is a regular file (never follows a final symlink)."""
    try:
        fd = os.open(src, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        return False
    with os.fdopen(fd, "rb") as fin:
        st = os.fstat(fin.fileno())
        if not stat.S_ISREG(st.st_mode):
            return False
        with dst.open("wb") as fout:
            shutil.copyfileobj(fin, fout)
    dst.chmod(stat.S_IMODE(st.st_mode) | stat.S_IWUSR | stat.S_IRUSR)
    return True


def _kind(mode: int) -> str:
    for check_, name in (
        (stat.S_ISFIFO, "fifo"),
        (stat.S_ISSOCK, "socket"),
        (stat.S_ISCHR, "chardev"),
        (stat.S_ISBLK, "blockdev"),
    ):
        if check_(mode):
            return name
    return "other"


def _copy_tree(src: Path, dst: Path, rel: str, report: CopyReport) -> None:
    dst.mkdir(exist_ok=True)
    try:
        entries = sorted(os.scandir(src), key=lambda e: e.name)
    except OSError:
        return
    for entry in entries:
        child = f"{rel}/{entry.name}" if rel else entry.name
        if entry.is_symlink():
            try:
                report.symlinks[child] = str(Path(entry.path).readlink())
            except OSError:  # pragma: no cover - raced away
                report.symlinks[child] = "?"
        elif entry.is_dir(follow_symlinks=False):
            _copy_tree(Path(entry.path), dst / entry.name, child, report)
        elif entry.is_file(follow_symlinks=False):
            if not _copy_regular(Path(entry.path), dst / entry.name):
                report.special[child] = "unreadable"
        else:
            report.special[child] = _kind(entry.stat(follow_symlinks=False).st_mode)


def _strip_tree(root: Path, report: CopyReport) -> None:
    """Remove ``.gitattributes``/``.gitmodules`` from the work tree copy (not from ``.git``)."""
    for dirpath, dirnames, filenames in os.walk(root):
        if Path(dirpath) == root and ".git" in dirnames:
            dirnames.remove(".git")
        for name in filenames:
            if name in _STRIP_TREE_NAMES:
                p = Path(dirpath) / name
                report.stripped[p.relative_to(root).as_posix()] = p.read_bytes()
                p.unlink()


def _sanitize_git_dir(git_dir: Path, pristine_git_config: bytes) -> None:
    (git_dir / "config").write_bytes(pristine_git_config)
    hooks = git_dir / "hooks"
    if hooks.is_dir():
        shutil.rmtree(hooks)
    hooks.mkdir(exist_ok=True)
    for rel in _STRIP_GIT_FILES:
        (git_dir / rel).unlink(missing_ok=True)


@contextmanager
def sanitized_copy_info(
    scratch: Path, pristine_git_config: bytes
) -> Generator[tuple[Path, CopyReport]]:
    """Like :func:`sanitized_copy` but also yields what was skipped/stripped."""
    if scratch.is_symlink():
        raise UnsafeRepoError("scratch root is a symlink")
    has_git = False
    try:
        st = (scratch / ".git").lstat()
    except FileNotFoundError:
        pass
    else:
        if not stat.S_ISDIR(st.st_mode):
            raise UnsafeRepoError(f".git is not a real directory ({_kind(st.st_mode)})")
        has_git = True
    report = CopyReport()
    with tempfile.TemporaryDirectory(prefix="mcp-chaos-snap-") as tmp:
        repo = Path(tmp) / "repo"
        _copy_tree(scratch, repo, "", report)
        if has_git:
            _sanitize_git_dir(repo / ".git", pristine_git_config)
        _strip_tree(repo, report)
        yield repo, report


@contextmanager
def sanitized_copy(scratch: Path, pristine_git_config: bytes) -> Generator[Path]:
    """Yield a sanitized copy of ``scratch`` that is safe to run git and readers against.

    Symlinks are skipped (never followed), ``.git`` must be a real directory, ``.git/config``
    is replaced with ``pristine_git_config`` and hooks, attributes, alternates,
    ``.gitattributes`` and ``.gitmodules`` are deleted from the copy.
    """
    with sanitized_copy_info(scratch, pristine_git_config) as (repo, _report):
        yield repo
