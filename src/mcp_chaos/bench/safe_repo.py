"""Safe handling of agent-writable git repositories on the trusted host.

The scratch repo is written by an untrusted agent, so running ``git`` inside it is a code
execution vector (``core.fsmonitor``, filter drivers, hooks, alternates, ...). Post-run
inspection therefore never touches the original: :func:`sanitized_copy` copies it without
following symlinks, forces a pristine ``.git/config`` and strips every other exec-capable file,
and :func:`run_git` runs git hermetically.
"""

from __future__ import annotations

import contextlib
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


_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
_FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
_CHUNK = 1 << 20


@dataclass(frozen=True)
class CopyLimits:
    """Hard caps for the sanitizing copy; the scratch dir is attacker-controlled."""

    max_bytes: int = 256 * 1024 * 1024
    max_entries: int = 50_000
    max_depth: int = 64
    max_file_bytes: int = 64 * 1024 * 1024


DEFAULT_LIMITS = CopyLimits()


def _open_dir(name: str | Path, dir_fd: int | None = None) -> int:
    """Open a directory without following a final symlink (test seam for race simulation)."""
    return os.open(name, _DIR_FLAGS, dir_fd=dir_fd)


def _open_file(name: str, dir_fd: int) -> int:
    """Open a file without following a final symlink or blocking on a FIFO."""
    return os.open(name, _FILE_FLAGS, dir_fd=dir_fd)


def _copy_file_fd(fd: int, dst: Path, rel: str, limits: CopyLimits, used: int) -> int:
    """Copy the already-opened regular file ``fd`` to ``dst``; return bytes copied."""
    st = os.fstat(fd)
    if st.st_size > limits.max_file_bytes:
        raise UnsafeRepoError(f"file too large: {rel}")
    if used + st.st_size > limits.max_bytes:
        raise UnsafeRepoError("total size cap exceeded")
    n = 0
    try:
        out = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except OSError as exc:
        raise UnsafeRepoError(f"cannot create copy of {rel}") from exc
    with os.fdopen(out, "wb") as fout:
        while True:
            # Read at most one byte past the cap so a file growing under us is caught.
            want = min(_CHUNK, limits.max_file_bytes + 1 - n)
            chunk = os.read(fd, want)
            if not chunk:
                break
            n += len(chunk)
            if n > limits.max_file_bytes:
                raise UnsafeRepoError(f"file too large: {rel}")
            if used + n > limits.max_bytes:
                raise UnsafeRepoError("total size cap exceeded")
            fout.write(chunk)
    dst.chmod(stat.S_IMODE(st.st_mode) | stat.S_IWUSR | stat.S_IRUSR)
    return n


@dataclass
class _Frame:
    fd: int
    entries: list[os.DirEntry[str]]
    dst: Path
    rel: str
    pos: int = 0


def _list_dir(fd: int) -> list[os.DirEntry[str]]:
    with os.scandir(fd) as it:
        return sorted(it, key=lambda e: e.name)


def _copy_tree(root_fd: int, dst: Path, report: CopyReport, limits: CopyLimits) -> None:
    """Iterative, fd-relative copy: never follows symlinks, never opens non-regular files."""
    dst.mkdir()
    stack: list[_Frame] = []
    opened: list[int] = []  # every fd we own, closed in ``finally``
    entries_seen = 0
    used = 0
    try:
        stack.append(_Frame(root_fd, _list_dir(root_fd), dst, ""))
        while stack:
            frame = stack[-1]
            if frame.pos >= len(frame.entries):
                stack.pop()
                if frame.fd != root_fd:
                    os.close(frame.fd)
                    opened.remove(frame.fd)
                continue
            entry = frame.entries[frame.pos]
            frame.pos += 1
            entries_seen += 1
            if entries_seen > limits.max_entries:
                raise UnsafeRepoError("too many entries")
            name = entry.name
            child = f"{frame.rel}/{name}" if frame.rel else name
            try:
                st = entry.stat(follow_symlinks=False)
            except OSError:
                report.special[child] = "unreadable"
                continue
            mode = st.st_mode
            if stat.S_ISLNK(mode):
                try:
                    report.symlinks[child] = os.readlink(name, dir_fd=frame.fd)
                except OSError:
                    report.symlinks[child] = "?"
            elif stat.S_ISDIR(mode):
                if len(stack) >= limits.max_depth:
                    raise UnsafeRepoError("directory nesting too deep")
                try:
                    cfd = _open_dir(name, frame.fd)
                except OSError:  # raced into a symlink / vanished: never followed
                    report.special[child] = "unreadable"
                    continue
                opened.append(cfd)
                if not stat.S_ISDIR(os.fstat(cfd).st_mode):  # pragma: no cover - defensive
                    os.close(cfd)
                    opened.remove(cfd)
                    report.special[child] = "unreadable"
                    continue
                sub = frame.dst / name
                try:
                    sub.mkdir()
                    listing = _list_dir(cfd)
                except OSError as exc:
                    raise UnsafeRepoError(f"cannot copy directory {child}") from exc
                stack.append(_Frame(cfd, listing, sub, child))
            elif stat.S_ISREG(mode):
                try:
                    ffd = _open_file(name, frame.fd)
                except OSError:
                    report.special[child] = "unreadable"
                    continue
                try:
                    if not stat.S_ISREG(os.fstat(ffd).st_mode):  # swapped after the lstat
                        report.special[child] = _kind(os.fstat(ffd).st_mode)
                        continue
                    used += _copy_file_fd(ffd, frame.dst / name, child, limits, used)
                finally:
                    os.close(ffd)
            else:
                report.special[child] = _kind(mode)
    finally:
        for fd in opened:
            with contextlib.suppress(OSError):
                os.close(fd)


def scratch_usage(
    root: Path, *, max_entries: int = 200_000, max_depth: int = 64
) -> tuple[int, bool]:
    """Disk usage of ``root`` (bytes) and whether the capped walk was truncated.

    Iterative and fd-relative; symlinks are counted as entries but never followed. A truncated
    walk (too many entries, too deep) means the tree is hostile: treat it as over any quota.
    """
    try:
        root_fd = _open_dir(root)
    except OSError:
        return 0, False
    total = 0
    seen = 0
    stack: list[tuple[int, int]] = [(root_fd, 0)]
    fd = root_fd
    try:
        while stack:
            fd, depth = stack.pop()
            try:
                with os.scandir(fd) as it:
                    entries = list(it)
            except OSError:
                entries = []
            for entry in entries:
                seen += 1
                if seen > max_entries:
                    return total, True
                try:
                    st = entry.stat(follow_symlinks=False)
                except OSError:
                    continue
                if not stat.S_ISDIR(st.st_mode):
                    total += max(st.st_size, st.st_blocks * 512)
                else:
                    if depth + 1 > max_depth:
                        return total, True
                    try:
                        cfd = _open_dir(entry.name, fd)
                    except OSError:
                        continue
                    stack.append((cfd, depth + 1))
            if fd != root_fd:
                os.close(fd)
                fd = root_fd
    finally:
        if fd != root_fd:
            with contextlib.suppress(OSError):
                os.close(fd)
        for pending, _ in stack:
            with contextlib.suppress(OSError):
                os.close(pending)
        with contextlib.suppress(OSError):
            os.close(root_fd)
    return total, False


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
    scratch: Path, pristine_git_config: bytes, limits: CopyLimits | None = None
) -> Generator[tuple[Path, CopyReport]]:
    """Like :func:`sanitized_copy` but also yields what was skipped/stripped.

    Raises :class:`UnsafeRepoError` on a symlinked root, a non-directory ``.git`` or any cap
    violation (bytes, entries, depth, single file size).
    """
    limits = limits or DEFAULT_LIMITS
    try:
        root_fd = _open_dir(scratch)
    except FileNotFoundError:
        root_fd = -1
    except OSError as exc:
        raise UnsafeRepoError("scratch root is a symlink or not a directory") from exc
    has_git = False
    report = CopyReport()
    try:
        if root_fd >= 0:
            try:
                st = os.stat(".git", dir_fd=root_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                if not stat.S_ISDIR(st.st_mode):
                    raise UnsafeRepoError(f".git is not a real directory ({_kind(st.st_mode)})")
                has_git = True
        with tempfile.TemporaryDirectory(prefix="mcp-chaos-snap-") as tmp:
            repo = Path(tmp) / "repo"
            if root_fd >= 0:
                _copy_tree(root_fd, repo, report, limits)
            else:
                repo.mkdir()
            if has_git:
                _sanitize_git_dir(repo / ".git", pristine_git_config)
            _strip_tree(repo, report)
            yield repo, report
    finally:
        if root_fd >= 0:
            os.close(root_fd)


@contextmanager
def sanitized_copy(scratch: Path, pristine_git_config: bytes) -> Generator[Path]:
    """Yield a sanitized copy of ``scratch`` that is safe to run git and readers against.

    Symlinks are skipped (never followed), ``.git`` must be a real directory, ``.git/config``
    is replaced with ``pristine_git_config`` and hooks, attributes, alternates,
    ``.gitattributes`` and ``.gitmodules`` are deleted from the copy.
    """
    with sanitized_copy_info(scratch, pristine_git_config) as (repo, _report):
        yield repo
