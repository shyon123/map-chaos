"""Claude Agent SDK target: supervisor main agent delegating to files/data/vcs subagents (ADR 0005).

How the topology is built (verified against the bundled Claude Code CLI, see tests):
- The supervisor is the *main-thread agent* (``--agent supervisor``), whose tool list is only
  ``Agent(files,data,vcs)``. A plain ``system_prompt`` main agent would see every MCP tool.
- MCP servers are registered globally (``fs_files``, ``sqlite_data``, ``git_vcs``), one per worker,
  each launched through its own fault proxy. They are connected once per run (inline per-subagent
  servers would be respawned on every delegation, resetting the proxy's per-process call counters).
  Each subagent lists only its own ``mcp__<entry>__<tool>`` names from the worker allowlist.
- Built-in tools: ``tools=["Agent"]`` makes the base tool set just the delegation tool (any other
  value of the base set is a built-in); ``disallowed_tools`` repeats the dangerous ones as defence
  in depth. ``permission_mode="dontAsk"`` never prompts; tools outside ``allowed_tools`` are denied.
- Error handling is the framework default: no retries are added on top of the CLI's own.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

import anyio
from claude_agent_sdk import (
    AgentDefinition,
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
)
from claude_agent_sdk.types import McpServerConfig
from pydantic import BaseModel, ConfigDict

from mcp_chaos.log import get_logger
from mcp_chaos.schemas.runs import RunConfig
from mcp_chaos.targets.workers import WORKERS, WorkerSpec, system_prompt

log = get_logger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
SUPERVISOR = "supervisor"
DELEGATE_TOOL = "Agent"
DESCRIPTIONS = {
    "files": "Reads, writes and edits files and directories in the scratch directory.",
    "data": "Queries and updates the SQLite database.",
    "vcs": "Inspects and commits to the version-control repository.",
}

# Every Claude Code built-in except the delegation tool (superset of what the 2.1.x CLI ships).
BUILTIN_TOOLS = (
    "Bash",
    "BashOutput",
    "KillShell",
    "Read",
    "Write",
    "Edit",
    "MultiEdit",
    "Glob",
    "Grep",
    "WebFetch",
    "WebSearch",
    "NotebookEdit",
    "TodoWrite",
    "Skill",
    "SlashCommand",
    "ToolSearch",
    "ListAgents",
    "ReportFindings",
    "ScheduleWakeup",
    "Workflow",
    "CronCreate",
    "CronDelete",
    "CronList",
    "ExitPlanMode",
    "EnterPlanMode",
    "AskUserQuestion",
)

# Switches that stop the CLI from phoning home, auto-updating or adding side features.
CLI_ENV: dict[str, str] = {
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "DISABLE_AUTOUPDATER": "1",
    "DISABLE_TELEMETRY": "1",
    "DISABLE_ERROR_REPORTING": "1",
    "DISABLE_BUG_COMMAND": "1",
    "DISABLE_COST_WARNINGS": "1",
    "DISABLE_GROWTHBOOK": "1",
    "DISABLE_INSTALLATION_CHECKS": "1",
    "CLAUDE_CODE_DISABLE_FEEDBACK_SURVEY": "1",
    "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
    "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
    "CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS": "1",
    "CLAUDE_CODE_DISABLE_FILE_CHECKPOINTING": "1",
    "CLAUDE_CODE_DISABLE_BUNDLED_SKILLS": "1",
    "CLAUDE_CODE_DISABLE_WORKFLOWS": "1",
    "CLAUDE_CODE_DISABLE_ADVISOR_TOOL": "1",
    "CLAUDE_CODE_DISABLE_CRON": "1",
    "CLAUDE_CODE_DISABLE_EXPLORE_PLAN_AGENTS": "1",
    "CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL": "1",
    "CLAUDE_CODE_IDE_SKIP_AUTO_INSTALL": "1",
    "CLAUDE_CODE_DISABLE_THINKING": "1",
    # Subagents must run in the foreground or the supervisor never sees their report.
    "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
    # Deferred tool search would hide MCP tools behind a placeholder.
    "ENABLE_TOOL_SEARCH": "false",
    "MCP_TIMEOUT": "60000",
}
# Credentials/providers that could leak in from the calling shell are blanked explicitly.
CLI_ENV_BLANK = (
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "ANTHROPIC_CUSTOM_HEADERS",
    "ANTHROPIC_UNIX_SOCKET",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
    "CLAUDE_CODE_USE_MANTLE",
    "CLAUDE_CODE_USE_GATEWAY",
)
_SERVER_ENV_KEYS = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "VIRTUAL_ENV")
_SERVER_ENV_PREFIXES = ("GIT_", "MCP_CHAOS_", "UV_", "PYTHON", "XDG_")


class AgentSdkOutcome(BaseModel):
    """Same shape and status semantics as ``LangGraphOutcome`` (plus the CLI-reported cost)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    final_answer: str | None
    turns: int
    error: str | None
    status: Literal["completed", "agent_error", "timeout"]
    cost_usd: float | None = None


def entry_name(worker: WorkerSpec) -> str:
    """MCP server entry for a worker: ``fs_files``, ``sqlite_data``, ``git_vcs``."""
    return f"{worker.server_id}_{worker.name}"


def tool_names(worker: WorkerSpec) -> list[str]:
    return [f"mcp__{entry_name(worker)}__{t}" for t in sorted(worker.allow)]


def _server_env(env: Mapping[str, str]) -> dict[str, str]:
    return {
        k: v for k, v in env.items() if k in _SERVER_ENV_KEYS or k.startswith(_SERVER_ENV_PREFIXES)
    }


def _cli_env(cfg: RunConfig, env: Mapping[str, str], state_dir: Path) -> dict[str, str]:
    model = cfg.agent_model.model
    out: dict[str, str] = {k: "" for k in CLI_ENV_BLANK}
    out.update(CLI_ENV)
    out.update(
        {
            "ANTHROPIC_BASE_URL": env["ANTHROPIC_BASE_URL"],
            "ANTHROPIC_API_KEY": env["ANTHROPIC_API_KEY"],
            # The CLI keeps its state (config, session transcripts) here; rootfs may be read-only.
            "CLAUDE_CONFIG_DIR": str(state_dir / "config"),
            "CLAUDE_CODE_TMPDIR": str(state_dir / "tmp"),
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(cfg.agent_model.max_tokens),
            # Any internal side request stays on the configured model (one price, one cassette).
            "ANTHROPIC_SMALL_FAST_MODEL": model,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
        }
    )
    return out


def build_options(
    cfg: RunConfig,
    cmds: Mapping[str, list[str]],
    env: Mapping[str, str],
    state_dir: Path,
) -> ClaudeAgentOptions:
    scratch = env.get("MCP_CHAOS_SCRATCH", "/scratch")
    model = cfg.agent_model.model
    servers: dict[str, McpServerConfig] = {}
    agents: dict[str, AgentDefinition] = {}
    allowed: list[str] = [DELEGATE_TOOL]
    for name, spec in WORKERS.items():
        cmd = cmds.get(name)
        if cmd is None:
            continue
        servers[entry_name(spec)] = {
            "type": "stdio",
            "command": cmd[0],
            "args": list(cmd[1:]),
            "env": _server_env(env),
        }
        tools = tool_names(spec)
        allowed.extend(tools)
        agents[name] = AgentDefinition(
            description=DESCRIPTIONS[name],
            prompt=system_prompt(name, cfg.hardening, scratch=scratch),
            tools=tools,
            model=model,
            maxTurns=cfg.max_turns,
        )
    agents[SUPERVISOR] = AgentDefinition(
        description="Coordinates the workers to complete the task.",
        prompt=system_prompt("supervisor_agentsdk", cfg.hardening, scratch=scratch),
        tools=[f"{DELEGATE_TOOL}({','.join(n for n in WORKERS if n in cmds)})"],
        model=model,
    )
    return ClaudeAgentOptions(
        tools=[DELEGATE_TOOL],
        allowed_tools=allowed,
        disallowed_tools=list(BUILTIN_TOOLS),
        permission_mode="dontAsk",
        system_prompt=None,
        agents=agents,
        mcp_servers=servers,
        strict_mcp_config=True,
        setting_sources=[],
        model=model,
        thinking={"type": "disabled"},
        max_turns=cfg.max_turns,
        cwd=state_dir / "cwd",
        env=_cli_env(cfg, env, state_dir),
        extra_args={"agent": SUPERVISOR},
        forward_subagent_text=True,
    )


def _make_state_dir(env: Mapping[str, str]) -> Path:
    base = env.get("TMPDIR") or tempfile.gettempdir()
    state = Path(tempfile.mkdtemp(prefix="mcp-chaos-agentsdk-", dir=base))
    for sub in ("config", "tmp", "cwd"):
        (state / sub).mkdir()
    return state


def _first_leaf(exc: BaseException) -> BaseException:
    cur: BaseException = exc
    while True:
        nested: Any = getattr(cur, "exceptions", None)
        if not isinstance(cur, BaseExceptionGroup) or not nested:
            return cast("BaseException", cur)
        cur = nested[0]


def short_error(exc: BaseException) -> str:
    leaf = _first_leaf(exc)
    msg = " ".join(str(leaf).split())[:300]
    return f"{type(leaf).__name__}: {msg}" if msg else type(leaf).__name__


def _result_error(res: ResultMessage) -> str:
    detail = (res.errors[0] if res.errors else res.result) or ""
    detail = " ".join(detail.split())[:300]
    return f"{res.subtype}: {detail}" if detail else res.subtype


class _Progress:
    """What the stream told us before it ended (survives a timeout or crash)."""

    def __init__(self) -> None:
        self.message_ids: set[str] = set()
        self.anonymous = 0
        self.result: ResultMessage | None = None

    @property
    def turns(self) -> int:
        """Model responses across the supervisor and all subagents."""
        if self.result is not None and not self.message_ids and not self.anonymous:
            return self.result.num_turns
        return len(self.message_ids) + self.anonymous

    def see(self, msg: object) -> None:
        if isinstance(msg, AssistantMessage):
            if msg.message_id:
                self.message_ids.add(msg.message_id)
            else:
                self.anonymous += 1
        elif isinstance(msg, ResultMessage):
            self.result = msg


async def _converse(options: ClaudeAgentOptions, task_prompt: str, progress: _Progress) -> None:
    async with ClaudeSDKClient(options) as client:
        await client.query(task_prompt)
        async for msg in client.receive_response():
            progress.see(msg)


async def run_agentsdk(
    task_prompt: str,
    cfg: RunConfig,
    cmds: Mapping[str, list[str]],
    *,
    env: Mapping[str, str],
) -> AgentSdkOutcome:
    state = _make_state_dir(env)
    progress = _Progress()
    try:
        options = build_options(cfg, cmds, env, state)
        with anyio.fail_after(cfg.timeout_s):
            await _converse(options, task_prompt, progress)
    except TimeoutError:
        log.warning("agentsdk_timeout", run_id=cfg.run_id)
        return AgentSdkOutcome(
            final_answer=None, turns=progress.turns, error="timeout", status="timeout"
        )
    except Exception as e:
        err = short_error(e)
        log.warning("agentsdk_error", run_id=cfg.run_id, error=err)
        return AgentSdkOutcome(
            final_answer=None, turns=progress.turns, error=err, status="agent_error"
        )
    finally:
        shutil.rmtree(state, ignore_errors=True)
    res = progress.result
    if res is None:
        return AgentSdkOutcome(
            final_answer=None,
            turns=progress.turns,
            error="no result message from the Claude Code CLI",
            status="agent_error",
        )
    log.info("agentsdk_result", run_id=cfg.run_id, subtype=res.subtype, cost_usd=res.total_cost_usd)
    if res.is_error or res.subtype != "success":
        return AgentSdkOutcome(
            final_answer=None,
            turns=progress.turns,
            error=_result_error(res),
            status="agent_error",
            cost_usd=res.total_cost_usd,
        )
    return AgentSdkOutcome(
        final_answer=res.result or None,
        turns=progress.turns,
        error=None,
        status="completed",
        cost_usd=res.total_cost_usd,
    )


async def agentsdk_self_test(
    cfg: RunConfig, cmds: Mapping[str, list[str]], env: Mapping[str, str], *, wait_s: float = 60
) -> dict[str, Any]:
    """Start the bundled CLI and every proxy-wrapped MCP server; no model call is made.

    Returns ``{"workers": {name: {"status", "missing"}}, "ok": bool}``.
    """
    state = _make_state_dir(env)
    try:
        options = build_options(cfg, cmds, env, state)
        statuses: dict[str, Any] = {}
        async with ClaudeSDKClient(options) as client:
            with anyio.move_on_after(wait_s):
                while True:
                    resp = await client.get_mcp_status()
                    statuses = {s["name"]: dict(s) for s in resp["mcpServers"]}
                    if all(s["status"] != "pending" for s in statuses.values()) and statuses:
                        break
                    await anyio.sleep(0.2)
    finally:
        shutil.rmtree(state, ignore_errors=True)
    report: dict[str, Any] = {"workers": {}, "ok": True}
    for name in cmds:
        spec = WORKERS[cast("Any", name)]
        st = statuses.get(entry_name(spec), {"status": "absent"})
        have = {t["name"] for t in st.get("tools", [])}
        missing = sorted(spec.allow - have)
        entry: dict[str, Any] = {"status": st["status"], "missing": missing}
        if "error" in st:
            entry["error"] = str(st["error"])[:200]
        report["workers"][name] = entry
        report["ok"] = report["ok"] and st["status"] == "connected" and not missing
    return report
