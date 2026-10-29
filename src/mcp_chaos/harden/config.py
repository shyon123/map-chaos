"""Guard runtime configuration: the ablation toggles plus timing parameters."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from mcp_chaos.schemas.runs import HardeningConfig

DEFAULT_DEADLINE_S = 20.0

# Read-only tools of the worker allowlists (targets/workers.py); only these are safe to re-issue,
# because a timeout / late reply / malformed reply happens AFTER the upstream may have executed.
READ_ONLY_TOOLS: frozenset[str] = frozenset(
    {
        "read_file",
        "read_text_file",
        "read_media_file",
        "read_multiple_files",
        "list_directory",
        "list_directory_with_sizes",
        "directory_tree",
        "search_files",
        "get_file_info",
        "list_allowed_directories",
        "list_tables",
        "describe_table",
        "read_query",
        "git_status",
        "git_log",
        "git_diff",
        "git_diff_staged",
        "git_diff_unstaged",
        "git_show",
    }
)


class GuardConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    hardening: HardeningConfig = Field(default_factory=HardeningConfig)
    deadline_s: float = Field(default=DEFAULT_DEADLINE_S, gt=0)
    backoff_base_s: float = Field(default=0.25, ge=0)
    backoff_max_s: float = Field(default=4.0, ge=0)
    breaker_threshold: int = Field(default=3, ge=1)
    breaker_cooldown_s: float = Field(default=30.0, ge=0)
    seed: int = 0
    retry_tools: frozenset[str] = READ_ONLY_TOOLS  # others are never retried
