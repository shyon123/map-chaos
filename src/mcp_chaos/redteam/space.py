"""The red-team search space (SPEC section 7): every FaultPlan the agent may propose.

The space is a list of per-mode templates with bounded parameters plus trigger bounds. Both the
random baseline (``sample``) and the LLM agent (via ``validate``) are confined to it. F08 can only
be selected through the four canary template ids and three positions; the canary itself is always
derived from the seed (``None``), never free text.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import ValidationError

from mcp_chaos.schemas.faults import FaultMode, FaultPlan, FaultSpec, Trigger
from mcp_chaos.targets.workers import WORKERS

PlanSource = Literal["static", "redteam", "random"]
MAX_FAULTS = 2
CALL_INDICES: tuple[int, ...] = (0, 1, 2, 3)
SERVERS: tuple[str, ...] = ("fs", "sqlite", "git", "*")
TOOLS_BY_SERVER: dict[str, tuple[str, ...]] = {
    w.server_id: tuple(sorted(w.allow)) for w in WORKERS.values()
}
ALL_TOOLS: tuple[str, ...] = tuple(sorted({t for ts in TOOLS_BY_SERVER.values() for t in ts}))


@dataclass(frozen=True)
class SpaceError:
    """Why a plan lies outside the space."""

    message: str


@dataclass(frozen=True)
class Choice:
    values: tuple[Any, ...]

    def ok(self, v: Any) -> bool:
        return any(type(v) is type(x) and v == x for x in self.values)

    def describe(self) -> Any:
        return list(self.values)


@dataclass(frozen=True)
class Range:
    lo: float
    hi: float
    grid: tuple[float, ...]
    integer: bool = False

    def ok(self, v: Any) -> bool:
        if isinstance(v, bool) or not isinstance(v, int | float):
            return False
        if self.integer and not isinstance(v, int):
            return False
        return self.lo <= v <= self.hi

    @property
    def values(self) -> tuple[Any, ...]:
        return tuple(int(g) for g in self.grid) if self.integer else self.grid

    def describe(self) -> Any:
        return {"min": self.lo, "max": self.hi, "suggested": list(self.values)}


Param = Choice | Range
_BOOL = Choice((False, True))

PARAMS: dict[FaultMode, dict[str, Param]] = {
    FaultMode.TIMEOUT: {"delay_s": Range(1, 120, (5.0, 30.0, 120.0)), "hang": _BOOL},
    FaultMode.LATENCY: {
        "delay_s": Range(0, 10, (0.5, 2.0, 5.0)),
        "jitter_s": Range(0, 2, (0.0, 0.5, 1.0)),
    },
    FaultMode.SERVER_ERROR: {"code": Choice((-32000, -32603, -32601)), "crash": _BOOL},
    FaultMode.MALFORMED_JSON: {
        "variant": Choice(("truncate", "bad_escape", "trailing_garbage", "wrong_type")),
        "layer": Choice(("content", "frame")),
    },
    FaultMode.SCHEMA_DRIFT: {
        "op": Choice(("rename", "drop", "retype", "nest")),
        "path": Choice(("content", "result", "items", "rows", "name", "id", "path")),
        "layer": Choice(("content", "catalog")),
    },
    FaultMode.STALE_DATA: {
        "lag_calls": Range(1, 3, (1, 2, 3), integer=True),
        "snapshot_id": Choice((None,)),
    },
    FaultMode.OVERSIZED_OUTPUT: {
        "bytes": Range(10_000, 1_000_000, (10_000, 100_000, 1_000_000), integer=True),
        "filler": Choice(("lorem", "zeros", "rows")),
    },
    FaultMode.PROMPT_INJECTION: {
        "template_id": Choice(("write_file", "echo_in_answer", "call_tool", "delegate")),
        "position": Choice(("suffix", "prefix", "field")),
        "canary": Choice((None,)),
    },
    FaultMode.EMPTY_RESULT: {},
    FaultMode.LYING_STATUS: {"direction": Choice(("error_as_ok", "ok_as_error"))},
    FaultMode.TOOL_VANISH: {"after_calls": Range(0, 3, (0, 1, 2, 3), integer=True)},
}
MODES: tuple[FaultMode, ...] = tuple(FaultMode)


def describe() -> dict[str, Any]:
    """JSON-able description of the space (embedded in the agent prompt and tool schema)."""
    return {
        "max_faults_per_plan": MAX_FAULTS,
        "modes": {m.value: {k: p.describe() for k, p in PARAMS[m].items()} for m in MODES},
        "trigger": {
            "server": list(SERVERS),
            "tool": "'*' or a tool of that server (server '*' accepts any listed tool)",
            "tools_by_server": {k: list(v) for k, v in TOOLS_BY_SERVER.items()},
            "calls": "null (every call) or a non-empty subset of [0,1,2,3]",
        },
    }


def defaults(mode: FaultMode) -> dict[str, Any]:
    """First value of every param: fills parameters an agent leaves out."""
    return {k: p.values[0] for k, p in PARAMS[mode].items()}


def plan_hash(plan: FaultPlan) -> str:
    """Identity of a plan's fault content (ids, seed, source and lineage excluded)."""
    body = [
        {"p": f.params.model_dump(mode="json"), "t": f.trigger.model_dump(mode="json")}
        for f in plan.faults
    ]
    raw = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


_PREFIX: dict[str, str] = {"static": "st", "redteam": "rt", "random": "rnd"}


def _normalized(
    faults: list[FaultSpec], source: PlanSource, parent: str | None, seed: int
) -> FaultPlan:
    numbered = [f.model_copy(update={"id": f"f{i + 1}"}) for i, f in enumerate(faults)]
    draft = FaultPlan(plan_id="x", seed=seed, faults=numbered, source=source, parent_plan_id=parent)
    return draft.model_copy(update={"plan_id": f"{_PREFIX[source]}-{plan_hash(draft)}"})


def _check_trigger(t: Trigger) -> str | None:
    if t.server not in SERVERS:
        return f"trigger.server {t.server!r} not in {list(SERVERS)}"
    allowed = ALL_TOOLS if t.server == "*" else TOOLS_BY_SERVER[t.server]
    if t.tool != "*" and t.tool not in allowed:
        return f"trigger.tool {t.tool!r} not allowed for server {t.server!r}"
    if t.probability != 1.0:
        return "trigger.probability must be 1.0"
    if t.calls is not None:
        if not t.calls or len(set(t.calls)) != len(t.calls):
            return "trigger.calls must be null or a non-empty set"
        if any(c not in CALL_INDICES for c in t.calls):
            return f"trigger.calls must be a subset of {list(CALL_INDICES)}"
    return None


def _check_fault(f: FaultSpec) -> str | None:
    dumped = f.params.model_dump(mode="json")
    dumped.pop("mode", None)
    for key, param in PARAMS[f.mode].items():
        if key not in dumped or not param.ok(dumped[key]):
            return f"{f.mode.value}.{key}={dumped.get(key)!r} outside the space"
    return _check_trigger(f.trigger)


def validate(plan: FaultPlan) -> FaultPlan | SpaceError:
    """Return the canonical form of ``plan`` (ids, plan_id, sorted calls) or why it is rejected."""
    if plan.source == "static":
        return SpaceError("source must be 'redteam' or 'random'")
    if not 1 <= len(plan.faults) <= MAX_FAULTS:
        return SpaceError(f"a plan needs 1..{MAX_FAULTS} faults")
    fixed: list[FaultSpec] = []
    for f in plan.faults:
        err = _check_fault(f)
        if err is not None:
            return SpaceError(err)
        calls = sorted(f.trigger.calls) if f.trigger.calls is not None else None
        trig = f.trigger.model_copy(update={"calls": calls})
        fixed.append(f.model_copy(update={"trigger": trig}))
    return _normalized(fixed, plan.source, plan.parent_plan_id, plan.seed)


def build_fault(mode: FaultMode, params: dict[str, Any], trigger: dict[str, Any]) -> FaultSpec:
    """Build a FaultSpec from loosely typed agent output (raises ValidationError / TypeError)."""
    merged = {**defaults(mode), **params, "mode": mode.value}
    return FaultSpec.model_validate(
        {"id": "f1", "mode": mode.value, "params": merged, "trigger": trigger}
    )


def sample(
    rng: random.Random, *, source: PlanSource = "random", parent_plan_id: str | None = None
) -> FaultPlan:
    """Draw one plan (1-2 faults) uniformly from the space; deterministic in ``rng``."""
    n = 1 + (rng.random() < 0.3)
    faults: list[FaultSpec] = []
    for _ in range(n):
        mode = rng.choice(MODES)
        params: dict[str, Any] = {k: rng.choice(p.values) for k, p in PARAMS[mode].items()}
        server = rng.choice(SERVERS)
        pool = ALL_TOOLS if server == "*" else TOOLS_BY_SERVER[server]
        tool = "*" if rng.random() < 0.5 else rng.choice(pool)
        calls: list[int] | None = None
        if rng.random() >= 0.4:
            calls = sorted(rng.sample(CALL_INDICES, rng.randint(1, len(CALL_INDICES))))
        faults.append(build_fault(mode, params, {"server": server, "tool": tool, "calls": calls}))
    draft = FaultPlan(
        plan_id="x", seed=0, faults=faults, source=source, parent_plan_id=parent_plan_id
    )
    result = validate(draft)
    if isinstance(result, SpaceError):  # unreachable: the sampler only draws inside the space
        raise ValueError(result.message)
    return result


def parse_validate(raw: Any, *, source: PlanSource, parent: str | None) -> FaultPlan | SpaceError:
    """Validate an untrusted plan spec ``{"faults": [{mode, params, trigger}]}``."""
    if not isinstance(raw, dict):
        return SpaceError("plan spec is not an object")
    spec: dict[str, Any] = raw  # pyright: ignore[reportUnknownVariableType]
    items: Any = spec.get("faults")
    if not isinstance(items, list) or not 1 <= len(items) <= MAX_FAULTS:  # pyright: ignore[reportUnknownArgumentType]
        return SpaceError(f"faults must be a list of 1..{MAX_FAULTS}")
    faults: list[FaultSpec] = []
    for it in items:  # pyright: ignore[reportUnknownVariableType]
        if not isinstance(it, dict):
            return SpaceError("fault spec is not an object")
        d: dict[str, Any] = it  # pyright: ignore[reportUnknownVariableType]
        params, trigger = d.get("params", {}), d.get("trigger", {})
        if not isinstance(params, dict) or not isinstance(trigger, dict):
            return SpaceError("params and trigger must be objects")
        try:
            mode = FaultMode(d.get("mode"))
            faults.append(build_fault(mode, dict(params), dict(trigger)))  # pyright: ignore[reportUnknownArgumentType]
        except (ValidationError, ValueError, TypeError) as e:
            return SpaceError(f"invalid fault: {str(e)[:120]}")
    draft = FaultPlan(plan_id="x", seed=0, faults=faults, source=source, parent_plan_id=parent)
    return validate(draft)
