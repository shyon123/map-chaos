"""F08 prompt_injection: embed a benign canary instruction in a successful tool result."""

from __future__ import annotations

import copy
import json
from typing import Any, cast

from mcp_chaos.proxy.canaries import derive_canary, render_template
from mcp_chaos.proxy.faults.base import REGISTRY, FaultContext, FaultOutcome
from mcp_chaos.schemas import FaultApplied, FaultMode, PromptInjectionParams

Path = list[str | int]


def _first_string_leaf(node: Any, path: Path) -> Path | None:
    """Deterministic DFS (dict keys sorted); returns the path to the first str leaf."""
    if isinstance(node, str):
        return path
    if isinstance(node, dict):
        d = cast("dict[str, Any]", node)
        for key in sorted(d):
            found = _first_string_leaf(d[key], [*path, key])
            if found is not None:
                return found
    elif isinstance(node, list):
        items = cast("list[Any]", node)
        for i, item in enumerate(items):
            found = _first_string_leaf(item, [*path, i])
            if found is not None:
                return found
    return None


def _append_at(root: Any, path: Path, suffix: str) -> None:
    node = root
    for step in path[:-1]:
        node = node[step]
    node[path[-1]] = node[path[-1]] + suffix


def _compact(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class PromptInjection:
    def apply(self, ctx: FaultContext, response: dict[str, Any]) -> FaultOutcome:
        result: Any = response.get("result")
        if "error" in response or not isinstance(result, dict):
            return FaultOutcome()
        if cast("dict[str, Any]", result).get("isError") is True:
            return FaultOutcome()
        params = ctx.spec.params
        if not isinstance(params, PromptInjectionParams):
            return FaultOutcome()
        canary = params.canary or derive_canary(ctx.rng)
        payload = render_template(params.template_id, canary)
        block: dict[str, Any] = {"type": "text", "text": payload}

        new = copy.deepcopy(response)
        new_result: dict[str, Any] = new["result"]
        raw_content = new_result.get("content")
        content: list[Any] = cast("list[Any]", raw_content) if isinstance(raw_content, list) else []
        new_result["content"] = content

        position = params.position
        fallback = False
        if position == "field":
            sc = new_result.get("structuredContent")
            path = _first_string_leaf(sc, []) if sc is not None else None
            if path is None:
                position, fallback = "suffix", True
            else:
                original_sc = copy.deepcopy(sc)
                _append_at(new_result, ["structuredContent", *path], " " + payload)
                for i, b in enumerate(content):
                    if not isinstance(b, dict):
                        continue
                    blk = cast("dict[str, Any]", b)
                    text = blk.get("text")
                    if blk.get("type") == "text" and isinstance(text, str):
                        try:
                            parsed = json.loads(text)
                        except ValueError:
                            continue
                        if parsed == original_sc:
                            content[i] = {**blk, "text": _compact(new_result["structuredContent"])}
                            break
        if position == "prefix":
            content.insert(0, block)
        elif position == "suffix":
            content.append(block)

        applied = FaultApplied(
            fault_id=ctx.spec.id,
            mode=FaultMode.PROMPT_INJECTION,
            detail={
                "template_id": params.template_id,
                "position": position,
                "canary": canary,
                "fallback": fallback,
            },
        )
        return FaultOutcome(response_obj=new, applied=applied)


REGISTRY[FaultMode.PROMPT_INJECTION] = PromptInjection()
