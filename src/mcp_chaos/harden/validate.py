"""Response validation: JSON-RPC shape, CallToolResult shape, ``outputSchema``, embedded JSON.

Pure functions. Each returns a list of human-readable problems; empty means valid.
"""

from __future__ import annotations

import json
from typing import Any, cast

import jsonschema
import jsonschema.validators

_MAX_PROBLEMS = 5
# After an opening ``[``, JSON continues with one of these; this keeps prose such as
# ``[main abc123] msg`` or an INI ``[section]`` header from being mistaken for truncated JSON.
_ARRAY_FOLLOW = set('{["-0123456789tfn]')


def looks_like_json(text: str) -> bool:
    """True for text that claims to be a JSON object/array (starts with ``{`` or ``[``)."""
    s = text.lstrip()
    if s.startswith("{"):
        return True
    if not s.startswith("["):
        return False
    rest = s[1:].lstrip()
    return bool(rest) and rest[0] in _ARRAY_FOLLOW


def is_incomplete(result: dict[str, Any]) -> bool:
    """A non-final result (e.g. ``resultType: "input_required"``) has no CallToolResult body."""
    rt: Any = result.get("resultType")
    return isinstance(rt, str) and rt != "complete"


def check_jsonrpc_response(obj: Any) -> list[str]:
    if not isinstance(obj, dict):
        return ["response is not a JSON object"]
    o = cast("dict[str, Any]", obj)
    problems: list[str] = []
    if o.get("jsonrpc") != "2.0":
        problems.append('missing or wrong "jsonrpc" (expected "2.0")')
    if "id" not in o:
        problems.append('missing "id"')
    has_result, has_error = "result" in o, "error" in o
    if has_result == has_error:
        problems.append('response must contain exactly one of "result" or "error"')
    if has_error:
        err: Any = o["error"]
        if not isinstance(err, dict):
            problems.append('"error" is not an object')
        else:
            e = cast("dict[str, Any]", err)
            code: Any = e.get("code")
            if not isinstance(code, int) or isinstance(code, bool):
                problems.append('"error.code" is not an integer')
            if not isinstance(e.get("message"), str):
                problems.append('"error.message" is not a string')
    return problems


def check_call_tool_result(result: Any) -> list[str]:
    if not isinstance(result, dict):
        return ["result is not an object"]
    r = cast("dict[str, Any]", result)
    if is_incomplete(r):
        return []
    problems: list[str] = []
    content: Any = r.get("content")
    if not isinstance(content, list):
        problems.append('"content" is missing or not a list')
    else:
        for i, block in enumerate(cast("list[Any]", content)):
            if not isinstance(block, dict):
                problems.append(f"content[{i}] is not an object")
                continue
            b = cast("dict[str, Any]", block)
            if not isinstance(b.get("type"), str):
                problems.append(f'content[{i}] has no string "type"')
            elif b["type"] == "text":
                text: Any = b.get("text")
                if not isinstance(text, str):
                    problems.append(f'content[{i}] text block has no string "text"')
                elif looks_like_json(text):
                    try:
                        json.loads(text)
                    except (ValueError, RecursionError):
                        problems.append(f"content[{i}] text looks like JSON but does not parse")
    is_error: Any = r.get("isError")
    if is_error is not None and not isinstance(is_error, bool):
        problems.append('"isError" is not a boolean')
    sc: Any = r.get("structuredContent")
    if sc is not None and not isinstance(sc, dict):
        problems.append('"structuredContent" is not an object')
    return problems


def check_output_schema(result: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    """Validate ``structuredContent`` against a declared ``outputSchema``."""
    if result.get("isError") is True:
        return []
    sc: Any = result.get("structuredContent")
    if sc is None:
        return ["tool declares an outputSchema but returned no structuredContent"]
    try:
        cls: Any = jsonschema.validators.validator_for(schema)
        validator: Any = cls(schema)
        errors: list[Any] = sorted(
            validator.iter_errors(sc), key=lambda e: [str(p) for p in e.path]
        )
    except Exception:
        return []  # a schema that cannot be enforced must not be blamed on the result
    out: list[str] = []
    for e in errors[:_MAX_PROBLEMS]:
        where = "/".join(str(p) for p in e.path) or "<root>"
        out.append(f"structuredContent at {where}: {str(e.message)[:200]}")
    return out


def validate_tool_response(obj: Any, output_schema: dict[str, Any] | None) -> list[str]:
    """Full check of a ``tools/call`` response. JSON-RPC errors are shape-checked only."""
    problems = check_jsonrpc_response(obj)
    if problems or not isinstance(obj, dict):
        return problems
    o = cast("dict[str, Any]", obj)
    if "result" not in o:
        return []
    if isinstance(o["result"], dict) and is_incomplete(cast("dict[str, Any]", o["result"])):
        return []
    problems = check_call_tool_result(o["result"])
    if problems:
        return problems
    if output_schema is not None:
        return check_output_schema(cast("dict[str, Any]", o["result"]), output_schema)
    return []
