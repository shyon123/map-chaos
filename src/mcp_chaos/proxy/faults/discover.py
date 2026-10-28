"""Import every fault module in this package so each one registers itself in REGISTRY."""

from __future__ import annotations

import importlib
import pkgutil

import mcp_chaos.proxy.faults as _pkg

_NON_FAULT = frozenset({"base", "discover"})


def load_all() -> None:
    for mod in pkgutil.iter_modules(_pkg.__path__):
        if mod.name not in _NON_FAULT and not mod.name.startswith("_"):
            importlib.import_module(f"{_pkg.__name__}.{mod.name}")
