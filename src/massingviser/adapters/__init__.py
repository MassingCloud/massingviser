"""``massingviser.adapters`` -- optional integrations, each behind a capability token.

This is the only layer allowed to depend on heavy third-party libraries, and every module in it is
**optional**. Importing this package never fails: it probes what is installed and reports it. A
deployment with none of the extras runs exactly as it did before, minus the capabilities those
extras provide.

That is the whole argument for the token architecture. IfcOpenShell is a C++ geometry kernel and
pyproj carries the PROJ database; neither belongs anywhere near a kernel that has to stay small and
stable. Behind a token they are swappable, absent-able, and invisible to the fifteen capability
families that consume what they produce.

    from massingviser.adapters import available, missing
    available()   # ("crs", "ifc", "solids")
    missing()     # {"solids": "No module named 'manifold3d'"}
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

#: Adapter name -> the extras that have to be installed for it.
REQUIREMENTS: Mapping[str, tuple[str, ...]] = {
    "ifc": ("ifcopenshell", "numpy"),
    "ifc_write": ("ifcopenshell", "numpy"),
    "solids": ("trimesh", "manifold3d", "numpy"),
    "crs": ("pyproj", "numpy"),
}

_FAILURES: dict[str, str] = {}


def _probe(name: str) -> bool:
    for requirement in REQUIREMENTS[name]:
        try:
            importlib.import_module(requirement)
        except Exception as thrown:  # noqa: BLE001
            _FAILURES[name] = str(thrown)
            return False
    _FAILURES.pop(name, None)
    return True


def available() -> tuple[str, ...]:
    """Adapters whose dependencies are all installed."""
    return tuple(name for name in sorted(REQUIREMENTS) if _probe(name))


def missing() -> dict[str, str]:
    """Adapters that cannot load, and why. Reported rather than raised."""
    for name in REQUIREMENTS:
        _probe(name)
    return dict(_FAILURES)


def load(name: str) -> Any:
    """Import an adapter module, with a message that names the missing extra.

    ``ModuleNotFoundError: manifold3d`` from four frames deep tells a user nothing about what to
    install or why they wanted it.
    """
    if name not in REQUIREMENTS:
        raise KeyError(f'No adapter "{name}". Available: {", ".join(sorted(REQUIREMENTS))}.')
    if not _probe(name):
        raise ImportError(
            f'The "{name}" adapter needs {", ".join(REQUIREMENTS[name])} '
            f'(pip install "massingviser[{name}]"): {_FAILURES.get(name, "unavailable")}'
        )
    return importlib.import_module(f"{__name__}.{name}")


__all__ = ["REQUIREMENTS", "available", "load", "missing"]
