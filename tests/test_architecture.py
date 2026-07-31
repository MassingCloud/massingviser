"""Architecture invariants, enforced rather than asserted in prose.

``massingifc`` ships a ``check:architecture`` script for exactly this reason: the claims on its
README -- only the adapter carries third-party dependencies, the kernel has none, no package
outside the adapter imports ``three`` -- cannot be held by documentation. They age badly and
silently. A check fails a build.

These are the same claims, restated for this port and checked by parsing the imports.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent / "src" / "massingviser"

#: Everything the standard library gives us. Anything else is a third-party runtime dependency.
STDLIB = set(sys.stdlib_module_names)

#: The one package allowed to render.
VIEWER = "viewer"

#: Third-party packages the viewer may use.
VIEWER_ALLOWED = {"viser", "numpy"}

#: Server-side compute. Allowed numpy and nothing else -- vectorising a BVH is the whole point of
#: the layer, and keeping numpy out of the kernel, schema, SDK and plugins is what keeps those
#: runnable anywhere.
COMPUTE = "geometry"
COMPUTE_ALLOWED = {"numpy"}

#: Optional integrations. The only layer allowed heavy third-party libraries, and every module in
#: it is optional -- importing the package probes what is installed rather than failing.
ADAPTERS = "adapters"
ADAPTERS_ALLOWED = {"numpy", "ifcopenshell", "trimesh", "manifold3d", "pyproj"}


def _modules() -> list[Path]:
    return sorted(ROOT.rglob("*.py"))


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _imports(path: Path) -> set[str]:
    """Top-level package name of every absolute import in a module."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:  # relative imports are internal
                found.add(node.module.split(".")[0])
    return found


def _third_party(path: Path) -> set[str]:
    return {name for name in _imports(path) if name not in STDLIB and name != "massingviser"}


def _internal_imports(path: Path) -> set[str]:
    """Which top-level ``massingviser`` subpackage each import resolves to.

    Parsed rather than grepped: an earlier version of this check matched the string
    ``massingviser.schema`` inside a docstring and failed a module that imports nothing of the
    kind. A test that cries wolf gets deleted, which costs more than having no test.
    """
    parts = path.relative_to(ROOT).parts
    package = list(parts[:-1]) if path.name != "__init__.py" else list(parts[:-1])

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    resolved: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module and node.module.split(".")[0] == "massingviser":
                    tail = node.module.split(".")[1:]
                    if tail:
                        resolved.add(tail[0])
                continue
            base = package[: len(package) - (node.level - 1)]
            target = base + (node.module.split(".") if node.module else [])
            if target:
                resolved.add(target[0])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                bits = alias.name.split(".")
                if bits[0] == "massingviser" and len(bits) > 1:
                    resolved.add(bits[1])
    return resolved


def test_the_kernel_has_no_runtime_dependencies():
    """Not even numpy.

    The kernel is the part that has to stay stable across releases. A dependency here is a
    dependency every host inherits forever.
    """
    offenders = {
        _relative(path): sorted(_third_party(path))
        for path in (ROOT / "kernel").rglob("*.py")
        if _third_party(path)
    }
    assert offenders == {}


@pytest.mark.parametrize("package", ["kernel", "schema", "sdk", "plugins", "storage", "vcs"])
def test_nothing_below_the_viewer_imports_a_rendering_library(package):
    """The rule that lets the same plugin run headless, in a browser, or behind an exporter."""
    offenders = {
        _relative(path): sorted(_third_party(path) & VIEWER_ALLOWED)
        for path in (ROOT / package).rglob("*.py")
        if _third_party(path) & VIEWER_ALLOWED
    }
    assert offenders == {}


def test_third_party_dependencies_stay_in_their_two_layers():
    """Only the renderer and the compute layer may reach outside the standard library."""
    offenders: dict[str, list[str]] = {}
    for path in _modules():
        relative = _relative(path)
        top = relative.split("/")[0]
        allowed = {
            VIEWER: VIEWER_ALLOWED,
            COMPUTE: COMPUTE_ALLOWED,
            ADAPTERS: ADAPTERS_ALLOWED,
        }.get(top, set())
        extra = _third_party(path) - allowed
        if extra:
            offenders[relative] = sorted(extra)
    assert offenders == {}


def test_version_control_has_no_dependencies_at_all():
    """A project's history must be readable by anything that can read JSON."""
    offenders = {
        _relative(path): sorted(_third_party(path))
        for path in (ROOT / "vcs").rglob("*.py")
        if _third_party(path)
    }
    assert offenders == {}


def test_only_the_composition_root_reaches_for_an_optional_adapter():
    """`adapters` carries the heavy extras, so a deployment without them must still import."""
    offenders: dict[str, list[str]] = {}
    for package in ("kernel", "schema", "sdk", "plugins", "storage", "vcs", "geometry", "viewer"):
        for path in (ROOT / package).rglob("*.py"):
            if "adapters" in _internal_imports(path):
                offenders[_relative(path)] = ["adapters"]
    assert offenders == {}


def test_every_optional_adapter_declares_what_it_needs():
    """A missing extra must produce a message naming it, not a ModuleNotFoundError four frames deep."""
    from massingviser.adapters import REQUIREMENTS

    modules = {
        path.stem
        for path in (ROOT / "adapters").glob("*.py")
        if path.stem not in ("__init__", "plugin")
    }
    assert modules == set(REQUIREMENTS)


def test_nothing_below_the_compute_layer_imports_it():
    """`geometry` carries numpy, so the dependency-free layers must not reach into it."""
    forbidden = {"geometry"}
    offenders: dict[str, list[str]] = {}
    for package in ("kernel", "schema", "sdk", "plugins", "storage", "vcs"):
        for path in (ROOT / package).rglob("*.py"):
            leaked = sorted(_internal_imports(path) & forbidden)
            if leaked:
                offenders[_relative(path)] = leaked
    assert offenders == {}


def test_the_kernel_never_imports_a_capability():
    """The kernel contains mechanisms, never features.

    Nothing in ``kernel`` may know what a massing story or a cost assembly is -- that is the
    property that lets it be versioned independently of everything above it.
    """
    forbidden = {"plugins", "schema", "sdk", "viewer", "app", "demo"}
    offenders = {
        _relative(path): sorted(_internal_imports(path) & forbidden)
        for path in (ROOT / "kernel").rglob("*.py")
        if _internal_imports(path) & forbidden
    }
    assert offenders == {}


def test_no_capability_plugin_imports_another():
    """Capabilities compose through tokens, never through imports.

    This is the claim the bridge plugin exists to demonstrate: estimating measures massing geometry
    without either package knowing the other is installed.
    """
    plugins = ROOT / "plugins"
    families = {p.name for p in plugins.iterdir() if p.is_dir() and not p.name.startswith("__")}
    offenders: dict[str, list[str]] = {}

    for family in sorted(families):
        siblings = families - {family}
        for path in (plugins / family).rglob("*.py"):
            # Inside `plugins/<family>/`, a level-2 relative import resolves to a sibling family.
            parts = path.relative_to(ROOT).parts
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            leaked: set[str] = set()
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom) or not node.module:
                    continue
                if node.level == 2 and node.module.split(".")[0] in siblings:
                    leaked.add(node.module.split(".")[0])
                if node.level == 0 and node.module.startswith("massingviser.plugins."):
                    candidate = node.module.split(".")[2]
                    if candidate in siblings:
                        leaked.add(candidate)
            if leaked:
                offenders[_relative(path)] = sorted(leaked)
    assert offenders == {}


def test_importing_the_package_does_not_pull_in_viser():
    """``import massingviser`` must stay cheap enough for a headless job."""
    source = (ROOT / "__init__.py").read_text(encoding="utf-8")
    assert "import viser" not in source
    # It is imported lazily inside `serve`, which is the only entry point that needs it.
    assert "from .viewer import serve" in source


def test_every_module_is_reachable_from_a_package_export():
    """No orphan modules -- a file nobody imports is a file nobody maintains."""
    for package in ("kernel", "schema", "sdk", "storage", "vcs", "geometry"):
        exports = (ROOT / package / "__init__.py").read_text(encoding="utf-8")
        for path in (ROOT / package).glob("*.py"):
            if path.name == "__init__.py":
                continue
            if path.stem == "codec":
                continue  # imported explicitly by the composition root, not part of the record surface
            assert f"from .{path.stem} import" in exports, f"{_relative(path)} is not re-exported"
