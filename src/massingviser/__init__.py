"""MassingViser -- a federated AEC platform as a kernel plus plugins, in pure Python.

Two lineages meet here.

``massingifc`` is a framework-agnostic kernel and plugin architecture for AEC: service container,
event bus, command bus with undo, versioned persistence, plugin host, capability registry -- and
deliberately **no viewer**.

``viser`` is the opposite shape: a pure-Python library whose whole point is that ``pip install``
gives you a browser 3D viewer driven from Python.

MassingViser is the join. The architecture is ported to Python, and viser supplies exactly the half
massingifc left out. The kernel's rules survive the move intact: the kernel contains mechanisms and
never features, no plugin can crash the host, everything persisted is versioned, and nothing below
``massingviser.viewer`` imports a rendering library.

    from massingviser import build_kernel, serve

    kernel = build_kernel()
    serve(kernel)                     # opens http://127.0.0.1:8080
"""

from typing import Any

from .app import DEFAULT_PLUGINS, build_kernel
from .kernel import KERNEL_API_VERSION, Kernel, KernelError, Result, create_kernel

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_PLUGINS",
    "KERNEL_API_VERSION",
    "Kernel",
    "KernelError",
    "Result",
    "__version__",
    "build_kernel",
    "create_kernel",
    "serve",
]


def serve(kernel: "Kernel[Any] | None" = None, **options: Any) -> Any:
    """Open a viser viewer over a kernel, building a default one if none is given.

    Imported lazily so that ``import massingviser`` stays free of viser -- a headless test run or a
    server-side job has no reason to pull in a websocket stack and numpy.
    """
    from .viewer import serve as _serve

    return _serve(kernel if kernel is not None else build_kernel(), **options)
