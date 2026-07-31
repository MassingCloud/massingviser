"""``massingviser.web`` -- the HTTP face of the engine bridge.

Four routes: the manifest, the payload buffers, picking and culling. Everything expensive happens
on this side of them, so the browser layer receives GlobalIds and geometry rather than a model.

Standard library only, like every package here except ``viewer``, ``geometry`` and ``adapters``.
"""

from .server import (
    REQUEST_TIMEOUT,
    WEB_ROOT,
    SceneBridge,
    create_handler,
    serve,
)

__all__ = [
    "REQUEST_TIMEOUT",
    "WEB_ROOT",
    "SceneBridge",
    "create_handler",
    "serve",
]
