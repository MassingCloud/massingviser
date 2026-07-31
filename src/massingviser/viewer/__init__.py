"""``massingviser.viewer`` -- the viser shell.

The only package that imports viser. Everything below it -- kernel, schema, SDK and every
capability plugin -- has no rendering dependency at all, which is what lets the same plugins run
headless in a test, in a desktop shell, or behind an engine exporter.
"""

from .app import MassingViserApp, serve
from .bridge import KernelBridge
from .scene import DEFAULT_COLOR, STORY_GAP, SceneSync, hex_to_rgb, mesh_arrays

__all__ = [
    "DEFAULT_COLOR",
    "STORY_GAP",
    "KernelBridge",
    "MassingViserApp",
    "SceneSync",
    "hex_to_rgb",
    "mesh_arrays",
    "serve",
]
