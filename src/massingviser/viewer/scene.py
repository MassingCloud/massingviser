"""Rendering massing records into a viser scene.

The kernel never learns that this file exists. It reads capability tokens the massing plugin
provides and writes meshes into viser -- so the same plugin runs unchanged in a headless test, and
a different shell could render the same records to a desktop viewport or an engine package.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..plugins.massing import (
    ContextToken,
    MassingToken,
    ProfileToken,
    StoryToken,
    extrude_stories,
    to_xy,
)
from ..plugins.massing.tessellate import Mesh
from ..schema import MassingObjectRecord

#: Default mass colour when a record carries none. Cool grey-blue reads as "conceptual" and stays
#: legible against both the light and dark viser themes.
DEFAULT_COLOR = (140, 165, 190)

#: Vertical gap left between storeys so a mass reads as stacked plates rather than one block. This
#: is the single visual decision that makes a massing model look like a massing model.
STORY_GAP = 0.35


def hex_to_rgb(value: str | None) -> tuple[int, int, int]:
    if not value:
        return DEFAULT_COLOR
    text = value.lstrip("#")
    if len(text) == 3:
        text = "".join(char * 2 for char in text)
    if len(text) != 6:
        return DEFAULT_COLOR
    try:
        return (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16))
    except ValueError:
        return DEFAULT_COLOR


def _shade(color: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    """Lighten or darken so adjacent storeys are distinguishable without a second palette."""
    return tuple(max(0, min(255, int(channel * factor))) for channel in color)  # type: ignore[return-value]


def mesh_arrays(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """Convert the plugin's plain tuples into the arrays viser wants.

    This is the *only* place numpy enters the massing path, which is what keeps the tessellator
    testable without it.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float32).reshape(-1, 3)
    faces = np.asarray(mesh.faces, dtype=np.uint32).reshape(-1, 3)
    return vertices, faces


@dataclass
class SceneSync:
    """Keeps the viser scene in step with kernel state.

    Rebuilds a mass wholesale when it changes rather than diffing storey by storey. A mass is tens
    of triangles; the diffing logic would be more code than the redraw costs, and a stale storey
    left behind by a partial update is exactly the bug that is hardest to see.
    """

    server: Any
    bridge: Any
    on_select: Callable[[str], None] | None = None
    _nodes: dict[str, list[Any]] = field(default_factory=dict)
    _labels: dict[str, Any] = field(default_factory=dict)
    _site: list[Any] = field(default_factory=list)

    # -- capability lookups ------------------------------------------------------------------

    def _capability(self, token: Any) -> Any:
        return self.bridge.read(lambda: self.bridge.kernel.capabilities.get(token))

    def masses(self) -> tuple[MassingObjectRecord, ...]:
        service = self._capability(MassingToken)
        if service is None:
            return ()
        return self.bridge.read(service.list)

    # -- rendering ---------------------------------------------------------------------------

    def clear(self) -> None:
        for handles in self._nodes.values():
            for handle in handles:
                handle.remove()
        self._nodes.clear()
        for handle in self._labels.values():
            handle.remove()
        self._labels.clear()

    def remove_mass(self, mass_id: str) -> None:
        for handle in self._nodes.pop(mass_id, ()):
            handle.remove()
        label = self._labels.pop(mass_id, None)
        if label is not None:
            label.remove()

    def sync(self) -> None:
        """Redraw every mass, dropping any whose record has gone."""
        masses = self.masses()
        live = {mass.id for mass in masses}
        for stale in [mass_id for mass_id in self._nodes if mass_id not in live]:
            self.remove_mass(stale)
        for mass in masses:
            self.draw_mass(mass)
        self.draw_site()

    def draw_mass(self, mass: MassingObjectRecord) -> None:
        profiles = self._capability(ProfileToken)
        stories_service = self._capability(StoryToken)
        if profiles is None or stories_service is None:
            return

        profile = self.bridge.read(lambda: profiles.get(mass.profile_id))
        if profile is None:
            return

        stories = self.bridge.read(lambda: stories_service.stories(mass.id))
        heights = [story.height for story in stories] or list(mass.story_heights)
        if not heights:
            self.remove_mass(mass.id)
            return

        outer = to_xy(profile.points)
        holes = [to_xy(hole) for hole in profile.holes]

        # Per-storey outline overrides -- a setback is a different profile on one floor, and
        # ignoring it here would draw a tower the metrics panel disagrees with.
        overrides: dict[int, Sequence[tuple[float, float]]] = {}
        for story in stories:
            if story.profile_id:
                override = self.bridge.read(lambda s=story: profiles.get(s.profile_id))
                if override is not None:
                    overrides[story.index] = to_xy(override.points)

        story_meshes = extrude_stories(
            outer,
            holes,
            heights,
            base_elevation=profile.base_elevation,
            story_outlines=overrides or None,
            slab_gap=STORY_GAP,
        )

        self.remove_mass(mass.id)
        base_color = hex_to_rgb(mass.color)
        opacity = mass.opacity if mass.opacity is not None else 1.0
        handles: list[Any] = []

        for story_mesh in story_meshes:
            if story_mesh.mesh.is_empty:
                continue
            vertices, faces = mesh_arrays(story_mesh.mesh)
            # Alternate a hair lighter/darker so floor lines read without an edge pass.
            factor = 1.0 if story_mesh.index % 2 == 0 else 0.92
            excluded = next(
                (s.excluded_from_gfa for s in stories if s.index == story_mesh.index), False
            )
            handle = self.server.scene.add_mesh_simple(
                f"/masses/{mass.id}/story_{story_mesh.index:03d}",
                vertices=vertices,
                faces=faces,
                color=_shade(base_color, factor * (0.7 if excluded else 1.0)),
                opacity=None if opacity >= 1.0 else opacity,
                flat_shading=True,
                side="double",
            )
            if self.on_select is not None:
                handle.on_click(lambda _event, mass_id=mass.id: self.on_select(mass_id))
            handles.append(handle)

        self._nodes[mass.id] = handles

        centre_x = sum(point[0] for point in outer) / len(outer)
        centre_y = sum(point[1] for point in outer) / len(outer)
        top = profile.base_elevation + sum(heights)
        self._labels[mass.id] = self.server.scene.add_label(
            f"/masses/{mass.id}/label",
            text=f"{mass.name}  ({len(heights)} storeys, {top:.1f} m)",
            position=(centre_x, centre_y, top + 2.0),
        )

    def draw_site(self) -> None:
        for handle in self._site:
            handle.remove()
        self._site = []

        context = self._capability(ContextToken)
        if context is None:
            return
        boundary = self.bridge.read(context.site_boundary)
        if boundary is None or len(boundary.points) < 3:
            return

        ring = [(point[0], point[1], 0.05) for point in boundary.points]
        segments = np.asarray(
            [[ring[i], ring[(i + 1) % len(ring)]] for i in range(len(ring))], dtype=np.float32
        )
        self._site.append(
            self.server.scene.add_line_segments(
                "/site/boundary", points=segments, colors=(220, 120, 60), line_width=3.0
            )
        )
