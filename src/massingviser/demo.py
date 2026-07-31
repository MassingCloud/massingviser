"""A small worked scheme, so the viewer opens on something rather than an empty grid.

Everything here goes through the command bus, exactly as the GUI does. That is deliberate: the demo
is also a check that the public path works, and anything it can do a user can undo.
"""

from __future__ import annotations

from typing import Any

from .plugins.massing import MASSING_COMMANDS

#: A courtyard block, a slab and a tower -- three masses that between them exercise holes,
#: setbacks-by-profile, and a tall stack.
SCHEME = (
    {
        "name": "Courtyard block",
        "points": [(0, 0, 0), (50, 0, 0), (50, 40, 0), (0, 40, 0)],
        "hole": [(16, 12, 0), (34, 12, 0), (34, 28, 0), (16, 28, 0)],
        "storeys": 6,
        "height": 3.6,
        "color": "#4C78A8",
    },
    {
        "name": "North slab",
        "points": [(0, 52, 0), (64, 52, 0), (64, 68, 0), (0, 68, 0)],
        "hole": None,
        "storeys": 9,
        "height": 3.4,
        "color": "#54A24B",
    },
    {
        "name": "Tower",
        "points": [(64, 6, 0), (90, 6, 0), (90, 32, 0), (64, 32, 0)],
        "hole": None,
        "storeys": 28,
        "height": 3.5,
        "color": "#F58518",
    },
)


def seed(bridge: Any) -> None:
    """Build the demo scheme through the command bus."""
    from .plugins.massing import ProfileToken

    bridge.execute(
        MASSING_COMMANDS.set_site_boundary,
        {
            "points": [(-12, -12, 0), (104, -12, 0), (104, 80, 0), (-12, 80, 0)],
            "name": "Site",
            "max_floor_area_ratio": 3.5,
            "max_height": 110.0,
        },
    )

    for block in SCHEME:
        sketched = bridge.execute(
            MASSING_COMMANDS.sketch_profile,
            {"points": block["points"], "name": block["name"]},
        )
        if not sketched.ok:
            continue
        if block["hole"] is not None:
            profiles = bridge.read(lambda: bridge.kernel.capabilities.get(ProfileToken))
            bridge.run(profiles.add_hole(sketched.value, block["hole"]))

        bridge.execute(
            MASSING_COMMANDS.create_mass,
            {
                "name": block["name"],
                "profile_id": sketched.value,
                "story_count": block["storeys"],
                "story_height": block["height"],
                "color": block["color"],
            },
        )
