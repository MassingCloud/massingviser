"""Write the fixtures the JavaScript tests read.

Run from the repository root:

    python web/test/generate_fixtures.py

These are checked in deliberately. The JS reader is tested against buffers the *Python* encoder
produced, so a change on either side that breaks the wire contract fails the Node suite -- which is
the only place the two implementations actually meet.
"""

from __future__ import annotations

import asyncio
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from massingviser import build_kernel  # noqa: E402
from massingviser.geometry.payload import (  # noqa: E402
    MeshInput,
    build_geometry_payloads,
    encode_mesh_batch,
)
from massingviser.plugins.engine import SceneExportToken, to_manifest  # noqa: E402
from massingviser.plugins.massing import MASSING_COMMANDS  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"

CUBE_VERTICES = [
    (0, 0, 0),
    (1, 0, 0),
    (1, 1, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 0, 1),
    (1, 1, 1),
    (0, 1, 1),
]
CUBE_FACES = [
    (0, 2, 1),
    (0, 3, 2),
    (4, 5, 6),
    (4, 6, 7),
    (0, 1, 5),
    (0, 5, 4),
    (1, 2, 6),
    (1, 6, 5),
    (2, 3, 7),
    (2, 7, 6),
    (3, 0, 4),
    (3, 4, 7),
]


def _offset(points, delta):
    return [(x + delta, y + delta, z + delta) for x, y, z in points]


async def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)

    shaded = build_geometry_payloads({"CUBE": (CUBE_VERTICES, CUBE_FACES)}, lod_budgets=())
    (FIXTURES / "cube-shaded.bin").write_bytes(shaded.payloads[0].data)

    plain = build_geometry_payloads(
        {"CUBE": (CUBE_VERTICES, CUBE_FACES)}, lod_budgets=(), shade=False
    )
    (FIXTURES / "cube-plain.bin").write_bytes(plain.payloads[0].data)

    two = encode_mesh_batch(
        [
            MeshInput("A", CUBE_VERTICES, CUBE_FACES),
            MeshInput("B", _offset(CUBE_VERTICES, 10.0), CUBE_FACES),
        ]
    )
    (FIXTURES / "two-cubes.bin").write_bytes(two.data)

    # A v1 buffer: identical layout, no normals block, version stamped back to 1.
    v1 = plain.payloads[0].data
    (FIXTURES / "cube-v1.bin").write_bytes(v1[:4] + struct.pack("<I", 1) + v1[8:])

    # A real scene manifest, from the same path a browser would fetch it through.
    kernel = build_kernel()
    await kernel.start()
    sketched = await kernel.commands.execute(
        MASSING_COMMANDS.sketch_profile,
        {"points": [(0, 0, 0), (30, 0, 0), (30, 18, 0), (0, 18, 0)], "name": "Block"},
    )
    await kernel.commands.execute(
        MASSING_COMMANDS.create_mass,
        {"name": "Block", "profile_id": sketched.value, "story_count": 6, "story_height": 3.6},
    )
    package = (await kernel.capabilities.get(SceneExportToken).build()).value
    manifest = to_manifest(package)
    # Pinned, so regenerating produces a byte-identical file. Everything else in the manifest is a
    # function of the model; this one field is a function of the clock, and leaving it live would
    # make the CI check that fixtures are current fail on every single run.
    manifest["generatedAt"] = "2026-01-01T00:00:00Z"
    (FIXTURES / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    await kernel.stop()

    for path in sorted(FIXTURES.iterdir()):
        print(f"  {path.name:20} {path.stat().st_size:>8,} bytes")


if __name__ == "__main__":
    asyncio.run(main())
