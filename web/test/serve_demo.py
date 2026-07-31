"""Serve the demo scheme for the render test.

A separate script rather than a fixture inside the Node test, because the server has to be a real
process listening on a real port -- which is the only configuration the browser can actually load.

    python web/test/serve_demo.py [port]
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from massingviser import build_kernel  # noqa: E402
from massingviser.demo import SCHEME  # noqa: E402
from massingviser.plugins.massing import MASSING_COMMANDS, ProfileToken  # noqa: E402
from massingviser.web import serve  # noqa: E402


def main(port: int) -> None:
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def run() -> None:
        asyncio.set_event_loop(loop)
        loop.call_soon(ready.set)
        loop.run_forever()

    threading.Thread(target=run, daemon=True).start()
    ready.wait(10)

    def submit(coroutine):
        return asyncio.run_coroutine_threadsafe(coroutine, loop).result(60)

    kernel = build_kernel()
    submit(kernel.start())

    profiles = kernel.capabilities.get(ProfileToken)
    for block in SCHEME:
        sketched = submit(
            kernel.commands.execute(
                MASSING_COMMANDS.sketch_profile,
                {"points": block["points"], "name": block["name"]},
            )
        )
        if block["hole"]:
            submit(profiles.add_hole(sketched.value, block["hole"]))
        submit(
            kernel.commands.execute(
                MASSING_COMMANDS.create_mass,
                {
                    "name": block["name"],
                    "profile_id": sketched.value,
                    "story_count": block["storeys"],
                    "story_height": block["height"],
                    "color": block["color"],
                },
            )
        )

    serve(kernel, loop, port=port)
    print(f"serving on {port}", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8137)
