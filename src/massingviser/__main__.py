"""``python -m massingviser`` -- open the viewer on a fresh project."""

from __future__ import annotations

import argparse

from .app import DEFAULT_PLUGINS, build_kernel


def main() -> None:
    parser = argparse.ArgumentParser(prog="massingviser", description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="default: loopback only")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--plugins",
        default=",".join(DEFAULT_PLUGINS),
        help=f"comma-separated; available: {', '.join(DEFAULT_PLUGINS)}",
    )
    parser.add_argument("--demo", action="store_true", help="seed a small scheme to look at")
    arguments = parser.parse_args()

    kernel = build_kernel(
        plugins=tuple(p.strip() for p in arguments.plugins.split(",") if p.strip())
    )

    from .viewer import serve

    app = serve(kernel, host=arguments.host, port=arguments.port, block=False)
    if arguments.demo:
        from .demo import seed

        seed(app.bridge)
        app.refresh()

    import time

    try:
        while True:
            time.sleep(0.25)
            app.poll()
    except KeyboardInterrupt:
        pass
    finally:
        app.bridge.close()


if __name__ == "__main__":
    main()
