"""Refresh the vendored three.js build.

three.js is vendored rather than loaded from a CDN. A CDN import map is convenient and makes the
client unusable on any machine that cannot reach unpkg -- which for AEC means a site office, a
secure network, or anyone running this behind a proxy that does not allow it. A viewer that needs
the public internet to draw a building is not a viewer you can take to a project.

Each file is pinned to a version **and** a SHA-256. A vendored dependency whose contents are not
checked is a supply-chain hole with extra steps: the point of committing the bytes is that they are
the bytes that were reviewed.

    python web/vendor/fetch_vendor.py            # verify what is checked in
    python web/vendor/fetch_vendor.py --update   # re-download and rewrite the digests
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

THREE_VERSION = "0.169.0"
HERE = Path(__file__).resolve().parent
LOCK = HERE / "vendor.lock.json"

SOURCES = {
    "three.module.min.js": f"https://unpkg.com/three@{THREE_VERSION}/build/three.module.min.js",
    "OrbitControls.js": (
        f"https://unpkg.com/three@{THREE_VERSION}/examples/jsm/controls/OrbitControls.js"
    ),
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def update() -> int:
    lock = {"three": THREE_VERSION, "files": {}}
    for name, url in SOURCES.items():
        data = urllib.request.urlopen(url, timeout=120).read()
        (HERE / name).write_bytes(data)
        lock["files"][name] = {"url": url, "sha256": digest(data), "bytes": len(data)}
        print(f"  fetched {name:24} {len(data):>9,} bytes")
    LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8", newline="\n")
    return 0


def verify() -> int:
    if not LOCK.is_file():
        print("No vendor.lock.json. Run with --update.", file=sys.stderr)
        return 1
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if lock.get("three") != THREE_VERSION:
        print(f"Lock is for three {lock.get('three')}, this script pins {THREE_VERSION}.")
        return 1

    failures = 0
    for name, expected in lock["files"].items():
        path = HERE / name
        if not path.is_file():
            print(f"  MISSING {name}", file=sys.stderr)
            failures += 1
            continue
        actual = digest(path.read_bytes())
        if actual != expected["sha256"]:
            print(f"  CHANGED {name}: {expected['sha256'][:16]} -> {actual[:16]}", file=sys.stderr)
            failures += 1
        else:
            print(f"  ok      {name:24} {expected['bytes']:>9,} bytes")
    return 1 if failures else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true", help="re-download and rewrite the lock")
    raise SystemExit(update() if parser.parse_args().update else verify())
