"""Minimal semantic-version handling.

Deliberately dependency-free and deliberately small: the kernel only ever needs to answer "is this
provider new enough for this consumer", which is a comparison plus a caret range. A full semver
implementation (pre-release ordering, complex range grammars) would be more surface to keep
correct than the question warrants.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_PATTERN = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?$")
_RANGE = re.compile(r"^(\^|>=|<=|>|<|=)?\s*(.+)$")


@dataclass(frozen=True)
class SemVer:
    major: int
    minor: int
    patch: int
    #: Pre-release tag, e.g. ``beta.1``. Compared only for equality, never ordered.
    prerelease: str | None = None


def parse_semver(value: str) -> SemVer | None:
    match = _PATTERN.match(value.strip())
    if not match:
        return None
    return SemVer(
        major=int(match.group(1)),
        minor=int(match.group(2)),
        patch=int(match.group(3)),
        prerelease=match.group(4),
    )


def compare_semver(a: SemVer, b: SemVer) -> int:
    """Return a negative number if ``a < b``, positive if ``a > b``, ``0`` when equal."""
    if a.major != b.major:
        return a.major - b.major
    if a.minor != b.minor:
        return a.minor - b.minor
    if a.patch != b.patch:
        return a.patch - b.patch
    # A pre-release sorts below its own release (1.0.0-beta < 1.0.0), matching the semver spec's
    # ordering rule without implementing the full identifier comparison.
    if a.prerelease == b.prerelease:
        return 0
    if a.prerelease is None:
        return 1
    if b.prerelease is None:
        return -1
    return -1 if a.prerelease < b.prerelease else 1


def satisfies(version: str, range_: str) -> bool:
    """Test ``version`` against a range.

    Supported forms: ``*`` (any), ``1.2.3`` (exact), ``>=1.2.3``, ``>1.2.3``, ``<=1.2.3``,
    ``<1.2.3``, and ``^1.2.3`` (compatible-with: same major, at least this version -- for ``0.x``,
    same minor).
    """
    trimmed = range_.strip()
    if trimmed in ("*", ""):
        return True

    parsed = parse_semver(version)
    if parsed is None:
        return False

    match = _RANGE.match(trimmed)
    if not match:
        return False
    operator = match.group(1) or "="
    target = parse_semver(match.group(2) or "")
    if target is None:
        return False

    comparison = compare_semver(parsed, target)
    if operator == "=":
        return comparison == 0
    if operator == ">":
        return comparison > 0
    if operator == ">=":
        return comparison >= 0
    if operator == "<":
        return comparison < 0
    if operator == "<=":
        return comparison <= 0
    if operator == "^":
        if comparison < 0:
            return False
        # Below 1.0.0 the minor acts as the breaking-change axis, so ^0.3.1 must not match 0.4.0.
        if target.major == 0:
            return parsed.major == 0 and parsed.minor == target.minor
        return parsed.major == target.major
    return False
