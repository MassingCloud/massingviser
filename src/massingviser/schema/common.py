"""Primitives shared by every record family."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

#: Opaque identifier. Stable across sessions and safe to persist and cross-reference.
Id = str

#: ISO-8601 timestamp, always UTC. Stored as a string so records stay JSON-round-trippable.
IsoTimestamp = str

Vec3 = "tuple[float, float, float]"

#: Column-major 4x4 transform, matching ``THREE.Matrix4.toArray`` -- translation at indices
#: 12, 13, 14.
#:
#: Stated explicitly because the two conventions are indistinguishable at the type level and a
#: transposed matrix does not fail, it just puts everything in the wrong place.
Matrix4 = "tuple[float, ...]"


@dataclass(frozen=True)
class Provenance:
    """Where a record came from.

    Carried by anything that can originate outside the authored model -- twin observations,
    imported schedules, external cost libraries. Without provenance a federated project quickly
    loses track of which numbers are measured, which are assumed, and which are somebody else's.
    """

    source: str
    imported_at: IsoTimestamp | None = None
    #: 0..1 where meaningful -- sensor accuracy, scan registration quality, mapping confidence.
    confidence: float | None = None
    external_id: str | None = None


@dataclass(frozen=True)
class ElementRef:
    """Reference to an element inside a model.

    ``global_id`` is the identity -- the IFC ``IfcGloballyUniqueId`` carried by the element itself.
    It survives re-export, re-conversion, a different viewer, and a new session.

    ``local_id`` is a *transient runtime handle* (a Fragments local id, an express id). It is valid
    only for one load of one model in one session: re-converting the same IFC can renumber it. It
    exists here purely so runtime code can avoid a lookup on hot paths.

    **Never persist or compare ``local_id`` across sessions, and never anchor a record to it.**
    Markup, clashes, 4D links, takeoff and field status all reference elements this way; a
    transient identity in this type would propagate silently into every one of them and only
    surface as "all the pins moved" after somebody re-issued a model.
    """

    model_id: Id
    #: IFC GlobalId. Stable, persistable identity.
    global_id: str
    #: Session-scoped runtime handle. Cache, never store.
    local_id: "int | str | None" = None


def same_element(a: ElementRef, b: ElementRef) -> bool:
    """Compare element references by their stable identity, ignoring transient handles."""
    return a.model_id == b.model_id and a.global_id == b.global_id


def element_key(element: ElementRef) -> str:
    """Stable key for maps and sets. Built from identity only, deliberately."""
    return f"{element.model_id}/{element.global_id}"


@dataclass(frozen=True)
class UnitizedValue:
    value: float
    #: Unit symbol, e.g. ``m``, ``m2``, ``m3``, ``kg``, ``each``.
    unit: str


@dataclass(frozen=True)
class Money:
    """Money kept as minor units plus a currency code -- never as a float.

    ``12.30`` in binary floating point is not 12.30, and a bill of quantities adds thousands of
    them together. Integer minor units (pence, cents) make the arithmetic exact.
    """

    amount_minor: int
    currency: str

    def __post_init__(self) -> None:
        if not isinstance(self.amount_minor, int) or isinstance(self.amount_minor, bool):
            raise TypeError("Money.amount_minor must be an int of minor units, not a float.")

    @property
    def major(self) -> float:
        """Presentation only. Never feed this back into arithmetic."""
        return self.amount_minor / 100.0

    def __add__(self, other: "Money") -> "Money":
        self._assert_same_currency(other)
        return Money(self.amount_minor + other.amount_minor, self.currency)

    def __sub__(self, other: "Money") -> "Money":
        self._assert_same_currency(other)
        return Money(self.amount_minor - other.amount_minor, self.currency)

    def scaled(self, factor: float) -> "Money":
        """Multiply by a quantity, rounding half-up to the nearest minor unit."""
        return Money(int(_round_half_up(self.amount_minor * factor)), self.currency)

    def _assert_same_currency(self, other: "Money") -> None:
        if self.currency != other.currency:
            raise ValueError(
                f"Cannot combine {self.currency} with {other.currency}; convert explicitly."
            )

    def __str__(self) -> str:
        return f"{self.amount_minor / 100:,.2f} {self.currency}"


def _round_half_up(value: float) -> float:
    import math

    return math.floor(value + 0.5) if value >= 0 else math.ceil(value - 0.5)


def zero_money(currency: str) -> Money:
    return Money(0, currency)


# `GeoReference` lives in `geo.py`, not here.
#
# It is shared by models, twins and site boundaries rather than owned by any one of them, and it
# comes with real behaviour -- CRS parsing, unit conversion, extent validation -- that belongs
# beside it. An abbreviated copy in `common` would be the version people imported by accident.
