"""Making records survive a round trip through JSON.

``massingifc`` never needs this: its records are plain object literals, so ``JSON.stringify`` is the
whole story. The Python port uses frozen dataclasses instead -- which buys real typing, field-name
validation on patch, and ``replace()`` -- and the price is that they have to be told how to
serialise.

The failure this prevents is specific and nasty. ``MemoryStorageAdapter`` deep-copies, so records
round-trip through it untouched and every test passes. Swap in a real adapter and the first save
raises. That is exactly the "only shows up once a real adapter is swapped in" bug the memory
adapter's own docstring warns about, and it is why the storage tests exercise both adapters.

Decoding resolves types through an explicit registry and **never imports by name**. A project file
naming ``os.system`` as its record type would otherwise be a code-execution vector; here an
unrecognised type decodes to a plain dict.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

RECORD_TAG = "$record"

_REGISTRY: dict[str, type] | None = None
_EXTRA: dict[str, type] = {}


def _registry() -> dict[str, type]:
    """Every dataclass this package exports, keyed by class name.

    Built lazily because the schema package imports this module, so it cannot be built at import
    time without a cycle.
    """
    global _REGISTRY
    if _REGISTRY is None:
        from . import __all__ as exported
        from . import __dict__ as namespace

        _REGISTRY = {
            name: value
            for name in exported
            if is_dataclass(value := namespace.get(name)) and isinstance(value, type)
        }
    return {**_REGISTRY, **_EXTRA}


def register_record_type(record_type: type) -> type:
    """Register a plugin's own record type so it round-trips too.

    Usable as a decorator. Names must be unique across the platform -- a collision is refused rather
    than resolved by load order, which would make a project file decode differently depending on
    which plugins happened to activate first.
    """
    name = record_type.__name__
    existing = _registry().get(name)
    if existing is not None and existing is not record_type:
        raise ValueError(
            f'A different record type named "{name}" is already registered '
            f"({existing.__module__}). Record type names must be unique."
        )
    _EXTRA[name] = record_type
    return record_type


def record_default(value: Any) -> Any:
    """``json.dump(default=...)`` hook: encode a dataclass as a tagged mapping."""
    if is_dataclass(value) and not isinstance(value, type):
        encoded: dict[str, Any] = {RECORD_TAG: type(value).__name__}
        for field in fields(value):
            encoded[field.name] = getattr(value, field.name)
        return encoded
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")


def _to_tuples(value: Any) -> Any:
    """Deep list-to-tuple conversion, descending through lists only.

    Nested collections are ordinary here -- ``ProfileRecord.points`` is a tuple of coordinate
    tuples, and a hole list is a tuple of those. Converting only the outer level leaves the inner
    ones as lists, which compares unequal to the record that was saved and fails nothing until
    something checks. Dicts are left alone: a ``Mapping`` field is not tuple-typed, so a list inside
    one is genuinely a list.
    """
    if isinstance(value, list):
        return tuple(_to_tuples(item) for item in value)
    return value


def record_object_hook(mapping: dict[str, Any]) -> Any:
    """``json.load(object_hook=...)`` hook: revive a tagged mapping into its dataclass."""
    name = mapping.get(RECORD_TAG)
    if not isinstance(name, str):
        return mapping
    record_type = _registry().get(name)
    if record_type is None:
        # Unknown type: hand back the raw mapping rather than importing something a file named.
        return mapping

    values = {key: value for key, value in mapping.items() if key != RECORD_TAG}
    # JSON has one sequence type, so a field declared as a tuple comes back as a list. Left alone,
    # `dataclasses.replace` would happily produce a record whose collection fields have drifted
    # type, and the state store's identity comparison would still work -- so nothing would fail
    # until something checked.
    for field in fields(record_type):
        current = values.get(field.name)
        if isinstance(current, list) and "tuple" in str(field.type):
            values[field.name] = _to_tuples(current)
    try:
        return record_type(**values)
    except TypeError:
        # A record written by a build with different fields. Returning the mapping keeps the data
        # rather than losing it, and the migration engine is the thing that should reconcile it.
        return mapping
