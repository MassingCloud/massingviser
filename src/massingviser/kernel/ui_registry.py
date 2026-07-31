from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from .disposable import Disposable, to_disposable

THost = TypeVar("THost")

#: Where a contribution attaches. Open-ended by design: a host shell is free to invent its own
#: points (``"status-bar"``, ``"viewport-gizmo"``) without a kernel change.
UIContributionPoint = str


@dataclass(frozen=True)
class UIMountContext(Generic[THost]):
    host: THost
    contribution_id: str


@dataclass(frozen=True, eq=False)
class UIContribution:
    """A UI contribution.

    Note there is no rendering type anywhere in this file. The kernel must stay usable from a
    desktop shell, a test harness, or a headless server-side session, so the host surface is
    generic and the viser shell narrows it at its own boundary.
    """

    id: str
    point: UIContributionPoint
    title: str | None = None
    icon: str | None = None
    #: Logical bucket within a point, e.g. a toolbar section.
    group: str | None = None
    #: Ascending sort key. Ties break on ``id`` so ordering is stable across sessions.
    order: int = 0
    #: Host-interpreted hint, e.g. ``"left"``, ``"right"``, ``"bottom"``, ``"modal"``.
    placement: str | None = None
    #: Command run on activation. Keeps UI declarative and routes actions through the command bus.
    command_id: str | None = None
    #: Visibility predicate evaluated against host-supplied context.
    when: Callable[[Mapping[str, Any]], bool] | None = None
    #: Attach to a host-provided surface, returning a teardown for the host to call.
    mount: Callable[[UIMountContext[Any]], Disposable | None] | None = None
    plugin_id: str | None = None

    def __hash__(self) -> int:
        return id(self)


class UIExtensionRegistry(Generic[THost]):
    """Registry of UI contributions.

    The kernel stores and orders descriptors; it never renders. That split is what lets the same
    plugin contribute to a viser shell and a desktop shell without knowing which one it is running
    in.
    """

    __slots__ = ("_by_point", "_listeners")

    def __init__(self) -> None:
        self._by_point: dict[str, list[UIContribution]] = {}
        self._listeners: dict[Callable[[str], None], None] = {}

    def register(self, contribution: UIContribution) -> Disposable:
        contributions = self._by_point.setdefault(contribution.point, [])
        contributions.append(contribution)
        contributions.sort(key=lambda c: (c.order, c.id))
        self._by_point[contribution.point] = contributions
        self._notify(contribution.point)

        def _remove() -> None:
            current = self._by_point.get(contribution.point)
            if current is None:
                return
            for index, candidate in enumerate(current):
                if candidate is contribution:
                    current.pop(index)
                    break
            if not current:
                self._by_point.pop(contribution.point, None)
            self._notify(contribution.point)

        return to_disposable(_remove)

    def register_all(
        self, contributions: list[UIContribution] | tuple[UIContribution, ...]
    ) -> Disposable:
        subscriptions = [self.register(contribution) for contribution in contributions]

        def _remove_all() -> None:
            for subscription in subscriptions:
                subscription.dispose()

        return to_disposable(_remove_all)

    def by_point(self, point: UIContributionPoint) -> tuple[UIContribution, ...]:
        """Everything registered at a point, in sort order, regardless of visibility."""
        return tuple(self._by_point.get(point, ()))

    def visible(
        self, point: UIContributionPoint, context: Mapping[str, Any] | None = None
    ) -> tuple[UIContribution, ...]:
        """Contributions at a point whose ``when`` predicate passes.

        A predicate that raises hides the contribution rather than propagating: a plugin with a
        broken visibility rule should lose its own button, not break the toolbar for everyone else.
        """
        ctx = context or {}
        result = []
        for contribution in self.by_point(point):
            if contribution.when is None:
                result.append(contribution)
                continue
            try:
                if contribution.when(ctx):
                    result.append(contribution)
            except Exception:  # noqa: BLE001
                pass
        return tuple(result)

    def points(self) -> tuple[str, ...]:
        return tuple(self._by_point)

    def on_did_change(self, listener: Callable[[str], None]) -> Disposable:
        self._listeners[listener] = None
        return to_disposable(lambda: self._listeners.pop(listener, None))

    def _notify(self, point: str) -> None:
        for listener in list(self._listeners):
            try:
                listener(point)
            except Exception:  # noqa: BLE001
                pass  # a failing observer must not corrupt the registry
