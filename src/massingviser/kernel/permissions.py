from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from .errors import KernelError
from .result import Err, Ok, Result, err, ok
from .result import resolve as _resolve


@dataclass(frozen=True)
class Identity:
    id: str
    roles: tuple[str, ...] = ()
    display_name: str | None = None
    attributes: Mapping[str, Any] | None = None


#: The anonymous subject used before a host installs a real identity.
ANONYMOUS_IDENTITY = Identity(id="anonymous", roles=(), display_name="Anonymous")


@dataclass(frozen=True)
class PermissionRequest:
    #: Dotted verb, e.g. ``"massing.story.edit"`` or ``"markup.issue.assign"``.
    action: str
    #: Optional target -- a model id, project id, or record id.
    resource: str | None = None
    context: Mapping[str, Any] | None = None


@runtime_checkable
class PermissionEvaluator(Protocol):
    def evaluate(self, identity: Identity, request: PermissionRequest) -> Any:
        """Return ``bool`` or an awaitable resolving to one."""
        ...


class _AllowAll:
    __slots__ = ()

    def evaluate(self, identity: Identity, request: PermissionRequest) -> bool:
        return True


#: Default posture: permit everything.
#:
#: The kernel ships permission *hooks*, not a policy. A standalone desktop session has no
#: meaningful authorization boundary, and a kernel that denied by default would force every host to
#: write a policy before the first command could run. Hosts that need enforcement install their own
#: evaluator; the enforcement path is identical either way, so it is exercised from day one.
ALLOW_ALL: PermissionEvaluator = _AllowAll()


class _RoleEvaluator:
    __slots__ = ("_grants",)

    def __init__(self, grants: Mapping[str, Sequence[str]]) -> None:
        self._grants = grants

    def evaluate(self, identity: Identity, request: PermissionRequest) -> bool:
        allowed = self._grants.get(request.action, self._grants.get("*"))
        if allowed is None:
            return False
        return any(role == "*" or role in identity.roles for role in allowed)


def create_role_evaluator(grants: Mapping[str, Sequence[str]]) -> PermissionEvaluator:
    """Grant an action if the identity holds any role listed for it (exact match or ``*``)."""
    return _RoleEvaluator(grants)


class PermissionService:
    __slots__ = ("_identity", "_evaluator")

    def __init__(self) -> None:
        self._identity: Identity = ANONYMOUS_IDENTITY
        self._evaluator: PermissionEvaluator = ALLOW_ALL

    @property
    def identity(self) -> Identity:
        return self._identity

    def set_identity(self, identity: Identity) -> None:
        self._identity = identity

    def set_evaluator(self, evaluator: PermissionEvaluator) -> None:
        self._evaluator = evaluator

    async def can(self, request: PermissionRequest) -> bool:
        """Never raises, and never turns an evaluator failure into an allow.

        A policy that raises is an *indeterminate* answer, and the only safe reading of
        indeterminate in an authorization check is "no" -- failing open here would turn a bug in a
        host's policy into a silent privilege escalation.
        """
        try:
            return bool(await _resolve(self._evaluator.evaluate(self._identity, request)))
        except Exception:  # noqa: BLE001
            return False

    async def require(self, request: PermissionRequest) -> Result[None, KernelError]:
        if await self.can(request):
            return ok(None)
        details: dict[str, Any] = {"action": request.action, "identity": self._identity.id}
        if request.resource is not None:
            details["resource"] = request.resource
        return err(
            KernelError(
                "PERMISSION_DENIED",
                f'Identity "{self._identity.id}" may not "{request.action}".',
                details,
            )
        )
