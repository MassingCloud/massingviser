"""The kernel's load-bearing invariants.

Each test here corresponds to a promise the architecture makes. If one of these fails, a claim in
the README has stopped being true.
"""

from __future__ import annotations

import pytest

from massingviser.kernel import (
    CommandDefinition,
    CommandInvocation,
    DisposableStore,
    EventBus,
    KernelError,
    MemoryStorageAdapter,
    PermissionRequest,
    PersistenceEngine,
    ServiceContainer,
    StateStore,
    UIContribution,
    create_kernel,
    create_role_evaluator,
    create_service_token,
    satisfies,
    to_disposable,
)
from massingviser.kernel.permissions import Identity, PermissionService
from massingviser.schema import MigrationDefinition, MigrationRegistry
from massingviser.sdk import define_plugin

# ---------------------------------------------------------------------------------------------
# Disposables
# ---------------------------------------------------------------------------------------------


def test_dispose_is_idempotent():
    calls = []
    disposable = to_disposable(lambda: calls.append(1))
    disposable.dispose()
    disposable.dispose()
    assert calls == [1]


def test_store_drains_in_reverse_order():
    order = []
    store = DisposableStore()
    store.add(to_disposable(lambda: order.append("first")))
    store.add(to_disposable(lambda: order.append("second")))
    store.dispose()
    assert order == ["second", "first"]


def test_one_failing_disposable_does_not_strand_the_rest():
    drained = []

    def boom() -> None:
        raise RuntimeError("teardown failed")

    store = DisposableStore()
    store.add(to_disposable(lambda: drained.append("early")))
    store.add(to_disposable(boom))
    errors = store.dispose_collecting()
    assert drained == ["early"]
    assert len(errors) == 1


def test_adding_to_a_disposed_store_disposes_immediately():
    disposed = []
    store = DisposableStore()
    store.dispose()
    store.add(to_disposable(lambda: disposed.append(1)))
    assert disposed == [1]  # otherwise a late registration leaks forever


# ---------------------------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------------------------


def test_emit_never_raises_and_reports_handler_failures():
    bus = EventBus()
    seen = []
    bus.on("x", lambda _p: (_ for _ in ()).throw(ValueError("bad handler")))
    bus.on("x", lambda p: seen.append(p))

    report = bus.emit("x", 42)
    assert seen == [42]  # the good handler still ran
    assert report.delivered == 1
    assert len(report.errors) == 1


def test_handlers_are_snapshotted_before_dispatch():
    bus = EventBus()
    delivered = []

    subscription = {}

    def unsubscribing(_payload):
        subscription["handle"].dispose()
        delivered.append("first")

    subscription["handle"] = bus.on("x", unsubscribing)
    bus.on("x", lambda _p: delivered.append("second"))

    bus.emit("x", None)
    assert delivered == ["first", "second"]  # mutating mid-iteration must not skip the second


# ---------------------------------------------------------------------------------------------
# Service container
# ---------------------------------------------------------------------------------------------


def test_circular_dependency_names_the_cycle():
    container = ServiceContainer()
    a = create_service_token("a")
    b = create_service_token("b")
    container.register(a, lambda c: c.resolve(b))
    container.register(b, lambda c: c.resolve(a))

    with pytest.raises(KernelError) as caught:
        container.resolve(a)
    assert caught.value.code == "SERVICE_CIRCULAR"
    assert "a -> b -> a" in caught.value.message


def test_singleton_is_cached_on_its_owning_container():
    container = ServiceContainer()
    token = create_service_token("shared")
    built = []
    container.register(token, lambda _c: built.append(1) or object())

    scope_one = container.create_scope("one")
    scope_two = container.create_scope("two")
    assert scope_one.resolve(token) is scope_two.resolve(token)
    assert len(built) == 1


def test_tokens_are_nominal_not_name_based():
    container = ServiceContainer()
    mine = create_service_token("logger")
    theirs = create_service_token("logger")
    container.register(mine, lambda _c: "mine")
    container.register(theirs, lambda _c: "theirs")  # same name, no collision
    assert container.resolve(mine) == "mine"
    assert container.resolve(theirs) == "theirs"


# ---------------------------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------------------------


def test_no_op_write_does_not_wake_subscribers():
    store = StateStore()
    slice_ = store.define_slice("s", 1)
    seen = []
    slice_.subscribe(lambda n, p: seen.append((n, p)))
    slice_.set(1)
    assert seen == []
    slice_.set(2)
    assert seen == [(2, 1)]


def test_transaction_collapses_to_one_notification_with_the_earliest_previous():
    store = StateStore()
    slice_ = store.define_slice("s", 0)
    seen = []
    slice_.subscribe(lambda n, p: seen.append((n, p)))

    def edit():
        slice_.set(1)
        slice_.set(2)
        slice_.set(3)

    store.transaction(edit)
    assert seen == [(3, 0)]


def test_restore_parks_state_for_a_plugin_that_has_not_activated_yet():
    store = StateStore()
    store.restore({"later/records": (1, 2, 3)})
    assert store.snapshot()["later/records"] == (1, 2, 3)
    slice_ = store.define_slice("later/records", ())
    assert slice_.get() == (1, 2, 3)  # load order must not matter


def test_a_raising_subscriber_does_not_block_the_others():
    store = StateStore()
    slice_ = store.define_slice("s", 0)
    seen = []
    slice_.subscribe(lambda _n, _p: (_ for _ in ()).throw(RuntimeError()))
    slice_.subscribe(lambda n, _p: seen.append(n))
    slice_.set(1)
    assert seen == [1]


# ---------------------------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------------------------


async def test_execute_returns_err_instead_of_raising():
    kernel = create_kernel()

    def boom(_params, _ctx):
        raise ValueError("handler exploded")

    kernel.commands.register(CommandDefinition(id="t.boom", handler=boom))
    result = await kernel.commands.execute("t.boom")
    assert not result.ok
    assert result.error.code == "COMMAND_FAILED"
    assert "handler exploded" in result.error.message


async def test_unknown_command_is_a_result_not_an_exception():
    kernel = create_kernel()
    result = await kernel.commands.execute("nope")
    assert not result.ok
    assert result.error.code == "COMMAND_NOT_FOUND"


async def test_undo_redo_round_trip():
    kernel = create_kernel()
    value = {"n": 0}

    def set_value(params, _ctx):
        previous = value["n"]
        value["n"] = params["to"]
        return previous

    kernel.commands.register(
        CommandDefinition(
            id="t.set",
            handler=set_value,
            create_inverse=lambda _p, previous: CommandInvocation("t.set", {"to": previous}),
        )
    )

    await kernel.commands.execute("t.set", {"to": 5})
    assert value["n"] == 5
    await kernel.commands.undo()
    assert value["n"] == 0
    await kernel.commands.redo()
    assert value["n"] == 5


async def test_a_fresh_action_invalidates_the_redo_branch():
    kernel = create_kernel()
    kernel.commands.register(
        CommandDefinition(
            id="t.x",
            handler=lambda _p, _c: None,
            create_inverse=lambda _p, _r: CommandInvocation("t.x", {}),
        )
    )
    await kernel.commands.execute("t.x")
    await kernel.commands.undo()
    assert kernel.commands.can_redo
    await kernel.commands.execute("t.x")
    assert not kernel.commands.can_redo


async def test_nested_commands_do_not_enter_the_history():
    kernel = create_kernel()

    async def composite(_params, _ctx):
        await kernel.commands.execute("t.child", {})

    kernel.commands.register(
        CommandDefinition(
            id="t.child",
            handler=lambda _p, _c: None,
            create_inverse=lambda _p, _r: CommandInvocation("t.child", {}),
        )
    )
    kernel.commands.register(
        CommandDefinition(
            id="t.parent",
            handler=composite,
            create_inverse=lambda _p, _r: CommandInvocation("t.parent", {}),
        )
    )

    await kernel.commands.execute("t.parent", {})
    # One user action must be one undo step, not two.
    assert kernel.commands.history_size["undo"] == 1


async def test_unregistering_a_command_purges_its_history_entries():
    kernel = create_kernel()
    subscription = kernel.commands.register(
        CommandDefinition(
            id="t.x",
            handler=lambda _p, _c: None,
            create_inverse=lambda _p, _r: CommandInvocation("t.x", {}),
        )
    )
    await kernel.commands.execute("t.x")
    assert kernel.commands.can_undo
    subscription.dispose()
    assert not kernel.commands.can_undo


async def test_middleware_that_raises_fails_only_the_command():
    kernel = create_kernel()
    kernel.commands.register(CommandDefinition(id="t.x", handler=lambda _p, _c: "ok"))

    async def broken(_invocation, _next):
        raise RuntimeError("middleware exploded")

    kernel.commands.use(broken)
    result = await kernel.commands.execute("t.x")
    assert not result.ok
    assert "middleware exploded" in result.error.message


# ---------------------------------------------------------------------------------------------
# Permissions
# ---------------------------------------------------------------------------------------------


async def test_a_policy_that_raises_denies_rather_than_allows():
    class Exploding:
        def evaluate(self, _identity, _request):
            raise RuntimeError("policy bug")

    service = PermissionService()
    service.set_evaluator(Exploding())
    assert await service.can(PermissionRequest(action="anything")) is False


async def test_role_evaluator_gates_a_command():
    kernel = create_kernel(
        permission_evaluator=create_role_evaluator({"t.restricted": ["admin"]}),
        identity=Identity(id="u1", roles=("viewer",)),
    )
    kernel.commands.register(
        CommandDefinition(id="t.x", permission="t.restricted", handler=lambda _p, _c: "done")
    )
    denied = await kernel.commands.execute("t.x")
    assert not denied.ok and denied.error.code == "PERMISSION_DENIED"

    kernel.permissions.set_identity(Identity(id="u2", roles=("admin",)))
    allowed = await kernel.commands.execute("t.x")
    assert allowed.ok and allowed.value == "done"


# ---------------------------------------------------------------------------------------------
# Plugin host
# ---------------------------------------------------------------------------------------------


async def test_a_plugin_that_raises_is_quarantined_and_rolled_back():
    kernel = create_kernel()

    def activate(context):
        context.commands.register(CommandDefinition(id="broken.partial", handler=lambda *_: None))
        context.ui.register(UIContribution(id="broken.panel", point="panel"))
        raise RuntimeError("activation exploded")

    kernel.use(define_plugin(id="broken", version="1.0.0", activate=activate))
    report = await kernel.start()

    assert report.failed and report.failed[0][0] == "broken"
    assert kernel.plugins.status("broken") == "quarantined"
    # Everything it managed to register before failing is gone.
    assert not kernel.commands.has("broken.partial")
    assert kernel.ui.by_point("panel") == ()


async def test_a_quarantined_plugin_is_not_retried_until_reset():
    kernel = create_kernel()
    attempts = []

    def activate(_context):
        attempts.append(1)
        raise RuntimeError("nope")

    kernel.use(define_plugin(id="broken", version="1.0.0", activate=activate))
    await kernel.start()
    retry = await kernel.plugins.activate("broken")
    assert not retry.ok and retry.error.code == "PLUGIN_QUARANTINED"
    assert len(attempts) == 1

    kernel.plugins.reset("broken")
    await kernel.plugins.activate("broken")
    assert len(attempts) == 2


async def test_deactivation_releases_every_registration():
    kernel = create_kernel()

    def activate(context):
        context.commands.register(CommandDefinition(id="p.cmd", handler=lambda *_: None))
        context.ui.register(UIContribution(id="p.panel", point="panel"))
        context.events.on("something", lambda _p: None)
        context.state.define_slice("records", ())

    kernel.use(define_plugin(id="p", version="1.0.0", activate=activate))
    await kernel.start()
    assert kernel.commands.has("p.cmd")

    await kernel.plugins.deactivate("p")
    assert not kernel.commands.has("p.cmd")
    assert kernel.ui.by_point("panel") == ()
    assert kernel.events.listener_count("something") == 0
    assert "p/records" not in kernel.state.snapshot()


async def test_a_dependency_cycle_is_reported_with_the_cycle():
    from massingviser.kernel import PluginDependency

    kernel = create_kernel()
    kernel.use(
        define_plugin(
            id="a",
            version="1.0.0",
            dependencies=[PluginDependency(id="b")],
            activate=lambda _c: None,
        )
    )
    kernel.use(
        define_plugin(
            id="b",
            version="1.0.0",
            dependencies=[PluginDependency(id="a")],
            activate=lambda _c: None,
        )
    )
    report = await kernel.start()
    assert report.failed[0][1].code == "PLUGIN_DEPENDENCY_CYCLE"
    assert "->" in report.failed[0][1].message


async def test_an_incompatible_api_version_is_refused_at_registration():
    kernel = create_kernel()
    result = kernel.use(
        define_plugin(id="future", version="1.0.0", api_version="^99.0.0", activate=lambda _c: None)
    )
    assert not result.ok
    assert result.error.code == "PLUGIN_API_INCOMPATIBLE"


async def test_dependents_activate_after_their_dependency():
    from massingviser.kernel import PluginDependency

    kernel = create_kernel()
    order = []
    kernel.use(
        define_plugin(
            id="consumer",
            version="1.0.0",
            dependencies=[PluginDependency(id="provider")],
            activate=lambda _c: order.append("consumer"),
        )
    )
    kernel.use(
        define_plugin(id="provider", version="1.0.0", activate=lambda _c: order.append("provider"))
    )
    await kernel.start()
    assert order == ["provider", "consumer"]


# ---------------------------------------------------------------------------------------------
# Persistence and migration
# ---------------------------------------------------------------------------------------------


async def test_a_document_from_a_newer_build_is_refused_not_misread():
    registry = MigrationRegistry().declare("demo", 1)
    storage = MemoryStorageAdapter()
    engine = PersistenceEngine(adapter=storage, migrator=registry)

    await engine.save("k", "demo", {"a": 1}, version=7)
    loaded = await engine.load("k")
    assert not loaded.ok
    assert loaded.error.code == "SCHEMA_VERSION_UNSUPPORTED"


async def test_load_migrates_without_rewriting_storage():
    registry = MigrationRegistry().register(
        MigrationDefinition(
            schema="demo",
            from_version=1,
            to_version=2,
            migrate=lambda data: {**data, "added": True},
        )
    )
    storage = MemoryStorageAdapter()
    engine = PersistenceEngine(adapter=storage, migrator=registry)
    await engine.save("k", "demo", {"a": 1}, version=1)

    loaded = await engine.load("k")
    assert loaded.value.version == 2 and loaded.value.data["added"] is True

    raw = await storage.get("k")
    assert raw["version"] == 1  # reads are pure: opening a project must not mutate it


async def test_migrate_in_place_persists_and_backs_up_first():
    registry = MigrationRegistry().register(
        MigrationDefinition(
            schema="demo", from_version=1, to_version=2, migrate=lambda data: {"v": 2}
        )
    )
    engine = PersistenceEngine(adapter=MemoryStorageAdapter(), migrator=registry)
    await engine.save("k", "demo", {"v": 1}, version=1)

    await engine.migrate_in_place("k")
    assert (await engine.load("k")).value.data == {"v": 2}
    assert len(await engine.list_backups("k")) == 1


def test_duplicate_migration_out_of_one_version_is_refused():
    registry = MigrationRegistry().register(
        MigrationDefinition(schema="d", from_version=1, to_version=2, migrate=lambda d: d)
    )
    with pytest.raises(KernelError) as caught:
        registry.register(
            MigrationDefinition(schema="d", from_version=1, to_version=3, migrate=lambda d: d)
        )
    assert caught.value.code == "MIGRATION_FAILED"


def test_a_migration_that_raises_fails_one_document_not_the_project():
    registry = MigrationRegistry().register(
        MigrationDefinition(
            schema="d",
            from_version=1,
            to_version=2,
            migrate=lambda _d: (_ for _ in ()).throw(KeyError("missing field")),
        )
    )
    from massingviser.kernel import VersionedDocument

    result = registry.migrate(VersionedDocument("d", 1, "", {}))
    assert not result.ok and result.error.code == "MIGRATION_FAILED"


# ---------------------------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------------------------


async def test_closing_over_unsaved_work_is_refused_unless_forced():
    from massingviser.kernel import ContainerCreateInit

    kernel = create_kernel()
    created = await kernel.containers.create(
        "massingviser.project", ContainerCreateInit(container_id="p1", name="Tower")
    )
    await created.value.write_document("project.json", "massingifc.project", {"name": "Tower"})

    refused = await kernel.containers.close()
    assert not refused.ok and "unsaved changes" in refused.error.message
    assert (await kernel.containers.close(force=True)).ok


async def test_a_container_round_trips_documents_and_blobs():
    from massingviser.kernel import ContainerCreateInit, ContainerSource

    storage = MemoryStorageAdapter()
    kernel = create_kernel(storage=storage)
    container = (
        await kernel.containers.create(
            "massingviser.project", ContainerCreateInit(container_id="p1", name="Tower")
        )
    ).value
    await container.write_document("project.json", "massingifc.project", {"name": "Tower"})
    await container.write_blob("models/tower.bin", b"\x00\x01\x02", "application/octet-stream")
    saved = await kernel.containers.save()
    assert saved.ok
    assert kernel.containers.active.dirty is False  # saving is not editing

    await kernel.containers.close()
    reopened = await kernel.containers.open(ContainerSource(name="p1"))
    assert reopened.ok
    document = (await reopened.value.read_document("project.json")).value
    assert document.data == {"name": "Tower"}
    assert (await reopened.value.read_blob("models/tower.bin")).value == b"\x00\x01\x02"
    assert reopened.value.dirty is False


# ---------------------------------------------------------------------------------------------
# Semver
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("version", "range_", "expected"),
    [
        ("1.2.3", "^1.0.0", True),
        ("2.0.0", "^1.0.0", False),
        ("0.4.0", "^0.3.1", False),  # below 1.0.0 the minor is the breaking axis
        ("0.3.5", "^0.3.1", True),
        ("1.0.0", ">=1.0.0", True),
        ("1.0.0-beta", "1.0.0", False),
        ("1.2.3", "*", True),
        ("not-a-version", "^1.0.0", False),
    ],
)
def test_semver_ranges(version, range_, expected):
    assert satisfies(version, range_) is expected
