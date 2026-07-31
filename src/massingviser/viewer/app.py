"""The viser shell.

``massingifc`` ships no viewer on purpose, and its ``ui-shell`` package is the *bookkeeping* half
of one -- which panels exist, which are open, what the layout was -- leaving rendering to a host.
This module is that host, and it is the only file in MassingViser that imports viser.

The important property is that it renders the **UI extension registry** rather than a hard-coded
layout. A plugin that contributes a panel gets a tab; a plugin that contributes a toolbar button
gets a button wired to its command id. Panels with a bespoke renderer get one; anything else falls
back to a generic surface built from that plugin's own registered commands. Installing a new
capability plugin therefore changes the browser UI without this file being touched, which is the
whole point of the architecture underneath.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import viser

from ..kernel import Kernel, KernelError, Result, UIContribution
from ..plugins.estimating import (
    ESTIMATING_COMMANDS,
    EstimateToken,
)
from ..plugins.markup import MARKUP_COMMANDS, IssueToken
from ..plugins.massing import (
    MASSING_COMMANDS,
    MassingToken,
    MetricsToken,
    ProfileToken,
)
from .bridge import KernelBridge
from .scene import DEFAULT_COLOR, SceneSync

PanelRenderer = Callable[["MassingViserApp", Any], None]


@dataclass
class MassingViserApp:
    """A browser session over a running kernel."""

    bridge: KernelBridge
    server: viser.ViserServer
    title: str = "MassingViser"
    _scene: SceneSync | None = None
    _selected_mass: str | None = None
    _status: Any = None
    _rerender: list[Callable[[], None]] = field(default_factory=list)
    _panel_bodies: dict[str, Any] = field(default_factory=dict)

    # -- lifecycle ---------------------------------------------------------------------------

    def build(self) -> MassingViserApp:
        self.server.gui.configure_theme(
            titlebar_content=None, control_layout="collapsible", dark_mode=True
        )
        self.server.gui.set_panel_label(self.title)

        self.server.scene.add_grid(
            "/ground", width=200.0, height=200.0, cell_size=5.0, section_size=25.0, plane="xy"
        )
        self.server.scene.set_up_direction("+z")

        self._scene = SceneSync(server=self.server, bridge=self.bridge, on_select=self._select_mass)

        self._build_header()
        self._build_panels()

        # A single kernel-side observer drives every redraw. Subscribing per-record would mean
        # remembering to unsubscribe on delete, and a leaked subscription on a deleted mass is a
        # redraw of geometry that no longer exists.
        self.bridge.read(lambda: self.bridge.kernel.events.observe(self._on_kernel_event))

        self.refresh()
        return self

    def _on_kernel_event(self, event_type: str, _payload: Any) -> None:
        # Runs on the kernel thread. Only cheap flags here -- redrawing from inside an emit would
        # re-enter the state store mid-notification.
        if event_type.startswith(("massing.", "markup.", "estimating.")):
            self._dirty = True

    _dirty: bool = False

    def refresh(self) -> None:
        if self._scene is not None:
            self._scene.sync()
        for callback in list(self._rerender):
            try:
                callback()
            except Exception:  # noqa: BLE001
                pass  # one broken panel must not blank the rest of the UI
        self._dirty = False

    def poll(self) -> None:
        """Redraw if a kernel event has landed since the last pass."""
        if self._dirty:
            self.refresh()

    # -- header ------------------------------------------------------------------------------

    def _build_header(self) -> None:
        diagnostics = self.bridge.read(self.bridge.kernel.diagnostics)
        active = [record for record in diagnostics.plugins if record.status == "active"]

        with self.server.gui.add_folder("Session", order=0):
            self.server.gui.add_markdown(
                f"**Kernel** `{diagnostics.api_version}` &nbsp; "
                f"**Plugins** {len(active)}/{len(diagnostics.plugins)} &nbsp; "
                f"**Commands** {diagnostics.commands}"
            )
            self._status = self.server.gui.add_markdown("Ready.")

            undo = self.server.gui.add_button("Undo", icon=viser.Icon.ARROW_BACK_UP)
            redo = self.server.gui.add_button("Redo", icon=viser.Icon.ARROW_FORWARD_UP)

            @undo.on_click
            def _(_event: Any) -> None:
                self._report(self.bridge.undo(), "Undone.")
                self.refresh()

            @redo.on_click
            def _(_event: Any) -> None:
                self._report(self.bridge.redo(), "Redone.")
                self.refresh()

        # Toolbar contributions become buttons. They carry a command id and nothing else, so the
        # shell can honour them without knowing what any of them do.
        toolbar = self.bridge.read(lambda: self.bridge.kernel.ui.by_point("toolbar"))
        if toolbar:
            with self.server.gui.add_folder("Toolbar", order=1, expand_by_default=False):
                for contribution in toolbar:
                    self._toolbar_button(contribution)

    def _toolbar_button(self, contribution: UIContribution) -> None:
        button = self.server.gui.add_button(contribution.title or contribution.id)

        @button.on_click
        def _(_event: Any) -> None:
            if not contribution.command_id:
                self._say(f"{contribution.title} has no command bound.")
                return
            # A contribution declares a command, not its arguments. Firing one that needs
            # parameters would fail confusingly, so the shell points at the panel that can.
            self._say(
                f"`{contribution.command_id}` -- use the "
                f"{(contribution.group or 'relevant').title()} panel to supply its inputs."
            )

    # -- panels ------------------------------------------------------------------------------

    def _build_panels(self) -> None:
        contributions = self.bridge.read(lambda: self.bridge.kernel.ui.by_point("panel"))
        renderers: Mapping[str, PanelRenderer] = {
            "massing.panel": MassingViserApp._render_massing_panel,
            "markup.panel": MassingViserApp._render_issues_panel,
            "estimating.panel": MassingViserApp._render_cost_panel,
        }

        for contribution in contributions:
            with self.server.gui.add_folder(
                contribution.title or contribution.id,
                order=10 + contribution.order,
                expand_by_default=contribution.id == "massing.panel",
            ) as folder:
                renderer = renderers.get(contribution.id)
                if renderer is not None:
                    renderer(self, folder)
                else:
                    self._render_generic_panel(contribution)

        with self.server.gui.add_folder("Diagnostics", order=900, expand_by_default=False):
            self._render_diagnostics()

    def _render_generic_panel(self, contribution: UIContribution) -> None:
        """Fallback surface for a panel with no bespoke renderer.

        Lists the contributing plugin's own commands. It is not a designed UI, but it means a newly
        installed capability is *reachable* the moment it activates rather than invisible until
        somebody writes a panel for it.
        """
        plugin_id = contribution.plugin_id or ""
        commands = self.bridge.read(self.bridge.kernel.commands.list)
        prefix = plugin_id.rsplit(".", 1)[-1]
        mine = [command for command in commands if command.id.startswith(prefix)]

        if not mine:
            self.server.gui.add_markdown(f"_{contribution.title} contributed no commands._")
            return
        self.server.gui.add_markdown(f"_Generic surface for `{plugin_id}`. {len(mine)} commands._")
        for command in mine[:12]:
            self.server.gui.add_markdown(f"- `{command.id}` -- {command.title or ''}")

    # -- massing panel -----------------------------------------------------------------------

    def _render_massing_panel(self, _folder: Any) -> None:
        gui = self.server.gui

        footprints: dict[str, list[tuple[float, float, float]]] = {
            "Rectangle 40 x 24": [(0, 0, 0), (40, 0, 0), (40, 24, 0), (0, 24, 0)],
            "L-shape": [(0, 0, 0), (44, 0, 0), (44, 18, 0), (20, 18, 0), (20, 40, 0), (0, 40, 0)],
            "Courtyard block": [(0, 0, 0), (50, 0, 0), (50, 40, 0), (0, 40, 0)],
            "Tower 26 x 26": [(0, 0, 0), (26, 0, 0), (26, 26, 0), (0, 26, 0)],
        }

        name = gui.add_text("Name", initial_value="Block A")
        shape = gui.add_dropdown("Footprint", tuple(footprints), initial_value="Rectangle 40 x 24")
        storeys = gui.add_slider("Storeys", min=1, max=60, step=1, initial_value=12)
        storey_height = gui.add_slider(
            "Floor to floor (m)", min=2.4, max=6.0, step=0.1, initial_value=3.5
        )
        offset = gui.add_vector2("Position (m)", initial_value=(0.0, 0.0), step=1.0)
        colour = gui.add_rgb("Colour", initial_value=DEFAULT_COLOR)
        create = gui.add_button("Create mass", icon=viser.Icon.BOX)

        @create.on_click
        def _(_event: Any) -> None:
            points = footprints[shape.value]
            dx, dy = offset.value
            moved = [(x + dx, y + dy, z) for x, y, z in points]

            sketched = self.bridge.execute(
                MASSING_COMMANDS.sketch_profile, {"points": moved, "name": shape.value}
            )
            if not sketched.ok:
                self._report(sketched, "")
                return

            if shape.value == "Courtyard block":
                # Adds a real hole, which exercises the bridged-hole tessellation path rather than
                # only ever drawing convex blocks.
                profiles = self.bridge.read(
                    lambda: self.bridge.kernel.capabilities.get(ProfileToken)
                )
                hole = [
                    (15 + dx, 12 + dy, 0),
                    (35 + dx, 12 + dy, 0),
                    (35 + dx, 28 + dy, 0),
                    (15 + dx, 28 + dy, 0),
                ]
                self.bridge.run(profiles.add_hole(sketched.value, hole))

            red, green, blue = colour.value
            created = self.bridge.execute(
                MASSING_COMMANDS.create_mass,
                {
                    "name": name.value,
                    "profile_id": sketched.value,
                    "story_count": int(storeys.value),
                    "story_height": float(storey_height.value),
                    "color": f"#{red:02x}{green:02x}{blue:02x}",
                },
            )
            if created.ok:
                self._selected_mass = created.value.id
                self._say(f"Created **{created.value.name}**.")
            else:
                self._report(created, "")
            self.refresh()

        gui.add_markdown("---")

        selection = gui.add_dropdown("Selected", ("(none)",), initial_value="(none)")
        edit_storeys = gui.add_slider("Storeys (selected)", min=1, max=60, step=1, initial_value=12)
        edit_opacity = gui.add_slider("Opacity", min=0.1, max=1.0, step=0.05, initial_value=1.0)
        duplicate = gui.add_button("Duplicate", icon=viser.Icon.COPY)
        delete = gui.add_button("Delete", icon=viser.Icon.TRASH, color="red")
        metrics_view = gui.add_markdown("_No mass selected._")

        # Guard so programmatic writes to the widgets (during a refresh) do not fire the handlers
        # and issue a command the user never asked for.
        syncing = {"value": False}

        def options() -> list[str]:
            service = self.bridge.read(lambda: self.bridge.kernel.capabilities.get(MassingToken))
            if service is None:
                return []
            return [mass.id for mass in self.bridge.read(service.list)]

        def redraw_selection() -> None:
            ids = options()
            syncing["value"] = True
            try:
                selection.options = tuple(ids) if ids else ("(none)",)
                if self._selected_mass not in ids:
                    self._selected_mass = ids[0] if ids else None
                selection.value = self._selected_mass or "(none)"

                mass = self._current_mass()
                if mass is None:
                    metrics_view.content = "_No mass selected._"
                    return
                edit_storeys.value = mass.story_count
                edit_opacity.value = mass.opacity if mass.opacity is not None else 1.0
                metrics_view.content = self._metrics_markdown(mass.id)
            finally:
                syncing["value"] = False

        self._rerender.append(redraw_selection)

        @selection.on_update
        def _(_event: Any) -> None:
            if syncing["value"]:
                return
            self._selected_mass = None if selection.value == "(none)" else selection.value
            redraw_selection()

        @edit_storeys.on_update
        def _(_event: Any) -> None:
            if syncing["value"] or self._selected_mass is None:
                return
            self._report(
                self.bridge.execute(
                    MASSING_COMMANDS.set_story_count,
                    {"id": self._selected_mass, "count": int(edit_storeys.value)},
                ),
                f"Set to {int(edit_storeys.value)} storeys.",
            )
            self.refresh()

        @edit_opacity.on_update
        def _(_event: Any) -> None:
            if syncing["value"] or self._selected_mass is None:
                return
            self._report(
                self.bridge.execute(
                    MASSING_COMMANDS.set_opacity,
                    {"id": self._selected_mass, "opacity": float(edit_opacity.value)},
                ),
                "",
            )
            self.refresh()

        @duplicate.on_click
        def _(_event: Any) -> None:
            if self._selected_mass is None:
                return
            result = self.bridge.execute(
                MASSING_COMMANDS.duplicate_mass, {"id": self._selected_mass}
            )
            if result.ok:
                self._selected_mass = result.value.id
            self._report(result, "Duplicated.")
            self.refresh()

        @delete.on_click
        def _(_event: Any) -> None:
            if self._selected_mass is None:
                return
            self._report(
                self.bridge.execute(MASSING_COMMANDS.remove_mass, {"id": self._selected_mass}),
                "Deleted -- Undo restores it exactly.",
            )
            self._selected_mass = None
            self.refresh()

        gui.add_markdown("---")
        site = gui.add_button("Set 80 x 60 site (FAR limit 4.0)")

        @site.on_click
        def _(_event: Any) -> None:
            self._report(
                self.bridge.execute(
                    MASSING_COMMANDS.set_site_boundary,
                    {
                        "points": [(-10, -10, 0), (70, -10, 0), (70, 50, 0), (-10, 50, 0)],
                        "name": "Site",
                        "max_floor_area_ratio": 4.0,
                        "max_height": 80.0,
                    },
                ),
                "Site boundary set; floor area ratio is now reported.",
            )
            self.refresh()

    def _current_mass(self) -> Any:
        if self._selected_mass is None:
            return None
        service = self.bridge.read(lambda: self.bridge.kernel.capabilities.get(MassingToken))
        if service is None:
            return None
        return self.bridge.read(lambda: service.get(self._selected_mass))

    def _metrics_markdown(self, mass_id: str) -> str:
        metrics = self.bridge.read(lambda: self.bridge.kernel.capabilities.get(MetricsToken))
        if metrics is None:
            return "_Massing metrics unavailable._"
        computed = self.bridge.run(metrics.compute(mass_id))
        if not computed.ok:
            return f"_{computed.error.message}_"
        value = computed.value
        far = (
            f"{value.floor_area_ratio:.2f}"
            if value.floor_area_ratio is not None
            else "not set (no site)"
        )
        return (
            f"| | |\n|---|---:|\n"
            f"| Footprint | {value.footprint_area:,.0f} m² |\n"
            f"| Gross floor area | {value.gross_floor_area:,.0f} m² |\n"
            f"| Volume | {value.volume:,.0f} m³ |\n"
            f"| Envelope | {value.envelope_area:,.0f} m² |\n"
            f"| Height | {value.height:,.1f} m |\n"
            f"| Storeys | {value.story_count} |\n"
            f"| Plot ratio | {far} |\n"
        )

    # -- issues panel ------------------------------------------------------------------------

    def _render_issues_panel(self, _folder: Any) -> None:
        gui = self.server.gui
        title = gui.add_text("Title", initial_value="Coordinate core with structure")
        priority = gui.add_dropdown(
            "Priority", ("low", "medium", "high", "critical"), initial_value="medium"
        )
        raise_issue = gui.add_button("Raise issue", icon=viser.Icon.FLAG)
        listing = gui.add_markdown("_No issues._")

        def redraw() -> None:
            service = self.bridge.read(lambda: self.bridge.kernel.capabilities.get(IssueToken))
            if service is None:
                listing.content = "_Markup capability not installed._"
                return
            issues = self.bridge.read(service.query)
            if not issues:
                listing.content = "_No issues._"
                return
            rows = "\n".join(
                f"- **{issue.title}** — `{issue.status}`"
                + (f" · {issue.priority}" if issue.priority else "")
                for issue in issues
            )
            listing.content = f"{len(issues)} issue(s):\n\n{rows}"

        self._rerender.append(redraw)

        @raise_issue.on_click
        def _(_event: Any) -> None:
            self._report(
                self.bridge.execute(
                    MARKUP_COMMANDS.create_issue,
                    {"title": title.value, "priority": priority.value},
                ),
                "Issue raised.",
            )
            self.refresh()

        gui.add_markdown("---")
        pin = gui.add_button("Pin on selected mass")

        @pin.on_click
        def _(_event: Any) -> None:
            if self._selected_mass is None:
                self._say("Select a mass first.")
                return
            self._report(
                self.bridge.execute(
                    MARKUP_COMMANDS.create,
                    {
                        "kind": "pin",
                        "model_id": "massing",
                        # A GlobalId, never the viewer's handle -- a pin anchored to a transient id
                        # survives exactly one session.
                        "element_ids": [self._selected_mass],
                        "text": title.value,
                    },
                ),
                "Pin added.",
            )
            self.refresh()

    # -- cost panel --------------------------------------------------------------------------

    def _render_cost_panel(self, _folder: Any) -> None:
        gui = self.server.gui
        rate = gui.add_number("Rate £/m³ (concrete frame)", initial_value=420.0, step=10.0)
        contingency = gui.add_slider(
            "Contingency %", min=0.0, max=25.0, step=0.5, initial_value=7.5
        )
        run = gui.add_button("Price the scheme", icon=viser.Icon.CALCULATOR)
        output = gui.add_markdown("_Not priced yet._")

        def redraw() -> None:
            service = self.bridge.read(lambda: self.bridge.kernel.capabilities.get(EstimateToken))
            if service is None:
                return
            estimates = self.bridge.read(service.list)
            if not estimates:
                return
            latest = estimates[-1]
            output.content = (
                f"**{latest.name}** — `{latest.status}`\n\n"
                f"| | |\n|---|---:|\n"
                f"| Subtotal | {latest.subtotal} |\n"
                f"| Contingency | {latest.contingency_percent:.1f}% |\n"
                f"| **Total** | **{latest.total}** |\n\n"
                f"_Basis: {', '.join(f'{v.model_id}@{v.version}' for v in latest.basis_model_versions) or 'n/a'}_"
            )

        self._rerender.append(redraw)

        @run.on_click
        def _(_event: Any) -> None:
            self._price(rate.value, contingency.value, output)
            self.refresh()

    def _price(self, rate_major: float, contingency: float, output: Any) -> None:
        from ..schema import Money

        # Every step is an ordinary command on the bus, so pricing is audited, permission-checked
        # and recorded exactly like a geometry edit. Fixed ids make each step an upsert, so
        # re-pricing after a design change replaces the previous run instead of stacking on it.
        setup: list[tuple[str, Any]] = [
            (
                ESTIMATING_COMMANDS.add_rule,
                {
                    "id": "rule-frame",
                    "name": "Superstructure volume",
                    "metric": "NetVolume",
                    "unit": "m3",
                    "filter": {"ifc_class": "IfcBuildingStorey"},
                    # Evaluated per storey by the safe parser -- never eval'd.
                    "expression": "Area * Height",
                },
            ),
            (
                ESTIMATING_COMMANDS.add_resource,
                {
                    "id": "res-frame",
                    "name": "Concrete frame",
                    "type": "material",
                    "unit": "m3",
                    "rate": Money(int(round(rate_major * 100)), "GBP"),
                },
            ),
            (
                ESTIMATING_COMMANDS.add_assembly,
                {
                    "id": "asm-frame",
                    "code": "SUPERSTRUCTURE",
                    "name": "Frame and envelope allowance",
                    "unit": "m3",
                    "components": [
                        {"resource_id": "res-frame", "factor": 1.0, "waste_percent": 5.0}
                    ],
                    "overhead_percent": 12.0,
                    "profit_percent": 6.0,
                },
            ),
            (ESTIMATING_COMMANDS.run_takeoff, {}),
            (ESTIMATING_COMMANDS.create_boq, {"name": "Concept bill", "currency": "GBP"}),
        ]

        boq_id: str | None = None
        for command_id, params in setup:
            result = self.bridge.execute(command_id, params)
            if not result.ok:
                self._report(result, "")
                return
            if command_id == ESTIMATING_COMMANDS.create_boq:
                boq_id = result.value.id
            if command_id == ESTIMATING_COMMANDS.run_takeoff:
                summary = result.value
                if not summary.quantities:
                    self._say(
                        "Nothing to price -- create a mass first. "
                        f"({summary.rules_run} rule(s) matched no element.)"
                    )
                    return

        assert boq_id is not None
        for command_id, params in (
            (
                ESTIMATING_COMMANDS.generate_boq,
                {"boq_id": boq_id, "assembly_by_code": {"SUPERSTRUCTURE": "asm-frame"}},
            ),
            (
                ESTIMATING_COMMANDS.create_estimate,
                {
                    "name": "Concept estimate",
                    "boq_id": boq_id,
                    "contingency_percent": float(contingency),
                },
            ),
        ):
            result = self.bridge.execute(command_id, params)
            if not result.ok:
                self._report(result, "")
                return

        self._say(f"Priced: **{result.value.total}**.")

    # -- diagnostics -------------------------------------------------------------------------

    def _render_diagnostics(self) -> None:
        view = self.server.gui.add_markdown("")

        def redraw() -> None:
            diagnostics = self.bridge.read(self.bridge.kernel.diagnostics)
            plugins = "\n".join(
                f"- `{record.manifest.id}` **{record.status}**"
                + (f" — {record.error.message}" if record.error else "")
                for record in diagnostics.plugins
            )
            capabilities = "\n".join(f"- `{token}`" for token in sorted(diagnostics.capabilities))
            view.content = (
                f"**Plugins**\n\n{plugins or '_none_'}\n\n"
                f"**Capabilities ({len(diagnostics.capabilities)})**\n\n"
                f"{capabilities or '_none_'}\n\n"
                f"**Undo** {diagnostics.history['undo']} · "
                f"**Redo** {diagnostics.history['redo']} · "
                f"**Commands** {diagnostics.commands}\n\n"
                f"**State namespaces**\n\n"
                + "\n".join(f"- `{ns}`" for ns in sorted(diagnostics.state_namespaces))
            )

        self._rerender.append(redraw)

    # -- helpers -----------------------------------------------------------------------------

    def _select_mass(self, mass_id: str) -> None:
        self._selected_mass = mass_id
        self._say(f"Selected `{mass_id}`.")
        self.refresh()

    def _say(self, message: str) -> None:
        if self._status is not None:
            self._status.content = message

    def _report(self, result: Result[Any, KernelError], success: str) -> None:
        """Surface a command outcome.

        Failures are shown, not swallowed. The bus returns them as values precisely so a shell can
        do this instead of discovering them in a log.
        """
        if result.ok:
            if success:
                self._say(success)
        else:
            self._say(f"⚠️ {result.error.message}")


def serve(
    kernel: Kernel[Any],
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    title: str = "MassingViser",
    start_plugins: bool = True,
    block: bool = True,
) -> MassingViserApp:
    """Run a kernel behind a viser viewer.

    Binds to loopback by default: the scene is an unauthenticated project model, and defaulting to
    ``0.0.0.0`` the way a bare ``ViserServer`` does would put it on every interface on the machine.
    Pass ``host="0.0.0.0"`` deliberately if that is what you want.
    """
    bridge = KernelBridge(kernel)
    if start_plugins:
        report = bridge.start()
        for plugin_id, error in report.failed:
            print(f"[massingviser] plugin {plugin_id} failed: {error.message}")

    server = viser.ViserServer(host=host, port=port, label=title)
    app = MassingViserApp(bridge=bridge, server=server, title=title).build()

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.position = (95.0, -85.0, 65.0)
        client.camera.look_at = (20.0, 12.0, 18.0)
        app.refresh()

    if block:
        try:
            while True:
                time.sleep(0.25)
                app.poll()
        except KeyboardInterrupt:
            pass
        finally:
            bridge.close()
    return app
