"""``massingviser.plugins.analytics`` -- aggregating what the other families know.

Analytics owns no numbers of its own. Every metric arrives from a provider capability, which is
what keeps this package from becoming the place where a copy of the GFA calculation quietly
diverges from the one in massing.

Forecasts carry **bounds**, always. A single projected number invites a decision it cannot support;
the interval is the honest part of the answer.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from ...kernel import (
    CapabilityToken,
    CommandDefinition,
    KernelError,
    PluginContext,
    Result,
    UIContribution,
    create_capability_token,
    err,
    ok,
)
from ...schema import Id, IsoTimestamp
from ...sdk import Clock, IdFactory, RecordStore, SequentialIdFactory, SystemClock
from ...sdk import create_record_store, define_plugin

PLUGIN_ID = "massingviser.analytics"
PLUGIN_VERSION = "0.1.0"


@dataclass(frozen=True)
class MetricValue:
    key: str
    value: float
    unit: str | None = None
    label: str | None = None


@runtime_checkable
class MetricProvider(Protocol):
    """Contributes metrics. Many-to-one: every family that knows a number registers one."""

    @property
    def namespace(self) -> str: ...
    def collect(self) -> Sequence[MetricValue]: ...


MetricProviderToken: CapabilityToken[MetricProvider] = create_capability_token(
    "analytics.provider"
)


@dataclass(frozen=True)
class MetricSample:
    id: Id
    key: str
    value: float
    captured_at: IsoTimestamp
    unit: str | None = None


@dataclass(frozen=True)
class Snapshot:
    id: Id
    captured_at: IsoTimestamp
    values: tuple[MetricValue, ...] = ()
    #: Providers that raised. One broken provider must not blank the dashboard.
    failed: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ReportSection:
    title: str
    values: tuple[MetricValue, ...]


@dataclass(frozen=True)
class Report:
    id: Id
    name: str
    generated_at: IsoTimestamp
    sections: tuple[ReportSection, ...] = ()


@dataclass(frozen=True)
class Forecast:
    key: str
    horizon: int
    #: Projected values, one per step.
    values: tuple[float, ...]
    #: Lower and upper bound per step, at the stated confidence.
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    confidence: float
    basis: int
    #: Set when the history is too short or too flat for the projection to mean much.
    caveat: str | None = None


@runtime_checkable
class AnalyticsService(Protocol):
    def providers(self) -> tuple[str, ...]: ...
    async def capture(self) -> Result[Snapshot, KernelError]: ...
    def history(self, key: str) -> tuple[MetricSample, ...]: ...
    def snapshots(self) -> tuple[Snapshot, ...]: ...
    async def report(self, name: str) -> Result[Report, KernelError]: ...
    async def forecast(
        self, key: str, *, horizon: int = 3, confidence: float = 0.95
    ) -> Result[Forecast, KernelError]: ...


AnalyticsToken: CapabilityToken[AnalyticsService] = create_capability_token("analytics.service")


class ANALYTICS_COMMANDS:
    capture = "analytics.capture"
    report = "analytics.report"
    forecast = "analytics.forecast"


class ANALYTICS_EVENTS:
    captured = "analytics.captured"
    reported = "analytics.reported"


#: Two-sided normal quantiles, so a forecast interval does not need scipy.
_Z = {0.80: 1.2816, 0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}


def _z_for(confidence: float) -> float:
    return min(_Z.items(), key=lambda item: abs(item[0] - confidence))[1]


def linear_forecast(
    values: Sequence[float], horizon: int, confidence: float
) -> tuple[list[float], list[float], list[float], str | None]:
    """Least-squares trend with a prediction interval.

    Deliberately the simplest thing that can carry an honest bound. A richer model would be easy
    to add and hard to justify against the two dozen samples a project actually accumulates -- and
    the interval, not the point estimate, is what a reader should be looking at either way.
    """
    n = len(values)
    if n < 2:
        return [], [], [], "not enough history to project"

    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(values) / n
    denominator = sum((x - mean_x) ** 2 for x in xs)
    slope = (
        sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, values)) / denominator
        if denominator
        else 0.0
    )
    intercept = mean_y - slope * mean_x

    residuals = [y - (intercept + slope * x) for x, y in zip(xs, values)]
    # n-2 degrees of freedom: two were spent fitting the line.
    dof = max(n - 2, 1)
    sigma = math.sqrt(sum(r * r for r in residuals) / dof)
    z = _z_for(confidence)

    projected: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    for step in range(1, horizon + 1):
        x = n - 1 + step
        point = intercept + slope * x
        # The interval widens with distance from the data, which is the property that stops a
        # far-horizon number being read as though it were a near one.
        spread = z * sigma * math.sqrt(
            1 + 1 / n + ((x - mean_x) ** 2 / denominator if denominator else 0.0)
        )
        projected.append(point)
        lower.append(point - spread)
        upper.append(point + spread)

    caveat = None
    if n < 4:
        caveat = f"only {n} samples; the interval is wide for a reason"
    elif sigma == 0.0:
        caveat = "history is perfectly linear, so the interval understates real uncertainty"
    return projected, lower, upper, caveat


class AnalyticsServiceImpl:
    __slots__ = ("_context", "_clock", "_ids", "_samples", "_snapshots", "_reports")

    def __init__(
        self,
        context: PluginContext,
        clock: Clock,
        ids: IdFactory,
        samples: RecordStore[MetricSample],
        snapshots: RecordStore[Snapshot],
        reports: RecordStore[Report],
    ) -> None:
        self._context = context
        self._clock = clock
        self._ids = ids
        self._samples = samples
        self._snapshots = snapshots
        self._reports = reports

    def _providers(self) -> list[Any]:
        return [p.value for p in self._context.capabilities.get_all(MetricProviderToken)]

    def providers(self) -> tuple[str, ...]:
        return tuple(sorted(provider.namespace for provider in self._providers()))

    async def capture(self) -> Result[Snapshot, KernelError]:
        captured_at = self._clock.iso()
        values: list[MetricValue] = []
        failed: list[tuple[str, str]] = []

        for provider in self._providers():
            try:
                collected = list(provider.collect())
            except Exception as thrown:  # noqa: BLE001
                # One broken provider must not blank the dashboard.
                failed.append((provider.namespace, str(thrown)))
                continue
            for metric in collected:
                # Namespaced on collection, so two families can both publish "total" without
                # silently overwriting each other.
                values.append(
                    MetricValue(
                        key=f"{provider.namespace}.{metric.key}",
                        value=metric.value,
                        unit=metric.unit,
                        label=metric.label,
                    )
                )

        snapshot = Snapshot(
            id=self._ids.next("snap"),
            captured_at=captured_at,
            values=tuple(values),
            failed=tuple(failed),
        )
        self._snapshots.add(snapshot)
        self._samples.add_many(
            [
                MetricSample(
                    id=self._ids.next("sample"),
                    key=metric.key,
                    value=metric.value,
                    captured_at=captured_at,
                    unit=metric.unit,
                )
                for metric in values
            ]
        )
        self._context.events.emit(ANALYTICS_EVENTS.captured, {"snapshot": snapshot})
        return ok(snapshot)

    def history(self, key: str) -> tuple[MetricSample, ...]:
        return tuple(
            sorted(
                self._samples.query(lambda sample: sample.key == key),
                key=lambda sample: sample.captured_at,
            )
        )

    def snapshots(self) -> tuple[Snapshot, ...]:
        return self._snapshots.all()

    async def report(self, name: str) -> Result[Report, KernelError]:
        captured = await self.capture()
        if not captured.ok:
            return err(captured.error)
        snapshot = captured.value

        grouped: dict[str, list[MetricValue]] = {}
        for metric in snapshot.values:
            grouped.setdefault(metric.key.split(".", 1)[0], []).append(metric)

        report = Report(
            id=self._ids.next("report"),
            name=name,
            generated_at=snapshot.captured_at,
            sections=tuple(
                ReportSection(title=namespace, values=tuple(values))
                for namespace, values in sorted(grouped.items())
            ),
        )
        self._reports.add(report)
        self._context.events.emit(ANALYTICS_EVENTS.reported, {"report": report})
        return ok(report)

    async def forecast(
        self, key: str, *, horizon: int = 3, confidence: float = 0.95
    ) -> Result[Forecast, KernelError]:
        if horizon < 1:
            return err(KernelError("COMMAND_FAILED", "A forecast needs a horizon of at least 1.", {}))
        series = [sample.value for sample in self.history(key)]
        if len(series) < 2:
            return err(
                KernelError(
                    "COMMAND_FAILED",
                    f'"{key}" has {len(series)} sample(s); a trend needs at least two.',
                    {"key": key, "samples": len(series)},
                )
            )

        values, lower, upper, caveat = linear_forecast(series, horizon, confidence)
        return ok(
            Forecast(
                key=key,
                horizon=horizon,
                values=tuple(values),
                lower=tuple(lower),
                upper=tuple(upper),
                confidence=confidence,
                basis=len(series),
                caveat=caveat,
            )
        )


def create_analytics_plugin(*, clock: Clock | None = None, ids: IdFactory | None = None) -> Any:
    resolved_clock = clock or SystemClock()
    resolved_ids = ids or SequentialIdFactory()

    def activate(context: PluginContext) -> None:
        service = AnalyticsServiceImpl(
            context,
            resolved_clock,
            resolved_ids,
            create_record_store(context.state, "samples"),
            create_record_store(context.state, "snapshots"),
            create_record_store(context.state, "reports"),
        )
        context.capabilities.provide(AnalyticsToken, service, version=PLUGIN_VERSION)

        async def capture(_params: Mapping[str, Any], _ctx: Any) -> Any:
            result = await service.capture()
            if not result.ok:
                raise result.error
            return result.value

        async def report(params: Mapping[str, Any], _ctx: Any) -> Any:
            result = await service.report(params.get("name", "Project report"))
            if not result.ok:
                raise result.error
            return result.value

        async def forecast(params: Mapping[str, Any], _ctx: Any) -> Any:
            result = await service.forecast(
                params["key"],
                horizon=params.get("horizon", 3),
                confidence=params.get("confidence", 0.95),
            )
            if not result.ok:
                raise result.error
            return result.value

        for command in (
            CommandDefinition(
                id=ANALYTICS_COMMANDS.capture, title="Capture metrics", handler=capture
            ),
            CommandDefinition(
                id=ANALYTICS_COMMANDS.report, title="Generate report", handler=report
            ),
            CommandDefinition(
                id=ANALYTICS_COMMANDS.forecast, title="Forecast metric", handler=forecast
            ),
        ):
            context.commands.register(command)

        context.ui.register(
            UIContribution(
                id="analytics.panel", point="panel", title="Analytics", placement="right", order=70
            )
        )
        context.logger.info("Analytics capability ready")

    return define_plugin(
        id=PLUGIN_ID,
        version=PLUGIN_VERSION,
        name="Analytics",
        description="Metric provider aggregation, history, snapshots, reports and forecasts "
        "with bounds.",
        activate=activate,
    )


analytics_plugin = create_analytics_plugin()
