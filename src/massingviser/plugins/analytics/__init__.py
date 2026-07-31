"""Analytics -- metric aggregation, history, snapshots, reports and bounded forecasts."""

from .plugin import (
    ANALYTICS_COMMANDS,
    ANALYTICS_EVENTS,
    PLUGIN_ID,
    AnalyticsService,
    AnalyticsToken,
    Forecast,
    MetricProvider,
    MetricProviderToken,
    MetricSample,
    MetricValue,
    Report,
    ReportSection,
    Snapshot,
    analytics_plugin,
    create_analytics_plugin,
    linear_forecast,
)

__all__ = [
    "ANALYTICS_COMMANDS",
    "ANALYTICS_EVENTS",
    "PLUGIN_ID",
    "AnalyticsService",
    "AnalyticsToken",
    "Forecast",
    "MetricProvider",
    "MetricProviderToken",
    "MetricSample",
    "MetricValue",
    "Report",
    "ReportSection",
    "Snapshot",
    "analytics_plugin",
    "create_analytics_plugin",
    "linear_forecast",
]
