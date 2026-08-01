"""Working calendars.

A programme's dates only mean something against a calendar. Two tasks that both span "1 March to
31 March" are not the same amount of work if one runs a five-day week and the other a six-day week
through a shutdown, and a cashflow curve that spreads cost evenly over calendar days puts money on
Christmas Day.

**What is read, and what is not.** The working week, whole-day exceptions, and hours per day --
which is what the rest of this platform actually consumes: the S-curve needs to know which days
count, and P6 stores lag in hours that only convert to days through this number. What is *not*
read is shift times within a day, resource-specific calendars, or inherited base calendars beyond
one level. Those change durations in ways a partial implementation would get subtly wrong, and a
schedule that is subtly wrong about dates is worse than one that is openly approximate.

The fallback is a five-day week at eight hours, which is what every scheduling tool defaults to and
what this platform assumed unconditionally before calendars existed.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, timedelta

#: Monday..Sunday as 0..6, matching `date.weekday()`.
MONDAY_TO_FRIDAY = frozenset({0, 1, 2, 3, 4})

#: What every scheduling tool defaults to, and what this platform assumed before it read calendars.
DEFAULT_HOURS_PER_DAY = 8.0


@dataclass(frozen=True)
class WorkCalendar:
    """A working week, its exceptions, and how many hours a working day holds."""

    id: str
    name: str = "Standard"
    #: `date.weekday()` values that are worked. Empty means nothing is worked, which is legal in a
    #: file and is why `working_days` has to cope with it rather than dividing by zero.
    working_weekdays: frozenset[int] = MONDAY_TO_FRIDAY
    #: Whole days that are not worked despite falling on a working weekday -- bank holidays, a
    #: Christmas shutdown, a site closure.
    holidays: frozenset[date] = field(default_factory=frozenset)
    hours_per_day: float = DEFAULT_HOURS_PER_DAY

    def is_working(self, day: date) -> bool:
        return day.weekday() in self.working_weekdays and day not in self.holidays

    def working_days(self, start: date, end: date) -> int:
        """Working days in ``[start, end)``.

        Half open deliberately: a task that finishes on the 5th does not consume the 5th, and
        counting it would give a one-day task on a Friday two days of work.
        """
        if end <= start or not self.working_weekdays:
            return 0
        total = 0
        cursor = start
        while cursor < end:
            if self.is_working(cursor):
                total += 1
            cursor += timedelta(days=1)
        return total

    def hours_to_days(self, hours: float) -> float:
        """Convert a duration P6 stored in hours. Lag is the one that matters here."""
        return hours / self.hours_per_day if self.hours_per_day else 0.0


#: Used for any task whose file named no calendar, and for a file that carried none at all.
DEFAULT_CALENDAR = WorkCalendar(id="default", name="Standard (assumed)")


def calendar_or_default(
    calendars: dict[str, WorkCalendar] | None, calendar_id: str | None
) -> WorkCalendar:
    """The named calendar, or the assumed one -- never a KeyError.

    A task naming a calendar the file did not include is a broken export, not a reason to refuse
    the whole programme; it falls back and the dates stay readable.
    """
    if not calendars or not calendar_id:
        return DEFAULT_CALENDAR
    return calendars.get(str(calendar_id), DEFAULT_CALENDAR)


def spread_over_working_days(
    calendar: WorkCalendar, start: date, end: date, days: Iterable[date]
) -> dict[date, float]:
    """Share one task's work equally across the working days it spans.

    Equal per working day, not per calendar day. A task that runs across a two-week shutdown puts
    none of its cost inside the shutdown, which is the whole reason to read a calendar at all.

    Returns an empty mapping when the task spans no working days -- a milestone, or a task placed
    entirely inside a shutdown. The caller decides what that means; silently spreading it over the
    calendar instead would put cost on days the file says nobody is working.
    """
    working = [day for day in days if calendar.is_working(day) and start <= day < end]
    if not working:
        return {}
    share = 1.0 / len(working)
    return {day: share for day in working}
