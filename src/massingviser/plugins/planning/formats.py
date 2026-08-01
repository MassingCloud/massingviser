"""Programme formats: Primavera P6 XER and MS Project XML (MSPDI).

Both reduce to the same thing -- a list of row mappings in the shape the CSV and JSON readers
already produce -- so the import, re-import and link-preservation logic downstream is untouched by
which tool the programme came out of.

**What these parsers deliberately do not do.** Neither reads calendars, resource assignments, cost
loading or constraint arithmetic. A schedule is a large, quirky, vendor-specific model and a parser
that half-reads a calendar produces dates that look right and are not. What is read is the part
this platform actually uses: task identity, name, planned and actual dates, percent complete, WBS,
criticality and the dependency graph. Anything else is left in the file, and a row that cannot be
read is **rejected by name** rather than guessed at.

Standard library only, like every other capability family.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ElementTree
from collections.abc import Sequence
from datetime import date, timedelta
from typing import Any

from .calendars import DEFAULT_HOURS_PER_DAY, MONDAY_TO_FRIDAY, WorkCalendar

#: XER stores dates as ``YYYY-MM-DD HH:MM``; the time half is optional in some exports.
_XER_DATE = re.compile(r"^(\d{4}-\d{2}-\d{2})(?:[ T](\d{2}:\d{2}))?")

#: P6 relationship codes, in the vocabulary the rest of the platform uses.
_XER_DEPENDENCY = {"PR_FS": "FS", "PR_SS": "SS", "PR_FF": "FF", "PR_SF": "SF"}

#: MSPDI encodes the relationship as an integer. The order is Microsoft's, not alphabetical, and
#: getting it wrong silently reverses half a programme's logic.
_MSPDI_DEPENDENCY = {0: "FF", 1: "FS", 2: "SF", 3: "SS"}

_DOCTYPE = re.compile(r"<!DOCTYPE", re.IGNORECASE)

#: Fallback when a file names no calendar. Real durations come from the task's own calendar.
_HOURS_PER_DAY = DEFAULT_HOURS_PER_DAY

#: P6 numbers weekdays 1..7 starting at Sunday; `date.weekday()` numbers them 0..6 starting at
#: Monday. Getting this off by one shifts an entire programme's working week by a day.
_XER_WEEKDAY = {1: 6, 2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5}

#: MSPDI uses the same 1..7-from-Sunday convention as P6.
_MSPDI_WEEKDAY = dict(_XER_WEEKDAY)

#: Excel's day zero, which is what P6 stores calendar exceptions as. Excel also believes 1900 was
#: a leap year, so its serials run one day ahead of reality from 1 March 1900 -- irrelevant for
#: construction dates, and noted so nobody 'fixes' the offset below.
_EXCEL_EPOCH = date(1899, 12, 30)

#: A day block inside a P6 calendar's DaysOfWeek section: `(0||<n>()(...))`.
_XER_DAY_BLOCK = re.compile(r"\(0\|\|(\d)\(\)\((.*?)\)\)", re.DOTALL)

#: A whole-day exception: `(0||d|<excel serial>()`.
_XER_EXCEPTION = re.compile(r"\(0\|\|d\|(\d+)\(\)")


def _xer_calendars(tables: dict[str, list[dict[str, str]]]) -> dict[str, WorkCalendar]:
    """Read the CALENDAR table.

    ``clndr_data`` is a nested parenthesised blob rather than a table, so this reads the two things
    the platform consumes -- which weekdays carry shift times, and which whole days are excepted --
    and leaves the shift times themselves alone. A day is a working day when its block contains a
    start time; that is what P6 writes and what every reader keys on.
    """
    calendars: dict[str, WorkCalendar] = {}
    for row in tables.get("CALENDAR", []):
        identifier = (row.get("clndr_id") or "").strip()
        if not identifier:
            continue
        blob = row.get("clndr_data") or ""

        working: set[int] = set()
        for match in _XER_DAY_BLOCK.finditer(blob):
            if "s|" in match.group(2):
                mapped = _XER_WEEKDAY.get(int(match.group(1)))
                if mapped is not None:
                    working.add(mapped)

        holidays: set[date] = set()
        for match in _XER_EXCEPTION.finditer(blob):
            try:
                holidays.add(_EXCEL_EPOCH + timedelta(days=int(match.group(1))))
            except (ValueError, OverflowError):
                continue

        try:
            hours = float(row.get("day_hr_cnt") or DEFAULT_HOURS_PER_DAY)
        except ValueError:
            hours = DEFAULT_HOURS_PER_DAY

        calendars[identifier] = WorkCalendar(
            id=identifier,
            name=(row.get("clndr_name") or "Standard").strip(),
            # A calendar whose blob named no working day is far more likely to be a blob this
            # reader did not understand than a calendar where nobody ever works.
            working_weekdays=frozenset(working) if working else MONDAY_TO_FRIDAY,
            holidays=frozenset(holidays),
            hours_per_day=hours if hours > 0 else DEFAULT_HOURS_PER_DAY,
        )
    return calendars


class ScheduleParseError(ValueError):
    """The payload is not the format it claimed to be."""


def _text(payload: str | bytes) -> str:
    if isinstance(payload, (bytes, bytearray)):
        # XER out of P6 is routinely cp1252, and a stray en-dash in a task name should not fail an
        # import. UTF-8 first because everything else is.
        for encoding in ("utf-8", "cp1252", "latin-1"):
            try:
                return bytes(payload).decode(encoding)
            except UnicodeDecodeError:
                continue
        raise ScheduleParseError("Could not decode the programme in UTF-8, cp1252 or latin-1.")
    return payload


def _xer_date(value: str | None) -> str | None:
    if not value:
        return None
    match = _XER_DATE.match(value.strip())
    if match is None:
        return None
    date, time = match.group(1), match.group(2)
    return f"{date}T{time}:00" if time else f"{date}T00:00:00"


def parse_xer(payload: str | bytes) -> tuple[list[dict[str, Any]], dict[str, WorkCalendar]]:
    """Read a Primavera P6 XER export.

    XER is a flat table dump: ``%T`` names a table, ``%F`` gives its column names, and each ``%R``
    is a row under them. Tabs separate fields, and a field may legitimately be empty.

    Identity is the **activity code** (``task_code``) rather than P6's internal ``task_id``, because
    the activity code is what a scheduler sees, types into a report and expects a re-import to match
    on. The internal id is used only to resolve the relationship table, then discarded.
    """
    text = _text(payload)
    if "%T" not in text and "ERMHDR" not in text:
        raise ScheduleParseError("No XER table markers; this does not look like a P6 export.")

    tables: dict[str, list[dict[str, str]]] = {}
    current: str | None = None
    fields: list[str] = []

    for line in text.splitlines():
        if not line.startswith("%"):
            continue
        parts = line.split("\t")
        marker = parts[0]
        if marker == "%T":
            current = parts[1].strip() if len(parts) > 1 else None
            fields = []
            if current:
                tables.setdefault(current, [])
        elif marker == "%F":
            fields = [field.strip() for field in parts[1:]]
        elif marker == "%R" and current and fields:
            values = parts[1:]
            # A short row is padded rather than dropped: P6 omits trailing empties.
            values += [""] * (len(fields) - len(values))
            tables[current].append(dict(zip(fields, values, strict=False)))
        elif marker == "%E":
            break

    task_rows = tables.get("TASK", [])
    if not task_rows:
        raise ScheduleParseError("The XER contains no TASK table.")

    calendars = _xer_calendars(tables)
    task_calendar = {
        row.get("task_id", ""): (row.get("clndr_id") or "").strip() for row in task_rows
    }

    # wbs_id -> a readable code, when the file carries the WBS table.
    wbs_names = {
        row.get("wbs_id", ""): row.get("wbs_short_name") or row.get("wbs_name") or ""
        for row in tables.get("PROJWBS", [])
    }

    identity = {
        row.get("task_id", ""): (row.get("task_code") or row.get("task_id") or "").strip()
        for row in task_rows
    }

    predecessors: dict[str, list[dict[str, Any]]] = {}
    for row in tables.get("TASKPRED", []):
        successor = identity.get(row.get("task_id", ""))
        predecessor = identity.get(row.get("pred_task_id", ""))
        if not successor or not predecessor:
            continue
        # Converted through the *successor's* calendar. Lag is stored in hours and a six-hour day
        # makes 16 hours of lag two and a bit days, not two -- the hardcoded eight was only ever
        # right for the default calendar.
        calendar = calendars.get(task_calendar.get(row.get("task_id", ""), ""))
        hours_per_day = calendar.hours_per_day if calendar else _HOURS_PER_DAY
        try:
            lag = float(row.get("lag_hr_cnt") or 0) / (hours_per_day or _HOURS_PER_DAY)
        except ValueError:
            lag = 0.0
        predecessors.setdefault(successor, []).append(
            {
                "predecessor": predecessor,
                "type": _XER_DEPENDENCY.get(row.get("pred_type", ""), "FS"),
                "lag": lag,
            }
        )

    rows: list[dict[str, Any]] = []
    for row in task_rows:
        code = identity.get(row.get("task_id", ""), "")
        if not code:
            continue
        percent = row.get("phys_complete_pct") or row.get("complete_pct")
        try:
            # P6 stores this as 0-100; the platform stores 0..1.
            fraction: float | None = float(percent) / 100.0 if percent else None
        except ValueError:
            fraction = None
        rows.append(
            {
                "id": code,
                "name": (row.get("task_name") or code).strip(),
                "planned_start": _xer_date(row.get("target_start_date")),
                "planned_finish": _xer_date(row.get("target_end_date")),
                "actual_start": _xer_date(row.get("act_start_date")),
                "actual_finish": _xer_date(row.get("act_end_date")),
                "percent_complete": fraction,
                "wbs_code": wbs_names.get(row.get("wbs_id", "")) or None,
                "critical": (row.get("driving_path_flag") or "").upper() == "Y",
                "calendar_id": task_calendar.get(row.get("task_id", "")) or None,
                "predecessors": predecessors.get(code, []),
            }
        )
    return rows, calendars


def _tag(element: ElementTree.Element) -> str:
    """The local name. MSPDI is namespaced and some exports drop the namespace entirely."""
    return element.tag.rsplit("}", 1)[-1]


def _child(element: ElementTree.Element, name: str) -> str | None:
    for child in element:
        if _tag(child) == name:
            return (child.text or "").strip() or None
    return None


def _children(element: ElementTree.Element, name: str) -> list[ElementTree.Element]:
    return [child for child in element if _tag(child) == name]


def _mspdi_calendars(root: ElementTree.Element) -> dict[str, WorkCalendar]:
    """Read `<Calendars>`.

    MS Project spells the same two facts differently from P6: a `<WeekDay>` carries a `DayType`
    (1..7 from Sunday) and a `DayWorking` flag, and an exception is a `<WeekDay>` with a
    `<TimePeriod>` instead of a day type. One level of `<BaseCalendarUID>` is followed; a chain
    deeper than that is left alone rather than half-resolved.
    """
    calendars: dict[str, WorkCalendar] = {}
    bases: dict[str, str] = {}

    for container in _children(root, "Calendars"):
        for element in _children(container, "Calendar"):
            uid = _child(element, "UID")
            if uid is None:
                continue
            base = _child(element, "BaseCalendarUID")
            if base and base != "-1":
                bases[uid] = base

            working: set[int] = set()
            holidays: set[date] = set()
            declared = False
            for days in _children(element, "WeekDays"):
                for day in _children(days, "WeekDay"):
                    day_type = _child(day, "DayType")
                    period = _children(day, "TimePeriod")
                    if day_type is not None and not period:
                        declared = True
                        try:
                            mapped = _MSPDI_WEEKDAY.get(int(day_type))
                        except ValueError:
                            continue
                        if mapped is not None and (_child(day, "DayWorking") or "0") == "1":
                            working.add(mapped)
                    elif period:
                        # An exception. Non-working ones are the ones that move a date.
                        if (_child(day, "DayWorking") or "0") == "1":
                            continue
                        for span in period:
                            start = _child(span, "FromDate") or _child(day, "FromDate")
                            finish = _child(span, "ToDate") or start
                            first, last = _iso_date(start), _iso_date(finish)
                            if first is None:
                                continue
                            cursor = first
                            while cursor <= (last or first):
                                holidays.add(cursor)
                                cursor += timedelta(days=1)

            hours = _child(element, "HoursPerDay")
            try:
                hours_per_day = float(hours) if hours else DEFAULT_HOURS_PER_DAY
            except ValueError:
                hours_per_day = DEFAULT_HOURS_PER_DAY

            calendars[uid] = WorkCalendar(
                id=uid,
                name=_child(element, "Name") or "Standard",
                working_weekdays=frozenset(working) if declared and working else MONDAY_TO_FRIDAY,
                holidays=frozenset(holidays),
                hours_per_day=hours_per_day if hours_per_day > 0 else DEFAULT_HOURS_PER_DAY,
            )

    # One level of inheritance: a calendar that declared no week of its own takes its base's.
    for uid, base in bases.items():
        parent = calendars.get(base)
        child = calendars.get(uid)
        if parent is None or child is None or child.working_weekdays != MONDAY_TO_FRIDAY:
            continue
        calendars[uid] = WorkCalendar(
            id=child.id,
            name=child.name,
            working_weekdays=parent.working_weekdays,
            holidays=frozenset(child.holidays | parent.holidays),
            hours_per_day=child.hours_per_day,
        )
    return calendars


def _iso_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def parse_mspdi(payload: str | bytes) -> tuple[list[dict[str, Any]], dict[str, WorkCalendar]]:
    """Read a Microsoft Project XML (MSPDI) export.

    Two things this gets right that a naive reader does not:

    - **UID 0 is skipped.** It is the project summary row, spanning the whole programme. Importing
      it would add a task as long as the project that nothing actually builds.
    - **Parents come from ``OutlineLevel``**, tracked as a stack in document order, because MSPDI
      expresses hierarchy by position and depth rather than by pointing at a parent.
    """
    text = _text(payload)
    if _DOCTYPE.search(text):
        # Same rule as the ICDD reader: a DOCTYPE in an untrusted file is the entry point for
        # entity expansion, and is refused rather than ignored.
        raise ScheduleParseError("A DOCTYPE in a programme file is refused, not ignored.")
    try:
        root = ElementTree.fromstring(text)
    except ElementTree.ParseError as thrown:
        raise ScheduleParseError(f"Not well-formed XML: {thrown}") from thrown

    task_containers = _children(root, "Tasks")
    if not task_containers:
        raise ScheduleParseError("The XML has no <Tasks> element; this is not an MSPDI export.")

    calendars = _mspdi_calendars(root)

    by_uid: dict[str, str] = {}
    elements: list[ElementTree.Element] = []
    for container in task_containers:
        for task in _children(container, "Task"):
            uid = _child(task, "UID")
            if uid is None or uid == "0":
                continue
            identity = _child(task, "WBS") or uid
            by_uid[uid] = identity
            elements.append(task)

    rows: list[dict[str, Any]] = []
    # (outline level, identity) of the open ancestors, shallowest first.
    stack: list[tuple[int, str]] = []

    for task in elements:
        uid = _child(task, "UID") or ""
        identity = by_uid[uid]
        try:
            level = int(_child(task, "OutlineLevel") or 1)
        except ValueError:
            level = 1
        while stack and stack[-1][0] >= level:
            stack.pop()
        parent = stack[-1][1] if stack else None
        stack.append((level, identity))

        percent = _child(task, "PercentComplete")
        try:
            fraction: float | None = float(percent) / 100.0 if percent else None
        except ValueError:
            fraction = None

        predecessors: list[dict[str, Any]] = []
        for link in _children(task, "PredecessorLink"):
            predecessor_uid = _child(link, "PredecessorUID")
            if predecessor_uid is None or predecessor_uid not in by_uid:
                continue
            try:
                kind = _MSPDI_DEPENDENCY.get(int(_child(link, "Type") or 1), "FS")
            except ValueError:
                kind = "FS"
            try:
                # LinkLag is in tenths of a minute, which is not a unit anyone expects.
                lag = float(_child(link, "LinkLag") or 0) / 10.0 / 60.0 / _HOURS_PER_DAY
            except ValueError:
                lag = 0.0
            predecessors.append({"predecessor": by_uid[predecessor_uid], "type": kind, "lag": lag})

        rows.append(
            {
                "id": identity,
                "name": _child(task, "Name") or identity,
                "planned_start": _child(task, "Start"),
                "planned_finish": _child(task, "Finish"),
                "actual_start": _child(task, "ActualStart"),
                "actual_finish": _child(task, "ActualFinish"),
                "percent_complete": fraction,
                "wbs_code": _child(task, "WBS"),
                "parent_id": parent,
                "critical": (_child(task, "Critical") or "0") in ("1", "true", "True"),
                "calendar_id": _child(task, "CalendarUID"),
                "predecessors": predecessors,
            }
        )
    return rows, calendars


def flatten_predecessors(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every ``(successor, predecessor)`` pair, as flat dicts.

    A real programme gives a task several predecessors, which the single ``predecessor`` column of
    the CSV reader cannot express. Keeping the list on the row and flattening here means both
    shapes reach the same downstream code.
    """
    pairs: list[dict[str, Any]] = []
    for row in rows:
        for link in row.get("predecessors") or ():
            pairs.append(
                {
                    "successor": row.get("id"),
                    "predecessor": link.get("predecessor"),
                    "type": (link.get("type") or "FS").upper(),
                    "lag": float(link.get("lag") or 0.0),
                }
            )
    return pairs
