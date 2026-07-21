"""Forex trading sessions and market-open logic.

Non-crypto markets (Forex, metals, energy, indices, stock CFDs) are **not** 24/7:
the FX week runs from Sunday 21:00 UTC to Friday 21:00 UTC, and liquidity rotates
through three main regional sessions plus quieter transitional windows.

Sessions (approximate, UTC):
  * Tokyo  (Asian)     00:00 - 09:00
  * London (European)  07:00 - 16:00
  * New York           12:00 - 21:00

Overlaps concentrate liquidity — notably Tokyo/London (07:00-09:00) and the largest,
London/New York (12:00-16:00). Between the New York close and the Tokyo open the market
is in a thin transitional phase. Terry uses this module to (a) know whether a live
Demo-Mode fetch should expect fresh data, and (b) label the current session for the UI.
Gaps in the underlying candle data (weekends/holidays) are handled separately by
:func:`terry.engine.candle_store.fill_1m_gaps`.
"""
from __future__ import annotations

from datetime import datetime, timezone

# name -> (open_hour_utc, close_hour_utc)
SESSIONS = {
    "Tokyo": (0, 9),
    "London": (7, 16),
    "New York": (12, 21),
}

# Friday/Sunday cutover of the 24-hour FX week, in UTC hours.
_WEEK_CLOSE_HOUR = 21   # Friday 21:00 UTC -> weekend
_WEEK_OPEN_HOUR = 21    # Sunday 21:00 UTC -> reopen


def _dt(ts_ms: int) -> datetime:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)


def is_forex_open(ts_ms: int) -> bool:
    """Whether the FX/CFD market is open at a millisecond UTC timestamp."""
    dt = _dt(ts_ms)
    weekday = dt.weekday()  # Mon=0 … Sun=6
    if weekday == 5:  # Saturday: always closed
        return False
    if weekday == 6:  # Sunday: opens at 21:00 UTC
        return dt.hour >= _WEEK_OPEN_HOUR
    if weekday == 4:  # Friday: closes at 21:00 UTC
        return dt.hour < _WEEK_CLOSE_HOUR
    return True


def active_sessions(ts_ms: int) -> list[str]:
    """Regional sessions open at ``ts_ms`` (empty when the market is closed)."""
    if not is_forex_open(ts_ms):
        return []
    hour = _dt(ts_ms).hour
    active = []
    for name, (open_h, close_h) in SESSIONS.items():
        if open_h <= hour < close_h:
            active.append(name)
    return active


def session_label(ts_ms: int) -> str:
    """Human-readable label for the current session/overlap/transition."""
    if not is_forex_open(ts_ms):
        return "Closed"
    active = active_sessions(ts_ms)
    if len(active) >= 2:
        return " + ".join(active) + " overlap"
    if len(active) == 1:
        return active[0]
    return "Transition"


def session_info(ts_ms: int) -> dict:
    """Compact session snapshot for the live payload / UI."""
    return {
        "is_open": is_forex_open(ts_ms),
        "active_sessions": active_sessions(ts_ms),
        "label": session_label(ts_ms),
    }
