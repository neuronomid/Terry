"""Dukascopy public ``.bi5`` historical feed driver.

Free and unauthenticated — no account required. For each hour Dukascopy serves one
LZMA-compressed file of ticks under::

    https://datafeed.dukascopy.com/datafeed/{INSTRUMENT}/{YYYY}/{MM}/{DD}/{HH}h_ticks.bi5

where ``MM`` is **zero-indexed** (January = ``00``). Each 20-byte big-endian record is
``>3I2f`` = ``(ms_from_hour, ask, bid, ask_volume, bid_volume)``; ask/bid are integers
scaled by the instrument's price ``scale`` (see :mod:`terry.data.instruments`).

Ticks are aggregated into Terry's ``[timestamp, open, close, high, low, volume]``
one-minute rows using the **bid** price. Weekends / holidays have no file (HTTP 404)
and simply contribute no candles — Forex and other session markets are naturally gappy.
"""
from __future__ import annotations

import lzma
import struct
import time

import numpy as np
import requests

from . import instruments

ONE_MIN_MS = 60_000
ONE_HOUR_MS = 3_600_000
_TICK = struct.Struct(">3I2f")
_BASE_URL = "https://datafeed.dukascopy.com/datafeed"
_HEADERS = {"User-Agent": "Mozilla/5.0 (Terry candle importer)"}
# Conservative earliest probe date; most FX reaches back much further, but this keeps
# ``get_starting_time`` honest without an extra network round-trip per instrument.
_DEFAULT_START = "2010-01-01"


def _session():
    session = requests.Session()
    session.headers.update(_HEADERS)
    return session


def _hour_url(instrument: str, hour_start_ms: int) -> str:
    from datetime import datetime, timezone
    dt = datetime.fromtimestamp(hour_start_ms / 1000, tz=timezone.utc)
    # Dukascopy months are zero-indexed in the path.
    return (f"{_BASE_URL}/{instrument}/{dt.year:04d}/{dt.month - 1:02d}/"
            f"{dt.day:02d}/{dt.hour:02d}h_ticks.bi5")


def _decompress(raw: bytes) -> bytes:
    if not raw:
        return b""
    try:
        return lzma.decompress(raw)
    except lzma.LZMAError:
        return lzma.decompress(raw, format=lzma.FORMAT_ALONE)


def fetch_hour_ticks(instrument, scale, hour_start_ms, session=None, retries=3):
    """Return ``[(timestamp_ms, price, volume), …]`` for one hour, empty if no data.

    A missing file (HTTP 404) means the market was closed that hour — normal for
    session markets — and yields no ticks. Transient network/5xx errors are retried
    a few times before the hour is skipped so one bad hour never aborts a long import.
    """
    session = session or _session()
    url = _hour_url(instrument, hour_start_ms)
    raw = None
    for attempt in range(retries):
        try:
            response = session.get(url, timeout=30)
        except requests.RequestException:
            if attempt == retries - 1:
                return []
            time.sleep(0.5 * (attempt + 1))
            continue
        if response.status_code == 404:
            return []
        if response.status_code // 100 == 2:
            raw = response.content
            break
        if response.status_code // 100 == 5 and attempt < retries - 1:
            time.sleep(0.5 * (attempt + 1))
            continue
        return []
    if not raw:
        return []
    data = _decompress(raw)
    ticks = []
    for offset in range(0, len(data) - len(data) % _TICK.size, _TICK.size):
        ms, _ask, bid, _ask_vol, bid_vol = _TICK.unpack_from(data, offset)
        ticks.append((hour_start_ms + ms, bid / scale, float(bid_vol)))
    return ticks


def aggregate_ticks_to_1m(ticks) -> np.ndarray:
    """Aggregate ``(ts_ms, price, volume)`` ticks into sorted 1m OHLCV rows."""
    buckets = {}
    for ts, price, volume in ticks:
        minute = ts // ONE_MIN_MS * ONE_MIN_MS
        row = buckets.get(minute)
        if row is None:
            # [open, close, high, low, volume]
            buckets[minute] = [price, price, price, price, volume]
        else:
            row[1] = price                    # close = latest tick
            if price > row[2]:
                row[2] = price                # high
            if price < row[3]:
                row[3] = price                # low
            row[4] += volume
    if not buckets:
        return np.empty((0, 6))
    out = np.empty((len(buckets), 6))
    for i, minute in enumerate(sorted(buckets)):
        o, c, h, l, v = buckets[minute]
        out[i] = [minute, o, c, h, l, v]
    return out


def fetch_1m_range(symbol, start_ts, finish_ts, on_progress=None,
                   should_stop=None, rate_limit_sleep=0.05) -> np.ndarray:
    """Fetch all 1m candles in ``[start_ts, finish_ts)`` for a Terry symbol.

    Mirrors :func:`terry.data.binance.fetch_1m_range` so the background importer and the
    session runner can call it interchangeably.
    """
    instrument, scale = instruments.dukascopy_instrument(symbol)
    session = _session()
    start_ts, finish_ts = int(start_ts), int(finish_ts)
    total = max(finish_ts - start_ts, 1)
    hour = start_ts // ONE_HOUR_MS * ONE_HOUR_MS
    ticks = []
    while hour < finish_ts:
        if should_stop and should_stop():
            break
        ticks.extend(fetch_hour_ticks(instrument, scale, hour, session))
        hour += ONE_HOUR_MS
        if on_progress:
            on_progress(min(hour - start_ts, total), total)
        if rate_limit_sleep:
            time.sleep(rate_limit_sleep)
    rows = aggregate_ticks_to_1m(ticks)
    if len(rows) == 0:
        return rows
    mask = (rows[:, 0] >= start_ts) & (rows[:, 0] < finish_ts)
    return rows[mask]


def get_starting_time(symbol):
    """Earliest timestamp Terry will offer for a Dukascopy instrument.

    Dukascopy's per-instrument history depth varies; this returns a safe floor rather
    than probing thousands of historical hours. Imports simply yield no candles before
    the instrument's real inception.
    """
    from .. import helpers as jh
    if instruments.resolve(symbol) is None:
        return None
    return jh.date_to_timestamp(_DEFAULT_START)
