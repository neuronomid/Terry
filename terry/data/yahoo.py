"""Yahoo Finance v8 chart driver for Demo Mode live data.

Provides the recent one-minute chart tail and the live forming-candle price for
non-crypto instruments. Uses the **unauthenticated** ``/v8/finance/chart`` endpoint
(no crumb/cookie handshake). Stock indices resolve to a futures/ETF proxy via
:mod:`terry.data.instruments` so the live candle tracks the market instead of Yahoo's
~15-minute delayed index level.

Yahoo only serves ~7 days of 1-minute history, so this feed is for *live* demos, not
the multi-year historical backfill (that is Dukascopy's job).
"""
from __future__ import annotations

import numpy as np
import requests

from . import instruments

ONE_MIN_MS = 60_000
SEVEN_DAYS_MS = 7 * 86_400_000
_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
_HEADERS = {"User-Agent": "Mozilla/5.0 (Terry live demo)"}


def _ticker(symbol: str) -> str:
    """Yahoo ticker for a Terry symbol (registry proxy, else the dashless symbol)."""
    resolved = instruments.yahoo_ticker(symbol)
    if resolved:
        return resolved
    return symbol.replace("-", "").upper()


def _session():
    session = requests.Session()
    session.headers.update(_HEADERS)
    return session


def _fetch_chart(ticker, start_ts, finish_ts, session):
    # Yahoo caps 1m history at ~7 days; clamp the window so period1 is never rejected.
    finish_ts = int(finish_ts)
    start_ts = max(int(start_ts), finish_ts - SEVEN_DAYS_MS)
    params = {
        "interval": "1m",
        "period1": start_ts // 1000,
        "period2": finish_ts // 1000 + 60,
        "includePrePost": "true",
    }
    response = session.get(_CHART_URL.format(ticker=ticker), params=params, timeout=30)
    if response.status_code // 100 != 2:
        raise RuntimeError(f"Yahoo chart error {response.status_code} for {ticker}")
    payload = response.json()
    result = (payload.get("chart") or {}).get("result") or []
    if not result:
        return None
    return result[0]


def fetch_1m_range(symbol, start_ts, finish_ts, on_progress=None,
                   should_stop=None, rate_limit_sleep=0.0) -> np.ndarray:
    """Fetch recent 1m candles in ``[start_ts, finish_ts)`` from Yahoo v8.

    Signature mirrors :func:`terry.data.binance.fetch_1m_range`. Rows with a null close
    (market closed / no print) are dropped so the chart never shows phantom candles.
    """
    del rate_limit_sleep  # single request; no pagination needed for a 7-day window
    if should_stop and should_stop():
        return np.empty((0, 6))
    session = _session()
    result = _fetch_chart(_ticker(symbol), start_ts, finish_ts, session)
    if not result:
        return np.empty((0, 6))
    timestamps = result.get("timestamp") or []
    quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
    opens = quote.get("open") or []
    highs = quote.get("high") or []
    lows = quote.get("low") or []
    closes = quote.get("close") or []
    volumes = quote.get("volume") or []
    start_ts, finish_ts = int(start_ts), int(finish_ts)
    rows = []
    for i, sec in enumerate(timestamps):
        close = closes[i] if i < len(closes) else None
        open_ = opens[i] if i < len(opens) else None
        if close is None or open_ is None:
            continue
        ts = int(sec) * 1000 // ONE_MIN_MS * ONE_MIN_MS
        if ts < start_ts or ts >= finish_ts:
            continue
        high = highs[i] if i < len(highs) and highs[i] is not None else max(open_, close)
        low = lows[i] if i < len(lows) and lows[i] is not None else min(open_, close)
        vol = volumes[i] if i < len(volumes) and volumes[i] is not None else 0.0
        rows.append([ts, float(open_), float(close), float(high), float(low), float(vol)])
    if on_progress:
        on_progress(1, 1)
    if not rows:
        return np.empty((0, 6))
    array = np.array(rows, dtype=float)
    _, unique = np.unique(array[:, 0], return_index=True)
    return array[np.sort(unique)]


def fetch_live_price(symbol):
    """Real-time last price for the live forming candle, or ``None`` on any failure.

    Prefers the full-precision close of the newest 1-minute bar over the meta
    ``regularMarketPrice``. Yahoo rounds that meta price (e.g. EUR/USD ``1.1410315`` is
    reported as ``1.141``), which flattens sub-pip FX moves and makes the forming candle
    look frozen even as the market ticks. Whichever reading carries the most recent
    timestamp wins; the precise 1m close breaks ties. Returns ``None`` when the market is
    closed or a request fails so the demo loop holds its last good price instead of stalling.
    """
    try:
        session = _session()
        params = {"interval": "1m", "range": "1d", "includePrePost": "true"}
        response = session.get(_CHART_URL.format(ticker=_ticker(symbol)),
                               params=params, timeout=10)
        if response.status_code // 100 != 2:
            return None
        result = ((response.json().get("chart") or {}).get("result") or [None])[0]
        if not result:
            return None
        meta = result.get("meta") or {}
        timestamps = result.get("timestamp") or []
        closes = ((result.get("indicators") or {}).get("quote") or [{}])[0].get("close") or []
        # Newest non-null 1m close (full precision).
        close_price = close_time = None
        for i in range(len(closes) - 1, -1, -1):
            if closes[i] is not None:
                close_price = closes[i]
                close_time = timestamps[i] if i < len(timestamps) else None
                break
        # (timestamp, precision_rank, value): rank breaks ties toward the precise 1m close.
        candidates = []
        if close_price is not None:
            candidates.append((close_time if close_time is not None else -1, 1, close_price))
        market_price = meta.get("regularMarketPrice")
        if market_price is not None:
            market_time = meta.get("regularMarketTime")
            candidates.append((market_time if market_time is not None else -1, 0, market_price))
        if not candidates:
            return None
        candidates.sort()
        return float(candidates[-1][2])
    except Exception:
        return None
