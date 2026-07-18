"""
Free candle data from Binance's public REST API (no API key required).

Endpoints used (klines): spot /api/v3/klines, USDT-M perpetual futures /fapi/v1/klines.
Binance US mirror is provided for regions where binance.com is geo-blocked (HTTP 451).
"""
import time

import numpy as np
import requests

from .. import helpers as jh

# exchange name -> (base_url, kline_path, market_type)
EXCHANGES = {
    "Binance Spot": ("https://api.binance.com", "/api/v3/klines", "spot"),
    "Binance": ("https://api.binance.com", "/api/v3/klines", "spot"),
    "Binance US Spot": ("https://api.binance.us", "/api/v3/klines", "spot"),
    "Binance Perpetual Futures": ("https://fapi.binance.com", "/fapi/v1/klines", "futures"),
    "Binance USDT Perpetual": ("https://fapi.binance.com", "/fapi/v1/klines", "futures"),
}

DEFAULT_EXCHANGE = "Binance Perpetual Futures"
MAX_LIMIT = 1000
ONE_MIN_MS = 60_000


def exchange_endpoint(exchange: str):
    if exchange not in EXCHANGES:
        raise ValueError(
            f"Unknown exchange '{exchange}'. Supported: {list(EXCHANGES)}"
        )
    return EXCHANGES[exchange]


def _session():
    s = requests.Session()
    return s


def fetch_1m_chunk(exchange, symbol, start_ts, session=None):
    """Fetch up to 1000 1m candles starting at start_ts. Returns list of candle rows."""
    base, path, _ = exchange_endpoint(exchange)
    session = session or _session()
    params = {
        "interval": "1m",
        "symbol": jh.dashless_symbol(symbol),
        "startTime": int(start_ts),
        "limit": MAX_LIMIT,
    }
    resp = session.get(base + path, params=params, timeout=30)
    if resp.status_code == 451:
        raise RuntimeError(
            "Binance returned HTTP 451 (geo-restricted). Try exchange 'Binance US Spot' "
            "or use a VPN."
        )
    if resp.status_code == 400:
        raise ValueError(f"Bad request for {symbol} on {exchange}: {resp.text[:200]}")
    if resp.status_code // 100 != 2:
        raise RuntimeError(f"Exchange error {resp.status_code}: {resp.reason}")
    data = resp.json()
    rows = []
    for d in data:
        rows.append([
            int(d[0]),          # timestamp
            float(d[1]),        # open
            float(d[4]),        # close
            float(d[2]),        # high
            float(d[3]),        # low
            float(d[5]),        # volume
        ])
    return rows


def fetch_1m_range(exchange, symbol, start_ts, finish_ts, on_progress=None,
                   should_stop=None, rate_limit_sleep=0.25):
    """
    Fetch all 1m candles in [start_ts, finish_ts). Paginates 1000-at-a-time.
    Calls on_progress(fetched_ms, total_ms) periodically. Returns a numpy array.
    """
    session = _session()
    total = max(finish_ts - start_ts, 1)
    cursor = start_ts
    all_rows = []
    while cursor < finish_ts:
        if should_stop and should_stop():
            break
        rows = fetch_1m_chunk(exchange, symbol, cursor, session)
        if not rows:
            break
        rows = [r for r in rows if r[0] < finish_ts]
        if not rows:
            break
        all_rows.extend(rows)
        last_ts = rows[-1][0]
        cursor = last_ts + ONE_MIN_MS
        if on_progress:
            on_progress(min(cursor - start_ts, total), total)
        if len(rows) < MAX_LIMIT:
            break
        time.sleep(rate_limit_sleep)
    if not all_rows:
        return np.empty((0, 6))
    arr = np.array(all_rows, dtype=float)
    # dedup + sort by timestamp
    _, unique_idx = np.unique(arr[:, 0], return_index=True)
    return arr[np.sort(unique_idx)]


def get_starting_time(exchange, symbol):
    """Approximate earliest available 1m timestamp for a symbol (weekly klines probe)."""
    base, path, _ = exchange_endpoint(exchange)
    params = {"interval": "1w", "symbol": jh.dashless_symbol(symbol), "limit": 1000}
    resp = _session().get(base + path, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if len(data) < 2:
        return int(data[0][0]) if data else None
    return int(data[1][0])
