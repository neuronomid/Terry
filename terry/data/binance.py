"""Free public candle drivers for exchanges Jesse supports in backtesting.

The module keeps its historical name for import compatibility. All drivers use unauthenticated
REST endpoints and normalize rows to Terry's ``[timestamp, open, close, high, low, volume]``
one-minute candle format.
"""

from __future__ import annotations

import time

import numpy as np
import requests

from .. import helpers as jh

# exchange name -> (base_url, endpoint, market_type). The tuple shape is retained because the
# runner and external callers use item 2 to derive spot/futures engine configuration.
EXCHANGES = {
    "Binance Spot": ("https://api.binance.com", "/api/v3/klines", "spot"),
    "Binance": ("https://api.binance.com", "/api/v3/klines", "spot"),
    "Binance US Spot": ("https://api.binance.us", "/api/v3/klines", "spot"),
    "Binance Perpetual Futures": (
        "https://fapi.binance.com", "/fapi/v1/klines", "futures"),
    "Binance USDT Perpetual": (
        "https://fapi.binance.com", "/fapi/v1/klines", "futures"),
    "Bitfinex Spot": ("https://api-pub.bitfinex.com", "/v2/candles", "spot"),
    "Coinbase Spot": (
        "https://api.coinbase.com", "/api/v3/brokerage/market/products", "spot"),
    "Bybit USDT Perpetual": ("https://api.bybit.com", "/v5/market/kline", "futures"),
    "Bybit USDC Perpetual": ("https://api.bybit.com", "/v5/market/kline", "futures"),
    "Bybit Spot": ("https://api.bybit.com", "/v5/market/kline", "spot"),
    "Gate USDT Perpetual": (
        "https://api.gateio.ws", "/api/v4/futures/usdt/candlesticks", "futures"),
    "Kraken Pro Futures": (
        "https://futures.kraken.com", "/api/charts/v1/trade", "futures"),
}

DEFAULT_EXCHANGE = "Binance Perpetual Futures"
MAX_LIMIT = 1000
ONE_MIN_MS = 60_000

_PAGE_LIMITS = {
    "Bitfinex Spot": 1440,
    "Coinbase Spot": 300,
    "Bybit USDT Perpetual": 200,
    "Bybit USDC Perpetual": 200,
    "Bybit Spot": 200,
    "Gate USDT Perpetual": 2000,
    "Kraken Pro Futures": 5000,
}


def exchange_endpoint(exchange: str):
    if exchange not in EXCHANGES:
        raise ValueError(f"Unknown exchange '{exchange}'. Supported: {list(EXCHANGES)}")
    return EXCHANGES[exchange]


def _session():
    return requests.Session()


def _validate_response(response, exchange, symbol):
    if response.status_code == 451:
        raise RuntimeError(
            f"{exchange} returned HTTP 451 (geo-restricted). Try an exchange available in "
            "your region or use a permitted network location."
        )
    if response.status_code in (400, 404):
        raise ValueError(
            f"Bad request for {symbol} on {exchange}: "
            f"{getattr(response, 'text', '')[:200]}"
        )
    if response.status_code // 100 != 2:
        raise RuntimeError(
            f"{exchange} error {response.status_code}: {getattr(response, 'reason', '')}"
        )


def fetch_1m_chunk(exchange, symbol, start_ts, session=None):
    """Fetch one page of 1m candles starting at ``start_ts``."""
    base, path, _ = exchange_endpoint(exchange)
    session = session or _session()
    page_limit = _PAGE_LIMITS.get(exchange, MAX_LIMIT)

    if exchange.startswith("Binance"):
        response = session.get(base + path, params={
            "interval": "1m", "symbol": jh.dashless_symbol(symbol),
            "startTime": int(start_ts), "limit": page_limit,
        }, timeout=30)
        _validate_response(response, exchange, symbol)
        data = response.json()
        rows = [[int(d[0]), float(d[1]), float(d[4]), float(d[2]),
                 float(d[3]), float(d[5])] for d in data]
    elif exchange.startswith("Bybit"):
        category = "spot" if exchange == "Bybit Spot" else "linear"
        response = session.get(base + path, params={
            "category": category, "symbol": jh.dashless_symbol(symbol),
            "interval": "1", "start": int(start_ts), "limit": page_limit,
        }, timeout=30)
        _validate_response(response, exchange, symbol)
        payload = response.json()
        if payload.get("retMsg") != "OK":
            raise ValueError(f"Bad request for {symbol} on {exchange}: {payload.get('retMsg')}")
        rows = [[int(d[0]), float(d[1]), float(d[4]), float(d[2]),
                 float(d[3]), float(d[5])] for d in reversed(payload["result"]["list"])]
    elif exchange == "Coinbase Spot":
        end_ts = start_ts + (page_limit - 1) * ONE_MIN_MS
        response = session.get(f"{base}{path}/{symbol}/candles", params={
            "granularity": "ONE_MINUTE", "start": int(start_ts / 1000),
            "end": int(end_ts / 1000),
        }, timeout=30)
        _validate_response(response, exchange, symbol)
        data = response.json().get("candles", [])
        rows = [[int(d["start"]) * 1000, float(d["open"]), float(d["close"]),
                 float(d["high"]), float(d["low"]), float(d["volume"])]
                for d in reversed(data)]
    elif exchange == "Bitfinex Spot":
        # Bitfinex returns HTTP 500 when ``end`` is beyond the latest closed candle.
        end_ts = min(start_ts + (page_limit - 1) * ONE_MIN_MS,
                     int(time.time() * 1000) // ONE_MIN_MS * ONE_MIN_MS)
        network_symbol = symbol.replace("-", "").upper()
        response = session.get(
            f"{base}{path}/trade:1m:t{network_symbol}/hist",
            params={"start": int(start_ts), "end": int(end_ts),
                    "limit": page_limit, "sort": 1}, timeout=30)
        _validate_response(response, exchange, symbol)
        rows = [[int(d[0]), float(d[1]), float(d[2]), float(d[3]),
                 float(d[4]), float(d[5])] for d in response.json()]
    elif exchange == "Gate USDT Perpetual":
        end_ts = start_ts + (page_limit - 1) * ONE_MIN_MS
        response = session.get(base + path, params={
            "contract": symbol.replace("-", "_").upper(), "interval": "1m",
            "from": int(start_ts / 1000), "to": int(end_ts / 1000),
        }, timeout=30)
        _validate_response(response, exchange, symbol)
        rows = [[int(d["t"]) * 1000, float(d["o"]), float(d["c"]),
                 float(d["h"]), float(d["l"]), float(d["v"])]
                for d in response.json()]
    elif exchange == "Kraken Pro Futures":
        base_asset, quote_asset = symbol.upper().split("-", 1)
        if base_asset == "BTC":
            base_asset = "XBT"
        network_symbol = f"PF_{base_asset}{quote_asset}"
        end_ts = start_ts + page_limit * ONE_MIN_MS
        response = session.get(
            f"{base}{path}/{network_symbol}/1m",
            params={"from": int(start_ts / 1000), "to": int(end_ts / 1000)},
            timeout=30)
        _validate_response(response, exchange, symbol)
        rows = [[int(d["time"]), float(d["open"]), float(d["close"]),
                 float(d["high"]), float(d["low"]), float(d["volume"])]
                for d in response.json().get("candles", [])]
    else:  # pragma: no cover - guarded by the registry and branches above
        raise ValueError(f"No candle driver is configured for {exchange}.")

    return sorted((row for row in rows if row[0] >= int(start_ts)), key=lambda row: row[0])


def fetch_1m_range(exchange, symbol, start_ts, finish_ts, on_progress=None,
                   should_stop=None, rate_limit_sleep=0.25):
    """Fetch all 1m candles in ``[start_ts, finish_ts)`` and normalize/deduplicate them."""
    session = _session()
    total = max(finish_ts - start_ts, 1)
    cursor = start_ts
    all_rows = []
    page_limit = _PAGE_LIMITS.get(exchange, MAX_LIMIT)
    while cursor < finish_ts:
        if should_stop and should_stop():
            break
        rows = fetch_1m_chunk(exchange, symbol, cursor, session)
        rows = [row for row in rows if cursor <= row[0] < finish_ts]
        if not rows:
            break
        all_rows.extend(rows)
        next_cursor = rows[-1][0] + ONE_MIN_MS
        if next_cursor <= cursor:
            raise RuntimeError(f"{exchange} candle API did not advance its cursor.")
        cursor = next_cursor
        if on_progress:
            on_progress(min(cursor - start_ts, total), total)
        if len(rows) < page_limit:
            break
        time.sleep(rate_limit_sleep)
    if not all_rows:
        return np.empty((0, 6))
    array = np.array(all_rows, dtype=float)
    _, unique_indices = np.unique(array[:, 0], return_index=True)
    return array[np.sort(unique_indices)]


def get_starting_time(exchange, symbol):
    """Return the earliest public timestamp where the exchange provides a reliable probe."""
    base, path, _ = exchange_endpoint(exchange)
    session = _session()
    if exchange.startswith("Binance"):
        response = session.get(base + path, params={
            "interval": "1w", "symbol": jh.dashless_symbol(symbol), "limit": 1000,
        }, timeout=30)
        _validate_response(response, exchange, symbol)
        data = response.json()
        return int(data[1][0]) if len(data) > 1 else (int(data[0][0]) if data else None)
    if exchange.startswith("Bybit"):
        category = "spot" if exchange == "Bybit Spot" else "linear"
        response = session.get(base + path, params={
            "category": category, "symbol": jh.dashless_symbol(symbol), "interval": "W",
            "limit": 200, "start": 1514811660000,
        }, timeout=30)
        _validate_response(response, exchange, symbol)
        data = list(reversed(response.json().get("result", {}).get("list", [])))
        return int(data[1][0]) if len(data) > 1 else (int(data[0][0]) if data else None)
    if exchange == "Coinbase Spot":
        return {"BTC-USD": 1438387200000, "ETH-USD": 1464739200000,
                "LTC-USD": 1477958400000}.get(symbol.upper())
    if exchange == "Bitfinex Spot":
        network_symbol = symbol.replace("-", "").upper()
        response = session.get(
            f"{base}{path}/trade:1D:t{network_symbol}/hist",
            params={"sort": 1, "limit": 5000}, timeout=30)
        _validate_response(response, exchange, symbol)
        data = response.json()
        return int(data[0][0]) + 86_400_000 if data else None
    if exchange == "Gate USDT Perpetual":
        response = session.get(base + path, params={
            "contract": symbol.replace("-", "_").upper(), "interval": "1w",
            "limit": 1000, "from": 1514811660,
        }, timeout=30)
        _validate_response(response, exchange, symbol)
        data = response.json()
        return int(data[0]["t"]) * 1000 if data else None
    if exchange == "Kraken Pro Futures":
        base_asset, quote_asset = symbol.upper().split("-", 1)
        base_asset = "XBT" if base_asset == "BTC" else base_asset
        response = session.get(
            f"{base}{path}/PF_{base_asset}{quote_asset}/1d",
            params={"from": 1577836800, "to": int(time.time())}, timeout=30)
        _validate_response(response, exchange, symbol)
        data = response.json().get("candles", [])
        return int(data[0]["time"]) if data else None
    return None
