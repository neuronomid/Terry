"""Synthetic candle factories (for tests and quick experiments). Analog of jesse.factories."""
import numpy as np

from . import helpers as jh

FIRST_TIMESTAMP = 1609459200000  # 2021-01-01T00:00:00Z


def candles_from_close_prices(prices) -> np.ndarray:
    """Build 1m candles from a list of close prices. open=prev close; high/low bracket them."""
    prices = list(prices)
    out = []
    prev = prices[0]
    for i, p in enumerate(prices):
        o = prev
        c = p
        hi = max(o, c)
        lo = min(o, c)
        out.append([FIRST_TIMESTAMP + i * 60_000, o, c, hi, lo, 1.0])
        prev = c
    return np.array(out, dtype=float)


def range_candles(count: int) -> np.ndarray:
    prices = np.cumsum(np.random.randn(count)) + 100
    return candles_from_close_prices(prices.tolist())
