"""Synthetic candle factories matching Jesse's notebook/test helpers."""
from random import randint

import numpy as np

FIRST_TIMESTAMP = 1_609_459_080_000
_timestamp = FIRST_TIMESTAMP
_open_price = randint(40, 100)
_close_price = randint(_open_price, 110) if randint(0, 1) else randint(30, _open_price)


def candles_from_close_prices(prices) -> np.ndarray:
    """Build Jesse-shaped 1m candles from an iterable of close prices."""
    global _timestamp
    fake_candle(reset=True)
    output = []
    previous = np.nan
    for price in prices:
        if np.isnan(previous):
            previous = price - 0.5
        _timestamp += 60_000
        output.append([
            _timestamp, previous, price, max(previous, price), min(previous, price),
            randint(0, 200),
        ])
        previous = price
    return np.asarray(output, dtype=float)


def range_candles(count: int) -> np.ndarray:
    fake_candle(reset=True)
    output = np.zeros((count, 6))
    for index in range(count):
        output[index] = fake_candle()
    return output


def fake_candle(attributes: dict | None = None, reset: bool = False) -> np.ndarray:
    """Generate the next stateful synthetic candle, with optional field overrides."""
    global _timestamp, _open_price, _close_price
    if reset:
        _timestamp = FIRST_TIMESTAMP
        _open_price = randint(40, 100)
        _close_price = randint(_open_price, 110)
    attributes = attributes or {}
    _timestamp += 60_000
    _open_price = _close_price
    _close_price += randint(1, 8)
    high = max(_open_price, _close_price)
    low = min(_open_price - 1, _close_price)
    return np.array([
        attributes.get("timestamp", _timestamp),
        attributes.get("open", _open_price),
        attributes.get("close", _close_price),
        attributes.get("high", high),
        attributes.get("low", low),
        attributes.get("volume", randint(1, 100)),
    ], dtype=np.float64)


__all__ = ["fake_candle", "range_candles", "candles_from_close_prices"]
