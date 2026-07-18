"""Notebook-friendly candle research functions compatible with Jesse's API."""

from __future__ import annotations

import time

import numpy as np

from .. import helpers as jh
from ..engine.candle_store import aggregate_candles
from ..factories import candles_from_close_prices, fake_candle, range_candles


def import_candles(exchange: str, symbol: str, start_date: str,
                   show_progressbar: bool = True) -> str:
    """Import public candles and wait for the local background job to finish."""
    ctx = _context()
    import_id = ctx.importer.start_import(exchange, symbol, start_date)
    last_progress = -1
    while True:
        status = ctx.importer.get_status(import_id)
        progress = int(status.get("progress", 0))
        if show_progressbar and progress != last_progress:
            print(f"\rImporting {exchange} {symbol}: {progress:3d}%", end="", flush=True)
            last_progress = progress
        if status["status"] == "finished":
            if show_progressbar:
                print()
            return status.get("message", "Import finished")
        if status["status"] in {"error", "canceled", "not_found"}:
            if show_progressbar:
                print()
            raise RuntimeError(status.get("message", f"Candle import {status['status']}"))
        time.sleep(0.05)


def store_candles(candles: np.ndarray, exchange: str, symbol: str) -> None:
    if not isinstance(candles, np.ndarray):
        raise TypeError("candles must be a numpy array.")
    if candles.ndim != 2 or candles.shape[1] != 6 or len(candles) < 2:
        raise ValueError("candles must have shape (n, 6) and contain at least two rows")
    difference = int(candles[1, 0] - candles[0, 0])
    if difference != 60_000:
        raise ValueError(
            "Candles passed to research.store_candles() must be 1m candles. "
            f"The timestamp difference is {difference} milliseconds."
        )
    _context().candle_db.store(exchange, symbol, candles)


def get_candles(exchange: str, symbol: str, timeframe: str,
                start_date_timestamp: int, finish_date_timestamp: int,
                warmup_candles_num: int = 0, caching: bool = False,
                is_for_jesse: bool = False):
    """Return ``(warmup_candles, trading_candles)`` from Terry's SQLite store."""
    del caching, is_for_jesse  # SQLite is already a local cache; Terry uses Jesse's array shape.
    if finish_date_timestamp <= start_date_timestamp:
        raise ValueError("finish_date_timestamp must be after start_date_timestamp")
    if warmup_candles_num < 0:
        raise ValueError("warmup_candles_num cannot be negative")
    timeframe_minutes = jh.timeframe_to_one_minutes(timeframe)
    raw_start = start_date_timestamp - warmup_candles_num * timeframe_minutes * 60_000
    raw = _context().candle_db.get(
        exchange, symbol, raw_start, finish_date_timestamp)
    if len(raw) == 0:
        return None, np.empty((0, 6))
    aggregated = aggregate_candles(raw, timeframe)
    trading = aggregated[aggregated[:, 0] >= start_date_timestamp]
    if warmup_candles_num:
        warmup = aggregated[aggregated[:, 0] < start_date_timestamp][-warmup_candles_num:]
    else:
        warmup = None
    return warmup, trading


def fake_range_candles(count: int) -> np.ndarray:
    return range_candles(count)


def _context():
    # Lazy import avoids context -> runner -> research -> candles -> context cycles.
    from ..context import get_context

    return get_context()


__all__ = [
    "import_candles", "store_candles", "get_candles", "fake_candle",
    "fake_range_candles", "candles_from_close_prices",
]
