"""Strategy utility functions with Jesse-compatible signatures and behaviour."""
import math
from decimal import Decimal

import numpy as np
import pandas as pd

from . import helpers as jh


def size_to_qty(position_size: float, entry_price: float, precision: int = 3,
                fee_rate: float = 0.0) -> float:
    """Convert a position size (in quote) to a base-asset quantity, accounting for fees."""
    if entry_price is None:
        raise TypeError("entry_price is None")
    if math.isnan(position_size) or math.isnan(entry_price):
        raise TypeError(f"position_size: {position_size}, entry_price: {entry_price}")
    if entry_price == 0:
        raise ValueError("entry_price cannot be zero")
    if fee_rate != 0:
        # Jesse reserves three fee legs so a generated quantity cannot exhaust margin.
        position_size *= 1 - fee_rate * 3
    return jh.floor_with_precision(position_size / entry_price, precision)


def qty_to_size(qty: float, price: float) -> float:
    if math.isnan(qty) or math.isnan(price):
        raise TypeError()
    return qty * price


def risk_to_size(capital_size: float, risk_percentage: float, risk_per_qty: float,
                 entry_price: float) -> float:
    if risk_per_qty == 0:
        raise ValueError("risk cannot be zero")
    risk_percentage /= 100
    temp_size = ((risk_percentage * capital_size) / risk_per_qty) * entry_price
    return min(temp_size, capital_size)


def risk_to_qty(capital: float, risk_per_capital: float, entry_price: float,
                stop_loss_price: float, precision: int = 8, fee_rate: float = 0.0) -> float:
    """Size so that a stop-out loses `risk_per_capital`% of capital. risk_per_capital is a %."""
    risk_per_qty = abs(entry_price - stop_loss_price)
    size = risk_to_size(capital, risk_per_capital, risk_per_qty, entry_price)
    if fee_rate != 0:
        size = size * (1 - fee_rate * 3)
    return size_to_qty(size, entry_price, precision=precision, fee_rate=fee_rate)


def estimate_risk(entry_price: float, stop_price: float) -> float:
    if math.isnan(entry_price) or math.isnan(stop_price):
        raise TypeError()
    return abs(entry_price - stop_price)


def limit_stop_loss(entry_price, stop_price, trade_type, max_allowed_risk_percentage):
    """Clamp a stop to a max risk %."""
    risk = abs(entry_price - stop_price) / entry_price * 100
    if risk > max_allowed_risk_percentage:
        risk = max_allowed_risk_percentage
        if trade_type == "long":
            return entry_price * (1 - risk / 100)
        return entry_price * (1 + risk / 100)
    return stop_price


def kelly_criterion(win_rate: float, ratio_avg_win_loss: float) -> float:
    if ratio_avg_win_loss == 0:
        return 0.0
    return win_rate - (1 - win_rate) / ratio_avg_win_loss


def crossed(series1, series2, direction=None, sequential=False):
    """Detect crossovers, returning a bool or a full boolean series like Jesse."""
    series1 = np.asarray(series1, dtype=float)
    scalar = np.isscalar(series2) or np.ndim(series2) == 0
    series2 = float(series2) if scalar else np.asarray(series2, dtype=float)

    if sequential:
        shifted1 = jh.np_shift(series1, 1, np.nan)
        shifted2 = series2 if scalar else jh.np_shift(series2, 1, np.nan)
        if direction is None or direction == "above":
            cross_above = np.logical_and(series1 > series2, shifted1 <= shifted2)
        if direction is None or direction == "below":
            cross_below = np.logical_and(series1 < series2, shifted1 >= shifted2)
        if direction is None:
            return np.logical_or(cross_above, cross_below)
        if direction == "above":
            return cross_above
        if direction == "below":
            return cross_below
        raise ValueError("direction must be 'above', 'below', or None")

    if len(series1) < 2 or (not scalar and len(series2) < 2):
        return False
    if scalar:
        series2 = np.array([series2, series2])
    if direction is None or direction == "above":
        cross_above = series1[-2] <= series2[-2] and series1[-1] > series2[-1]
    if direction is None or direction == "below":
        cross_below = series1[-2] >= series2[-2] and series1[-1] < series2[-1]

    if direction == "above":
        return bool(cross_above)
    if direction == "below":
        return bool(cross_below)
    if direction is None:
        return bool(cross_above or cross_below)
    raise ValueError("direction must be 'above', 'below', or None")


def subtract_floats(float1: float, float2: float) -> float:
    """Subtract decimal values without binary floating-point drift."""
    return float(Decimal(str(float1)) - Decimal(str(float2)))


def sum_floats(float1: float, float2: float) -> float:
    """Add decimal values without binary floating-point drift."""
    return float(Decimal(str(float1)) + Decimal(str(float2)))


def strictly_increasing(series: np.ndarray, lookback: int) -> bool:
    return bool(np.all(np.diff(np.asarray(series)[-lookback:]) > 0))


def strictly_decreasing(series: np.ndarray, lookback: int) -> bool:
    return bool(np.all(np.diff(np.asarray(series)[-lookback:]) < 0))


def streaks(series: np.ndarray, use_diff=True) -> np.ndarray:
    series = np.asarray(series)
    original_length = len(series)
    if use_diff:
        series = np.diff(series)
    pos = np.clip(series, 0, 1).astype(bool).cumsum()
    neg = np.clip(series, -1, 0).astype(bool).cumsum()
    streak = np.where(
        series >= 0,
        pos - np.maximum.accumulate(np.where(series <= 0, pos, 0)),
        -neg + np.maximum.accumulate(np.where(series >= 0, neg, 0)),
    )
    return np.concatenate((np.full(original_length - len(streak), np.nan), streak))


def signal_line(series: np.ndarray, period: int = 10, matype: int = 0) -> np.ndarray:
    """Return a Jesse-style moving-average signal line for a one-dimensional series."""
    from .indicators.ma import ma

    values = np.asarray(series, dtype=float)
    candles = np.column_stack((
        np.arange(len(values), dtype=float), values, values, values, values,
        np.ones(len(values), dtype=float),
    ))
    return ma(candles, period=period, matype=matype, sequential=True)


def numpy_candles_to_dataframe(candles, name_date="date", name_open="open",
                               name_high="high", name_low="low", name_close="close",
                               name_volume="volume") -> pd.DataFrame:
    df = pd.DataFrame(
        candles[:, 1:], index=pd.to_datetime(candles[:, 0], unit="ms"),
        columns=[name_open, name_close, name_high, name_low, name_volume],
    )
    df[name_date] = df.index
    return df[[name_date, name_open, name_high, name_low, name_close, name_volume]]


# ------------------------------------------------------------------ pairs stats
def prices_to_returns(price_series: np.ndarray) -> np.ndarray:
    price_series = np.asarray(price_series, dtype=float)
    ret = np.diff(price_series) / price_series[:-1] * 100
    return np.insert(ret, 0, np.nan)


def z_score(series: np.ndarray) -> np.ndarray:
    series = np.asarray(series, dtype=float)
    std = np.nanstd(series)
    if std == 0:
        return np.zeros_like(series)
    return (series - np.nanmean(series)) / std


def calculate_alpha_beta(returns1: np.ndarray, returns2: np.ndarray):
    import statsmodels.api as sm

    model = sm.OLS(returns1, sm.add_constant(returns2)).fit()
    return float(model.params[0]), float(model.params[1])


def are_cointegrated(price_returns_1, price_returns_2, cutoff=0.05) -> bool:
    """Return whether the Engle-Granger cointegration p-value is below ``cutoff``."""
    from statsmodels.tsa.stattools import coint

    return bool(coint(price_returns_1, price_returns_2)[1] < cutoff)


def combinations_without_repeat(a: np.ndarray, n: int = 2) -> np.ndarray:
    if n <= 1:
        raise ValueError("n must be >= 2")
    from itertools import permutations

    return np.array(list(permutations(np.asarray(a), n)))


def dd(msg: str) -> None:
    """Print a debugging value and terminate the current Python run."""
    print(msg)
    raise SystemExit(1)


def timeframe_to_one_minutes(timeframe: str) -> int:
    """Convert a Jesse timeframe string to its number of one-minute candles."""
    try:
        return jh.timeframe_to_one_minutes(timeframe)
    except ValueError as exc:
        from .exceptions import InvalidTimeframe

        raise InvalidTimeframe(str(exc)) from exc


def anchor_timeframe(timeframe: str) -> str:
    mapping = {
        "1m": "5m", "3m": "15m", "5m": "30m", "15m": "2h", "30m": "3h", "45m": "3h",
        "1h": "4h", "2h": "6h", "3h": "1D", "4h": "1D", "6h": "1D", "8h": "1D",
        "12h": "1D",
    }
    if timeframe not in mapping:
        raise KeyError(timeframe)
    return mapping[timeframe]
