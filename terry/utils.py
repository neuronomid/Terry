"""Strategy utility functions — sizing, crossovers, pairs stats (jesse.utils analog)."""
import numpy as np
import pandas as pd


def size_to_qty(position_size: float, entry_price: float, precision: int = 3,
                fee_rate: float = 0.0) -> float:
    """Convert a position size (in quote) to a base-asset quantity, accounting for fees."""
    if entry_price == 0:
        return 0.0
    if fee_rate != 0:
        position_size *= 1 - fee_rate
    qty = position_size / entry_price
    return round(qty, precision)


def qty_to_size(qty: float, price: float) -> float:
    return qty * price


def risk_to_size(capital_size: float, risk_percentage: float, risk_per_qty: float,
                 entry_price: float) -> float:
    if risk_per_qty == 0:
        return 0.0
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
    return size_to_qty(size, entry_price, precision=precision, fee_rate=0)


def estimate_risk(entry_price: float, stop_price: float) -> float:
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


def crossed(series1, series2, direction=None, sequential=False) -> bool:
    """Detect a crossover between two series (or a series and a constant)."""
    series1 = np.asarray(series1, dtype=float)
    if np.isscalar(series2) or (hasattr(series2, "ndim") and np.ndim(series2) == 0):
        series2 = np.full_like(series1, float(series2))
    else:
        series2 = np.asarray(series2, dtype=float)

    if len(series1) < 2 or len(series2) < 2:
        return False

    if direction is None or direction == "above":
        cross_above = (series1[-2] <= series2[-2]) & (series1[-1] > series2[-1])
    if direction is None or direction == "below":
        cross_below = (series1[-2] >= series2[-2]) & (series1[-1] < series2[-1])

    if direction == "above":
        return bool(cross_above)
    if direction == "below":
        return bool(cross_below)
    return bool(cross_above or cross_below)


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
    returns1 = np.asarray(returns1, dtype=float)
    returns2 = np.asarray(returns2, dtype=float)
    mask = np.isfinite(returns1) & np.isfinite(returns2)
    x, y = returns2[mask], returns1[mask]
    if len(x) < 2:
        return 0.0, 0.0
    beta, alpha = np.polyfit(x, y, 1)
    return float(alpha), float(beta)


def are_cointegrated(price_returns_1, price_returns_2, cutoff=0.05) -> bool:
    """
    Lightweight cointegration proxy: regress series 1 on series 2, then check whether the
    residual spread mean-reverts (lag-1 autocorrelation well below 1). Good enough for
    pairs screening without the statsmodels dependency.
    """
    a = np.asarray(price_returns_1, dtype=float)
    b = np.asarray(price_returns_2, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 30:
        return False
    beta = np.polyfit(b, a, 1)[0]
    spread = a - beta * b
    spread = spread - spread.mean()
    if len(spread) < 3 or np.std(spread) == 0:
        return False
    rho = np.corrcoef(spread[:-1], spread[1:])[0, 1]
    return bool(rho < (1 - cutoff))


def combinations_without_repeat(a: np.ndarray, n: int = 2) -> np.ndarray:
    a = np.asarray(a)
    if n == 2:
        out = np.array([[x, y] for i, x in enumerate(a) for j, y in enumerate(a) if i != j])
        return out
    from itertools import permutations
    return np.array(list(permutations(a, n)))


def anchor_timeframe(timeframe: str) -> str:
    mapping = {
        "1m": "5m", "3m": "15m", "5m": "30m", "15m": "2h", "30m": "3h", "45m": "4h",
        "1h": "4h", "2h": "6h", "3h": "12h", "4h": "1D", "6h": "1D", "8h": "1D",
        "12h": "1D", "1D": "1W",
    }
    return mapping.get(timeframe, "1D")
