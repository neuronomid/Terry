"""Helper functions — a focused analog of jesse.helpers (jh)."""
import random
import string
import uuid
from datetime import datetime, timezone

import numpy as np

# ---------------------------------------------------------------------------
# Timeframe math
# ---------------------------------------------------------------------------
_TF_MINUTES = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "45m": 45,
    "1h": 60, "2h": 120, "3h": 180, "4h": 240, "6h": 360, "8h": 480,
    "12h": 720, "1D": 1440, "1d": 1440, "3D": 4320, "3d": 4320,
    "1W": 10080, "1w": 10080, "1M": 43_200,
}


def timeframe_to_one_minutes(timeframe: str) -> int:
    try:
        return _TF_MINUTES[timeframe]
    except KeyError:
        raise ValueError(
            f"Timeframe '{timeframe}' is invalid. Supported: {list(_TF_MINUTES)}"
        )


def max_timeframe(timeframes_list) -> str:
    order = list(_TF_MINUTES.keys())
    best = order[0]
    for tf in timeframes_list:
        if order.index(tf) > order.index(best):
            best = tf
    return best


# ---------------------------------------------------------------------------
# Symbols
# ---------------------------------------------------------------------------
def dashless_symbol(symbol: str) -> str:
    return symbol.replace("-", "")


def dashy_symbol(symbol: str) -> str:
    if "-" in symbol:
        return symbol
    # try common quote assets
    for quote in ("USDT", "USDC", "BUSD", "USD", "BTC", "ETH", "EUR"):
        if symbol.endswith(quote):
            return f"{symbol[:-len(quote)]}-{quote}"
    return symbol


def base_asset(symbol: str) -> str:
    return symbol.split("-")[0]


def quote_asset(symbol: str) -> str:
    return symbol.split("-")[1]


def key(exchange: str, symbol: str, timeframe: str = None) -> str:
    if timeframe is None:
        return f"{exchange}-{symbol}"
    return f"{exchange}-{symbol}-{timeframe}"


# ---------------------------------------------------------------------------
# Time / dates  (all timestamps are milliseconds since epoch, UTC)
# ---------------------------------------------------------------------------
def now_to_timestamp() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def today_to_timestamp() -> int:
    d = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    return int(d.timestamp() * 1000)


def date_to_timestamp(date_str: str) -> int:
    """'2021-01-01' -> ms timestamp (UTC midnight)."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def timestamp_to_date(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def timestamp_to_iso8601(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S+00:00"
    )


def timestamp_to_time(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def date_diff_in_days(start_ts: int, end_ts: int) -> int:
    return int((end_ts - start_ts) / (1000 * 60 * 60 * 24))


# ---------------------------------------------------------------------------
# IDs
# ---------------------------------------------------------------------------
def generate_unique_id() -> str:
    return str(uuid.uuid4())


def random_str(num_characters: int = 8) -> str:
    return "".join(random.choice(string.ascii_letters) for _ in range(num_characters))


def string_after_character(s: str, character: str) -> str:
    return s.split(character, 1)[-1]


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------
def round_price_for_live_mode(price, precision: int = 8) -> float:
    return round(float(price), precision)


def floor_with_precision(num: float, precision: int = 0) -> float:
    temp = 10 ** precision
    return __import__("math").floor(num * temp) / temp


def prepare_qty(qty, side: str) -> float:
    if side.lower() in ("sell", "short"):
        return -abs(qty)
    if side.lower() in ("buy", "long"):
        return abs(qty)
    if side.lower() == "close":
        return 0.0
    raise ValueError(f"{side} is not a valid input")


def np_shift(arr: np.ndarray, num: int, fill_value=0) -> np.ndarray:
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


def np_ffill(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    idx_shape = tuple([slice(None)] + [np.newaxis] * (len(arr.shape) - axis - 1))
    idx = np.where(~np.isnan(arr), np.arange(arr.shape[axis])[idx_shape], 0)
    np.maximum.accumulate(idx, axis=axis, out=idx)
    slc = [
        np.arange(k)[
            tuple(slice(None) if dim == i else np.newaxis for dim in range(len(arr.shape)))
        ]
        for i, k in enumerate(arr.shape)
    ]
    slc[axis] = idx
    return arr[tuple(slc)]


def same_length(bigger: np.ndarray, shorter: np.ndarray) -> np.ndarray:
    return np.concatenate((np.full((bigger.shape[0] - shorter.shape[0]), np.nan), shorter))


# ---------------------------------------------------------------------------
# Candle source selection (used by the indicator library)
# ---------------------------------------------------------------------------
CANDLE_SOURCE_MAPPING = {
    "open":   lambda c: c[:, 1],
    "close":  lambda c: c[:, 2],
    "high":   lambda c: c[:, 3],
    "low":    lambda c: c[:, 4],
    "volume": lambda c: c[:, 5],
    "hl2":    lambda c: (c[:, 3] + c[:, 4]) / 2,
    "hlc3":   lambda c: (c[:, 3] + c[:, 4] + c[:, 2]) / 3,
    "ohlc4":  lambda c: (c[:, 1] + c[:, 3] + c[:, 4] + c[:, 2]) / 4,
}

# number of trailing candles kept when computing a non-sequential indicator value
WARMUP_CANDLES_NUM = 240


def get_candle_source(candles: np.ndarray, source_type: str = "close") -> np.ndarray:
    """Return the price series for the requested source type."""
    try:
        return CANDLE_SOURCE_MAPPING[source_type](candles)
    except KeyError:
        raise ValueError(f"Source type '{source_type}' not recognised")


def slice_candles(candles: np.ndarray, sequential: bool) -> np.ndarray:
    """For non-sequential calls, trim to the last WARMUP_CANDLES_NUM candles (matches Jesse)."""
    if not sequential and candles.shape[0] > WARMUP_CANDLES_NUM:
        candles = candles[-WARMUP_CANDLES_NUM:]
    return candles
