"""
Candle storage + on-the-fly timeframe aggregation with candle-close (causal) visibility.

Candle row format (matches Jesse): [timestamp_ms, open, close, high, low, volume].
Input candles are always 1-minute; larger timeframes are aggregated from them and only
revealed once fully closed relative to the simulator clock.
"""
import numpy as np

from .. import helpers as jh

ONE_MIN_MS = 60_000


def aggregate_candles(candles_1m: np.ndarray, timeframe: str) -> np.ndarray:
    """Aggregate a 1m candle array into `timeframe` candles (full array, non-causal)."""
    tf = jh.timeframe_to_one_minutes(timeframe)
    if tf == 1:
        return candles_1m
    n = len(candles_1m)
    # align to the first bucket boundary
    usable = (n // tf) * tf
    if usable == 0:
        return np.empty((0, 6))
    arr = candles_1m[:usable]
    reshaped = arr.reshape(-1, tf, 6)
    out = np.empty((reshaped.shape[0], 6))
    out[:, 0] = reshaped[:, 0, 0]              # timestamp = first
    out[:, 1] = reshaped[:, 0, 1]              # open = first open
    out[:, 2] = reshaped[:, -1, 2]             # close = last close
    out[:, 3] = reshaped[:, :, 3].max(axis=1)  # high
    out[:, 4] = reshaped[:, :, 4].min(axis=1)  # low
    out[:, 5] = reshaped[:, :, 5].sum(axis=1)  # volume
    return out


class CandleStore:
    def __init__(self):
        self.raw_1m = {}      # key "exchange-symbol" -> full 1m np.ndarray
        self._agg_cache = {}  # key "exchange-symbol-tf" -> full aggregated np.ndarray
        self.app = None       # set by store (for the clock)

    def reset(self):
        self.raw_1m = {}
        self._agg_cache = {}

    def init_from_dict(self, candles_dict: dict):
        """candles_dict: {"exchange-symbol": {"exchange","symbol","candles": np.ndarray(1m)}}"""
        self._agg_cache = {}
        for k, v in candles_dict.items():
            self.raw_1m[jh.key(v["exchange"], v["symbol"])] = np.asarray(v["candles"], dtype=float)

    def inject_warmup(self, exchange, symbol, warmup_1m: np.ndarray):
        """Prepend warm-up candles to the raw 1m array."""
        k = jh.key(exchange, symbol)
        if k in self.raw_1m and len(self.raw_1m[k]):
            self.raw_1m[k] = np.vstack([np.asarray(warmup_1m, dtype=float), self.raw_1m[k]])
        else:
            self.raw_1m[k] = np.asarray(warmup_1m, dtype=float)
        self._agg_cache = {}

    def _full_agg(self, exchange, symbol, timeframe):
        ck = jh.key(exchange, symbol, timeframe)
        if ck not in self._agg_cache:
            base = self.raw_1m[jh.key(exchange, symbol)]
            self._agg_cache[ck] = aggregate_candles(base, timeframe)
        return self._agg_cache[ck]

    def get_candles(self, exchange, symbol, timeframe) -> np.ndarray:
        """Return candles fully closed by the current simulator clock (candle-close semantics)."""
        full = self._full_agg(exchange, symbol, timeframe)
        if self.app is None or len(full) == 0:
            return full
        tf_ms = jh.timeframe_to_one_minutes(timeframe) * ONE_MIN_MS
        cutoff_end = self.app.time + ONE_MIN_MS  # end of the current 1m candle
        visible_mask = (full[:, 0] + tf_ms) <= cutoff_end
        return full[visible_mask]

    def current_1m_price(self, exchange, symbol) -> float:
        """Close of the current 1m candle at the simulator clock."""
        base = self.raw_1m[jh.key(exchange, symbol)]
        idx = self.app.index_1m
        return float(base[idx, 2])
