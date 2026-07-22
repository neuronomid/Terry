"""
Candle storage + on-the-fly timeframe aggregation with candle-close (causal) visibility.

Candle row format (matches Jesse): [timestamp_ms, open, close, high, low, volume].
Input candles are always 1-minute; larger timeframes are aggregated from them and only
revealed once fully closed relative to the simulator clock.
"""
import numpy as np

from .. import helpers as jh

ONE_MIN_MS = 60_000


def fill_1m_gaps(candles_1m: np.ndarray) -> np.ndarray:
    """Return a contiguous 1m array, filling missing minutes with flat candles.

    Session markets (Forex, indices, commodities, stock CFDs) close on weekends and
    holidays, so their stored 1m candles are non-contiguous. Terry's timeframe
    aggregation reshapes fixed blocks of consecutive rows and its daily-balance /
    monthly-return attribution assume one row per minute, so gaps would misalign both.
    Filling gaps with flat candles (open=close=high=low=previous close, volume=0) keeps
    every downstream engine assumption identical to the always-on crypto case. Closed
    periods carry a flat price, so strategies see no movement and place no fills there.
    """
    if len(candles_1m) < 2:
        return candles_1m
    ts = candles_1m[:, 0].astype(np.int64)
    start, end = int(ts[0]), int(ts[-1])
    full_ts = np.arange(start, end + 1, ONE_MIN_MS, dtype=np.int64)
    if len(full_ts) == len(candles_1m):
        return candles_1m  # already contiguous — no-op (the crypto path)
    pos = ((ts - start) // ONE_MIN_MS).astype(np.int64)
    out = np.zeros((len(full_ts), 6))
    out[:, 0] = full_ts
    out[pos, 1:] = candles_1m[:, 1:]
    valid = np.zeros(len(full_ts), dtype=bool)
    valid[pos] = True
    # forward-fill each missing row's flat price from the previous real close
    last_valid = np.where(valid, np.arange(len(full_ts)), 0)
    np.maximum.accumulate(last_valid, out=last_valid)
    prev_close = out[last_valid, 2]
    missing = ~valid
    for col in (1, 2, 3, 4):  # open, close, high, low -> flat at previous close
        out[missing, col] = prev_close[missing]
    out[missing, 5] = 0.0  # volume
    return out


def aggregate_candles_anchored(candles_1m: np.ndarray, timeframe: str,
                               origin: int) -> np.ndarray:
    """Aggregate 1m candles into `timeframe` buckets anchored at ``origin`` (ms).

    Each row is grouped by ``floor((ts - origin) / tf_ms)`` and the bucket timestamp is the
    aligned boundary ``origin + k * tf_ms`` — never the first present row. For a contiguous
    series that starts at ``origin`` this is identical to :func:`aggregate_candles` (the
    positional reshape the engine uses), but it stays correct when the 1m feed has gaps.

    Positional reshaping assumes one row per minute; a session market (Forex/metals/indices/
    stock CFDs) skips every closed minute, so reshaping drifts each bucket's timestamp away
    from real clock time. That desyncs the chart from the demo's live forming candle — whose
    time is the true ``floor(minute / tf) * tf`` bucket — making the last bar jump or freeze.
    Anchoring keeps every bucket on its real boundary so the live candle always lands on it.
    """
    tf = jh.timeframe_to_one_minutes(timeframe)
    if tf == 1 or len(candles_1m) == 0:
        return candles_1m
    tf_ms = tf * ONE_MIN_MS
    arr = np.asarray(candles_1m, dtype=float)
    ts = arr[:, 0].astype(np.int64)
    bucket = (ts - int(origin)) // tf_ms
    # Rows are time-sorted, so bucket ids are non-decreasing and np.unique's first-occurrence
    # indices come back already ascending — exactly what reduceat needs for per-bucket spans.
    uniq, first_idx = np.unique(bucket, return_index=True)
    last_idx = np.append(first_idx[1:] - 1, len(arr) - 1)
    out = np.empty((len(uniq), 6))
    out[:, 0] = int(origin) + uniq * tf_ms          # aligned boundary, gap-proof
    out[:, 1] = arr[first_idx, 1]                    # open  = first present open
    out[:, 2] = arr[last_idx, 2]                     # close = last present close
    out[:, 3] = np.maximum.reduceat(arr[:, 3], first_idx)  # high
    out[:, 4] = np.minimum.reduceat(arr[:, 4], first_idx)  # low
    out[:, 5] = np.add.reduceat(arr[:, 5], first_idx)      # volume
    return out


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
        # Defensive: if a secondary route/data feed is shorter than the base
        # timeline, hold its last known price instead of indexing out of bounds.
        if idx >= len(base):
            idx = len(base) - 1
        return float(base[idx, 2])
