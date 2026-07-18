"""Moving-block bootstrap pipeline preserving local multivariate price structure."""

from __future__ import annotations

import numpy as np

from .base_candles import BaseCandlesPipeline
from .gaussian_noise import _enforce_ohlc


class MovingBlockBootstrapCandlesPipeline(BaseCandlesPipeline):
    def __init__(self, batch_size: int, seed: int | None = None, **_ignored) -> None:
        super().__init__(batch_size)
        self._block_size = max(1, min(batch_size - 1, max(10, batch_size // 10)))
        self._rng = np.random.default_rng(seed)

    def _bootstrap(self, values: np.ndarray, count: int) -> np.ndarray:
        if len(values) == 0:
            return np.zeros((count, 3), dtype=float)
        block_size = max(1, min(self._block_size, len(values)))
        starts = self._rng.integers(
            0, len(values) - block_size + 1,
            size=int(np.ceil(count / block_size)) + 1,
        )
        return np.vstack([values[start:start + block_size] for start in starts])[:count]

    def process(self, original_1m_candles: np.ndarray, out: np.ndarray) -> bool:
        out[:] = original_1m_candles
        count = len(out)
        close_delta = np.diff(original_1m_candles[:, 2], prepend=self.last_price)
        high_delta = original_1m_candles[:, 3] - original_1m_candles[:, 2]
        low_delta = original_1m_candles[:, 2] - original_1m_candles[:, 4]
        values = np.column_stack([close_delta[1:], high_delta[1:], low_delta[1:]])
        sampled = self._bootstrap(values, count)
        closes = np.maximum(np.cumsum(sampled[:, 0]) + self.last_price, 1e-12)
        out[:, 2] = closes
        out[1:, 1] = closes[:-1]
        out[0, 1] = max(self.last_price, 1e-12)
        out[:, 3] = closes + sampled[:, 1]
        out[:, 4] = closes - sampled[:, 2]
        _enforce_ohlc(out)
        return True
