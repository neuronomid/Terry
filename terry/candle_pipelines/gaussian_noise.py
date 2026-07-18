"""Gaussian-noise candle pipeline."""

from __future__ import annotations

import numpy as np

from .base_candles import BaseCandlesPipeline


class GaussianNoiseCandlesPipeline(BaseCandlesPipeline):
    def __init__(self, batch_size: int, *, close_sigma: float,
                 high_sigma: float, low_sigma: float, close_mu: float = 0.0,
                 high_mu: float = 0.0, low_mu: float = 0.0,
                 seed: int | None = None) -> None:
        super().__init__(batch_size)
        self.close_mu = close_mu
        self.close_sigma = close_sigma
        self.high_mu = high_mu
        self.high_sigma = high_sigma
        self.low_mu = low_mu
        self.low_sigma = low_sigma
        self._rng = np.random.default_rng(seed)

    def process(self, original_1m_candles: np.ndarray, out: np.ndarray) -> bool:
        eps = 1e-12
        out[:] = original_1m_candles
        count = len(out)
        noise = self._rng.normal(self.close_mu, self.close_sigma, size=count).cumsum()
        out[:, 2] = np.maximum(out[:, 2] + noise, eps)
        out[1:, 1] = out[:-1, 2]
        out[0, 1] = max(self.last_price, eps)
        out[:, 3] += self.high_mu + self._rng.normal(0, self.high_sigma, size=count)
        out[:, 4] += self.low_mu + self._rng.normal(0, self.low_sigma, size=count)
        _enforce_ohlc(out, eps)
        return True


def _enforce_ohlc(out: np.ndarray, eps: float = 1e-12) -> None:
    out[:, 1] = np.maximum(out[:, 1], eps)
    out[:, 2] = np.maximum(out[:, 2], eps)
    out[:, 3] = np.maximum.reduce([out[:, 1], out[:, 2], out[:, 3], out[:, 4]])
    out[:, 4] = np.maximum(
        np.minimum.reduce([out[:, 1], out[:, 2], out[:, 3], out[:, 4]]), eps
    )
