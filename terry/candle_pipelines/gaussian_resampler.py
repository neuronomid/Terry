"""Gaussian price-delta resampling pipeline."""

from __future__ import annotations

import numpy as np

from .base_candles import BaseCandlesPipeline
from .gaussian_noise import _enforce_ohlc


class GaussianResamplerCandlesPipeline(BaseCandlesPipeline):
    def __init__(self, batch_size: int, *, mu: float = 0.0,
                 sigma: float | None = None, seed: int | None = None) -> None:
        super().__init__(batch_size)
        self.mu = mu
        self.sigma = sigma
        self._rng = np.random.default_rng(seed)

    def process(self, original_1m_candles: np.ndarray, out: np.ndarray) -> bool:
        eps = 1e-12
        out[:] = original_1m_candles
        closes = original_1m_candles[:, 2]
        count = len(out)
        deltas = np.diff(closes, prepend=self.last_price)
        mean_delta = float(np.nan_to_num(np.mean(deltas[1:]), nan=0.0))
        delta_std = float(np.nan_to_num(np.std(deltas[1:]), nan=0.0))
        median = float(np.nan_to_num(np.median(closes), nan=0.0))
        if self.sigma is None:
            relative = np.diff(closes) / np.maximum(closes[:-1], eps)
            relative_std = float(np.nan_to_num(np.std(relative), nan=0.0))
            target = max(relative_std * median if relative_std else median * 0.0005, eps)
            scale = target / max(delta_std, eps)
        else:
            scale = self.sigma
        out[:, 2] = np.maximum(
            self._rng.normal(mean_delta + self.mu, delta_std * scale, count).cumsum()
            + self.last_price,
            eps,
        )
        out[1:, 1] = out[:-1, 2]
        out[0, 1] = max(self.last_price, eps)
        high_delta = original_1m_candles[:, 3] - original_1m_candles[:, 2]
        low_delta = original_1m_candles[:, 2] - original_1m_candles[:, 4]
        out[:, 3] = out[:, 2] + self._rng.normal(
            float(np.mean(high_delta)) + self.mu, float(np.std(high_delta)) * scale, count)
        out[:, 4] = out[:, 2] - self._rng.normal(
            float(np.mean(low_delta)) + self.mu, float(np.std(low_delta)) * scale, count)
        _enforce_ohlc(out, eps)
        return True
