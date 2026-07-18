"""Base class shared by Terry's 1-minute candle pipelines."""

from __future__ import annotations

import numpy as np


class BaseCandlesPipeline:
    def __init__(self, batch_size: int) -> None:
        if not isinstance(batch_size, int) or batch_size < 2:
            raise ValueError("batch_size must be an integer of at least 2")
        self._batch_size = batch_size
        self._output = np.zeros((batch_size, 6), dtype=float)
        self.last_price = 0.0

    def get_candles(self, candles: np.ndarray, index: int,
                    candles_step: int = -1) -> np.ndarray:
        """Process a batch lazily and return one candle or a consecutive slice."""
        candles = np.asarray(candles, dtype=float)
        local_index = index % self._batch_size
        if local_index == 0:
            length = len(candles)
            if length == 0:
                raise ValueError("candles cannot be empty")
            self.last_price = (candles[0, 1] if self.last_price == 0.0
                               else self._output[min(length, self._batch_size) - 1, 2])
            target = self._output[:length]
            if not self.process(candles, target):
                target[:] = candles
        if candles_step == -1:
            return self._output[local_index]
        if local_index + candles_step <= min(len(candles), self._batch_size):
            return self._output[local_index:local_index + candles_step]
        raise ValueError(
            "Candle pipeline batch_size must be a multiple of the minimum route timeframe."
        )

    def transform(self, candles: np.ndarray) -> np.ndarray:
        """Transform a complete 1m array in deterministic-size batches."""
        source = np.asarray(candles, dtype=float)
        if source.ndim != 2 or source.shape[1] != 6:
            raise ValueError("candles must have shape (n, 6)")
        transformed = np.empty_like(source)
        self.last_price = 0.0
        for start in range(0, len(source), self._batch_size):
            original = source[start:start + self._batch_size]
            output = original.copy()
            if self.last_price == 0.0:
                self.last_price = float(original[0, 1])
            if not self.process(original, output):
                output[:] = original
            transformed[start:start + len(original)] = output
            self.last_price = float(output[-1, 2])
        return transformed

    def process(self, original_1m_candles: np.ndarray, out: np.ndarray) -> bool:
        """Mutate ``out`` and return True, or return False to keep the originals."""
        return False
