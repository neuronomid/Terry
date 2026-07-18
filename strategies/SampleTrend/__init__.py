from terry.strategies import Strategy
import terry.indicators as ta
from terry import utils


class SampleTrend(Strategy):
    """A simple EMA-cross trend follower with an ATR stop/target (futures)."""

    def should_long(self) -> bool:
        return ta.ema(self.candles, 20) > ta.ema(self.candles, 50)

    def should_short(self) -> bool:
        return ta.ema(self.candles, 20) < ta.ema(self.candles, 50)

    def go_long(self):
        qty = utils.size_to_qty(self.available_margin * 0.5, self.price, fee_rate=self.fee_rate)
        self.buy = qty, self.price

    def go_short(self):
        qty = utils.size_to_qty(self.available_margin * 0.5, self.price, fee_rate=self.fee_rate)
        self.sell = qty, self.price

    def on_open_position(self, order):
        atr = ta.atr(self.candles)
        if self.is_long:
            self.stop_loss = self.position.qty, self.price - 2 * atr
            self.take_profit = self.position.qty, self.price + 4 * atr
        elif self.is_short:
            self.stop_loss = self.position.qty, self.price + 2 * atr
            self.take_profit = self.position.qty, self.price - 4 * atr

    def update_position(self):
        pass
