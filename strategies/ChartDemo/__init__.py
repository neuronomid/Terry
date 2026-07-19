from terry.strategies import Strategy
import terry.indicators as ta
from terry import utils


class ChartDemo(Strategy):
    """EMA-cross trend follower that plots its indicators onto the backtest chart.

    Demonstrates Terry's candlestick-chart overlays: two EMA lines drawn on the
    price chart, plus an RSI sub-chart with 30/70 guide lines.
    """

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

    def after(self):
        # Overlay the two EMAs on the candlestick chart.
        self.add_line_to_candle_chart("EMA 20", float(ta.ema(self.candles, 20)), color="#f9b537")
        self.add_line_to_candle_chart("EMA 50", float(ta.ema(self.candles, 50)), color="#72a6ff")
        # RSI in its own sub-chart with overbought/oversold guides.
        self.add_extra_line_chart("RSI", "RSI 14", float(ta.rsi(self.candles, 14)), color="#a78bfa")
        self.add_horizontal_line_to_extra_chart("RSI", "Overbought", 70.0, color="#ff6b6b")
        self.add_horizontal_line_to_extra_chart("RSI", "Oversold", 30.0, color="#4dd49b")
