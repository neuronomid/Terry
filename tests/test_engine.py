"""Unit tests for Terry's engine, indicators, sizing, and metrics."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from terry.research.backtest import backtest
from terry.factories import candles_from_close_prices
from terry.strategy import Strategy
import terry.indicators as ta
from terry import utils


# --------------------------------------------------------------------- fixtures
def _trend_candles(n=60 * 24 * 20):
    t = np.arange(n)
    prices = 20000 + 1500 * np.sin(t / 1500.0) + 400 * np.sin(t / 230.0) + t * 0.02
    return candles_from_close_prices(prices.tolist())


class SmaCross(Strategy):
    def should_long(self):
        return ta.sma(self.candles, 10) > ta.sma(self.candles, 30)

    def should_short(self):
        return ta.sma(self.candles, 10) < ta.sma(self.candles, 30)

    def go_long(self):
        self.buy = utils.size_to_qty(self.available_margin * 0.5, self.price, fee_rate=self.fee_rate), self.price

    def go_short(self):
        self.sell = utils.size_to_qty(self.available_margin * 0.5, self.price, fee_rate=self.fee_rate), self.price

    def update_position(self):
        f, s = ta.sma(self.candles, 10), ta.sma(self.candles, 30)
        if (self.is_long and f < s) or (self.is_short and f > s):
            self.liquidate()


FUT_CONFIG = {"starting_balance": 10_000, "fee": 0.001, "type": "futures",
              "futures_leverage": 2, "futures_leverage_mode": "cross",
              "exchange": "B", "warm_up_candles": 0}


def _run(strategy_cls, config=None, candles=None, tf="1h"):
    config = config or FUT_CONFIG
    candles = candles if candles is not None else _trend_candles()
    routes = [{"exchange": "B", "symbol": "BTC-USDT", "timeframe": tf, "strategy": strategy_cls.__name__}]
    cndl = {"B-BTC-USDT": {"exchange": "B", "symbol": "BTC-USDT", "candles": candles}}
    return backtest(config, routes, [], cndl, generate_equity_curve=True,
                    strategy_classes={strategy_cls.__name__: strategy_cls})


# --------------------------------------------------------------------- indicators
def test_sma_matches_numpy():
    c = candles_from_close_prices(list(range(1, 51)))
    # close prices are 1..50; SMA(5) of last = mean(46..50) = 48
    assert abs(ta.sma(c, 5) - 48.0) < 1e-9


def test_ema_is_scalar_and_sequential():
    c = _trend_candles(2000)
    assert isinstance(ta.ema(c, 20), float)
    seq = ta.ema(c, 20, sequential=True)
    assert isinstance(seq, np.ndarray) and len(seq) == len(c)


def test_rsi_bounds():
    c = _trend_candles(3000)
    r = ta.rsi(c, 14, sequential=True)
    r = r[np.isfinite(r)]
    assert (r >= 0).all() and (r <= 100).all()


def test_macd_named_tuple():
    c = _trend_candles(3000)
    m = ta.macd(c)
    assert hasattr(m, "macd") and hasattr(m, "signal") and hasattr(m, "hist")
    assert m[0] == m.macd


def test_bollinger_ordering():
    c = _trend_candles(3000)
    bb = ta.bollinger_bands(c, 20)
    assert bb.upperband >= bb.middleband >= bb.lowerband


# --------------------------------------------------------------------- sizing
def test_size_to_qty():
    # $1000 at price 100, no fee -> 10 units
    assert utils.size_to_qty(1000, 100, fee_rate=0) == 10.0


def test_risk_to_qty_positive():
    q = utils.risk_to_qty(10000, 3, 100, 95, fee_rate=0)
    assert q > 0


def test_crossed():
    # cross happens at the LAST two points: a goes from below b to above b
    a_up = np.array([1, 1, 1, 2])
    b_up = np.array([2, 2, 2, 1])
    assert utils.crossed(a_up, b_up, direction="above") is True
    assert utils.crossed(a_up, b_up, direction="below") is False
    # reverse for a downward cross
    assert utils.crossed(b_up, a_up, direction="below") is True


# --------------------------------------------------------------------- engine
def test_backtest_returns_44_metrics():
    res = _run(SmaCross)
    assert len(res["metrics"]) == 44
    assert res["metrics"]["total"] > 0


def test_metric_keys_match_jesse():
    res = _run(SmaCross)
    expected = {"total", "win_rate", "net_profit", "net_profit_percentage", "starting_balance",
                "finishing_balance", "sharpe_ratio", "sortino_ratio", "calmar_ratio", "omega_ratio",
                "serenity_index", "max_drawdown", "annual_return", "expectancy", "fee",
                "longs_count", "shorts_count", "gross_profit", "gross_loss", "average_holding_period"}
    assert expected.issubset(set(res["metrics"].keys()))


def test_trade_schema():
    res = _run(SmaCross)
    assert res["trades"], "should produce trades"
    keys = set(res["trades"][0].keys())
    assert {"PNL", "PNL_percentage", "entry_price", "exit_price", "qty", "size",
            "fee", "holding_period", "opened_at", "closed_at", "type", "symbol"}.issubset(keys)


def test_balance_conservation_closed_trades():
    """finishing_balance == starting + sum(closed PNL) - open-position entry fees."""
    res = _run(SmaCross)
    m = res["metrics"]
    sum_pnl = sum(t["PNL"] for t in res["trades"])
    # gap is at most a couple of entry fees for any position still open at the end
    gap = abs(m["finishing_balance"] - (m["starting_balance"] + sum_pnl))
    assert gap < 50  # << one trade's notional*fee


def test_long_only_spot():
    class SpotLong(Strategy):
        def should_long(self):
            return ta.rsi(self.candles, 14) < 35
        def should_short(self):
            return False
        def go_long(self):
            self.buy = utils.size_to_qty(self.available_margin * 0.9, self.price, fee_rate=self.fee_rate), self.price
        def on_open_position(self, order):
            self.take_profit = self.position.qty, self.price * 1.03
            self.stop_loss = self.position.qty, self.price * 0.98

    cfg = {**FUT_CONFIG, "type": "spot", "futures_leverage": 1}
    res = _run(SpotLong, config=cfg)
    assert res["metrics"]["shorts_count"] in (0, None) or res["metrics"].get("shorts_count", 0) == 0


def test_stop_loss_take_profit_fill():
    """A position with a tight TP should close in profit; verify exits fire."""
    class TPStrat(Strategy):
        def should_long(self):
            return self.index == 1
        def should_short(self):
            return False
        def go_long(self):
            self.buy = utils.size_to_qty(self.available_margin * 0.5, self.price, fee_rate=self.fee_rate), self.price
        def on_open_position(self, order):
            self.take_profit = self.position.qty, self.price * 1.01
            self.stop_loss = self.position.qty, self.price * 0.95
        def update_position(self):
            pass

    # steadily rising prices → TP hits
    prices = np.linspace(100, 130, 60 * 24 * 5)
    c = candles_from_close_prices(prices.tolist())
    res = _run(TPStrat, candles=c)
    assert res["metrics"]["total"] >= 1
    # first trade should be a winner (TP hit)
    assert res["trades"][0]["PNL"] > 0


def test_no_trades_when_never_entering():
    class Never(Strategy):
        def should_long(self):
            return False
        def go_long(self):
            pass
    res = _run(Never)
    assert res["metrics"] == {"total": 0, "win_rate": 0, "net_profit_percentage": 0}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
