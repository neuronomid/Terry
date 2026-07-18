"""
Terry MCP resources — concise reference docs agents can read (terry:// URIs), mirroring
Jesse's jesse:// resources. Kept short and high-signal.
"""

STRATEGY = """# Terry Strategy Reference

Subclass `Strategy`. Logic runs once per candle **after it closes** (no look-ahead).

```python
from terry.strategies import Strategy
import terry.indicators as ta
from terry import utils

class MyStrategy(Strategy):
    def should_long(self) -> bool:      # required
        return ta.sma(self.candles, 10) > ta.sma(self.candles, 30)
    def should_short(self) -> bool:     # default False; on spot MUST stay False
        return ta.sma(self.candles, 10) < ta.sma(self.candles, 30)
    def go_long(self):                  # required — set self.buy
        qty = utils.size_to_qty(self.available_margin * 0.5, self.price, fee_rate=self.fee_rate)
        self.buy = qty, self.price
    def go_short(self):
        qty = utils.size_to_qty(self.available_margin * 0.5, self.price, fee_rate=self.fee_rate)
        self.sell = qty, self.price
    def should_cancel_entry(self) -> bool:
        return True
    def update_position(self):          # runs each candle while a position is open
        if self.is_long and ta.sma(self.candles, 10) < ta.sma(self.candles, 30):
            self.liquidate()
```

## Smart orders
Set `self.buy = qty, price` / `self.sell = qty, price`; the type is inferred from price vs
current price: equal→MARKET; BUY below→LIMIT, above→STOP (mirror for SELL). Never pick the type.

## Exits
`self.stop_loss = qty, price`, `self.take_profit = qty, price`, or `self.liquidate()`.
- Futures: may set stop/take in go_long/go_short, preferably in `on_open_position(order)`.
- Spot: cannot set stop/take inside go_long — set them in `on_open_position` (check is_long/is_short).
- Trailing stop: reassign `self.stop_loss` in `update_position()`; read current with
  `self.average_stop_loss` (NOT `self.stop_loss[1]`). There is no `self.trailing_stop`.

## Sizing
Never hardcode qty. Use `utils.size_to_qty(size, price, fee_rate=self.fee_rate)` or
`utils.risk_to_qty(capital, risk_pct, entry, stop, fee_rate=self.fee_rate)` (risk_pct is a %).

## Hooks
`before/after`, `on_open_position(order)`, `on_close_position(order, closed_trade)` (TWO args),
`on_increased_position(order)`, `on_reduced_position(order)`.

## Key self.* members
price/open/close/high/low/volume, candles, get_candles(ex,sym,tf); available_margin, balance,
fee_rate, leverage, portfolio_value; position, is_open/is_close/is_long/is_short,
average_entry_price/average_stop_loss/average_take_profit; entry_orders/exit_orders/orders/trades;
symbol/timeframe/exchange/index; hp (dict), vars, shared_vars; is_spot_trading/is_futures_trading.

## Optimization prep
Define `hyperparameters()` returning dicts with name/type(int|float|'categorical')/min/max/
step/options/default; read via `self.hp['name']`.
"""

INDICATOR = """# Indicators

Call `list_indicators` then `get_indicator_details(name)`. Usage:
```python
import terry.indicators as ta
v = ta.sma(self.candles, 20)                 # latest scalar
series = ta.sma(self.candles, 20, sequential=True)   # full np.ndarray
up, mid, low = ta.bollinger_bands(self.candles, 20)  # multi-line = named tuple
```
Multi-line indicators (macd, bollinger_bands, stoch, srsi, supertrend, keltner, donchian, ppo)
return named tuples — index and attribute access both work. Default source is close; don't pass
period/source unless asked. Use `utils.crossed(a, b, direction='above'|'below')` for crossovers.
"""

BACKTEST_METRICS = """# Backtest metrics reference

Returned in `metrics` (44 keys). Most important:
- total, win_rate (0..1), net_profit, net_profit_percentage
- starting_balance, finishing_balance
- sharpe_ratio, sortino_ratio, calmar_ratio, omega_ratio, serenity_index (annualized, 365d)
- max_drawdown (%, negative), max_underwater_period (days), annual_return (%)
- expectancy, expectancy_percentage, gross_profit, gross_loss
- longs_count/shorts_count, win_rate_longs/shorts, average_holding_period (seconds)
- winning_streak/losing_streak, avg_trades_per_day/week/month, fee

Interpretation: prefer a positive expectancy, sharpe > 1, drawdown you can stomach, and enough
trades (>~30) for the stats to mean anything. High sharpe on few trades or a short window is a
red flag — validate with a significance test and Monte Carlo.
"""

SIGNIFICANCE_TEST = """# Rule Significance Testing

Proves whether an ENTRY rule has a real edge vs random, via a bootstrap p-value on
signal*detrended-log-return. Workflow: write a MINIMAL strategy with only the entry signal →
create_significance_test_draft(n_simulations>=2000) → run_significance_test → poll.
Interpret p_value: <0.05 significant (proceed); 0.05–0.10 borderline (flag, ask user);
>0.10 HARD STOP (indistinguishable from random — don't silently proceed). Always run this before
building out a NEW/changed entry rule (skip only if the user says so, or only exits changed).
"""

MONTE_CARLO = """# Monte Carlo robustness

Candles mode (default): block-bootstrap the price path, re-run, compare `original` to the
percentile bands. Overfit check on sharpe (higher=better): original > best_5 → overfit_suspect;
best_5 >= original > median → borderline; original <= median → robust. Always report worst_5 for
downside tail. Trades mode (only on explicit request): shuffles trade order — only max_drawdown /
calmar are informative (return & win_rate are shuffle-invariant). Defaults: num_scenarios=200,
run_candles=True, run_trades=False.
"""

CANDLE = """# Candle data

import_candles(exchange, symbol, start_date, finish_date?) returns immediately with an import_id.
Then poll get_candle_import_status(import_id) every few seconds until status=="finished".
Already-stored candles are skipped, so re-running from the same start_date is safe. Free source:
public exchange APIs (no key). Historical drivers cover Binance Spot/US/Perpetual, Bitfinex Spot,
Coinbase Spot, Bybit Spot/USDT/USDC Perpetual, Gate USDT Perpetual, and Kraken Pro Futures. Don't
pre-check candles before a backtest — run it; if it stops with missing_candles, import starting
~2 months before start_date, then re-run.
"""

CONFIGURATION = """# Configuration

get_config / update_config(json). Keys: exchange, starting_balance, fee, type('futures'|'spot'),
futures_leverage, futures_leverage_mode, quote_asset, warm_up_candles, plus optimization/
monte_carlo/significance_test defaults. Only use update_config for user-driven changes, never to
work around a tool error. Terry uses SQLite locally — no Postgres/Redis needed. Live trading is
out of scope.
"""

POSITION_RISK = """# Position sizing & risk

`utils.size_to_qty(position_size, entry_price, precision=3, fee_rate=0)` — fraction of margin to qty.
`utils.risk_to_qty(capital, risk_per_capital_pct, entry, stop, precision=8, fee_rate=0)` — size so a
stop-out loses a fixed % (risk_per_capital is a percent, e.g. 3 = 3%).
`utils.qty_to_size`, `utils.estimate_risk`, `utils.limit_stop_loss`, `utils.kelly_criterion`.
Always size from available_margin/price/fee_rate; branch on is_long/is_short when setting stops.
"""

UTILITIES = """# Utilities (from terry import utils)

size_to_qty, qty_to_size, risk_to_qty, risk_to_size, estimate_risk, limit_stop_loss,
kelly_criterion, crossed(a,b,direction,sequential=False), signal_line, streaks,
strictly_increasing, strictly_decreasing, sum_floats, subtract_floats,
numpy_candles_to_dataframe, anchor_timeframe.
Pairs/stat-arb: prices_to_returns, z_score, are_cointegrated, calculate_alpha_beta,
combinations_without_repeat.
"""

OPTIMIZATION = """# Hyperparameter optimization

Define `hyperparameters()` on the strategy, then create an optimization draft and poll the
session after starting it. Jesse-compatible drafts accept JSON `routes` / `data_routes`, separate
training and testing date windows, `trials` per hyperparameter, and an `objective_function` of
sharpe, calmar, sortino, omega, serenity, smart sharpe, or smart sortino. The research API also
accepts explicit training/testing candle dictionaries and returns `best_trials` with both
in-sample `training_metrics` and out-of-sample `testing_metrics`.

Prefer candidates that remain strong on the testing window. A large train/test performance gap
is an overfitting warning; optimization cannot prove future profitability.
"""

BACKTEST_MANAGEMENT = """# Backtest workflow & pitfalls

1) create_backtest_draft(strategy, ...) — all params default from config; finish_date defaults to
   yesterday. 2) run_backtest(session_id) — returns immediately. 3) poll get_backtest_session until
   status finished/stopped. On stopped+missing_candles: import (2 months before start) then re-run.
Pitfalls: don't put two routes on the same exchange-symbol; size from available_margin (not fixed
qty); `self.position.qty` is 0 inside go_long/go_short; read the stop with self.average_stop_loss.
Every finished session carries a `dashboard_url` (a local HTML report) — surface it to the user.
"""

STRATEGY_EXAMPLES = '''# Example strategies

## Trend-following (futures, EMA cross + ATR stop)
```python
from terry.strategies import Strategy
import terry.indicators as ta
from terry import utils

class EmaTrend(Strategy):
    def should_long(self):  return ta.ema(self.candles, 20) > ta.ema(self.candles, 50)
    def should_short(self): return ta.ema(self.candles, 20) < ta.ema(self.candles, 50)
    def go_long(self):
        self.buy = utils.size_to_qty(self.available_margin*0.5, self.price, fee_rate=self.fee_rate), self.price
    def go_short(self):
        self.sell = utils.size_to_qty(self.available_margin*0.5, self.price, fee_rate=self.fee_rate), self.price
    def on_open_position(self, order):
        atr = ta.atr(self.candles)
        if self.is_long:
            self.stop_loss = self.position.qty, self.price - 2*atr
            self.take_profit = self.position.qty, self.price + 4*atr
        elif self.is_short:
            self.stop_loss = self.position.qty, self.price + 2*atr
            self.take_profit = self.position.qty, self.price - 4*atr
    def update_position(self):
        pass
```

## Mean-reversion (RSI, spot long-only)
```python
class RsiMeanReversion(Strategy):
    def should_long(self):  return ta.rsi(self.candles, 14) < 30
    def should_short(self): return False
    def go_long(self):
        self.buy = utils.size_to_qty(self.available_margin*0.9, self.price, fee_rate=self.fee_rate), self.price
    def on_open_position(self, order):
        self.take_profit = self.position.qty, self.price * 1.03
        self.stop_loss   = self.position.qty, self.price * 0.98
    def update_position(self):
        if ta.rsi(self.candles, 14) > 55:
            self.liquidate()
```
'''

_RESOURCES = {
    "terry://strategy": ("Strategy structure, lifecycle, orders, sizing", STRATEGY),
    "terry://strategy_examples": ("Complete runnable example strategies", STRATEGY_EXAMPLES),
    "terry://indicator": ("Indicator discovery and usage", INDICATOR),
    "terry://backtest_metrics": ("Backtest metric definitions", BACKTEST_METRICS),
    "terry://backtest_management": ("Backtest workflow and pitfalls", BACKTEST_MANAGEMENT),
    "terry://significance_test": ("Rule significance testing", SIGNIFICANCE_TEST),
    "terry://monte_carlo": ("Monte Carlo robustness analysis", MONTE_CARLO),
    "terry://candle": ("Candle import and management", CANDLE),
    "terry://configuration": ("Configuration reference", CONFIGURATION),
    "terry://position_risk": ("Position sizing and risk", POSITION_RISK),
    "terry://utilities": ("Utility functions", UTILITIES),
    "terry://optimization": ("Optimization workflow and result interpretation", OPTIMIZATION),
}


def register_resources(mcp):
    for uri, (desc, content) in _RESOURCES.items():
        _make(mcp, uri, desc, content)


def _make(mcp, uri, desc, content):
    @mcp.resource(uri, description=desc)
    def _resource() -> str:
        return content
