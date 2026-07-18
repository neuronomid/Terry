---
name: terry-strategy-tests
description: Use when writing or modifying Terry backend tests tied to strategy behavior, including entries, exits, smart orders, position hooks, multi-route state, and closed-trade metrics. Provides Terry's deterministic strategy-driven test pattern and testing_utils helpers.
---

# Write Terry strategy tests

Use a small purpose-built `Strategy` class and run it through Terry's real historical
engine. Put assertions in lifecycle hooks when the behavior is observable there; keep
the outer pytest test focused on arranging candles and invoking the helper.

## Preferred pattern

Define the strategy in the test for isolation, then pass it through
`strategy_classes`:

```python
from terry.strategy import Strategy
from terry import utils
from terry.testing_utils import single_route_backtest


class TestOnClosePosition(Strategy):
    def should_long(self):
        return self.price == 10

    def go_long(self):
        self.buy = utils.size_to_qty(10, self.price), self.price

    def on_open_position(self, order):
        self.take_profit = self.position.qty, 12

    def on_close_position(self, order, closed_trade):
        assert closed_trade.entry_price == 10
        assert closed_trade.exit_price == 12
        assert closed_trade.qty == 1


def test_on_close_position():
    result = single_route_backtest(
        "TestOnClosePosition",
        strategy_classes={"TestOnClosePosition": TestOnClosePosition},
    )
    assert result["metrics"]["total"] == 1
```

`single_route_backtest()` uses deterministic close prices `1..99` by default. Use
`trend="down"` for descending prices, `is_futures_trading=False` for spot, and
`fee`, `leverage`, `leverage_mode`, `candles_count`, or `timeframe` when relevant.

Use `strategies_dir=` instead of `strategy_classes=` when deliberately testing the
on-disk loader. Terry accepts both native `terry.*` imports and Jesse-style imports
such as `from jesse.strategies import Strategy` in those strategy files.

## Multi-route behavior

Use `two_routes_backtest()` for two trading routes and
`two_data_routes_backtest()` when extra non-trading feeds are required. Pass a
name-to-class mapping that contains every route strategy. Assert route-local state in
`before()` or `after()` at a chosen `self.index`.

## Pick the observing hook

| Hook | Observe |
|---|---|
| `should_long` / `should_short` | entry decisions |
| `go_long` / `go_short` | submitted entry orders |
| `should_cancel_entry` | pending-entry cancellation |
| `on_open_position(order)` | initial fills and exit placement |
| `on_increased_position(order)` | scale-ins |
| `on_reduced_position(order)` | partial exits |
| `on_close_position(order, closed_trade)` | final PnL, fees, and trade metadata |
| `update_position()` | per-candle open-position behavior |
| `before()` / `after()` | engine or route state around each cycle |

Read `terry/strategy.py` before using an unfamiliar property or hook. Do not guess
API names. Use `self.take_profit = qty, price`, `self.stop_loss = qty, price`, and
`self.liquidate()` exactly as production strategies do.

## Conventions

- Keep synthetic runs deterministic and short.
- Use `pytest.approx()` for floating-point accounting.
- Assert that `closed_trade.pnl` agrees with the simulated wallet change in fee/PnL tests.
- Exercise the real engine for lifecycle behavior; use direct model/exchange unit tests only
  for low-level accounting that cannot be expressed cleanly through a strategy.
- Never use real exchange credentials or live execution. Terry is historical research only.

Run the targeted test first, then the full suite:

```bash
.venv/bin/python -m pytest tests/test_engine.py -q
.venv/bin/python -m pytest -q
```
