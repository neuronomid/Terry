"""
The Strategy base class — Terry's developer-facing API, source-compatible with Jesse.

Subclass it and implement should_long()/go_long() (and optionally should_short/go_short/
update_position/on_open_position/...). Set orders with the smart-order tuples
`self.buy = qty, price`, `self.sell = qty, price`, `self.stop_loss`, `self.take_profit`.
"""
from abc import ABC, abstractmethod

import numpy as np

from . import helpers as jh
from .enums import sides, order_roles, exchange_types, trade_types
from .exceptions import InvalidStrategy, InvalidShortSellOnSpot


def _normalize_orders(value):
    """Accept (qty, price) or [(qty, price), ...] → np.ndarray [[qty, price], ...]."""
    if value is None:
        return None
    arr = np.array(value, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, 2)
    return arr


class Strategy(ABC):
    def __init__(self):
        self.name = None
        self.symbol = None
        self.exchange = None
        self.timeframe = None
        self.index = 0
        self.vars = {}
        self.hp = {}

        # engine wiring (set right after instantiation)
        self.position = None          # Position
        self.simulator = None         # Simulator (broker)
        self.store = None
        self.trades_route = None

        # pending order specs
        self._buy = None
        self._sell = None
        self._stop_loss = None
        self._take_profit = None

        self._cache = {}
        self._chart_lines = []

    # ================================================================= abstract
    @abstractmethod
    def should_long(self) -> bool:
        raise NotImplementedError

    def should_short(self) -> bool:
        return False

    @abstractmethod
    def go_long(self) -> None:
        raise NotImplementedError

    def go_short(self) -> None:
        pass

    def should_cancel_entry(self) -> bool:
        return True

    def update_position(self) -> None:
        pass

    def before(self) -> None:
        pass

    def after(self) -> None:
        pass

    # ================================================================= hooks
    def on_open_position(self, order) -> None:
        pass

    def on_close_position(self, order, closed_trade) -> None:
        pass

    def on_increased_position(self, order) -> None:
        pass

    def on_reduced_position(self, order) -> None:
        pass

    def on_cancel(self) -> None:
        pass

    def on_route_open_position(self, strategy) -> None:
        pass

    def on_route_close_position(self, strategy) -> None:
        pass

    def on_route_increased_position(self, strategy) -> None:
        pass

    def on_route_reduced_position(self, strategy) -> None:
        pass

    def on_route_canceled(self, strategy) -> None:
        pass

    def before_terminate(self):
        pass

    def terminate(self):
        pass

    # optional config methods
    def filters(self) -> list:
        return []

    def hyperparameters(self) -> list:
        return []

    def dna(self) -> str:
        return None

    def watch_list(self) -> list:
        return []

    # ================================================================= order setters
    @property
    def buy(self):
        return self._buy

    @buy.setter
    def buy(self, value):
        self._buy = _normalize_orders(value)

    @property
    def sell(self):
        return self._sell

    @sell.setter
    def sell(self, value):
        self._sell = _normalize_orders(value)

    @property
    def stop_loss(self):
        return self._stop_loss

    @stop_loss.setter
    def stop_loss(self, value):
        if self.is_spot_trading and not self.is_open and self._in_go_entry:
            raise InvalidStrategy(
                "On spot you cannot set stop_loss inside go_long(); set it in on_open_position()."
            )
        self._stop_loss = _normalize_orders(value)
        if self.is_open:
            self._sync_exit_orders()

    @property
    def take_profit(self):
        return self._take_profit

    @take_profit.setter
    def take_profit(self, value):
        if self.is_spot_trading and not self.is_open and self._in_go_entry:
            raise InvalidStrategy(
                "On spot you cannot set take_profit inside go_long(); set it in on_open_position()."
            )
        self._take_profit = _normalize_orders(value)
        if self.is_open:
            self._sync_exit_orders()

    _in_go_entry = False

    def liquidate(self) -> None:
        """Close the open position immediately at market."""
        if not self.is_open:
            return
        qty = abs(self.position.qty)
        side = sides.SELL if self.position.qty > 0 else sides.BUY
        price = self.price
        self.simulator.submit_order(self, side, qty, price,
                                    role=order_roles.CLOSE_POSITION, reduce_only=True)

    # ================================================================= engine driver
    def _execute(self):
        self.index += 1
        self._cache = {}
        self.before()

        if self.position.is_open:
            self.update_position()
            if self.position.is_open:
                self._sync_exit_orders()

        # If there is no open position (either there wasn't one, or update_position() just
        # closed it via liquidate), evaluate entries on the SAME candle — this matches Jesse's
        # same-candle position-flip behaviour.
        if self.position.is_close:
            self._evaluate_entries()

        self.after()

    def _evaluate_entries(self):
        active_entries = [o for o in self._active_orders()
                          if o.role == order_roles.OPEN_POSITION]
        if active_entries:
            if self.should_cancel_entry():
                self.simulator.cancel_entry_orders(self.symbol)
                self.on_cancel()
            return
        self._reset_pending()
        want_long = self.should_long()
        want_short = False
        if not want_long:
            want_short = self.should_short()
        if want_short and self.is_spot_trading:
            raise InvalidShortSellOnSpot("Shorting is not supported on spot.")
        if want_long or want_short:
            if self._filters_pass():
                self._in_go_entry = True
                try:
                    if want_long:
                        self.go_long()
                        self._submit_entry(sides.BUY)
                    elif want_short:
                        self.go_short()
                        self._submit_entry(sides.SELL)
                finally:
                    self._in_go_entry = False
                # futures may have set stop/take in go_* — synced on open

    def _filters_pass(self):
        for f in self.filters():
            if not f():
                return False
        return True

    def _submit_entry(self, side):
        spec = self._buy if side == sides.BUY else self._sell
        if spec is None:
            return
        for qty, price in spec:
            self.simulator.submit_order(self, side, qty, price,
                                        role=order_roles.OPEN_POSITION)

    def _sync_exit_orders(self):
        """Ensure active exit orders match the current stop_loss/take_profit specs."""
        if not self.is_open:
            return
        exit_side = sides.SELL if self.position.qty > 0 else sides.BUY
        # cancel existing exit orders, resubmit from specs (handles trailing reassignment)
        for o in self._active_orders():
            if o.role == order_roles.CLOSE_POSITION:
                o.cancel()
        for spec, tag in ((self._stop_loss, "stop_loss"), (self._take_profit, "take_profit")):
            if spec is None:
                continue
            for qty, price in spec:
                order = self.simulator.submit_order(
                    self, exit_side, abs(qty), price,
                    role=order_roles.CLOSE_POSITION, reduce_only=True)
                order.submitted_via = tag

    def _reset_pending(self):
        self._buy = None
        self._sell = None
        self._stop_loss = None
        self._take_profit = None

    # engine event bridges (called by simulator)
    def _on_open_position(self, order):
        self.on_open_position(order)
        self._sync_exit_orders()
        for r in self._other_routes():
            r.strategy.on_route_open_position(self)

    def _on_close_position(self, order, closed_trade):
        self._reset_pending()
        self.on_close_position(order, closed_trade)
        for r in self._other_routes():
            r.strategy.on_route_close_position(self)

    def _on_increased_position(self, order):
        self.on_increased_position(order)
        for r in self._other_routes():
            r.strategy.on_route_increased_position(self)

    def _on_reduced_position(self, order):
        self.on_reduced_position(order)
        for r in self._other_routes():
            r.strategy.on_route_reduced_position(self)

    def _other_routes(self):
        if self.store is None:
            return []
        return [r for r in self.store_routes if r.strategy is not self]

    store_routes = []

    def _active_orders(self):
        return self.store.orders.active_orders(self.symbol)

    # ================================================================= candles/prices
    @property
    def candles(self) -> np.ndarray:
        return self.store.candles.get_candles(self.exchange, self.symbol, self.timeframe)

    def get_candles(self, exchange, symbol, timeframe) -> np.ndarray:
        return self.store.candles.get_candles(exchange, symbol, timeframe)

    @property
    def current_candle(self) -> np.ndarray:
        return self.candles[-1]

    @property
    def price(self) -> float:
        return float(self.current_candle[2])

    @property
    def close(self) -> float:
        return float(self.current_candle[2])

    @property
    def open(self) -> float:
        return float(self.current_candle[1])

    @property
    def high(self) -> float:
        return float(self.current_candle[3])

    @property
    def low(self) -> float:
        return float(self.current_candle[4])

    @property
    def volume(self) -> float:
        return float(self.current_candle[5])

    @property
    def time(self) -> int:
        return self.store.app.time

    # ================================================================= account
    @property
    def _exchange(self):
        return self.store.exchanges[self.exchange]

    @property
    def balance(self) -> float:
        return self._exchange.balance

    @property
    def available_margin(self) -> float:
        return self._exchange.available_margin

    @property
    def leveraged_available_margin(self) -> float:
        return self._exchange.leveraged_available_margin

    @property
    def fee_rate(self) -> float:
        return self._exchange.fee_rate

    @property
    def leverage(self) -> int:
        return self._exchange.leverage

    @property
    def portfolio_value(self) -> float:
        return self.store.portfolio_value()

    @property
    def exchange_type(self) -> str:
        return self._exchange.type

    @property
    def is_spot_trading(self) -> bool:
        return self._exchange.is_spot

    @property
    def is_futures_trading(self) -> bool:
        return self._exchange.is_futures

    @property
    def base_asset(self) -> str:
        return jh.base_asset(self.symbol)

    @property
    def quote_asset(self) -> str:
        return jh.quote_asset(self.symbol)

    # ================================================================= position state
    @property
    def is_open(self) -> bool:
        return self.position.is_open

    @property
    def is_close(self) -> bool:
        return self.position.is_close

    @property
    def is_long(self) -> bool:
        return self.position.type == trade_types.LONG

    @property
    def is_short(self) -> bool:
        return self.position.type == trade_types.SHORT

    @property
    def average_entry_price(self):
        if self.is_open:
            return self.position.entry_price
        spec = self._buy if self._buy is not None else self._sell
        if spec is None:
            return None
        qty = spec[:, 0].sum()
        return float((spec[:, 0] * spec[:, 1]).sum() / qty) if qty else None

    @property
    def average_stop_loss(self) -> float:
        if self._stop_loss is None:
            raise InvalidStrategy("You have not set a stop_loss.")
        qty = self._stop_loss[:, 0].sum()
        return float((self._stop_loss[:, 0] * self._stop_loss[:, 1]).sum() / qty)

    @property
    def average_take_profit(self) -> float:
        if self._take_profit is None:
            raise InvalidStrategy("You have not set a take_profit.")
        qty = self._take_profit[:, 0].sum()
        return float((self._take_profit[:, 0] * self._take_profit[:, 1]).sum() / qty)

    @property
    def has_long_entry_orders(self) -> bool:
        return any(o.is_buy and o.role == order_roles.OPEN_POSITION
                   for o in self._active_orders())

    @property
    def has_short_entry_orders(self) -> bool:
        return any(o.is_sell and o.role == order_roles.OPEN_POSITION
                   for o in self._active_orders())

    # ================================================================= orders/trades
    @property
    def orders(self):
        return self.store.orders.get_orders(self.symbol)

    @property
    def entry_orders(self):
        return [o for o in self.orders if o.role == order_roles.OPEN_POSITION]

    @property
    def exit_orders(self):
        return [o for o in self.orders if o.role == order_roles.CLOSE_POSITION]

    @property
    def active_exit_orders(self):
        return [o for o in self.exit_orders if o.is_active]

    @property
    def trades(self):
        return [t for t in self.store.closed_trades if t.symbol == self.symbol]

    @property
    def metrics(self) -> dict:
        from .engine.metrics import trades_metrics
        ex = self._exchange
        return trades_metrics(self.store.closed_trades, self.store.app.daily_balance,
                              ex.starting_balance, ex.balance,
                              self.store.app.starting_time, self.store.app.ending_time)

    # ================================================================= env / mode
    @property
    def is_backtesting(self) -> bool:
        return self.store.app.trading_mode == "backtest"

    @property
    def is_livetrading(self) -> bool:
        return False

    @property
    def is_papertrading(self) -> bool:
        return False

    @property
    def is_live(self) -> bool:
        return False

    @property
    def shared_vars(self) -> dict:
        return _SHARED_VARS

    # ================================================================= misc
    @staticmethod
    def log(msg, log_type="info", send_notification=False, webhook=None):
        print(f"[{log_type}] {msg}")

    def add_line_to_candle_chart(self, title, value, color=None):
        self._chart_lines.append(("candle_line", title, value, color))

    def add_horizontal_line_to_candle_chart(self, title, value, color=None,
                                            line_width=1.5, line_style="solid"):
        self._chart_lines.append(("candle_hline", title, value, color))

    def add_extra_line_chart(self, chart_name, title, value, color=None):
        self._chart_lines.append(("extra_line", chart_name, title, value, color))

    def add_horizontal_line_to_extra_chart(self, chart_name, title, value, color=None,
                                           line_width=1.5, line_style="solid"):
        self._chart_lines.append(("extra_hline", chart_name, title, value, color))


_SHARED_VARS = {}
