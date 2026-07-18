"""
The Strategy base class — Terry's developer-facing API, source-compatible with Jesse.

Subclass it and implement should_long()/go_long() (and optionally should_short/go_short/
update_position/on_open_position/...). Set orders with the smart-order tuples
`self.buy = qty, price`, `self.sell = qty, price`, `self.stop_loss`, `self.take_profit`.
"""
from abc import ABC, abstractmethod
import csv
from functools import wraps
import os
import sys

import numpy as np

from . import helpers as jh
from .enums import sides, order_roles, order_submitted_via, trade_types
from .exceptions import ConflictingRules, InvalidStrategy, InvalidShortSellOnSpot


def cached(method):
    """Cache a strategy method for the current candle, matching Jesse's decorator."""
    @wraps(method)
    def decorated(self, *args, **kwargs):
        try:
            key = (method, args, tuple(sorted(kwargs.items())))
            hash(key)
        except TypeError:
            return method(self, *args, **kwargs)
        if key not in self._cache:
            self._cache[key] = method(self, *args, **kwargs)
        return self._cache[key]

    return decorated


def _normalize_orders(value):
    """Accept (qty, price) or [(qty, price), ...] → np.ndarray [[qty, price], ...]."""
    if value is None:
        return None
    arr = np.array(value, dtype=float)
    if arr.size == 0:
        return np.empty((0, 2), dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, 2)
    if np.any(arr[:, 1] <= 0):
        raise InvalidStrategy("Order price must be greater than zero.")
    return arr


class Strategy(ABC):
    def __init__(self):
        self.id = jh.generate_unique_id()
        self.name = None
        self.symbol = None
        self.exchange = None
        self.timeframe = None
        self.index = 0
        self.last_trade_index = 0
        self.vars = {}
        self.hp = {}
        self.increased_count = 0
        self.reduced_count = 0
        self.trades_count = 0
        self.trade = None

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
        self._current_route_index = None
        self._evaluating_filters = False

        # Jesse 2.5 machine-learning gather/deploy state.
        self.ml_mode = getattr(type(self), "ml_mode", "gather")
        self._ml_data_points = []
        self._current_ml_point = None
        self._ml_model = None
        self._ml_scaler = None
        self._ml_feature_importance = None

    def candles_pipeline(self):
        """Override to transform 1m candles for scenario/Monte Carlo research."""
        return None

    def record_features(self, features_dict: dict) -> None:
        if not isinstance(features_dict, dict):
            raise TypeError("features_dict must be a dict")
        if self._current_ml_point is None:
            self._current_ml_point = {
                "time": int(self.current_candle[0] / 1000),
                "features": {},
                "label": None,
            }
        self._current_ml_point["features"].update(features_dict)

    def record_label(self, name: str, value) -> None:
        if self._current_ml_point is None:
            return
        self._current_ml_point["label"] = {"name": name, "value": value}
        self._ml_data_points.append(self._current_ml_point)
        self._current_ml_point = None

    def export_ml_data(self, directory: str | None = None) -> bool:
        """Export completed ML samples to ``ml_data/<Strategy>_data.csv``."""
        try:
            if directory is None:
                module = sys.modules.get(self.__class__.__module__)
                module_file = getattr(module, "__file__", None)
                directory = os.path.dirname(os.path.abspath(module_file)) if module_file else os.getcwd()
            ml_dir = os.path.join(directory, "ml_data")
            os.makedirs(ml_dir, exist_ok=True)
            data_path = os.path.join(ml_dir, f"{self.name}_data.csv")
            feature_names = sorted({
                key for point in self._ml_data_points for key in point["features"]
            })
            with open(data_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["time", "label_name", "label_value", *feature_names])
                for point in self._ml_data_points:
                    if point.get("label") is None:
                        continue
                    writer.writerow([
                        point["time"], point["label"]["name"], point["label"]["value"],
                        *(point["features"].get(name, "") for name in feature_names),
                    ])
            return True
        except (OSError, TypeError, ValueError):
            return False

    def _load_ml_artifacts(self) -> None:
        if self._ml_model is not None:
            return
        module = sys.modules.get(self.__class__.__module__)
        module_file = getattr(module, "__file__", None)
        if not module_file:
            raise FileNotFoundError(
                f"Could not determine strategy directory from module '{self.__class__.__module__}'"
            )
        from .research.ml import load_ml_model

        artifacts = load_ml_model(os.path.dirname(os.path.abspath(module_file)))
        self._ml_model = artifacts["model"]
        self._ml_scaler = artifacts["scaler"]
        self._ml_feature_importance = artifacts.get("feature_importance")

    def ml_features(self) -> dict:
        raise NotImplementedError(
            "Override ml_features() in your strategy and return {feature_name: value}."
        )

    def _ml_input(self):
        features = self.ml_features()
        if not isinstance(features, dict) or not features:
            raise ValueError("ml_features() must return a non-empty dict")
        return np.array([[features[key] for key in sorted(features)]], dtype=float)

    def ml_predict(self) -> float:
        self._load_ml_artifacts()
        values = self._ml_scaler.transform(self._ml_input())
        return float(self._ml_model.predict(values)[0])

    def ml_predict_proba(self) -> dict:
        self._load_ml_artifacts()
        values = self._ml_scaler.transform(self._ml_input())
        probabilities = self._ml_model.predict_proba(values)[0]
        return {_class_label(label): float(probability)
                for label, probability in zip(self._ml_model.classes_, probabilities)}

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
        self._replace_entry_orders(sides.BUY, self._buy)

    @property
    def sell(self):
        return self._sell

    @sell.setter
    def sell(self, value):
        self._sell = _normalize_orders(value)
        self._replace_entry_orders(sides.SELL, self._sell)

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
            self._sync_exit_orders(order_submitted_via.STOP_LOSS)

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
            self._sync_exit_orders(order_submitted_via.TAKE_PROFIT)

    _in_go_entry = False

    def liquidate(self) -> None:
        """Close the open position immediately at market."""
        if not self.is_open:
            return
        for order in self._active_orders():
            if order.role == order_roles.CLOSE_POSITION:
                self.simulator.cancel_order(order)
        qty = abs(self.position.qty)
        side = sides.SELL if self.position.qty > 0 else sides.BUY
        price = self.price
        self.simulator.submit_order(self, side, qty, price,
                                    role=order_roles.CLOSE_POSITION, reduce_only=True)

    # ================================================================= engine driver
    def _execute(self):
        self._cache = {}
        self.before()

        if self.position.is_open:
            self.update_position()

        # If there is no open position (either there wasn't one, or update_position() just
        # closed it via liquidate), evaluate entries on the SAME candle — this matches Jesse's
        # same-candle position-flip behaviour.
        if self.position.is_close:
            self._evaluate_entries()

        self.after()
        # Jesse exposes index 0 during the first strategy execution and advances
        # it only after before/check/after have completed.
        self.index += 1

    def _evaluate_entries(self):
        active_entries = [o for o in self._active_orders()
                          if o.role == order_roles.OPEN_POSITION]
        if active_entries:
            if self.should_cancel_entry():
                self.simulator.cancel_entry_orders(self.symbol)
                self.on_cancel()
                for route in self._other_routes():
                    route.strategy.on_route_canceled(self)
            return
        self._reset_pending()
        want_long = self.should_long()
        want_short = self.should_short()
        if want_long and want_short:
            raise ConflictingRules(
                "should_short and should_long cannot both return True.")
        if want_short and self.is_spot_trading:
            raise InvalidShortSellOnSpot("Shorting is not supported on spot.")
        if want_long or want_short:
            self._in_go_entry = True
            try:
                if want_long:
                    self.go_long()
                else:
                    self.go_short()
            finally:
                self._in_go_entry = False
            if self._filters_pass():
                self._submit_entry(sides.BUY if want_long else sides.SELL)
            else:
                # Rejected smart-order specs are not visible on later candles.
                self._reset_pending()
            # futures may have set stop/take in go_* — synced on open

    def _filters_pass(self):
        self._evaluating_filters = True
        try:
            for f in self.filters():
                try:
                    passed = f()
                except TypeError as exc:
                    raise InvalidStrategy(
                        "Invalid filter format. You need to pass filter methods "
                        "WITHOUT calling them (no parentheses must be present at "
                        "the end)."
                    ) from exc
                if not passed:
                    return False
            return True
        finally:
            self._evaluating_filters = False

    def _submit_entry(self, side):
        spec = self._buy if side == sides.BUY else self._sell
        if spec is None:
            return
        self._submit_entry_specs(side, spec)

    def _submit_entry_specs(self, side, spec):
        submitted = []
        for qty, price in spec:
            submitted.append(self.simulator.submit_order(
                self, side, qty, price, role=order_roles.OPEN_POSITION,
                defer_execution=True))
        # Jesse reserves/submits the complete batch before executing market
        # orders, so callbacks can inspect all pending orders and reservations.
        for order in submitted:
            self.simulator.execute_market_order(order)

    def _replace_entry_orders(self, side, spec):
        if (self.position is None or self.simulator is None or
                not self.position.is_open or self._in_go_entry):
            return
        for order in self._active_orders():
            if order.role == order_roles.OPEN_POSITION:
                self.simulator.cancel_order(order)
        if spec is not None and len(spec):
            self._submit_entry_specs(side, spec)

    def _sync_exit_orders(self, changed=None):
        """Ensure active exit orders match the current stop_loss/take_profit specs."""
        if not self.is_open:
            return
        self._validate_exit_specs()
        exit_side = sides.SELL if self.position.qty > 0 else sides.BUY
        # A stop reassignment replaces stops only; a take-profit reassignment
        # replaces takes only. This matters for tiered exits where one side is
        # updated after a partial fill. ``changed=None`` performs the initial
        # full synchronization when a position opens.
        tags = ((order_submitted_via.STOP_LOSS,)
                if changed == order_submitted_via.STOP_LOSS else
                (order_submitted_via.TAKE_PROFIT,)
                if changed == order_submitted_via.TAKE_PROFIT else
                (order_submitted_via.STOP_LOSS,
                 order_submitted_via.TAKE_PROFIT))
        for o in self._active_orders():
            if (o.role == order_roles.CLOSE_POSITION and
                    o.submitted_via in tags):
                self.simulator.cancel_order(o)
        for spec, tag in (
                (self._stop_loss, order_submitted_via.STOP_LOSS),
                (self._take_profit, order_submitted_via.TAKE_PROFIT)):
            if tag not in tags or spec is None:
                continue
            for qty, price in spec:
                submit_price = price
                if tag == order_submitted_via.STOP_LOSS and (
                        (self.is_long and price >= self.price) or
                        (self.is_short and price <= self.price)):
                    submit_price = self.price
                elif tag == order_submitted_via.TAKE_PROFIT and (
                        (self.is_long and price <= self.price) or
                        (self.is_short and price >= self.price)):
                    submit_price = self.price
                order = self.simulator.submit_order(
                    self, exit_side, abs(qty), submit_price,
                    role=order_roles.CLOSE_POSITION, reduce_only=True,
                    submitted_via=tag)

    def _validate_exit_specs(self):
        if (self._stop_loss is not None and self._take_profit is not None and
                len(self._stop_loss) and
                np.array_equal(self._stop_loss, self._take_profit)):
            raise InvalidStrategy(
                "stop-loss and take-profit should not be exactly the same. "
                "Just use either one of them and it will do."
            )

    def _reset_pending(self):
        self._buy = None
        self._sell = None
        self._stop_loss = None
        self._take_profit = None

    # engine event bridges (called by simulator)
    def _on_open_position(self, order):
        self.increased_count = 1
        self.trade = self.simulator._open_trade.get(self.symbol)
        # Futures strategies may define protective orders in go_long/go_short.
        # Jesse submits those orders before invoking on_open_position(), so the
        # hook can inspect or replace them. Spot strategies define protection in
        # on_open_position(); the property setters synchronize those immediately.
        self._sync_exit_orders()
        self.on_open_position(order)
        for r in self._other_routes():
            r.strategy.on_route_open_position(self)

    def _on_close_position(self, order, closed_trade):
        self.last_trade_index = self.index
        self.trades_count += 1
        self.increased_count = 0
        self.reduced_count = 0
        self._reset_pending()
        self.on_close_position(order, closed_trade)
        for r in self._other_routes():
            r.strategy.on_route_close_position(self)

    def _on_increased_position(self, order):
        self.increased_count += 1
        self.on_increased_position(order)
        for r in self._other_routes():
            r.strategy.on_route_increased_position(self)

    def _on_reduced_position(self, order):
        self.reduced_count += 1
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
    def capital(self) -> float:
        raise NotImplementedError("self.capital was removed; use self.balance instead")

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

    @property
    def routes(self):
        return list(self.store_routes)

    @property
    def data_routes(self):
        return list(self.simulator.data_routes) if self.simulator is not None else []

    @property
    def current_route_index(self) -> int:
        if self._current_route_index is None:
            for index, route in enumerate(self.routes):
                if (route.exchange, route.symbol, route.timeframe) == (
                        self.exchange, self.symbol, self.timeframe):
                    self._current_route_index = index
                    break
            else:
                self._current_route_index = -1
        return self._current_route_index

    @property
    def mark_price(self) -> float:
        return self.position.mark_price

    @property
    def funding_rate(self) -> float:
        return self.position.funding_rate

    @property
    def next_funding_timestamp(self):
        return self.position.next_funding_timestamp

    @property
    def liquidation_price(self) -> float:
        return self.position.liquidation_price

    @property
    def all_positions(self) -> dict:
        return {route.symbol: route.strategy.position for route in self.routes}

    @property
    def daily_balances(self) -> list:
        return self.store.app.daily_balance

    @property
    def min_qty(self) -> float:
        if not self.is_live:
            raise ValueError("self.min_qty is only available in live modes")
        return None

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
        spec = self._buy if self._buy is not None else self._sell
        if spec is None:
            return self.position.entry_price if self.is_open else None
        qty = np.abs(spec[:, 0]).sum()
        return float(np.abs(spec[:, 0] * spec[:, 1]).sum() / qty) if qty else None

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
        orders = [o for o in self._active_orders()
                  if o.role == order_roles.OPEN_POSITION]
        if not orders:
            return ((self._in_go_entry or self._evaluating_filters) and
                    self._buy is not None and len(self._buy) > 0)
        return orders[0].is_buy

    @property
    def has_short_entry_orders(self) -> bool:
        orders = [o for o in self._active_orders()
                  if o.role == order_roles.OPEN_POSITION]
        if not orders:
            return ((self._in_go_entry or self._evaluating_filters) and
                    self._sell is not None and len(self._sell) > 0)
        return orders[0].is_sell

    # ================================================================= orders/trades
    @property
    def orders(self):
        return self.store.orders.get_orders(self.symbol)

    @property
    def entry_orders(self):
        return [o for o in self.orders
                if o.role == order_roles.OPEN_POSITION and not o.is_canceled]

    @property
    def exit_orders(self):
        return [o for o in self.orders
                if o.role == order_roles.CLOSE_POSITION and not o.is_canceled]

    @property
    def active_exit_orders(self):
        return [o for o in self.exit_orders if o.is_active]

    @property
    def trades(self):
        return self.store.closed_trades

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
        return self.store.app.trading_mode == "livetrade"

    @property
    def is_papertrading(self) -> bool:
        return self.store.app.trading_mode == "papertrade"

    @property
    def is_live(self) -> bool:
        return self.is_livetrading or self.is_papertrading

    @property
    def shared_vars(self) -> dict:
        if self.store is not None:
            return self.store.vars
        from .store import get_current_store
        return get_current_store().vars

    # ================================================================= misc
    @staticmethod
    def log(msg, log_type="info", send_notification=False, webhook=None):
        from .services import logger
        msg = str(msg)
        if log_type == "info":
            logger.info(msg, send_notification=send_notification, webhook=webhook)
        elif log_type == "error":
            logger.error(msg, send_notification=send_notification)
        else:
            raise ValueError(
                f'log_type should be either "info" or "error". You passed {log_type}')

    def add_line_to_candle_chart(self, title, value, color=None):
        self._validate_chart_value(value)
        self._chart_lines.append(("candle_line", title, value, color))

    def add_horizontal_line_to_candle_chart(self, title, value, color=None,
                                            line_width=1.5, line_style="solid"):
        self._validate_chart_value(value)
        self._validate_line_style(line_style)
        self._chart_lines.append(("candle_hline", title, value, color, line_width, line_style))

    def add_extra_line_chart(self, chart_name, title, value, color=None):
        self._validate_chart_value(value)
        self._chart_lines.append(("extra_line", chart_name, title, value, color))

    def add_horizontal_line_to_extra_chart(self, chart_name, title, value, color=None,
                                           line_width=1.5, line_style="solid"):
        self._validate_chart_value(value)
        self._validate_line_style(line_style)
        self._chart_lines.append(("extra_hline", chart_name, title, value, color,
                                  line_width, line_style))

    @staticmethod
    def _validate_chart_value(value):
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"Invalid value type: {type(value)}. The value must be either "
                f"int or float; you're passing {value}")

    @staticmethod
    def _validate_line_style(line_style):
        if line_style not in ("solid", "dotted"):
            raise ValueError(f"Invalid line_style: {line_style}")

def _class_label(value):
    """Preserve numeric/string sklearn class labels while unboxing NumPy scalars."""
    return value.item() if isinstance(value, np.generic) else value
