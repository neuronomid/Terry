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
from .enums import sides, order_roles, trade_types
from .exceptions import InvalidStrategy, InvalidShortSellOnSpot


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
    if arr.ndim == 1:
        arr = arr.reshape(1, 2)
    return arr


class Strategy(ABC):
    def __init__(self):
        self.id = jh.generate_unique_id()
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
        self._current_route_index = None

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
                for route in self._other_routes():
                    route.strategy.on_route_canceled(self)
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
        return self.store.vars

    # ================================================================= misc
    @staticmethod
    def log(msg, log_type="info", send_notification=False, webhook=None):
        if log_type not in ("info", "error"):
            raise ValueError(f'log_type should be either "info" or "error". You passed {log_type}')
        print(f"[{log_type}] {msg}")

    def add_line_to_candle_chart(self, title, value, color=None):
        self._chart_lines.append(("candle_line", title, value, color))

    def add_horizontal_line_to_candle_chart(self, title, value, color=None,
                                            line_width=1.5, line_style="solid"):
        self._chart_lines.append(("candle_hline", title, value, color, line_width, line_style))

    def add_extra_line_chart(self, chart_name, title, value, color=None):
        self._chart_lines.append(("extra_line", chart_name, title, value, color))

    def add_horizontal_line_to_extra_chart(self, chart_name, title, value, color=None,
                                           line_width=1.5, line_style="solid"):
        self._chart_lines.append(("extra_hline", chart_name, title, value, color,
                                  line_width, line_style))

def _class_label(value):
    """Preserve numeric/string sklearn class labels while unboxing NumPy scalars."""
    return value.item() if isinstance(value, np.generic) else value
