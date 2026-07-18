import numpy as np

from .. import helpers as jh
from ..enums import trade_types
from ._compat import CallableDict


class ClosedTrade:
    """
    A completed round-trip trade. Accumulates the buy/sell orders that made it up and
    computes PnL, fees, and holding period. `to_dict` mirrors Jesse's trade schema.
    """

    def __init__(self):
        self.id = jh.generate_unique_id()
        self.strategy_name = None
        self.symbol = None
        self.exchange = None
        self.type = None                 # 'long' | 'short'
        self.opened_at = None            # ms
        self.closed_at = None            # ms
        self.orders = []                 # list of Order
        self.buy_orders = []             # list of (qty, price)
        self.sell_orders = []            # list of (qty, price)
        self.leverage = 1

    # ---- accumulation ----
    def add_order(self, order):
        self.orders.append(order)
        if order.is_buy:
            self.buy_orders.append((abs(order.qty), order.price))
        else:
            self.sell_orders.append((abs(order.qty), order.price))

    # ---- computed ----
    @property
    def qty(self) -> float:
        if self.type == trade_types.LONG:
            return float(sum(q for q, _ in self.buy_orders))
        return float(sum(q for q, _ in self.sell_orders))

    @property
    def entry_price(self) -> float:
        orders = self.buy_orders if self.type == trade_types.LONG else self.sell_orders
        total_qty = sum(q for q, _ in orders)
        if total_qty == 0:
            return np.nan
        return float(sum(q * p for q, p in orders) / total_qty)

    @property
    def exit_price(self) -> float:
        orders = self.sell_orders if self.type == trade_types.LONG else self.buy_orders
        total_qty = sum(q for q, _ in orders)
        if total_qty == 0:
            return np.nan
        return float(sum(q * p for q, p in orders) / total_qty)

    @property
    def fee(self) -> float:
        # fee is computed from the exchange fee rate captured at close time
        return float(self._fee)

    @fee.setter
    def fee(self, v):
        self._fee = v

    @property
    def size(self) -> float:
        return float(self.qty * self.entry_price)

    @property
    def pnl(self) -> float:
        """Net PnL including fees."""
        qty = self.qty
        if self.type == trade_types.LONG:
            gross = qty * (self.exit_price - self.entry_price)
        else:
            gross = qty * (self.entry_price - self.exit_price)
        return float(gross - self._fee)

    @property
    def pnl_percentage(self) -> float:
        return self.roi

    @property
    def roi(self) -> float:
        if self.size == 0:
            return 0.0
        return float((self.pnl / (self.size / self.leverage)) * 100)

    @property
    def total_cost(self) -> float:
        return self.entry_price * abs(self.qty) / self.leverage

    @property
    def is_long(self) -> bool:
        return self.type == trade_types.LONG

    @property
    def is_short(self) -> bool:
        return self.type == trade_types.SHORT

    @property
    def is_open(self) -> bool:
        return self.opened_at is not None

    @property
    def current_qty(self) -> float:
        return float(sum(order.qty for order in self.orders if order.is_executed))

    @property
    def holding_period(self) -> int:
        """Seconds the trade was held."""
        if self.opened_at is None or self.closed_at is None:
            return 0
        return int((self.closed_at - self.opened_at) / 1000)

    @property
    def to_dict(self) -> dict:
        return CallableDict({
            "id": self.id,
            "strategy_name": self.strategy_name,
            "symbol": self.symbol,
            "exchange": self.exchange,
            "type": self.type,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "qty": self.qty,
            "size": self.size,
            "PNL": self.pnl,
            "PNL_percentage": self.pnl_percentage,
            "fee": self._fee,
            "holding_period": self.holding_period,
            "opened_at": self.opened_at,
            "closed_at": self.closed_at,
            "orders": [o.to_dict() for o in self.orders],
        })

    @property
    def to_json(self) -> dict:
        result = self.to_dict()
        result.pop("orders", None)
        return result

    @property
    def to_dict_with_orders(self) -> dict:
        return self.to_dict()

    _fee = 0.0
