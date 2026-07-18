from .. import helpers as jh
from ..enums import order_statuses, order_submitted_via, sides
from ._compat import CallableDict


class Order:
    """A single order. Type (market/limit/stop) is resolved by the broker via smart-routing."""

    def __init__(self, attributes: dict = None):
        attributes = attributes or {}
        self.id = attributes.get("id", jh.generate_unique_id())
        self.symbol = attributes.get("symbol")
        self.exchange = attributes.get("exchange")
        self.side = attributes.get("side")          # 'buy' | 'sell'
        self.type = attributes.get("type")          # MARKET | LIMIT | STOP
        self.reduce_only = attributes.get("reduce_only", False)
        self.qty = attributes.get("qty", 0.0)       # signed: + for buy, - for sell
        self.filled_qty = attributes.get("filled_qty", 0.0)
        self.price = attributes.get("price")
        self.status = attributes.get("status", order_statuses.ACTIVE)
        self.created_at = attributes.get("created_at")
        self.executed_at = attributes.get("executed_at")
        self.role = attributes.get("role")
        self.submitted_via = attributes.get("submitted_via")
        self.reserved_quote = attributes.get("reserved_quote", 0.0)
        self.trade_id = attributes.get("trade_id")

    # ---- status helpers ----
    @property
    def is_active(self):
        return self.status == order_statuses.ACTIVE

    @property
    def is_executed(self):
        return self.status == order_statuses.EXECUTED

    @property
    def is_filled(self):
        return self.is_executed

    @property
    def is_new(self):
        return self.is_active

    @property
    def is_queued(self):
        return self.status == order_statuses.QUEUED

    @property
    def is_partially_filled(self):
        return self.status == order_statuses.PARTIALLY_FILLED

    @property
    def is_cancellable(self):
        return self.is_active or self.is_partially_filled or self.is_queued

    @property
    def is_canceled(self):
        return self.status == order_statuses.CANCELED

    @property
    def is_buy(self):
        return self.side == sides.BUY

    @property
    def is_sell(self):
        return self.side == sides.SELL

    @property
    def is_stop_loss(self):
        return self.submitted_via == order_submitted_via.STOP_LOSS

    @property
    def is_take_profit(self):
        return self.submitted_via == order_submitted_via.TAKE_PROFIT

    @property
    def value(self):
        return abs(self.qty) * self.price

    @property
    def remaining_qty(self):
        return jh.prepare_qty(abs(self.qty) - abs(self.filled_qty), self.side)

    def execute(self, price=None, timestamp=None):
        if price is not None:
            self.price = price
        self.filled_qty = self.qty
        self.status = order_statuses.EXECUTED
        self.executed_at = timestamp

    def cancel(self):
        if self.is_executed:
            return
        self.status = order_statuses.CANCELED

    @property
    def to_dict(self):
        return CallableDict({
            "id": self.id,
            "symbol": self.symbol,
            "exchange": self.exchange,
            "side": self.side,
            "type": self.type,
            "qty": self.qty,
            "filled_qty": self.filled_qty,
            "price": self.price,
            "status": self.status,
            "created_at": self.created_at,
            "executed_at": self.executed_at,
            "role": self.role,
            "reduce_only": self.reduce_only,
            "trade_id": self.trade_id,
        })
