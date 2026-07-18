from .. import helpers as jh
from ..enums import order_statuses, order_types, sides


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
        self.price = attributes.get("price")
        self.status = attributes.get("status", order_statuses.ACTIVE)
        self.created_at = attributes.get("created_at")
        self.executed_at = attributes.get("executed_at")
        self.role = attributes.get("role")
        self.submitted_via = attributes.get("submitted_via")

    # ---- status helpers ----
    @property
    def is_active(self):
        return self.status == order_statuses.ACTIVE

    @property
    def is_executed(self):
        return self.status == order_statuses.EXECUTED

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
        return self.role == "CLOSE POSITION" and self.type == order_types.STOP

    @property
    def is_take_profit(self):
        return self.role == "CLOSE POSITION" and self.type == order_types.LIMIT

    @property
    def value(self):
        return abs(self.qty) * self.price

    def execute(self, price=None, timestamp=None):
        if price is not None:
            self.price = price
        self.status = order_statuses.EXECUTED
        self.executed_at = timestamp

    def cancel(self):
        if self.is_executed:
            return
        self.status = order_statuses.CANCELED

    def to_dict(self):
        return {
            "id": self.id,
            "symbol": self.symbol,
            "exchange": self.exchange,
            "side": self.side,
            "type": self.type,
            "qty": self.qty,
            "price": self.price,
            "status": self.status,
            "created_at": self.created_at,
            "executed_at": self.executed_at,
            "role": self.role,
            "reduce_only": self.reduce_only,
        }
