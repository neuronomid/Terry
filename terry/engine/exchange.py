from ..enums import exchange_types
from ..exceptions import InsufficientBalance, InsufficientMargin
from ..enums import order_types
from ..utils import subtract_floats, sum_floats


class Exchange:
    """
    A simulated exchange holding wallet balance and driving margin/fee accounting for
    one exchange (spot or futures). One Exchange instance per configured exchange.
    """

    def __init__(self, name, starting_balance, fee_rate, exchange_type,
                 futures_leverage=1, futures_leverage_mode="cross", quote_asset="USDT"):
        self.name = name
        self.type = exchange_type            # 'spot' | 'futures'
        self.fee_rate = fee_rate
        self.futures_leverage = futures_leverage
        self.futures_leverage_mode = futures_leverage_mode
        self.quote_asset = quote_asset

        self.starting_assets = {quote_asset: float(starting_balance)}
        self.assets = {quote_asset: float(starting_balance)}
        # spot base holdings keyed by base asset
        self.base_assets = {}

        self.store = None  # set by engine (for open position/order lookups)

    # ------------------------------------------------------------------ props
    @property
    def is_futures(self):
        return self.type == exchange_types.FUTURES

    @property
    def is_spot(self):
        return self.type == exchange_types.SPOT

    @property
    def leverage(self):
        return self.futures_leverage if self.is_futures else 1

    @property
    def balance(self):
        return self.assets[self.quote_asset]

    @property
    def starting_balance(self):
        return self.starting_assets[self.quote_asset]

    # --------------------------------------------------------------- mutation
    def charge_fee(self, fee):
        self.assets[self.quote_asset] = subtract_floats(
            self.assets[self.quote_asset], fee)

    def add_realized_pnl(self, pnl):
        self.assets[self.quote_asset] = sum_floats(
            self.assets[self.quote_asset], pnl)

    def spend_quote(self, amount):
        self.assets[self.quote_asset] = subtract_floats(
            self.assets[self.quote_asset], amount)

    def receive_quote(self, amount):
        self.assets[self.quote_asset] = sum_floats(
            self.assets[self.quote_asset], amount)

    def add_base(self, symbol, qty):
        self.base_assets[symbol] = sum_floats(
            self.base_assets.get(symbol, 0.0), qty)
        base_asset = symbol.split("-", 1)[0]
        value = sum_floats(self.assets.get(base_asset, 0.0), qty)
        self.assets[base_asset] = 0.0 if abs(value) < 1e-12 else value

    def ensure_base_asset(self, symbol):
        base_asset = symbol.split("-", 1)[0]
        self.assets.setdefault(base_asset, 0.0)
        self.base_assets.setdefault(symbol, 0.0)

    def reserve_spot_buy(self, order):
        if not self.is_spot or not order.is_buy:
            return
        amount = abs(order.qty) * order.price
        available = self.balance
        if amount > available + 1e-12:
            raise InsufficientBalance(
                f"Not enough balance. Available balance at {self.name} for "
                f"{self.quote_asset} is {available} but you're trying to spend {amount}")
        self.spend_quote(amount)
        order.reserved_quote = amount

    def validate_order_submission(self, order):
        """Reject orders that exceed Jesse's simulated margin/asset limits."""
        if self.is_futures:
            if not order.reduce_only:
                required_margin = order.value / self.leverage
                if required_margin > self.available_margin + 1e-12:
                    raise InsufficientMargin(
                        f"Cannot submit an order with a value of "
                        f"${round(order.qty * order.price)} when your available "
                        f"margin is ${round(self.available_margin)}. Consider "
                        "increasing leverage number from the settings or reducing "
                        "the order size."
                    )
            return

        if not order.is_sell:
            return
        base_asset = order.symbol.split("-", 1)[0]
        base_balance = self.assets.get(base_asset, 0.0)
        active_sells = [
            candidate for candidate in self.store.orders.active_orders(order.symbol)
            if candidate.is_sell
        ]
        if order.type == order_types.MARKET:
            pending = sum(abs(candidate.qty) for candidate in active_sells
                          if candidate.type == order_types.LIMIT)
        else:
            pending = sum(abs(candidate.qty) for candidate in active_sells
                          if candidate.type == order.type)
        requested = sum_floats(abs(order.qty), pending)
        if requested > base_balance + 1e-12:
            raise InsufficientBalance(
                f"Not enough balance. Available balance at {self.name} for "
                f"{base_asset} is {base_balance} but you're trying to sell "
                f"{requested}"
            )

    def release_spot_reservation(self, order):
        amount = getattr(order, "reserved_quote", 0.0) or 0.0
        if amount:
            self.receive_quote(amount)
            order.reserved_quote = 0.0

    def store_time(self):
        # fills are recorded at the CLOSE time of the current 1m candle (matches Jesse)
        if self.store is not None and self.store.app.time is not None:
            return self.store.app.time + 60_000
        return None

    # ----------------------------------------------------------- available
    def _reserved_and_position_margin(self):
        """Return (open_position_entry_value, active_entry_order_value) for this exchange."""
        pos_value = 0.0
        order_value = 0.0
        if self.store is None:
            return pos_value, order_value
        for pos in self.store.positions.storage.values():
            if pos.exchange_name == self.name and pos.is_open:
                pos_value += pos.entry_value
        for order in self.store.orders.get_orders():
            if order.exchange == self.name and order.is_active and not order.reduce_only:
                order_value += order.value
        return pos_value, order_value

    @property
    def available_margin(self):
        pos_value, order_value = self._reserved_and_position_margin()
        if self.is_futures:
            used = (pos_value + order_value) / self.leverage
            unrealized = sum(
                position.pnl for position in self.store.positions.storage.values()
                if position.exchange_name == self.name and position.is_open)
            return self.balance - used + unrealized
        # Spot buy orders are reserved from wallet balance on submission, as in Jesse.
        return self.balance

    @property
    def leveraged_available_margin(self):
        return self.available_margin * self.leverage

    @property
    def wallet_balance(self):
        return self.balance
