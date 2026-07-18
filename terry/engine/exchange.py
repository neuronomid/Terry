from ..enums import exchange_types


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
        self.assets[self.quote_asset] -= fee

    def add_realized_pnl(self, pnl):
        self.assets[self.quote_asset] += pnl

    def spend_quote(self, amount):
        self.assets[self.quote_asset] -= amount

    def receive_quote(self, amount):
        self.assets[self.quote_asset] += amount

    def add_base(self, symbol, qty):
        self.base_assets[symbol] = self.base_assets.get(symbol, 0.0) + qty

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
            return self.balance - used
        # spot
        return self.balance - order_value

    @property
    def leveraged_available_margin(self):
        return self.available_margin * self.leverage

    def wallet_balance(self):
        return self.balance
