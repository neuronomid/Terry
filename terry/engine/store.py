"""Global engine state store (per-backtest, reset between runs)."""
from .candle_store import CandleStore


class AppState:
    def __init__(self):
        self.time = None            # ms timestamp of the current 1m candle
        self.index_1m = 0           # index into the raw 1m arrays
        self.starting_time = None
        self.ending_time = None
        self.daily_balance = []     # portfolio value snapshots (one per day)
        self.total_open_trades = 0
        self.total_open_pl = 0.0
        self.session_id = None
        self.trading_mode = "backtest"

    def reset(self):
        self.__init__()


class PositionsState:
    def __init__(self):
        self.storage = {}           # symbol -> Position

    def reset(self):
        self.storage = {}

    def get_position(self, symbol):
        return self.storage.get(symbol)


class OrdersState:
    def __init__(self):
        self.storage = {}           # symbol -> [Order]

    def reset(self):
        self.storage = {}

    def add_order(self, order):
        self.storage.setdefault(order.symbol, []).append(order)

    def get_orders(self, symbol=None):
        if symbol is not None:
            return self.storage.get(symbol, [])
        out = []
        for lst in self.storage.values():
            out.extend(lst)
        return out

    def active_orders(self, symbol=None):
        return [o for o in self.get_orders(symbol) if o.is_active]


class Store:
    def __init__(self):
        self.app = AppState()
        self.candles = CandleStore()
        self.candles.app = self.app
        self.exchanges = {}         # name -> Exchange
        self.positions = PositionsState()
        self.orders = OrdersState()
        self.closed_trades = []
        # Jesse exposes a per-run dictionary through Strategy.shared_vars so
        # routes in the same simulation can communicate without leaking state
        # into later or concurrently executing backtests.
        self.vars = {}

    def reset(self):
        self.app.reset()
        self.candles.reset()
        self.candles.app = self.app
        self.exchanges = {}
        self.positions.reset()
        self.orders.reset()
        self.closed_trades = []
        self.vars = {}

    # convenience
    def add_exchange(self, exchange):
        exchange.store = self
        self.exchanges[exchange.name] = exchange

    def portfolio_value(self):
        """Total equity across all exchanges (wallet balance + unrealized PnL)."""
        total = 0.0
        for ex in self.exchanges.values():
            total += ex.balance
            if ex.is_spot:
                # add value of held base assets
                for sym, pos in self.positions.storage.items():
                    if pos.exchange_name == ex.name and pos.is_open and pos.qty > 0:
                        total += pos.value
        # futures: add unrealized pnl of open positions
        for pos in self.positions.storage.values():
            ex = self.exchanges.get(pos.exchange_name)
            if ex and ex.is_futures and pos.is_open:
                total += pos.pnl
        return total
