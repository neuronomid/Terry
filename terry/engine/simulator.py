"""
The backtest simulator: steps 1-minute candles, fills orders using smart-order routing,
dispatches position lifecycle events to strategies, and builds closed trades.
"""
import numpy as np

from .. import helpers as jh
from ..enums import order_types, order_statuses, sides, order_roles, trade_types
from ..models import Order, ClosedTrade

ONE_MIN_MS = 60_000
DAY_MS = 86_400_000


class Simulator:
    def __init__(self, store, routes, data_routes, run_silently=True):
        self.store = store
        self.routes = routes
        self.data_routes = data_routes
        self.run_silently = run_silently
        self.logs = []
        self.warmup_1m = 0
        self.signal_only = False
        self.signal_log = []        # list of (timestamp, close, signal) for significance testing
        # per-symbol open-trade builders
        self._open_trade = {}       # symbol -> ClosedTrade (in progress)

    # ------------------------------------------------------------------ util
    def _log(self, msg):
        if not self.run_silently:
            print(msg)
        self.logs.append({"time": self.store.app.time, "message": msg})

    def _exchange_for(self, name):
        return self.store.exchanges[name]

    def _fill_time(self):
        # fills happen at the close of the current 1m candle
        return self.store.app.time + ONE_MIN_MS

    # ---------------------------------------------------------- order submit
    def submit_order(self, strategy, side, qty, price, role, reduce_only=False):
        """Create an order, infer its type via smart routing, execute market orders now."""
        symbol = strategy.symbol
        exchange = strategy.exchange
        current = self.store.candles.current_1m_price(exchange, symbol)
        qty = abs(qty)
        signed_qty = qty if side == sides.BUY else -qty

        otype = self._infer_type(side, price, current)
        order = Order({
            "symbol": symbol, "exchange": exchange, "side": side, "type": otype,
            "qty": signed_qty, "price": float(price), "role": role,
            "reduce_only": reduce_only, "created_at": self.store.app.time,
            "status": order_statuses.ACTIVE,
        })
        self.store.orders.add_order(order)
        if otype == order_types.MARKET:
            self._execute_order(order, fill_price=current)
        return order

    @staticmethod
    def _infer_type(side, price, current):
        # near current price → market (0.01% tolerance)
        if abs(price - current) <= current * 1e-4:
            return order_types.MARKET
        if side == sides.BUY:
            return order_types.LIMIT if price < current else order_types.STOP
        else:  # sell
            return order_types.LIMIT if price > current else order_types.STOP

    # ------------------------------------------------------------- execution
    def _execute_order(self, order, fill_price):
        if not order.is_active:
            return
        order.execute(price=fill_price, timestamp=self._fill_time())
        symbol = order.symbol
        position = self.store.positions.get_position(symbol)
        strategy = position.strategy

        was_open = position.is_open
        prev_type = position.type

        # start a trade builder if opening from flat
        if not was_open:
            trade = ClosedTrade()
            trade.strategy_name = strategy.name
            trade.symbol = symbol
            trade.exchange = order.exchange
            trade.type = trade_types.LONG if order.is_buy else trade_types.SHORT
            trade.leverage = self._exchange_for(order.exchange).leverage
            trade.opened_at = self._fill_time()
            trade._fee = 0.0
            self._open_trade[symbol] = trade

        result = position._on_executed_order(order)

        # accumulate into the trade builder
        trade = self._open_trade.get(symbol)
        if trade is not None:
            trade.add_order(order)
            trade._fee += result["fee"]

        # dispatch events
        if not was_open and position.is_open:
            self._cancel_other_active(symbol, keep=None)  # entries stay; handled by strategy
            strategy._on_open_position(order)
        elif was_open and position.is_close:
            # finalize the trade
            if trade is not None:
                trade.closed_at = self._fill_time()
                self.store.closed_trades.append(trade)
                self._open_trade.pop(symbol, None)
            self._cancel_symbol_orders(symbol)
            strategy._on_close_position(order, trade)
        elif was_open and position.is_open:
            # increased or reduced
            if position.type == prev_type and abs(position.qty) > 0:
                if order.role == order_roles.REDUCE_POSITION or order.reduce_only:
                    strategy._on_reduced_position(order)
                else:
                    # heuristic: same direction as position → increase
                    increased = (order.is_buy and position.qty > 0) or (order.is_sell and position.qty < 0)
                    if increased:
                        strategy._on_increased_position(order)
                    else:
                        strategy._on_reduced_position(order)

    def _cancel_symbol_orders(self, symbol):
        for o in self.store.orders.active_orders(symbol):
            o.cancel()

    def _cancel_other_active(self, symbol, keep):
        pass

    def cancel_entry_orders(self, symbol):
        for o in self.store.orders.active_orders(symbol):
            if o.role == order_roles.OPEN_POSITION:
                o.cancel()

    # --------------------------------------------------------------- fills
    def _process_fills(self, i):
        """Fill active limit/stop orders against 1m candle i's OHLC."""
        for symbol, orders in list(self.store.orders.storage.items()):
            exchange = None
            position = self.store.positions.get_position(symbol)
            if position is None:
                continue
            exchange = position.exchange_name
            base = self.store.candles.raw_1m[jh.key(exchange, symbol)]
            _, o, c, high, low, _ = base[i]
            for order in list(orders):
                if not order.is_active:
                    continue
                p = order.price
                filled = False
                if order.type == order_types.LIMIT:
                    if order.is_buy and low <= p:
                        filled = True
                    elif order.is_sell and high >= p:
                        filled = True
                elif order.type == order_types.STOP:
                    if order.is_buy and high >= p:
                        filled = True
                    elif order.is_sell and low <= p:
                        filled = True
                if filled:
                    self._execute_order(order, fill_price=p)

    def _collect_signal(self, route):
        """Signal-only pass for rule significance testing: record +1/-1/0 per bar."""
        strat = route.strategy
        strat.index += 1
        strat._cache = {}
        try:
            long = strat.should_long()
        except Exception:
            long = False
        short = False
        if not long:
            try:
                short = strat.should_short()
            except Exception:
                short = False
        signal = 1 if long else (-1 if short else 0)
        close = float(strat.candles[-1][2])
        self.signal_log.append((int(self.store.app.time), close, signal))

    # --------------------------------------------------------------- run
    def run(self, generate_equity_curve=False):
        store = self.store
        app = store.app

        # reference 1m length from the first trading route
        first = self.routes[0]
        base_key = jh.key(first.exchange, first.symbol)
        n = len(store.candles.raw_1m[base_key])
        warmup_1m = self.warmup_1m

        # trading period begins after warmup
        first_trade_idx = min(warmup_1m, max(n - 1, 0))
        app.starting_time = store.candles.raw_1m[base_key][first_trade_idx, 0]
        app.ending_time = store.candles.raw_1m[base_key][-1, 0] + ONE_MIN_MS

        equity_curve = []
        last_day = None

        for i in range(n):
            app.index_1m = i
            app.time = store.candles.raw_1m[base_key][i, 0]

            # update current prices on every position
            for sym, pos in store.positions.storage.items():
                ex = pos.exchange_name
                pos.current_price = store.candles.current_1m_price(ex, sym)

            if i < warmup_1m:
                continue

            if not self.signal_only:
                # fill pending orders against this 1m candle
                self._process_fills(i)

            # run each trading strategy when its route timeframe candle closes
            for route in self.routes:
                tf = jh.timeframe_to_one_minutes(route.timeframe)
                if (i + 1) % tf == 0:
                    if self.signal_only:
                        self._collect_signal(route)
                    else:
                        route.strategy._execute()

            # daily balance snapshot (trading period only)
            day = app.time // DAY_MS
            if last_day is None:
                last_day = day
                app.daily_balance.append(store.portfolio_value())
            elif day != last_day:
                app.daily_balance.append(store.portfolio_value())
                last_day = day
            if generate_equity_curve:
                equity_curve.append({"time": int(app.time), "value": store.portfolio_value()})

        # End of backtest (matches Jesse's _terminate): count positions still open, then
        # force-close each at the last price so it appears as a closed trade too.
        open_trades = 0
        open_pl = 0.0
        if not self.signal_only:
            for sym, pos in list(store.positions.storage.items()):
                if pos.is_open:
                    open_trades += 1
                    open_pl += pos.pnl
                    strat = pos.strategy
                    price = store.candles.current_1m_price(pos.exchange_name, sym)
                    side = sides.SELL if pos.qty > 0 else sides.BUY
                    self.submit_order(strat, side, abs(pos.qty), price,
                                      role=order_roles.CLOSE_POSITION, reduce_only=True)
            # cancel any still-active entry orders that never filled
            for order in store.orders.get_orders():
                if order.is_active:
                    order.cancel()
        app.total_open_trades = open_trades
        app.total_open_pl = open_pl

        return {"equity_curve": equity_curve}
