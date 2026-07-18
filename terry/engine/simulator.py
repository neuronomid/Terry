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
    def submit_order(self, strategy, side, qty, price, role, reduce_only=False,
                     defer_execution=False, submitted_via=None):
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
            "status": order_statuses.ACTIVE, "submitted_via": submitted_via,
        })
        exchange_model = self._exchange_for(exchange)
        exchange_model.validate_order_submission(order)
        exchange_model.reserve_spot_buy(order)
        self.store.orders.add_order(order)
        if otype == order_types.MARKET and not defer_execution:
            self._execute_order(order, fill_price=current)
        return order

    def execute_market_order(self, order):
        if order.is_active and order.type == order_types.MARKET:
            current = self.store.candles.current_1m_price(
                order.exchange, order.symbol)
            self._execute_order(order, fill_price=current)

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
        symbol = order.symbol
        position = self.store.positions.get_position(symbol)
        if order.reduce_only:
            reduces_long = position.is_long and order.is_sell
            reduces_short = position.is_short and order.is_buy
            if not position.is_open or not (reduces_long or reduces_short):
                self.cancel_order(order)
                return
            fill_qty = min(abs(order.qty), abs(position.qty))
            order.qty = fill_qty if order.is_buy else -fill_qty
        order.execute(price=fill_price, timestamp=self._fill_time())
        strategy = position.strategy

        was_open = position.is_open
        prev_type = position.type

        # start a trade builder if opening from flat
        if not was_open:
            trade = ClosedTrade()
            trade.strategy_name = strategy.name
            trade.symbol = symbol
            trade.exchange = order.exchange
            trade.timeframe = strategy.timeframe
            trade.session_id = self.store.app.session_id
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
                for trade_order in trade.orders:
                    if not trade_order.is_canceled:
                        trade_order.trade_id = trade.id
                self.store.closed_trades.append(trade)
                self._open_trade.pop(symbol, None)
            self._cancel_symbol_orders(symbol)
            # Strategy.orders exposes current-trade orders in Jesse. Closed
            # orders remain available through the ClosedTrade object.
            self.store.orders.storage[symbol] = []
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
            self.cancel_order(o)

    def _cancel_other_active(self, symbol, keep):
        pass

    def cancel_entry_orders(self, symbol):
        for o in self.store.orders.active_orders(symbol):
            if o.role == order_roles.OPEN_POSITION:
                self.cancel_order(o)

    def cancel_order(self, order):
        if not order.is_active:
            return
        self._exchange_for(order.exchange).release_spot_reservation(order)
        order.cancel()

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
            fillable = []
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
                    fillable.append(order)
            on_open = [order for order in fillable if order.price == o]
            above = [order for order in fillable if order.price > o]
            below = [order for order in fillable if order.price < o]
            above.sort(key=lambda order: order.price)
            below.sort(key=lambda order: order.price, reverse=True)
            ordered = ([*on_open, *above, *below] if o > c
                       else [*on_open, *below, *above])
            for order in ordered:
                if order.is_active:
                    self._execute_order(order, fill_price=order.price)

    def _collect_signal(self, route):
        """Signal-only pass for rule significance testing: record +1/-1/0 per bar."""
        strat = route.strategy
        strat._cache = {}
        try:
            strat.before()
            long = strat.should_long()
            short = strat.should_short()
            strat.after()
        finally:
            strat.index += 1
        signal = 1 if long else (-1 if short else 0)
        close = float(strat.candles[-1][2])
        self.signal_log.append((int(self.store.app.time), close, signal))

    def _process_liquidations(self, i):
        """Liquidate isolated futures positions whose maintenance level was crossed."""
        for position in list(self.store.positions.storage.values()):
            if (not position.is_open or not position.exchange.is_futures or
                    position.mode != "isolated"):
                continue
            base = self.store.candles.raw_1m[
                jh.key(position.exchange_name, position.symbol)]
            _, _, close, high, low, _ = base[i]
            liquidation_price = position.liquidation_price
            crossed = ((position.is_long and low <= liquidation_price) or
                       (position.is_short and high >= liquidation_price))
            if not crossed:
                continue

            # Liquidation is triggered at the maintenance threshold and settles
            # at the bankruptcy price. This consumes the isolated collateral but
            # cannot invert the position or make the wallet negative.
            fill_price = float(position.bankruptcy_price)
            side = sides.SELL if position.is_long else sides.BUY
            order = self.submit_order(
                position.strategy, side, abs(position.qty), float(close),
                role=order_roles.CLOSE_POSITION, reduce_only=True,
                defer_execution=True,
            )
            order.submitted_via = "liquidation"
            self._execute_order(order, fill_price=fill_price)
            self.store.app.total_liquidations += 1

    # --------------------------------------------------------------- run
    def run(self, generate_equity_curve=False, should_cancel=None):
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
            if should_cancel and should_cancel():
                raise InterruptedError("Research run canceled")
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
                # Liquidation precedes Strategy.before(), matching the state a
                # strategy observes when the current candle breaches the level.
                self._process_liquidations(i)

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

        # End of backtest (matches Jesse's Strategy._terminate lifecycle): give each
        # strategy a final mutation hook, account for positions that remain open,
        # force-close them, then call terminate().
        open_trades = 0
        open_pl = 0.0
        if not self.signal_only:
            for route in self.routes:
                strat = route.strategy
                strat.before_terminate()
                pos = strat.position
                if pos.is_open:
                    open_trades += 1
                    open_pl += pos.pnl
                    for order in list(strat.active_exit_orders):
                        self.cancel_order(order)
                    price = store.candles.current_1m_price(
                        pos.exchange_name, pos.symbol)
                    side = sides.SELL if pos.qty > 0 else sides.BUY
                    self.submit_order(strat, side, abs(pos.qty), price,
                                      role=order_roles.CLOSE_POSITION, reduce_only=True)
                elif any(order.is_active for order in strat.entry_orders):
                    self.cancel_entry_orders(pos.symbol)
                    strat.on_cancel()
                strat.terminate()
            # cancel any still-active entry orders that never filled
            for order in store.orders.get_orders():
                if order.is_active:
                    self.cancel_order(order)
            store.app.daily_balance.append(store.portfolio_value())
        app.total_open_trades = open_trades
        app.total_open_pl = open_pl

        return {"equity_curve": equity_curve}
