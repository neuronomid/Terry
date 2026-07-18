import numpy as np

from .. import helpers as jh
from ..enums import trade_types


class Position:
    """
    Tracks a single symbol's position and drives balance accounting on its exchange.

    Accounting model (validated to match Jesse's net-profit definition):
      - Fees are charged to the wallet balance immediately on every fill.
      - Futures: gross realized PnL is credited to the balance when a position is
        reduced/closed; open margin is *reserved* (not spent) via available_margin.
      - Spot: a buy spends quote + fee and adds base; a sell returns quote - fee and
        removes base. No shorting.
    Over a full trade, balance change == gross_pnl - fees == the trade's net PnL.
    """

    def __init__(self, exchange, symbol: str):
        self.exchange = exchange
        self.exchange_name = exchange.name
        self.symbol = symbol
        self.qty = 0.0                 # signed base-asset quantity
        self.entry_price = None
        self.opened_at = None
        self.closed_at = None
        self._current_price = None
        # bookkeeping for the currently-open trade
        self.increased_count = 0
        self.reduced_count = 0
        self.strategy = None

    # ------------------------------------------------------------------ price
    @property
    def current_price(self):
        return self._current_price

    @current_price.setter
    def current_price(self, p):
        self._current_price = p

    # ------------------------------------------------------------------ state
    @property
    def is_open(self):
        return self.qty != 0

    @property
    def is_close(self):
        return self.qty == 0

    @property
    def type(self):
        if self.qty > 0:
            return trade_types.LONG
        if self.qty < 0:
            return trade_types.SHORT
        return "close"

    @property
    def value(self):
        if self._current_price is None or self.qty == 0:
            return 0.0
        return abs(self.qty) * self._current_price

    @property
    def entry_value(self):
        if self.entry_price is None or self.qty == 0:
            return 0.0
        return abs(self.qty) * self.entry_price

    @property
    def pnl(self):
        """Unrealized PnL of the open position (gross)."""
        if not self.is_open or self.entry_price is None or self._current_price is None:
            return 0.0
        return self.qty * (self._current_price - self.entry_price)

    @property
    def pnl_percentage(self):
        if not self.is_open or self.entry_price is None:
            return 0.0
        margin = self.entry_value / self.exchange.leverage
        if margin == 0:
            return 0.0
        return (self.pnl / margin) * 100

    @property
    def liquidation_price(self):
        return np.nan  # not modelled

    # --------------------------------------------------------------- mutation
    def _on_executed_order(self, order):
        """Update the position and wallet balance for a filled order."""
        qty_change = order.qty              # signed
        price = order.price
        fee = abs(qty_change) * price * self.exchange.fee_rate

        prev_qty = self.qty
        new_qty = prev_qty + qty_change

        # charge the fee immediately (both spot & futures)
        self.exchange.charge_fee(fee)

        if self.exchange.is_spot:
            # spot cash accounting
            if qty_change > 0:      # buy → spend quote
                self.exchange.spend_quote(abs(qty_change) * price)
            else:                   # sell → receive quote
                self.exchange.receive_quote(abs(qty_change) * price)

        opened = prev_qty == 0 and new_qty != 0
        increased = prev_qty != 0 and (np.sign(new_qty) == np.sign(prev_qty)) and abs(new_qty) > abs(prev_qty)
        reduced_or_closed = prev_qty != 0 and (
            abs(new_qty) < abs(prev_qty) or np.sign(new_qty) != np.sign(prev_qty)
        )

        realized = 0.0
        if opened:
            self.entry_price = price
            self.opened_at = self.exchange.store_time()
            self.increased_count = 1
            self.reduced_count = 0
        elif increased:
            total = abs(prev_qty) + abs(qty_change)
            self.entry_price = (abs(prev_qty) * self.entry_price + abs(qty_change) * price) / total
            self.increased_count += 1
        elif reduced_or_closed:
            closing_qty = min(abs(qty_change), abs(prev_qty))
            # realized gross pnl on the closed portion
            realized = closing_qty * np.sign(prev_qty) * (price - self.entry_price)
            if self.exchange.is_futures:
                self.exchange.add_realized_pnl(realized)
            self.reduced_count += 1
            if abs(qty_change) >= abs(prev_qty):
                # fully closed (possibly flipped)
                remaining = abs(qty_change) - abs(prev_qty)
                if remaining > 1e-12:
                    self.entry_price = price
                    self.opened_at = self.exchange.store_time()
                    self.increased_count = 1
                    self.reduced_count = 0
                else:
                    self.entry_price = None
                    self.closed_at = self.exchange.store_time()

        self.qty = 0.0 if abs(new_qty) < 1e-12 else new_qty
        return {"fee": fee, "realized": realized}

    def reset(self):
        self.qty = 0.0
        self.entry_price = None
        self.opened_at = None
        self.closed_at = None
        self.increased_count = 0
        self.reduced_count = 0

    def to_dict(self):
        return {
            "symbol": self.symbol,
            "exchange": self.exchange_name,
            "qty": self.qty,
            "entry_price": self.entry_price,
            "current_price": self._current_price,
            "type": self.type,
            "value": self.value,
            "pnl": self.pnl,
            "pnl_percentage": self.pnl_percentage,
            "is_open": self.is_open,
        }
