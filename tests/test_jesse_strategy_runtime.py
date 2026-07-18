"""Regression tests for Jesse strategy lifecycle and accounting compatibility."""

from __future__ import annotations

import inspect

import pytest

from terry import helpers as jh
from terry import enums as jesse_enums
from terry import exceptions as jesse_exceptions
from terry.exceptions import InsufficientBalance, InsufficientMargin, InvalidStrategy
from terry.factories import candles_from_close_prices
from terry.models import ClosedTrade, Order, Position, Route
from terry.models.ClosedTrade import ClosedTrade as CaseClosedTrade
from terry.models.Order import Order as CaseOrder
from terry.models.Position import Position as CasePosition
from terry.models.Route import Route as CaseRoute
from terry.research.backtest import backtest
from terry.strategy import Strategy
from terry.testing_utils import single_route_backtest


JESSE_PUBLIC_HELPERS = set("""
app_currency app_mode arrow_to_timestamp base_asset binary_search class_iter
clean_orderbook_list color convert_number dashless_symbol dashy_symbol
underline_to_dashy_symbol dashy_to_underline get_base_asset get_quote_asset
date_diff_in_days date_to_timestamp dna_to_hp dump_exception
estimate_average_price estimate_PNL estimate_PNL_percentage file_exists
clear_file make_directory floor_with_precision format_currency format_price
generate_unique_id generate_short_unique_id get_arrow get_candle_source get_config
get_store get_strategy_class insecure_hash insert_list is_backtesting
is_significance_testing is_debuggable is_debugging is_importing_candles is_live
is_livetrading is_optimizing is_paper_trading is_unit_testing is_valid_uuid key
max_timeframe normalize now now_to_timestamp now_to_datetime
current_1m_candle_timestamp np_ffill np_shift opposite_side opposite_type
orderbook_insertion_index_search orderbook_trim_price prepare_qty python_version
quote_asset random_str readable_duration relative_to_absolute round_or_none
round_price_for_live_mode round_qty_for_live_mode round_decimals_down
is_almost_equal same_length secure_hash should_execute_silently side_to_type
string_after_character slice_candles style terminate_app error timestamp_to_arrow
timestamp_to_date timestamp_to_time timestamp_to_iso8601 iso8601_to_timestamp
today_to_timestamp timeframe_to_one_minutes type_to_side unique_list closing_side
merge_dicts computer_name validate_response get_session_id get_pid is_jesse_project
dd dump debug terminal_debug float_or_none str_or_none cpu_cores_count
convert_to_env_name is_notebook get_os is_docker clear_output clean_nan_values
clean_infinite_values get_class_name next_candle_timestamp
get_candle_start_timestamp_based_on_timeframe is_price_near gzip_compress
compressed_response validate_cwd has_live_trade_plugin normalize_bool
""".split())


def test_lifecycle_index_trade_metadata_and_termination_hooks():
    events = []

    class LifecycleStrategy(Strategy):
        def before(self):
            if self.index == 0:
                events.append(("first-index", self.index))

        def should_long(self):
            return self.price == 10

        def go_long(self):
            self.buy = 1, self.price
            self.take_profit = 1, 12

        def on_close_position(self, order, closed_trade):
            events.append(("closed", self.trades_count))
            assert closed_trade.timeframe == "1m"
            assert all(
                item.trade_id == closed_trade.id
                for item in closed_trade.orders if not item.is_canceled
            )

        def before_terminate(self):
            events.append(("before-terminate", self.is_close))

        def terminate(self):
            events.append(("terminate", self.trades_count))

    result = single_route_backtest(
        "LifecycleStrategy",
        leverage=2,
        strategy_classes={"LifecycleStrategy": LifecycleStrategy},
    )

    trade = result["trades"][0]
    assert jh.is_valid_uuid(trade["id"])
    assert trade["timeframe"] == "1m"
    assert events[0] == ("first-index", 0)
    assert ("closed", 1) in events
    assert events[-2:] == [("before-terminate", True), ("terminate", 1)]


def test_isolated_positions_liquidate_before_strategy_before_hook():
    observed = []

    class IsolatedLiquidation(Strategy):
        def before(self):
            if self.price == 40:
                observed.append((self.is_close, self.balance, self.available_margin))

        def should_long(self):
            return self.price == 80

        def go_long(self):
            self.buy = 250, self.price

    result = single_route_backtest(
        "IsolatedLiquidation",
        leverage=2,
        leverage_mode="isolated",
        trend="down",
        strategy_classes={"IsolatedLiquidation": IsolatedLiquidation},
    )

    assert observed == [(True, 0.0, 0.0)]
    assert result["metrics"]["total"] == 1
    assert result["trades"][0]["exit_price"] == 40


def test_spot_partial_take_profit_preserves_other_exit_tiers():
    class TieredSpotExit(Strategy):
        def should_long(self):
            return self.price == 10

        def go_long(self):
            self.buy = 2, self.price

        def on_open_position(self, order):
            self.take_profit = [(1, 15), (1, 20)]

        def on_reduced_position(self, order):
            assert self.position.qty == 1
            self.stop_loss = 1, self.price - 1

    result = single_route_backtest(
        "TieredSpotExit",
        is_futures_trading=False,
        strategy_classes={"TieredSpotExit": TieredSpotExit},
    )

    assert result["trades"][0]["entry_price"] == 10
    assert result["trades"][0]["exit_price"] == 17.5
    assert result["trades"][0]["qty"] == 2


def test_order_priority_pending_entry_views_and_filter_error():
    opened_at = []

    class SortedEntries(Strategy):
        def should_long(self):
            return self.price == 10

        def go_long(self):
            self.buy = [(1, 10.2), (1, 10.3), (1, 10.1)]

        def on_open_position(self, order):
            opened_at.append(order.price)

    single_route_backtest(
        "SortedEntries",
        strategy_classes={"SortedEntries": SortedEntries},
    )
    assert opened_at == [10.1]

    class PendingEntries(Strategy):
        def should_long(self):
            return self.price == 10

        def go_long(self):
            self.buy = [(1, 9), (1, 8)]

        def filters(self):
            return [self.pending_filter]

        def pending_filter(self):
            assert self.has_long_entry_orders is True
            assert self.has_short_entry_orders is False
            return False

        def before(self):
            if self.price == 11:
                assert self.has_long_entry_orders is False

    single_route_backtest(
        "PendingEntries",
        strategy_classes={"PendingEntries": PendingEntries},
    )

    class InvalidFilter(Strategy):
        def should_long(self):
            return True

        def go_long(self):
            self.buy = 1, self.price

        def filters(self):
            return [False]

    with pytest.raises(InvalidStrategy, match="^Invalid filter format"):
        single_route_backtest(
            "InvalidFilter",
            strategy_classes={"InvalidFilter": InvalidFilter},
        )


def test_jesse_case_sensitive_model_imports_and_strategy_logs():
    assert (CaseOrder, CaseClosedTrade, CasePosition, CaseRoute) == (
        Order, ClosedTrade, Position, Route)

    class LoggingStrategy(Strategy):
        def before(self):
            if self.index == 0:
                self.log("strategy info")
                self.log("strategy error", log_type="error")

        def should_long(self):
            return False

        def go_long(self):
            pass

    config = {
        "starting_balance": 10_000, "fee": 0, "type": "futures",
        "futures_leverage": 1, "futures_leverage_mode": "cross",
        "exchange": "Sandbox", "warm_up_candles": 0,
    }
    candles = candles_from_close_prices(range(1, 5))
    result = backtest(
        config,
        [{"exchange": "Sandbox", "symbol": "BTC-USDT", "timeframe": "1m",
          "strategy": "LoggingStrategy"}],
        [],
        {"Sandbox-BTC-USDT": {
            "exchange": "Sandbox", "symbol": "BTC-USDT", "candles": candles}},
        strategy_classes={"LoggingStrategy": LoggingStrategy},
        generate_logs=True,
    )
    messages = [entry["message"] for entry in result["logs"]]
    assert messages == ["strategy info", "strategy error", "strategy error"]


def test_jesse_submission_and_strategy_validations():
    class TooMuchMargin(Strategy):
        def should_long(self):
            return self.index == 0

        def go_long(self):
            self.buy = 10_001 / self.price, self.price

    with pytest.raises(InsufficientMargin):
        single_route_backtest(
            "TooMuchMargin", leverage=1,
            strategy_classes={"TooMuchMargin": TooMuchMargin})
    single_route_backtest(
        "TooMuchMargin", leverage=2,
        strategy_classes={"TooMuchMargin": TooMuchMargin})

    class OversizedSpotExit(Strategy):
        def should_long(self):
            return self.price == 10

        def go_long(self):
            self.buy = 1, self.price

        def on_open_position(self, order):
            self.take_profit = 1.01, 12

    with pytest.raises(InsufficientBalance):
        single_route_backtest(
            "OversizedSpotExit", is_futures_trading=False,
            strategy_classes={"OversizedSpotExit": OversizedSpotExit})

    class InvalidOrders(Strategy):
        def should_long(self):
            return self.index == 0

        def go_long(self):
            self.buy = 1, 0

    with pytest.raises(InvalidStrategy, match="greater than zero"):
        single_route_backtest(
            "InvalidOrders", strategy_classes={"InvalidOrders": InvalidOrders})

    class ConflictingExits(Strategy):
        def should_long(self):
            return self.index == 0

        def go_long(self):
            self.buy = 1, 2

        def update_position(self):
            if self.index == 5:
                self.stop_loss = 1, 3
                self.take_profit = 1, 3

    with pytest.raises(InvalidStrategy, match="stop-loss and take-profit"):
        single_route_backtest(
            "ConflictingExits",
            strategy_classes={"ConflictingExits": ConflictingExits})

    class InvalidChartValue(Strategy):
        def should_long(self):
            return False

        def go_long(self):
            pass

        def after(self):
            self.add_line_to_candle_chart("bad", [1, 2])

    with pytest.raises(ValueError, match="Invalid value type"):
        single_route_backtest(
            "InvalidChartValue",
            strategy_classes={"InvalidChartValue": InvalidChartValue})


def test_jesse_helper_enum_and_exception_surface():
    actual_helpers = {
        name for name, value in vars(jh).items()
        if not name.startswith("_") and callable(value)
        and getattr(value, "__module__", None) == jh.__name__
    }
    assert actual_helpers == JESSE_PUBLIC_HELPERS

    keyword_sensitive_signatures = {
        "date_diff_in_days": ["date1", "date2"],
        "date_to_timestamp": ["date"],
        "normalize_bool": ["v"],
        "opposite_side": ["s"],
        "opposite_type": ["t"],
        "orderbook_trim_price": ["p", "ascending", "unit"],
        "round_or_none": ["x", "digits"],
        "round_price_for_live_mode": ["price", "precision"],
        "side_to_type": ["s"],
        "type_to_side": ["t"],
    }
    for name, parameters in keyword_sensitive_signatures.items():
        assert list(inspect.signature(getattr(jh, name)).parameters) == parameters

    start = jh.get_arrow(jh.date_to_timestamp("2024-01-01"))
    finish = jh.get_arrow(jh.date_to_timestamp("2024-01-04"))
    assert jh.date_diff_in_days(start, finish) == 3
    with pytest.raises(TypeError, match="Arrow instances"):
        jh.date_diff_in_days(0, 1)
    assert jh.estimate_PNL(2, 100, 110, "long", .001) == pytest.approx(19.58)
    assert jh.orderbook_insertion_index_search(
        [[10, 1], [20, 1]], [15, 1]) == (False, 1)
    assert jh.has_live_trade_plugin() is False

    upstream_enum_names = {
        "colors", "exchanges", "live_session_modes", "live_session_statuses",
        "migration_actions", "order_statuses", "order_submitted_via",
        "order_types", "sides", "timeframes", "trade_types",
    }
    assert upstream_enum_names <= set(vars(jesse_enums))
    assert jesse_enums.order_types.FOK == "FOK"
    assert jesse_enums.order_statuses.LIQUIDATED == "LIQUIDATED"
    assert jesse_enums.order_submitted_via.STOP_LOSS == "stop-loss"

    upstream_exception_names = {
        "CandleNotFoundInDatabase", "CandleNotFoundInExchange",
        "CandlesNotFound", "ConflictingRules", "EmptyPosition",
        "ExchangeError", "ExchangeInMaintenance", "ExchangeNotResponding",
        "ExchangeOrderNotFound", "ExchangeRejectedLeverageNumber",
        "ExchangeRejectedOrder", "InsufficientBalance", "InsufficientMargin",
        "InvalidConfig", "InvalidDateRange", "InvalidExchangeApiKeys",
        "InvalidRoutes", "InvalidShape", "InvalidStrategy", "InvalidSymbol",
        "InvalidTimeframe", "NegativeBalance", "NotSupportedError",
        "OpenPositionError", "OrderNotAllowed", "RouteNotFound",
        "SymbolNotFound", "Termination",
    }
    assert upstream_exception_names <= set(vars(jesse_exceptions))
