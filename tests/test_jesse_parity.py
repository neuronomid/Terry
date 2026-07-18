"""Regression coverage for Terry APIs added to match Jesse 2.5's public surface."""

from __future__ import annotations

import inspect
import importlib
import asyncio
import json
import math
import threading
import time
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from terry import utils
from terry.candle_pipelines import (
    GaussianNoiseCandlesPipeline,
    GaussianResamplerCandlesPipeline,
    MovingBlockBootstrapCandlesPipeline,
)
from terry.data.binance import EXCHANGES, fetch_1m_chunk
from terry.data.storage import CandleDB
from terry.factories import candles_from_close_prices
from terry.mcp.resources import _RESOURCES
from terry.context import TerryContext
from terry.context import set_context
from terry.data.importer import Importer
from terry.enums import order_statuses
from terry.models import Order
from terry.loader import load_strategy_from_source
from terry.mcp.server import build_server
from terry.mcp.tools import _common as session_tools
from terry.research.ml import load_ml_model, train_model
from terry.research.monte_carlo import monte_carlo_candles, monte_carlo_trades
from terry.research.optimize import _score, optimize, print_optimize_summary
from terry.research.significance import plot_significance_test, rule_significance_test
from terry.strategy import Strategy, cached
import terry.testing_utils as testing_utils
from terry.testing_utils import single_route_backtest


def test_jesse_utility_compatibility():
    above = utils.crossed([1, 3, 1, 4], 2, direction="above", sequential=True)
    assert above.tolist() == [False, True, False, True]
    signal = utils.signal_line([0, 1, 1, 0], period=2)
    assert np.isnan(signal[0]) and signal[1:].tolist() == [.5, 1, .5]
    streak = utils.streaks(np.array([0, 1, 2, 1, 2]))
    assert np.isnan(streak[0]) and streak[1:].tolist() == [1, 2, -1, 1]
    assert utils.strictly_increasing(np.array([1, 2, 3]), 3) is True
    assert utils.strictly_decreasing(np.array([3, 2, 1]), 3) is True
    assert utils.sum_floats(0.1, 0.2) == 0.3
    assert utils.subtract_floats(0.3, 0.1) == 0.2
    # Jesse reserves three fees and floors rather than rounding quantity precision.
    assert utils.size_to_qty(1000, 100, precision=3, fee_rate=.001) == 9.97


def test_cached_decorator_caches_for_one_strategy_cycle():
    class CachedStrategy(Strategy):
        calls = 0

        @cached
        def value(self, multiplier=1):
            self.calls += 1
            return self.calls * multiplier

        def should_long(self):
            return False

        def go_long(self):
            pass

    strategy = CachedStrategy()
    assert strategy.value(2) == strategy.value(2) == 2
    assert strategy.calls == 1
    strategy._cache = {}
    assert strategy.value(2) == 4


def test_unchanged_jesse_strategy_imports_run_with_terry_testing_helper():
    assert list(inspect.signature(
        testing_utils.get_btc_and_eth_candles).parameters) == []
    assert list(inspect.signature(testing_utils.single_route_backtest).parameters)[:8] == [
        "strategy_name", "is_futures_trading", "leverage", "leverage_mode",
        "trend", "fee", "candles_count", "timeframe",
    ]
    source = '''from jesse.strategies import Strategy
import jesse.indicators as ta
from jesse import utils

class JesseSourceStrategy(Strategy):
    def should_long(self):
        return self.price == 10 and ta.sma(self.candles, 2) > 0
    def should_short(self):
        return False
    def go_long(self):
        self.buy = utils.size_to_qty(10, self.price), self.price
    def on_open_position(self, order):
        self.take_profit = self.position.qty, 12
    def on_close_position(self, order, closed_trade):
        assert closed_trade.entry_price == 10
        assert closed_trade.exit_price == 12
'''
    strategy_class = load_strategy_from_source("JesseSourceStrategy", source)
    result = single_route_backtest(
        "JesseSourceStrategy",
        strategy_classes={"JesseSourceStrategy": strategy_class},
    )
    assert result["metrics"]["total"] == 1
    assert result["trades"][0]["entry_price"] == 10
    assert result["trades"][0]["exit_price"] == 12


def test_importer_can_retry_with_the_same_jesse_import_id(tmp_path, monkeypatch):
    module = importlib.import_module("terry.data.importer")

    def fake_fetch(_exchange, _symbol, start_ts, finish_ts, **_kwargs):
        return np.array([[max(start_ts, finish_ts - 60_000), 1, 1, 1, 1, 1]],
                        dtype=float)

    monkeypatch.setattr(module, "fetch_1m_range", fake_fetch)
    importer = Importer(CandleDB(tmp_path / "candles.db"))
    identifier = "same-import-id"
    assert importer.start_import(
        "Binance Spot", "BTC-USDT", "2024-01-01", "2024-01-02",
        import_id=identifier) == identifier
    deadline = time.monotonic() + 2
    while importer.get_status(identifier)["status"] not in {"finished", "error"}:
        assert time.monotonic() < deadline
        time.sleep(.005)
    assert importer.get_status(identifier)["status"] == "finished"
    assert importer.start_import(
        "Binance Spot", "BTC-USDT", "2024-01-01", "2024-01-02",
        import_id=identifier) == identifier
    deadline = time.monotonic() + 2
    while importer.get_status(identifier)["status"] not in {"finished", "error"}:
        assert time.monotonic() < deadline
        time.sleep(.005)
    assert importer.get_status(identifier)["status"] == "finished"


def test_mcp_session_envelopes_notes_snapshots_and_filters(tmp_path):
    strategy_path = tmp_path / "strategies" / "EnvelopeStrategy" / "__init__.py"
    strategy_path.parent.mkdir(parents=True)
    strategy_path.write_text(
        "from jesse.strategies import Strategy\n"
        "class EnvelopeStrategy(Strategy):\n"
        "    def should_long(self): return False\n"
        "    def go_long(self): pass\n",
        encoding="utf-8",
    )
    ctx = set_context(TerryContext(str(tmp_path)))
    state = {
        "strategy": "EnvelopeStrategy", "exchange": "Binance Spot",
        "symbol": "BTC-USDT", "timeframe": "1h",
        "start_date": "2024-01-01", "finish_date": "2024-02-01",
    }
    draft = session_tools.create_draft(
        "backtest", state, title="Envelope audit", description="Original note")
    assert draft["status"] == "success"
    assert draft["session_status"] == "draft"
    assert draft["backtest_id"] == draft["session_id"]
    assert draft["draft_state"]["form"]["routes"][0]["strategy"] == "EnvelopeStrategy"
    assert draft["draft_state"]["results"]["selectedRoute"]["symbol"] == "BTC-USDT"
    assert draft["draft_state"]["results"]["logsModal"] is False
    assert draft["notes"]["title"] == "MCP Backtest: Envelope audit"
    assert draft["notes"]["strategy_codes_captured"] == 1
    assert draft["dashboard_url"].endswith(f'{draft["session_id"]}.html')

    significance_draft = session_tools.create_draft(
        "significance_test", state, title="Signal audit")
    assert significance_draft["draft_state"]["form"]["id"] == (
        significance_draft["session_id"])
    assert significance_draft["draft_state"]["results"] == {
        "alert": {"message": "", "type": ""}}
    significance_session = session_tools.get_session(
        significance_draft["session_id"])
    assert significance_session["data"]["session"]["state"]["form"]["id"] == (
        significance_draft["session_id"])

    updated = session_tools.update_notes(
        draft["session_id"], title="Filtered title", description="Conclusion",
        strategy_codes=json.dumps({"Binance Spot-BTC-USDT": "snapshot"}))
    assert updated["status"] == "success"
    assert updated["strategy_code_keys"] == ["Binance Spot-BTC-USDT"]
    listed = session_tools.list_sessions(
        "backtest", limit=10, offset=0, title_search="filtered",
        status_filter="draft")
    assert listed["status"] == "success" and listed["count"] == 1
    session = session_tools.get_session(draft["session_id"])
    assert session["error"] is None
    assert session["data"]["session"]["state"]["form"]["routes"][0]["symbol"] == (
        "BTC-USDT")
    assert ctx.sessions.get(draft["session_id"])["notes_metadata"]["description"] == "Conclusion"


def test_mcp_schemas_and_basic_results_follow_jesse_contract(tmp_path):
    async def exercise():
        mcp = build_server(project_root=str(tmp_path))
        tools = await mcp.list_tools()
        schemas = {tool.name: list(tool.inputSchema["properties"])
                   for tool in tools}
        assert schemas["create_backtest_draft"][:17] == [
            "exchange", "routes", "data_routes", "start_date", "finish_date",
            "debug_mode", "export_csv", "export_json", "export_chart",
            "export_tradingview", "fast_mode", "benchmark", "title",
            "description", "strategy_summary", "change_summary", "rationale",
        ]
        assert schemas["create_optimization_draft"][:19] == [
            "exchange", "routes", "data_routes", "training_start_date",
            "training_finish_date", "testing_start_date", "testing_finish_date",
            "optimal_total", "objective_function", "trials",
            "best_candidates_count", "warm_up_candles", "fast_mode", "cpu_cores",
            "title", "description", "strategy_summary", "hypothesis", "rationale",
        ]
        for name in (
                "get_backtest_sessions", "get_significance_test_sessions",
                "get_monte_carlo_sessions", "get_optimization_sessions"):
            assert schemas[name] == [
                "limit", "offset", "title_search", "status_filter", "date_filter"]
        assert schemas["import_candles"][:4] == [
            "exchange", "symbol", "start_date", "import_id"]
        for name in (
                "update_backtest_notes", "update_significance_test_notes",
                "update_monte_carlo_notes", "update_optimization_notes"):
            assert schemas[name][:4] == [
                "session_id", "title", "description", "strategy_codes"]

        def decode(blocks):
            return json.loads(blocks[0].text)

        greeting = decode(await mcp.call_tool("greet_user", {"name": "Ada"}))
        assert greeting == {
            **greeting, "status": "success", "action": "greeting", "user_name": "Ada"}
        config = decode(await mcp.call_tool("get_config", {}))
        assert config["status"] == "success" and config["config"]["fee"] == config["fee"]
        indicators = decode(await mcp.call_tool("list_indicators", {}))
        assert indicators["status"] == "success" and indicators["count"] == 174
        details = decode(await mcp.call_tool(
            "get_indicator_details", {"indicator_name": "sma"}))
        assert details["status"] == "success"
        assert isinstance(details["parameters"], dict)

    asyncio.run(exercise())


def test_candle_pipelines_preserve_valid_ohlc():
    source = candles_from_close_prices(np.linspace(100, 130, 120).tolist())
    pipelines = [
        GaussianNoiseCandlesPipeline(
            30, close_sigma=.05, high_sigma=.01, low_sigma=.01, seed=3),
        GaussianResamplerCandlesPipeline(30, seed=3),
        MovingBlockBootstrapCandlesPipeline(30, seed=3),
    ]
    for pipeline in pipelines:
        output = pipeline.transform(source)
        assert output.shape == source.shape
        assert np.array_equal(output[:, 0], source[:, 0])
        assert np.all(output[:, 3] >= np.maximum(output[:, 1], output[:, 2]))
        assert np.all(output[:, 4] <= np.minimum(output[:, 1], output[:, 2]))
        assert np.all(output[:, 4] > 0)


def test_research_candle_factories_match_jesse_stateful_semantics():
    from terry.research import fake_candle

    first = fake_candle(reset=True)
    second = fake_candle()
    assert second[0] - first[0] == 60_000
    assert second[1] == first[2]
    overridden = fake_candle({"timestamp": 7, "close": 12, "volume": 4})
    assert overridden[[0, 2, 5]].tolist() == [7, 12, 4]
    candles = candles_from_close_prices([10, 11])
    assert candles[0, 1] == 9.5 and candles[1, 1] == 10
    assert candles[1, 0] - candles[0, 0] == 60_000


class _Response:
    status_code = 200
    reason = "OK"
    text = ""

    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


class _Session:
    def __init__(self, payload):
        self.payload = payload

    def get(self, *_args, **_kwargs):
        return _Response(self.payload)


@pytest.mark.parametrize(("exchange", "symbol", "payload"), [
    ("Binance Spot", "BTC-USDT", [[1000, "1", "3", ".5", "2", "4"]]),
    ("Bybit USDT Perpetual", "BTC-USDT",
     {"retMsg": "OK", "result": {"list": [["1000", "1", "3", ".5", "2", "4"]]}}),
    ("Coinbase Spot", "BTC-USD", {"candles": [{
        "start": "1", "open": "1", "close": "2", "high": "3", "low": ".5",
        "volume": "4"}]}),
    ("Bitfinex Spot", "BTC-USD", [[1000, 1, 2, 3, .5, 4]]),
    ("Gate USDT Perpetual", "BTC-USDT",
     [{"t": "1", "o": "1", "c": "2", "h": "3", "l": ".5", "v": "4"}]),
    ("Kraken Pro Futures", "BTC-USD", {"candles": [{
        "time": 1000, "open": "1", "close": "2", "high": "3", "low": ".5",
        "volume": "4"}]}),
])
def test_public_exchange_payloads_normalize_to_jesse_candle_shape(exchange, symbol, payload):
    row = fetch_1m_chunk(exchange, symbol, 1000, session=_Session(payload))[0]
    assert row == [1000, 1.0, 2.0, 3.0, .5, 4.0]


def test_supported_backtest_exchange_set_matches_jesse_public_drivers():
    expected = {
        "Binance Spot", "Binance US Spot", "Binance Perpetual Futures",
        "Bitfinex Spot", "Coinbase Spot", "Bybit USDT Perpetual",
        "Bybit USDC Perpetual", "Bybit Spot", "Gate USDT Perpetual",
        "Kraken Pro Futures",
    }
    assert expected.issubset(EXCHANGES)


def test_order_surface_and_sqlite_dedup_match_jesse_contract(tmp_path):
    order = Order({"side": "sell", "qty": -3, "filled_qty": -1,
                   "price": 10, "status": order_statuses.PARTIALLY_FILLED})
    assert order.is_partially_filled and order.is_cancellable
    assert order.remaining_qty == -2
    assert isinstance(order.to_dict, dict) and order.to_dict()["filled_qty"] == -1

    database = CandleDB(tmp_path / "candles.db")
    rows = candles_from_close_prices([1, 2, 3])
    assert database.store("B", "BTC-USDT", rows) == 3
    assert database.store("B", "BTC-USDT", rows) == 0
    assert database.coverage("B", "BTC-USDT")["count"] == 3


def test_runner_executes_multi_route_and_data_route_state(tmp_path):
    strategy_source = '''from terry.strategies import Strategy
from terry import utils

class MultiTrade(Strategy):
    def should_long(self): return self.index == 1
    def should_short(self): return False
    def go_long(self): self.buy = utils.size_to_qty(self.available_margin * .1, self.price), self.price
    def update_position(self):
        if self.index >= 5: self.liquidate()
'''
    path = tmp_path / "strategies" / "MultiTrade" / "__init__.py"
    path.parent.mkdir(parents=True)
    path.write_text(strategy_source, encoding="utf-8")
    context = TerryContext(str(tmp_path))
    start = 1_704_067_200_000  # 2024-01-01 UTC
    for index, symbol in enumerate(("BTC-USDT", "ETH-USDT", "SOL-USDT")):
        prices = np.linspace(100 + index * 20, 110 + index * 20, 1_440)
        rows = candles_from_close_prices(prices.tolist())
        rows[:, 0] += start - rows[0, 0]
        context.candle_db.store("Binance Perpetual Futures", symbol, rows)
    result = context.runner._run_backtest("parity-test", {
        "strategy": "MultiTrade", "symbol": "BTC-USDT", "timeframe": "1m",
        "exchange": "Binance Perpetual Futures", "start_date": "2024-01-01",
        "finish_date": "2024-01-02", "config": {"warm_up_candles": 0},
        "routes": [
            {"exchange": "Binance Perpetual Futures", "strategy": "MultiTrade",
             "symbol": "BTC-USDT", "timeframe": "1m"},
            {"exchange": "Binance Perpetual Futures", "strategy": "MultiTrade",
             "symbol": "ETH-USDT", "timeframe": "1m"},
        ],
        "data_routes": [{"exchange": "Binance Perpetual Futures",
                         "symbol": "SOL-USDT", "timeframe": "5m"}],
        "export_chart": False, "benchmark": True,
    })
    assert result["metrics"]["total"] == 2
    assert result["benchmark"]["return_percentage"] == pytest.approx(10)


def test_ml_training_artifacts_round_trip(tmp_path):
    points = [{
        "time": 1_700_000_000 + index,
        "features": {"momentum": float(index % 5), "volume": float(index)},
        "label": {"name": "up", "value": index % 2 == 0},
    } for index in range(40)]
    result = train_model(
        points, LogisticRegression(), save_to=str(tmp_path), verbose=False)
    loaded = load_ml_model(str(tmp_path))
    assert result["train_test_info"] == {
        **result["train_test_info"], "train_size": 32, "test_size": 8,
    }
    assert type(loaded["model"]).__name__ == "LogisticRegression"
    assert loaded["scaler"].n_features_in_ == 2
    assert {"rfe_ranking", "anova_f_values", "correlations", "cv_baseline",
            "cv_impacts", "cv_scores_without_feature", "consensus_ranks"}.issubset(
        result["feature_importance"])
    assert {item["feature"] for item in result["feature_impact"]} == {
        "momentum", "volume",
    }
    assert all("diff" in bucket for bucket in result["calibration"])


def test_optuna_optimizer_returns_jesse_and_terry_result_shapes():
    class Tunable(Strategy):
        def hyperparameters(self):
            return [{"name": "period", "type": int, "min": 2, "max": 4,
                     "default": 3}]

        def should_long(self):
            return self.index % self.hp["period"] == 0

        def should_short(self):
            return False

        def go_long(self):
            self.buy = utils.size_to_qty(self.available_margin * .1, self.price), self.price

        def update_position(self):
            if self.index % self.hp["period"] == 1:
                self.liquidate()

    config = {"starting_balance": 10_000, "fee": 0, "type": "futures",
              "futures_leverage": 1, "exchange": "B", "warm_up_candles": 0}
    route = [{"exchange": "B", "symbol": "BTC-USDT", "timeframe": "1m",
              "strategy": "Tunable"}]
    candles = candles_from_close_prices(
        np.linspace(100, 120, 60 * 24 * 4).tolist())
    dataset = {"B-BTC-USDT": {
        "exchange": "B", "symbol": "BTC-USDT", "candles": candles,
    }}
    result = optimize(
        config, route, [], candles=dataset, n_trials=3, best_candidates_count=2,
        min_trades=1, strategy_classes={"Tunable": Tunable}, progress_bar=False)
    assert result["total_trials"] == 3
    assert result["completed_trials"] == 3
    assert result["best_trials"] == result["candidates"]
    assert result["best"] in result["best_trials"]
    assert {"training_metrics", "testing_metrics", "params", "dna"}.issubset(
        result["best"])


def test_optimizer_uses_jesse_fitness_normalization_and_cpu_workers(monkeypatch):
    metrics = {
        "total": 10, "sharpe_ratio": 1.0, "win_rate": .5,
        "net_profit_percentage": 3.0, "calmar_ratio": .8,
        "max_drawdown": -2.0, "sortino_ratio": 1.1, "omega_ratio": 1.2,
    }
    expected = (math.log10(10) / math.log10(200)) * (1.5 / 5.5)
    assert _score(metrics, "sharpe_ratio", 5, optimal_total=200,
                  objective_function="sharpe") == pytest.approx(expected)
    assert _score({**metrics, "total": 5}, "sharpe_ratio", 5,
                  objective_function="sharpe") == 0.0001
    assert _score({**metrics, "sharpe_ratio": -0.1}, "sharpe_ratio", 5,
                  objective_function="sharpe") == 0.0001

    class Tunable(Strategy):
        def hyperparameters(self):
            return [{"name": "period", "type": int, "min": 2, "max": 3,
                     "default": 2}]

        def should_long(self):
            return False

        def should_short(self):
            return False

        def go_long(self):
            pass

    barrier = threading.Barrier(2, timeout=3)
    lock = threading.Lock()
    active = 0
    maximum = 0

    def fake_backtest(*_args, **_kwargs):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        barrier.wait()
        with lock:
            active -= 1
        return {"metrics": metrics}

    module = importlib.import_module("terry.research.optimize")
    monkeypatch.setattr(module, "backtest", fake_backtest)
    dataset = {"B-BTC-USDT": {
        "exchange": "B", "symbol": "BTC-USDT", "candles": np.zeros((4, 6)),
    }}
    result = optimize(
        {"starting_balance": 10_000, "fee": 0, "type": "futures",
         "futures_leverage": 1, "exchange": "B", "warm_up_candles": 0},
        [{"exchange": "B", "symbol": "BTC-USDT", "timeframe": "1m",
          "strategy": "Tunable"}], [], training_candles=dataset,
        testing_candles=dataset, n_trials=2, cpu_cores=2, min_trades=5,
        best_candidates_count=2, strategy_classes={"Tunable": Tunable},
        progress_bar=False,
    )
    assert maximum == 2
    assert result["cpu_cores"] == 2
    assert result["completed_trials"] == 2


def test_mcp_resource_surface_includes_optimization():
    assert len(_RESOURCES) == 12
    assert "terry://optimization" in _RESOURCES


def test_research_signatures_match_jesse_2_5_leading_parameters():
    assert list(inspect.signature(monte_carlo_candles).parameters)[:14] == [
        "config", "routes", "data_routes", "candles", "warmup_candles",
        "hyperparameters", "fast_mode", "num_scenarios", "progress_bar",
        "candles_pipeline_class", "candles_pipeline_kwargs", "cpu_cores",
        "progress_callback", "result_callback",
    ]
    assert list(inspect.signature(monte_carlo_trades).parameters)[:13] == [
        "config", "routes", "data_routes", "candles", "warmup_candles",
        "benchmark", "hyperparameters", "fast_mode", "num_scenarios",
        "progress_bar", "cpu_cores", "progress_callback", "result_callback",
    ]
    assert list(inspect.signature(rule_significance_test).parameters)[:11] == [
        "config", "routes", "data_routes", "candles", "warmup_candles",
        "hyperparameters", "n_simulations", "random_seed", "progress_bar",
        "cpu_cores", "progress_callback",
    ]
    assert list(inspect.signature(print_optimize_summary).parameters) == [
        "result", "show_params",
    ]


def test_monte_carlo_and_significance_return_jesse_research_shapes(tmp_path):
    class ResearchStrategy(Strategy):
        def should_long(self):
            return self.index % 20 == 1

        def should_short(self):
            return False

        def go_long(self):
            self.buy = utils.size_to_qty(self.available_margin * .1, self.price), self.price

        def update_position(self):
            if self.index % 20 == 10:
                self.liquidate()

    config = {"starting_balance": 10_000, "fee": 0, "type": "futures",
              "futures_leverage": 1, "exchange": "B", "warm_up_candles": 0}
    routes = [{"exchange": "B", "symbol": "BTC-USDT", "timeframe": "1m",
               "strategy": "ResearchStrategy"}]
    source = candles_from_close_prices(
        (100 + np.sin(np.arange(600) / 10) + np.arange(600) * .01).tolist())
    candles = {"B-BTC-USDT": {
        "exchange": "B", "symbol": "BTC-USDT", "candles": source,
    }}
    progress = []
    streamed = []
    candle_result = monte_carlo_candles(
        config, routes, [], candles, None, None, True, 4, False, None, None,
        None, lambda done: progress.append(done), streamed.append,
        strategy_classes={"ResearchStrategy": ResearchStrategy},
    )
    assert candle_result["total_requested"] == 4
    assert candle_result["num_scenarios"] == 3
    assert candle_result["original"]["scenario_index"] == 0
    assert len(candle_result["scenarios"]) == 3
    assert len(streamed) == 4 and progress[-1] == 4
    assert {"summary", "metrics", "interpretation"}.issubset(
        candle_result["confidence_analysis"])
    assert "summary_metrics" in candle_result and "overfit_verdict" in candle_result

    trade_result = monte_carlo_trades(
        config, routes, [], candles, None, False, None, True, 4, False, None,
        None, None, strategy_classes={"ResearchStrategy": ResearchStrategy},
    )
    assert trade_result["total_requested"] == trade_result["num_scenarios"] == 4
    assert len(trade_result["scenarios"]) == 4
    assert trade_result["original"]["trades"]
    assert {"summary", "metrics", "interpretation"}.issubset(
        trade_result["confidence_analysis"])
    assert "max_drawdown" in trade_result

    significance = rule_significance_test(
        config, routes, [], candles, None, None, 200, 7, False, None, None,
        strategy_classes={"ResearchStrategy": ResearchStrategy},
    )
    assert isinstance(significance["simulated_means"], np.ndarray)
    assert len(significance["simulated_means"]) == 200
    chart_path = plot_significance_test(significance, str(tmp_path), theme="dark")
    assert chart_path.endswith(".png")
    assert (tmp_path / chart_path.rsplit("/", 1)[-1]).is_file()
