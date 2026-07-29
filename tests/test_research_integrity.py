"""Statistical-integrity coverage for the Optimization and Rule Significance Test paths.

These guard the two properties the research surface exists to provide: a rule is only
ever scored on bars it could actually have traded, and out-of-sample validation runs
under the same conditions as the training window it is compared against.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from terry.dashboard import app as dashboard_app
from terry.dashboard.app import create_app
from terry.factories import candles_from_close_prices
from terry.mcp.tools import optimization as optimization_tools
from terry.research import OBJECTIVE_FUNCTIONS
from terry.research.optimize import _OBJECTIVES, _split_candles, optimize
from terry.research.significance import rule_significance_test
from terry.strategy import Strategy


CONFIG = {"starting_balance": 10_000, "fee": 0, "type": "futures",
          "futures_leverage": 1, "exchange": "B", "warm_up_candles": 0}
ROUTE = [{"exchange": "B", "symbol": "BTC-USDT", "timeframe": "1m",
          "strategy": "LongAfterUpBar"}]


class LongAfterUpBar(Strategy):
    """Long whenever the bar that just closed was an up bar."""

    def should_long(self):
        return self.close > self.open

    def should_short(self):
        return False

    def go_long(self):
        pass


def _significance(prices, **kwargs):
    dataset = {"B-BTC-USDT": {
        "exchange": "B", "symbol": "BTC-USDT",
        "candles": candles_from_close_prices(list(prices)),
    }}
    return rule_significance_test(
        CONFIG, ROUTE, [], dataset, n_simulations=2_000, random_seed=1,
        strategy_classes={"LongAfterUpBar": LongAfterUpBar}, **kwargs)


def test_significance_scores_a_signal_against_the_bar_that_follows_it():
    """On a strictly alternating series every up bar is followed by a down bar, so a
    rule that buys after an up bar must score negatively. Pairing each signal with the
    return that preceded it instead would flip the sign and manufacture an edge."""
    result = _significance([100, 101] * 100)

    assert result["observed_mean"] < 0
    assert result["p_value"] > 0.5
    assert result["verdict"] == "not_significant"
    assert result["n_observations"] == 199


def test_significance_rejects_a_hindsight_rule_on_an_unpredictable_series():
    """A random walk gives "buy after an up bar" no predictive power whatsoever."""
    rng = np.random.default_rng(7)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, 3_000)))

    assert _significance(prices)["p_value"] > 0.05


def test_significance_still_detects_a_genuine_edge():
    """The same rule on an autocorrelated series does predict the next bar."""
    rng = np.random.default_rng(11)
    shocks = rng.normal(0, 0.002, 3_000)
    returns = np.empty_like(shocks)
    returns[0] = shocks[0]
    for index in range(1, len(shocks)):
        returns[index] = 0.7 * returns[index - 1] + 0.3 * shocks[index]

    result = _significance(100 * np.exp(np.cumsum(returns)))
    assert result["observed_mean"] > 0
    assert result["p_value"] < 0.05
    assert result["verdict"] == "significant"


def test_significance_annualizes_over_bars_per_year_not_days_per_year():
    """`observed_mean` is per bar, so a 4h route compounds 2,190 bars a year — not the
    365 a flat days-per-year constant assumes, which understates it 6x."""
    from terry.research.significance import _bars_per_year

    assert _bars_per_year("1D") == 365
    assert _bars_per_year("4h") == 2_190
    assert _bars_per_year("1h") == 8_760
    assert _bars_per_year("1m") == 525_600

    result = _significance([100, 101] * 100)
    assert result["bars_per_year"] == 525_600  # the fixture runs on a 1m route
    assert result["annualized_return"] == pytest.approx(
        result["observed_mean"] * 525_600)


def _dataset(rows=1_000):
    return {"B-BTC-USDT": {
        "exchange": "B", "symbol": "BTC-USDT",
        "candles": np.arange(rows * 6, dtype=float).reshape(rows, 6),
    }}


def test_split_warms_up_the_testing_window_from_explicit_warmup_candles():
    """Terry's session runner resolves warm-up into candles and zeroes the config key.
    The testing window must still start warm, carved from the tail of training."""
    warmup = {"B-BTC-USDT": {"exchange": "B", "symbol": "BTC-USDT",
                             "candles": np.zeros((120, 6))}}

    _, testing, testing_warmup = _split_candles(
        _dataset(), 0.75, ROUTE, {"warm_up_candles": 0}, warmup)

    assert testing_warmup is not None
    assert len(testing_warmup["B-BTC-USDT"]["candles"]) == 120
    # It is the 120 rows immediately preceding the testing window — no leakage.
    assert testing_warmup["B-BTC-USDT"]["candles"][-1][0] == 749 * 6
    assert testing["B-BTC-USDT"]["candles"][0][0] == 750 * 6


def test_split_warmup_falls_back_to_the_config_and_never_exceeds_training():
    _, _, from_config = _split_candles(
        _dataset(), 0.75, ROUTE, {"warm_up_candles": 100})
    assert len(from_config["B-BTC-USDT"]["candles"]) == 100

    huge = {"B-BTC-USDT": {"exchange": "B", "symbol": "BTC-USDT",
                           "candles": np.zeros((5_000, 6))}}
    _, _, clamped = _split_candles(
        _dataset(), 0.75, ROUTE, {"warm_up_candles": 0}, huge)
    assert len(clamped["B-BTC-USDT"]["candles"]) == 750

    _, _, none_configured = _split_candles(
        _dataset(), 0.75, ROUTE, {"warm_up_candles": 0})
    assert none_configured is None


def test_optimizer_hands_the_testing_backtest_the_same_warmup_as_training(monkeypatch):
    """End to end: both windows of every trial must run under warmed-up indicators."""
    class Tunable(Strategy):
        def hyperparameters(self):
            return [{"name": "period", "type": int, "min": 2, "max": 4, "default": 3}]

        def should_long(self):
            return False

        def should_short(self):
            return False

        def go_long(self):
            pass

    seen = []

    def fake_backtest(_config, _routes, _data_routes, candles, warmup_candles=None,
                      **_kwargs):
        seen.append((len(candles["B-BTC-USDT"]["candles"]),
                     None if warmup_candles is None
                     else len(warmup_candles["B-BTC-USDT"]["candles"])))
        return {"metrics": {"total": 50, "sharpe_ratio": 2.0}}

    # `terry.research.optimize` resolves to the re-exported function, so reach the
    # module itself before patching the backtest it calls.
    monkeypatch.setattr(
        importlib.import_module("terry.research.optimize"), "backtest", fake_backtest)
    warmup = {"B-BTC-USDT": {"exchange": "B", "symbol": "BTC-USDT",
                             "candles": np.zeros((120, 6))}}

    optimize({**CONFIG, "warm_up_candles": 0},
             [{**ROUTE[0], "strategy": "Tunable"}], [],
             candles=_dataset(), warmup_candles=warmup, n_trials=1,
             strategy_classes={"Tunable": Tunable}, progress_bar=False)

    training, testing = seen
    assert training == (750, 120)
    assert testing == (250, 120)


def test_every_layer_validates_optimization_objectives_against_the_engine():
    """A name accepted by the optimizer must not be rejected by the MCP or HTTP edge."""
    assert OBJECTIVE_FUNCTIONS == frozenset(_OBJECTIVES)
    assert dashboard_app._OBJECTIVES == OBJECTIVE_FUNCTIONS
    assert {"calmar_ratio", "sortino_ratio", "omega_ratio", "serenity_index",
            "smart_sharpe", "net_profit_percentage"} <= OBJECTIVE_FUNCTIONS


def test_mcp_optimization_draft_accepts_every_engine_objective(tmp_path: Path,
                                                               monkeypatch):
    from terry.context import TerryContext, set_context

    set_context(TerryContext(str(tmp_path)))
    (tmp_path / "strategies" / "Tuned").mkdir(parents=True)
    (tmp_path / "strategies" / "Tuned" / "__init__.py").write_text(
        "from terry.strategies import Strategy\n\n"
        "class Tuned(Strategy):\n"
        "    def should_long(self): return False\n"
        "    def should_short(self): return False\n"
        "    def go_long(self): pass\n")

    registered = {}

    class _Recorder:
        def tool(self, *_args, **_kwargs):
            def decorate(function):
                registered[function.__name__] = function
                return function
            return decorate

    optimization_tools.register_optimization_tools(_Recorder())
    create_draft = registered["create_optimization_draft"]

    for objective in sorted(OBJECTIVE_FUNCTIONS):
        result = create_draft(strategy="Tuned", objective_function=objective)
        assert result.get("status") == "success", (objective, result)
    assert create_draft(strategy="Tuned",
                        objective_function="profit")["error"] == "invalid_config"


def test_dashboard_research_defaults_follow_the_saved_config(tmp_path: Path):
    """`get_optimization_config` advertises these defaults, so omitting a field in a
    session request must actually use them."""
    client = TestClient(create_app(str(tmp_path)))
    client.post("/api/strategies", json={
        "name": "Tuned",
        "content": ("from terry.strategies import Strategy\n\n"
                    "class Tuned(Strategy):\n"
                    "    def should_long(self): return False\n"
                    "    def should_short(self): return False\n"
                    "    def go_long(self): pass\n"),
    })
    updated = client.patch("/api/config", json={
        "optimization": {"objective": "calmar_ratio", "n_trials": 7,
                         "train_test_split": 0.6},
        "significance_test": {"n_simulations": 5_000},
        "monte_carlo": {"num_scenarios": 25},
    })
    assert updated.status_code == 200

    common = {"strategy": "Tuned", "symbol": "BTC-USDT", "timeframe": "1h",
              "start_date": "2024-01-01", "finish_date": "2024-01-10", "start": False}
    optimization = client.post("/api/sessions/optimization", json=dict(common))
    assert optimization.status_code == 200
    state = optimization.json()["state"]
    assert (state["objective"], state["n_trials"], state["train_test_split"]) == (
        "calmar_ratio", 7, 0.6)

    rule = client.post("/api/sessions/significance_test", json=dict(common))
    assert rule.json()["state"]["n_simulations"] == 5_000
    monte = client.post("/api/sessions/monte_carlo", json=dict(common))
    assert monte.json()["state"]["num_scenarios"] == 25

    # An explicit value still wins over the saved default.
    explicit = client.post("/api/sessions/optimization",
                           json={**common, "n_trials": 3})
    assert explicit.json()["state"]["n_trials"] == 3


TUNED_SOURCE = '''from terry.strategies import Strategy
from terry import utils

class Tuned(Strategy):
    def hyperparameters(self):
        return [{"name": "hold", "type": int, "min": 2, "max": 6, "default": 3}]
    def should_long(self):
        return self.index % (self.hp["hold"] * 2) == 0
    def should_short(self):
        return False
    def go_long(self):
        self.buy = utils.size_to_qty(self.available_margin * 0.3, self.price, fee_rate=self.fee_rate), self.price
    def update_position(self):
        if self.index % (self.hp["hold"] * 2) == self.hp["hold"]:
            self.liquidate()
'''

ENTRY_SOURCE = '''from terry.strategies import Strategy

class Entry(Strategy):
    def should_long(self):
        return self.close > self.open
    def should_short(self):
        return self.close < self.open
    def go_long(self):
        pass
'''


def _seed_rising_candles(project: Path) -> None:
    """An upward drift with a slow oscillation, starting days before the session window
    so warm-up candles exist on disk and daily returns vary enough to define a Sharpe."""
    from terry import helpers as jh
    from terry.data.storage import CandleDB

    (project / "storage").mkdir(parents=True, exist_ok=True)
    database = CandleDB(project / "storage" / "candles.db")
    start = jh.date_to_timestamp("2023-12-28")
    rows = []
    previous = 100.0
    for index in range(60 * 24 * 17):
        price = 100.0 + index * 0.02 + 8.0 * np.sin(index / 700.0)
        rows.append([start + index * 60_000, previous, price,
                     max(previous, price) + 0.01, min(previous, price) - 0.01, 10])
        previous = price
    database.store("Binance Perpetual Futures", "BTC-USDT", rows)


def _await_session(client: TestClient, session_id: str, limit: float = 120) -> dict:
    import time

    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        session = client.get(f"/api/session/{session_id}").json()
        if session["status"] in {"finished", "stopped", "canceled", "terminated"}:
            return session
        time.sleep(0.05)
    raise AssertionError(f"session {session_id} never became terminal")


def test_optimization_session_runs_end_to_end_and_validates_out_of_sample(
        tmp_path: Path):
    _seed_rising_candles(tmp_path)
    client = TestClient(create_app(str(tmp_path)))
    assert client.post("/api/strategies",
                       json={"name": "Tuned", "content": TUNED_SOURCE}).status_code == 200

    created = client.post("/api/sessions/optimization", json={
        "strategy": "Tuned", "symbol": "BTC-USDT", "timeframe": "15m",
        "start_date": "2024-01-01", "finish_date": "2024-01-13",
        "n_trials": 4, "optimal_total": 20, "best_candidates_count": 3,
        "cpu_cores": 2, "config": {"fee": 0, "warm_up_candles": 4},
    })
    assert created.status_code == 200

    session = _await_session(client, created.json()["session_id"])
    results = session["results"]
    assert session["status"] == "finished", results
    assert results["failed_trials"] == 0, results["trial_errors"]
    assert results["completed_trials"] == results["total_trials"] == 4
    assert results["objective"] == "sharpe_ratio"

    best = results["best"]
    assert best is not None, results.get("message")
    assert set(best["hp"]) == {"hold"}
    assert 2 <= best["hp"]["hold"] <= 6
    assert best["train_score"] > 0.0001
    # The candidate was re-scored on the unseen window, not just reported from training.
    assert best["test_score"] is not None
    assert best["test_metrics"]["total"] > 0
    assert results["candidates"][0] == best
    assert session["dashboard_url"]
    report = Path(session["dashboard_url"].removeprefix("file://"))
    assert report.exists() and "sharpe_ratio" in report.read_text()


def test_rule_significance_session_runs_end_to_end(tmp_path: Path):
    _seed_rising_candles(tmp_path)
    client = TestClient(create_app(str(tmp_path)))
    assert client.post("/api/strategies",
                       json={"name": "Entry", "content": ENTRY_SOURCE}).status_code == 200

    created = client.post("/api/sessions/significance_test", json={
        "strategy": "Entry", "symbol": "BTC-USDT", "timeframe": "15m",
        "start_date": "2024-01-01", "finish_date": "2024-01-13",
        "n_simulations": 2_000, "random_seed": 42, "cpu_cores": 2,
    })
    assert created.status_code == 200

    session = _await_session(client, created.json()["session_id"])
    assert session["status"] == "finished", session["results"]
    result = session["results"]["results"]
    assert 0.0 <= result["p_value"] <= 1.0
    assert result["verdict"] in {"significant", "borderline", "not_significant"}
    assert result["n_observations"] > 100
    assert result["n_simulations"] == 2_000
    # A 15m route packs 35,040 candles into a 24/7 year.
    assert result["bars_per_year"] == 35_040
    # The raw bootstrap samples stay out of the session payload.
    assert "simulated_means" not in result

    report = Path(session["dashboard_url"].removeprefix("file://")).read_text()
    assert "Rule Significance Test" in report
    # Flags read as flags, and counts are not rendered with six decimal places.
    assert f"<td>{result['significant']}</td>" in report
    assert "<td>2,000</td>" in report
    assert "<td>35,040</td>" in report


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
