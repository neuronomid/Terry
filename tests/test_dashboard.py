"""Integration coverage for Terry's browser dashboard API and local research workflow."""
from __future__ import annotations

import time
import asyncio
import threading
from pathlib import Path

from fastapi.testclient import TestClient

from terry import helpers as jh
from terry.context import TerryContext
from terry.dashboard.app import create_app
from terry.data.storage import CandleDB
from terry.mcp.server import build_server
from terry.mcp.tools import _common as session_tools
import terry.indicators as ta


STRATEGY = '''from terry.strategies import Strategy
from terry import utils

class DashboardTrade(Strategy):
    def should_long(self):
        return self.index == 1
    def should_short(self):
        return False
    def go_long(self):
        self.buy = utils.size_to_qty(self.available_margin * 0.5, self.price, fee_rate=self.fee_rate), self.price
    def update_position(self):
        if self.index >= 8:
            self.liquidate()
'''


def _seed_candles(project: Path) -> None:
    (project / "storage").mkdir(parents=True, exist_ok=True)
    db = CandleDB(project / "storage" / "candles.db")
    start = jh.date_to_timestamp("2024-01-01")
    rows = []
    for index in range(1_440):
        price = 100 + index * 0.02
        rows.append([start + index * 60_000, price, price, price + 0.1, price - 0.1, 10])
    db.store("Binance Perpetual Futures", "BTC-USDT", rows)


def _wait_for_session(client: TestClient, session_id: str) -> dict:
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        session = client.get(f"/api/session/{session_id}").json()
        if session["status"] in {"finished", "stopped", "canceled", "terminated"}:
            return session
        time.sleep(0.05)
    raise AssertionError("session did not become terminal")


def test_dashboard_serves_navigation_and_strategy_crud(tmp_path: Path):
    client = TestClient(create_app(str(tmp_path)))

    page = client.get("/")
    assert page.status_code == 200
    assert "Terry" in page.text
    assert client.get("/app.js").status_code == 200
    assert list(client.get("/api/status").json()["sessions"]) == [
        "backtest", "optimization", "monte_carlo", "significance_test",
    ]

    invalid = client.post("/api/strategies", json={"name": "../unsafe"})
    assert invalid.status_code == 422
    assert client.post("/api/strategies", json={"name": "BadContent", "content": 123}).status_code == 422
    assert client.put("/api/strategies/DoesNotExist", json={"content": STRATEGY}).status_code == 404

    created = client.post("/api/strategies", json={"name": "DashboardTrade", "content": STRATEGY})
    assert created.status_code == 200
    assert created.json()["validation_error"] is None
    assert client.get("/api/strategies").json()["strategies"][0]["name"] == "DashboardTrade"

    read = client.get("/api/strategies/DashboardTrade")
    assert "class DashboardTrade" in read.json()["content"]
    assert client.post("/api/strategies/DashboardTrade/fork", json={"name": "DashboardFork"}).status_code == 200
    assert client.delete("/api/strategies/DashboardFork").json()["status"] == "deleted"


def test_dashboard_config_indicator_and_auth_validation(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TERRY_DASHBOARD_PASSWORD", "local-only")
    client = TestClient(create_app(str(tmp_path)))

    assert client.get("/api/status").status_code == 401
    assert client.post("/api/auth/login", json={"password": "wrong"}).status_code == 401
    token = client.post("/api/auth/login", json={"password": "local-only"}).json()["auth_token"]
    auth = {"Authorization": f"Bearer {token}"}
    assert client.get("/api/status", headers=auth).json()["auth_required"] is True

    invalid = client.patch("/api/config", headers=auth, json={"unknown": 1})
    assert invalid.status_code == 422
    updated = client.patch("/api/config", headers=auth, json={"starting_balance": 12_500, "warm_up_candles": 0})
    assert updated.status_code == 200
    assert updated.json()["config"]["starting_balance"] == 12_500

    details = client.get("/api/indicators/sma", headers=auth)
    assert details.status_code == 200
    assert "sma(" in details.json()["signature"]
    assert client.get("/api/live", headers=auth).status_code == 501


def test_dashboard_auth_cookie_security_headers_and_no_token_in_query(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TERRY_DASHBOARD_PASSWORD", "local-only")
    app = create_app(str(tmp_path))
    client = TestClient(app)

    assert client.get("/api/auth/status").json() == {"auth_required": True, "authenticated": False}
    page = client.get("/")
    assert page.headers["x-frame-options"] == "DENY"
    assert "frame-ancestors 'none'" in page.headers["content-security-policy"]
    assert "access-control-allow-origin" not in page.headers

    login = client.post("/api/auth/login", json={"password": "local-only"})
    token = login.json()["auth_token"]
    cookie = login.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie
    assert client.get("/api/auth/status").json()["authenticated"] is True
    assert client.get("/api/status").status_code == 200  # HttpOnly cookie authenticates links/reports.

    stranger = TestClient(app)
    assert stranger.get(f"/api/status?token={token}").status_code == 401
    assert client.post("/api/auth/logout").json()["status"] == "signed_out"
    assert client.get("/api/status").status_code == 401


def test_dashboard_rejects_malformed_research_and_config_payloads(tmp_path: Path):
    client = TestClient(create_app(str(tmp_path)))
    assert client.post("/api/strategies", json={"name": "DashboardTrade", "content": STRATEGY}).status_code == 200
    base = {
        "strategy": "DashboardTrade", "exchange": "Binance Perpetual Futures",
        "symbol": "BTC-USDT", "timeframe": "1m", "start_date": "2024-01-01",
        "finish_date": "2024-01-02", "start": False,
    }

    assert client.post("/api/sessions/backtest", json={**base, "symbol": "not-a-pair"}).status_code == 422
    assert client.post("/api/sessions/backtest", json={**base, "hyperparameters": []}).status_code == 422
    assert client.post("/api/sessions/backtest", json={**base, "start": "false"}).status_code == 422
    assert client.post("/api/sessions/backtest", json={**base, "typo_field": True}).status_code == 422
    assert client.post("/api/sessions/backtest", json={**base, "benchmark": "yes"}).status_code == 422
    assert client.get("/api/sessions/backtest").json()["total"] == 0
    assert client.post("/api/sessions/optimization", json={**base, "n_trials": "many"}).status_code == 422
    assert client.post("/api/sessions/optimization", json={**base, "objective": "made_up"}).status_code == 422
    assert client.post("/api/sessions/monte_carlo", json={
        **base, "run_candles": "false", "run_trades": False,
    }).status_code == 422
    assert client.post("/api/sessions/monte_carlo", json={
        **base, "run_candles": False, "run_trades": False,
    }).status_code == 422

    assert client.patch("/api/config", json={"starting_balance": 0}).status_code == 422
    assert client.patch("/api/config", json={"warm_up_candles": 1.5}).status_code == 422
    assert client.patch("/api/config", json={"optimization": "invalid"}).status_code == 422
    assert client.patch("/api/config", json={"optimization": {"unknown": 1}}).status_code == 422
    assert client.patch("/api/config", json={
        "monte_carlo": {"run_candles": False, "run_trades": False},
    }).status_code == 422

    draft = client.post("/api/sessions/backtest", json=base).json()
    sid = draft["session_id"]
    assert draft["status"] == "draft"
    assert session_tools.cancel_session(sid, "optimization")["error"] == "wrong_kind"
    assert session_tools.cancel_session(sid, "backtest")["error"] == "not_running"
    assert client.post(f"/api/session/{sid}/cancel").status_code == 409
    assert client.patch(f"/api/session/{sid}", json={"unexpected": True}).status_code == 422
    assert client.patch(f"/api/session/{sid}", json={"notes": "x" * 20_001}).status_code == 422
    assert client.delete(f"/api/session/{sid}").status_code == 200
    assert client.delete("/api/candles/Binance%20Spot/BTC-USDT").status_code == 404


def test_dashboard_static_regressions_cover_accessibility_and_result_keys():
    source = Path("terry/dashboard/static/app.js").read_text(encoding="utf-8")
    styles = Path("terry/dashboard/static/styles.css").read_text(encoding="utf-8")
    assert "mc.summary_metrics?.sharpe_ratio" in source
    assert "tokenQuery" not in source
    assert "strategy-search')?.addEventListener('input'" in source
    assert 'class="skip-link"' in source
    assert "prefers-reduced-motion" in styles
    assert ":focus-visible" in styles
    assert ".pill.finished" in styles


def test_dashboard_backtest_runs_and_exports_end_to_end(tmp_path: Path):
    _seed_candles(tmp_path)
    client = TestClient(create_app(str(tmp_path)))
    assert client.patch("/api/config", json={"warm_up_candles": 0}).status_code == 200
    assert client.post("/api/strategies", json={"name": "DashboardTrade", "content": STRATEGY}).status_code == 200

    created = client.post("/api/sessions/backtest", json={
        "strategy": "DashboardTrade", "exchange": "Binance Perpetual Futures", "symbol": "BTC-USDT",
        "timeframe": "1m", "start_date": "2024-01-01", "finish_date": "2024-01-02",
    })
    assert created.status_code == 200
    session = _wait_for_session(client, created.json()["session_id"])
    assert session["status"] == "finished", session
    assert session["results"]["metrics"]["total"] >= 1
    assert session["results"]["benchmark"]["return_percentage"] > 0
    assert Path(session["results"]["charts_folder"]).is_dir()

    listed = client.get("/api/sessions/backtest?query=DashboardTrade").json()
    assert listed["total"] == 1
    assert client.patch(f"/api/session/{session['session_id']}", json={"notes": "checked in dashboard"}).json()["notes"] == "checked in dashboard"
    assert client.get(f"/api/session/{session['session_id']}/export?format=json").status_code == 200
    csv = client.get(f"/api/session/{session['session_id']}/export?format=csv")
    assert csv.status_code == 200 and "entry_price" in csv.text
    assert client.get(f"/reports/{session['session_id']}").status_code == 200


def test_dashboard_candle_error_states_are_explicit(tmp_path: Path):
    client = TestClient(create_app(str(tmp_path)))
    assert client.post("/api/candles/import", json={"exchange": "Unknown", "symbol": "BTC-USDT", "start_date": "2024-01-01"}).status_code == 422
    assert client.post("/api/candles/import", json={"exchange": "Binance Spot", "symbol": "not a pair", "start_date": "2024-01-01"}).status_code == 422
    assert client.get("/api/candles/import/not-a-real-job").status_code == 404


def test_runner_cancellation_is_terminal_and_cleans_worker_state(tmp_path: Path, monkeypatch):
    ctx = TerryContext(str(tmp_path))
    sid = ctx.sessions.create("backtest", {"strategy": "Slow"})
    stopping = threading.Event()
    release = threading.Event()

    def slow_run(session_id, _state):
        while not ctx.runner._should_cancel(session_id)():
            time.sleep(0.005)
        stopping.set()
        release.wait(timeout=2)
        raise InterruptedError("Research run canceled")

    monkeypatch.setattr(ctx.runner, "_run_backtest", slow_run)
    ctx.runner.run(sid)
    assert ctx.runner.cancel(sid)["status"] == "canceled"
    assert stopping.wait(timeout=1)
    assert ctx.runner.run(sid)["error"] == "worker_active"
    release.set()
    deadline = time.monotonic() + 2
    while sid in ctx.runner._threads and time.monotonic() < deadline:
        time.sleep(0.01)

    assert ctx.sessions.get(sid)["status"] == "canceled"
    assert sid not in ctx.runner._threads
    assert sid not in ctx.runner._canceled
    assert ctx.runner.cancel(sid)["error"] == "not_running"


def test_session_store_can_list_without_truncating_history(tmp_path: Path):
    ctx = TerryContext(str(tmp_path))
    for index in range(55):
        ctx.sessions.create("backtest", {"strategy": f"Strategy{index}"})
    assert len(ctx.sessions.list("backtest", 10)) == 10
    assert len(ctx.sessions.list("backtest", 10, offset=50)) == 5
    assert len(ctx.sessions.list("backtest", None)) == 55
    assert ctx.sessions.count("backtest") == 55


def test_indicator_and_mcp_surfaces_are_complete(tmp_path: Path):
    """Protect the documented Jesse-compatible 174 indicators and 58 MCP tools."""
    assert len(ta.__all__) == 174
    assert all(callable(getattr(ta, name)) for name in ta.__all__)

    mcp = build_server(project_root=str(tmp_path))
    names = {tool.name for tool in asyncio.run(mcp.list_tools())}
    assert len(names) == 58
    assert {
        "create_strategy", "read_strategy", "write_strategy", "import_candles",
        "create_backtest_draft", "run_backtest", "get_backtest_session",
        "create_significance_test_draft", "run_significance_test",
        "create_monte_carlo_draft", "run_monte_carlo",
        "create_optimization_draft", "run_optimization", "list_indicators",
        "get_indicator_details", "get_config", "update_config",
    }.issubset(names)
