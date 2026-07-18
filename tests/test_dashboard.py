"""Integration coverage for Terry's browser dashboard API and local research workflow."""
from __future__ import annotations

import time
import asyncio
from pathlib import Path

from fastapi.testclient import TestClient

from terry import helpers as jh
from terry.dashboard.app import create_app
from terry.data.storage import CandleDB
from terry.mcp.server import build_server
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

    invalid = client.post("/api/strategies", json={"name": "../unsafe"})
    assert invalid.status_code == 422

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
