"""Integration coverage for Terry's browser dashboard API and local research workflow."""
from __future__ import annotations

import time
import asyncio
import shutil
import subprocess
import threading
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from terry import helpers as jh
from terry.context import TerryContext, get_context
from terry.dashboard.app import create_app
from terry.data.storage import CandleDB
from terry.mcp.server import build_server
from terry.mcp.tools import _common as session_tools
import terry.indicators as ta


STRATEGY = '''from terry.strategies import Strategy
from terry import utils

class DashboardTrade(Strategy):
    def should_long(self):
        return self.index == 0
    def should_short(self):
        return False
    def go_long(self):
        self.buy = utils.size_to_qty(self.available_margin * 0.5, self.price, fee_rate=self.fee_rate), self.price
    def update_position(self):
        if self.index >= 8:
            self.liquidate()
'''


def _seed_candles(project: Path, symbols=("BTC-USDT",)) -> None:
    (project / "storage").mkdir(parents=True, exist_ok=True)
    db = CandleDB(project / "storage" / "candles.db")
    start = jh.date_to_timestamp("2024-01-01")
    rows = []
    for index in range(1_440):
        price = 100 + index * 0.02
        rows.append([start + index * 60_000, price, price, price + 0.1, price - 0.1, 10])
    for symbol in symbols:
        db.store("Binance Perpetual Futures", symbol, rows)


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
    for asset in ("/", "/app.js", "/charts.js", "/styles.css"):
        response = client.get(asset)
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
    status = client.get("/api/status")
    assert status.headers["cache-control"] == "no-store"
    assert list(status.json()["sessions"]) == [
        "backtest", "demo", "optimization", "monte_carlo", "significance_test",
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


def test_dashboard_strategy_import_accepts_zip_and_folder(tmp_path: Path):
    """Import must accept the exported bundle (.zip), a plain zip of the strategy folder,
    and a whole folder uploaded file-by-file — extra files (e.g. a report/) are preserved."""
    import base64
    import io
    import zipfile

    client = TestClient(create_app(str(tmp_path)))
    portable_src = STRATEGY.replace("DashboardTrade", "Portable")
    assert client.post("/api/strategies", json={"name": "Portable", "content": portable_src}).status_code == 200
    # Drop an extra report file so we can prove non-source files survive a round trip.
    (tmp_path / "strategies" / "Portable" / "report").mkdir()
    (tmp_path / "strategies" / "Portable" / "report" / "summary.md").write_text("# report\n")

    def b64(data: bytes) -> str:
        return base64.b64encode(data).decode()

    # 1) The exported bundle re-imports under a new name (class is renamed to match).
    bundle = client.get("/api/strategies/Portable/export").content
    r1 = client.post("/api/strategies/import", json={"data": b64(bundle), "name": "FromBundle"})
    assert r1.status_code == 200 and r1.json()["name"] == "FromBundle"
    assert "report/summary.md" in r1.json()["files"]
    assert "class FromBundle" in client.get("/api/strategies/FromBundle").json()["content"]

    # 2) A plain zip of the strategy folder (no manifest) is tolerated.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("PlainZip/__init__.py", STRATEGY.replace("DashboardTrade", "PlainZip"))
        z.writestr("PlainZip/report/notes.md", "notes")
        z.writestr("PlainZip/__pycache__/x.pyc", "junk")  # must be skipped
    r2 = client.post("/api/strategies/import", json={"data": b64(buf.getvalue())})
    assert r2.status_code == 200 and r2.json()["name"] == "PlainZip"
    assert r2.json()["files"] == ["__init__.py", "report/notes.md"]

    # 3) A whole folder uploaded as a files[] list (browser webkitdirectory upload).
    files = [
        {"path": "FolderUp/__init__.py", "data": b64(STRATEGY.replace("DashboardTrade", "FolderUp").encode())},
        {"path": "FolderUp/report/r.md", "data": b64(b"# r")},
    ]
    r3 = client.post("/api/strategies/import", json={"files": files})
    assert r3.status_code == 200 and r3.json()["name"] == "FolderUp"
    assert (tmp_path / "strategies" / "FolderUp" / "report" / "r.md").exists()

    # A zip with no __init__.py anywhere is rejected with a clear message.
    empty = io.BytesIO()
    with zipfile.ZipFile(empty, "w") as z:
        z.writestr("readme.txt", "hi")
    assert client.post("/api/strategies/import", json={"data": b64(empty.getvalue())}).status_code == 422
    # A traversal entry is silently dropped, never written outside the strategies dir.
    evil = [{"path": "../escape.py", "data": b64(b"x")},
            {"path": "Ok/__init__.py", "data": b64(b"x")}]
    client.post("/api/strategies/import", json={"files": evil})
    assert not (tmp_path / "escape.py").exists()
    assert not (tmp_path / "strategies" / "escape.py").exists()


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
    assert client.post("/api/sessions/optimization", json={
        **base, "optimal_total": 1,
    }).status_code == 422
    assert client.post("/api/sessions/monte_carlo", json={
        **base, "run_candles": "false", "run_trades": False,
    }).status_code == 422
    assert client.post("/api/sessions/monte_carlo", json={
        **base, "run_candles": False, "run_trades": False,
    }).status_code == 422
    assert client.post("/api/sessions/monte_carlo", json={
        **base, "pipeline_type": "unknown",
    }).status_code == 422
    assert client.post("/api/sessions/monte_carlo", json={
        **base, "pipeline_params": [],
    }).status_code == 422
    assert client.post("/api/sessions/significance_test", json={
        **base, "n_simulations": 2000, "random_seed": -1,
    }).status_code == 422
    assert client.post("/api/sessions/significance_test", json={
        **base, "n_simulations": 2000, "cpu_cores": 0,
    }).status_code == 422

    route_payload = {
        "exchange": "Binance Perpetual Futures",
        "start_date": "2024-01-01", "finish_date": "2024-01-02",
        "start": False,
        "routes": [
            {"strategy": "DashboardTrade", "symbol": "BTC-USDT", "timeframe": "1m"},
            {"strategy": "DashboardTrade", "symbol": "ETH-USDT", "timeframe": "5m"},
        ],
        "data_routes": [{"symbol": "SOL-USDT", "timeframe": "15m"}],
        "cpu_cores": 2,
    }
    multi = client.post("/api/sessions/backtest", json=route_payload)
    assert multi.status_code == 200
    multi_state = multi.json()["state"]
    assert multi_state["strategy"] == "DashboardTrade"
    assert multi_state["symbol"] == "BTC-USDT"
    assert multi_state["timeframe"] == "1m"
    assert multi_state["cpu_cores"] == 2
    assert len(multi_state["routes"]) == 2
    assert multi_state["data_routes"][0]["exchange"] == "Binance Perpetual Futures"
    duplicate = {**route_payload, "routes": [route_payload["routes"][0]] * 2}
    assert client.post("/api/sessions/backtest", json=duplicate).status_code == 422
    mixed_exchange = {**route_payload, "routes": [
        {**route_payload["routes"][0], "exchange": "Binance Spot"},
    ]}
    assert client.post("/api/sessions/backtest", json=mixed_exchange).status_code == 422
    assert client.post("/api/sessions/significance_test", json={
        **route_payload, "n_simulations": 2000,
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
    assert "c.summary_metrics?.sharpe_ratio" in source
    assert "tokenQuery" not in source
    assert "strategy-search')?.addEventListener('input'" in source
    assert 'class="skip-link"' in source
    assert "function bindCodeEditor()" in source
    assert "Ctrl/⌘+S to save" in source
    assert "Trading routes" in source and "Data routes" in source
    assert "JSON.parse(pipelineRaw)" in source
    assert "Candle Pipeline" in source and "pipeline_params_json" in source
    # Live Demo Mode + strategy import/export + table-header alignment
    assert "function demoForm()" in source and "paperAccountPanel" in source
    assert "Starting Paper Balance" in source and "History Lookback" in source
    assert "Start Live Demo" in source and "liveConnectingBlock" in source
    assert "function importStrategyZip" in source and "function importStrategyFolder" in source
    assert "/api/strategies/import" in source and "webkitdirectory" in source
    assert "/export" in source and 'id="import-strategy"' in source and 'id="import-folder"' in source
    assert ".trades-table th.num" in styles and ".mc-summary th" in styles
    assert "Session Title" in source and "Research Notes" in source
    assert "notes_metadata?.title" in source
    assert "new Intl.NumberFormat" in source and "new Intl.DateTimeFormat" in source
    assert "data-delete-session" in source and "Delete this saved session?" in source
    assert "data-title" in source and "Save Details" in source
    assert "research note changes" not in source
    assert "session detail changes" in source
    assert "Cancel this research session?" in source
    assert "prefers-reduced-motion" in styles
    assert ":focus-visible" in styles
    assert "pointer:coarse" in styles
    assert ".pill.finished" in styles
    assert ".line-numbers" in styles and ".route-builder" in styles
    # Live candle mutation + non-destructive trade connectors + modern trade sorting.
    assert "activePriceChart?.updateCandle?.(live.candle)" in source
    assert "data-trade-lines" in source and "setTradeLinesVisible" in source
    assert "data-live-feed" in source and "Feed delayed" in source
    assert "const TRADE_SORTS" in source and "defaultDirection:'desc'" in source
    assert "tradeSort={key:'entry',direction:'desc'}" in source
    assert ".trade-sort-popover" in styles and ".dotted-swatch" in styles


def test_price_chart_preserves_native_chart_and_exposes_live_controller_methods():
    """Exercise the actual browser-module return contract with a lightweight chart stub."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the dashboard module contract test")
    script = r"""
import fs from 'node:fs';
const source = fs.readFileSync(process.argv[1], 'utf8');
const charts = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
const seriesByKind = { candle: [], volume: [], line: [] };
const makeSeries = (kind, options = {}) => {
  const series = {
    kind, options, dataCalls: [], updateCalls: [], optionCalls: [], markerCalls: [],
    setData(data) { this.dataCalls.push(data); },
    update(data) { this.updateCalls.push(data); },
    setMarkers(data) { this.markerCalls.push(data); },
    createPriceLine() {},
    applyOptions(data) { this.optionCalls.push(data); },
  };
  seriesByKind[kind].push(series);
  return series;
};
const timeScale = {
  fitContent() {}, setVisibleLogicalRange() {}, subscribeVisibleLogicalRangeChange() {},
};
const nativeChart = {
  addCandlestickSeries(options) { return makeSeries('candle', options); },
  addHistogramSeries(options) { return makeSeries('volume', options); },
  addLineSeries(options) { return makeSeries('line', options); },
  priceScale() { return { applyOptions() {} }; },
  timeScale() { return timeScale; },
  removeSeries() {}, remove() {},
};
globalThis.window = { LightweightCharts: {
  LineStyle: { Dashed: 2, Dotted: 1 },
  createChart() { return nativeChart; },
} };
globalThis.document = { body: { classList: { contains() { return false; } } } };
const controller = charts.priceChart(
  { innerHTML: '', clientHeight: 460 },
  {
    candles: [{ time: 1, open: 10, high: 12, low: 9, close: 11, volume: 5 }],
    trade_lines: [
      { id: 'open', side: 'long', open_time: 1, close_time: 2,
        entry_price: 10, exit_price: 11, is_open: true },
      { id: 'same-tick', side: 'short', open_time: 3, close_time: 3,
        entry_price: 12, exit_price: 11, is_open: false },
    ],
  },
);
if (controller !== nativeChart) throw new Error('priceChart no longer returns the native chart');
for (const method of ['updateCandle', 'setMarkers', 'setTradeLines', 'setTradeLinesVisible']) {
  if (typeof controller[method] !== 'function') throw new Error(`missing ${method}`);
}
if (seriesByKind.line.length !== 2) throw new Error('one connector per trade was not created');
for (const series of seriesByKind.line) {
  if (series.options.lineStyle !== 1 || series.options.lineWidth !== 2)
    throw new Error('connector is not a visible dotted line');
}
const sameTickData = seriesByKind.line[1].dataCalls[0];
if (sameTickData[0].time !== 3 || sameTickData[1].time !== 4)
  throw new Error('same-tick connector was not retained');
controller.setTradeLinesVisible(false);
if (seriesByKind.line.some(series => series.optionCalls.at(-1)?.visible !== false))
  throw new Error('connector toggle did not hide every trade line');
controller.updateCandle({ time: 2, tick_time: 5, open: 11, high: 13, low: 10,
  close: 12, volume: 6 });
if (seriesByKind.candle[0].updateCalls.at(-1)?.close !== 12 ||
    seriesByKind.volume[0].updateCalls.at(-1)?.value !== 6)
  throw new Error('live OHLCV did not update both chart series');
if (seriesByKind.line[0].dataCalls.at(-1)?.[1]?.time !== 5)
  throw new Error('open connector did not advance with the live candle');
controller.setMarkers([]);
controller.setTradeLines([], false);
"""
    completed = subprocess.run(
        [node, "--input-type=module", "-e", script,
         str(Path("terry/dashboard/static/charts.js").resolve())],
        check=False, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_live_candle_refresh_updates_the_forming_row(tmp_path: Path, monkeypatch):
    """A live refresh must request the newest timestamp again and replace its OHLCV."""
    ctx = TerryContext(str(tmp_path))
    exchange, symbol = "Binance Perpetual Futures", "BTC-USDT"
    start = jh.date_to_timestamp("2024-01-01")
    ctx.candle_db.store(exchange, symbol, [
        [start, 100, 101, 102, 99, 10],
    ])
    called = {}

    def fake_fetch(ex, sym, start_ts, finish_ts, should_stop=None):
        called.update(exchange=ex, symbol=sym, start=start_ts, finish=finish_ts)
        assert should_stop is not None and should_stop() is False
        return np.asarray([
            [start, 100, 102, 103, 99, 12],
            [start + 60_000, 102, 104, 105, 101, 7],
        ], dtype=float)

    monkeypatch.setattr("terry.data.binance.fetch_1m_range", fake_fetch)
    ctx.runner._refresh_live_candles(
        exchange, symbol, start, start + 120_000, "live-session")

    assert called["start"] == start  # not start + 60s: the open row must be re-fetched
    rows = ctx.candle_db.get(exchange, symbol, start, start + 120_000)
    assert rows[:, 2].tolist() == [102, 104]
    candle = ctx.runner._demo_live_candle(
        exchange, symbol, "5m", start, start + 60_000)
    assert candle == {
        "time": int(start / 1000), "open": 100.0, "close": 104.0,
        "high": 105.0, "low": 99.0, "volume": 19.0, "timeframe": "5m",
    }


def test_demo_prepare_uses_exact_intraday_boundaries(tmp_path: Path):
    """Demo strategy replay must not truncate its moving window back to UTC midnight."""
    ctx = TerryContext(str(tmp_path))
    exchange, symbol = "Binance Perpetual Futures", "BTC-USDT"
    day = jh.date_to_timestamp("2024-01-01")
    rows = [[day + i * 60_000, 100 + i, 100 + i, 101 + i, 99 + i, 1]
            for i in range(12)]
    ctx.candle_db.store(exchange, symbol, rows)
    state = {
        "exchange": exchange, "symbol": symbol, "timeframe": "1m",
        "strategy": "DashboardTrade", "start_date": "2024-01-01",
        "finish_date": "2024-01-02", "config": {"warm_up_candles": 0},
    }
    start, finish = day + 2 * 60_000, day + 9 * 60_000
    _, _, _, candles, _ = ctx.runner._prepare(
        state, start_ts=start, finish_ts=finish)
    selected = candles[jh.key(exchange, symbol)]["candles"]
    assert selected[0, 0] == start
    assert selected[-1, 0] == finish - 60_000
    assert len(selected) == 7


def test_demo_candle_endpoint_includes_live_candle_and_trade_connectors(tmp_path: Path):
    client = TestClient(create_app(str(tmp_path)))
    ctx = get_context()
    exchange, symbol = "Binance Perpetual Futures", "BTC-USDT"
    start = jh.date_to_timestamp("2024-01-01")
    ctx.candle_db.store(exchange, symbol, [
        [start, 100, 101, 102, 99, 10],
        [start + 60_000, 101, 102, 103, 100, 11],
        [start + 120_000, 102, 103, 104, 101, 5],
    ])
    state = {
        "exchange": exchange, "symbol": symbol, "timeframe": "1m",
        "strategy": "DashboardTrade", "start_date": "2024-01-01",
        "finish_date": "2024-01-02",
    }
    sid = ctx.sessions.create("demo", state)
    live_candle = {"time": int((start + 120_000) / 1000), "open": 102,
                   "close": 104, "high": 105, "low": 101, "volume": 8}
    trade = {
        "id": "open-trade", "symbol": symbol, "type": "long",
        "entry_price": 101, "exit_price": 104, "opened_at": start + 60_000,
        "closed_at": start + 180_000, "is_open_at_end": True,
        "orders": [
            {"side": "buy", "qty": 1, "executed_at": start + 60_000,
             "reduce_only": False},
            {"side": "sell", "qty": -1, "executed_at": start + 180_000,
             "reduce_only": True},
        ],
    }
    ctx.sessions.set_results(sid, {
        "trades": [trade], "live": {
            "is_live": True, "window_start_ts": start,
            "updated_at": start + 150_000, "candle": live_candle,
        },
    }, status="running")

    payload = client.get(f"/api/session/{sid}/candles").json()
    assert payload["candles"][-1] == live_candle
    assert len(payload["markers"]) == 1  # synthetic terminal close stays hidden
    assert payload["markers"][0]["text"].startswith("● Open Long")
    assert payload["trade_lines"] == [{
        "id": "open-trade", "side": "long",
        "open_time": int((start + 60_000) / 1000),
        "close_time": int((start + 150_000) / 1000),
        "entry_price": 101, "exit_price": 104, "is_open": True,
    }]


def test_trade_connectors_keep_exact_times_and_cover_every_displayed_trade(tmp_path: Path):
    """1h round trips must not collapse or disappear once a result exceeds 300 trades."""
    client = TestClient(create_app(str(tmp_path)))
    ctx = get_context()
    exchange, symbol = "Binance Perpetual Futures", "SOL-USDT"
    start = jh.date_to_timestamp("2024-01-01")
    ctx.candle_db.store(exchange, symbol, [
        [start + minute * 60_000, 70, 70, 71, 69, 1]
        for minute in range(61)
    ])
    state = {
        "exchange": exchange, "symbol": symbol, "timeframe": "1h",
        "strategy": "ConnectorTest", "start_date": "2024-01-01",
        "finish_date": "2024-01-02",
    }
    sid = ctx.sessions.create("backtest", state)
    trades = [{
        "id": f"trade-{index}", "symbol": symbol, "type": "long",
        "entry_price": 70, "exit_price": 71,
        "opened_at": start + 10 * 60_000,
        "closed_at": start + 20 * 60_000,
        "orders": [],
    } for index in range(301)]
    ctx.sessions.set_results(sid, {"trades": trades}, status="finished")

    lines = client.get(f"/api/session/{sid}/candles").json()["trade_lines"]

    assert len(lines) == len(trades)
    assert lines[0]["id"] == "trade-0"  # oldest displayed trade is no longer truncated
    assert lines[-1]["id"] == "trade-300"
    assert lines[0]["open_time"] == int((start + 10 * 60_000) / 1000)
    assert lines[0]["close_time"] == int((start + 20 * 60_000) / 1000)
    assert lines[0]["open_time"] != lines[0]["close_time"]


def test_demo_loop_publishes_each_market_tick_and_surfaces_feed_errors(
        tmp_path: Path, monkeypatch):
    """Demo price/candle revisions must advance independently of strategy replay."""
    ctx = TerryContext(str(tmp_path))
    state = {
        "exchange": "Binance Perpetual Futures", "symbol": "SOL-USDT",
        "timeframe": "1m", "strategy": "TickTest", "lookback_days": 1,
        "start_date": "2024-01-01", "finish_date": "2024-01-02",
    }
    sid = ctx.sessions.create("demo", state)
    now = jh.date_to_timestamp("2024-01-02") + 30_000
    monkeypatch.setattr(jh, "now_to_timestamp", lambda force_fresh=False: now)
    refreshes = 0

    def refresh(*_args, **_kwargs):
        nonlocal refreshes
        refreshes += 1
        if refreshes == 3:
            raise RuntimeError("temporary market feed failure")

    prices = iter((100.0, 101.0))
    monkeypatch.setattr(ctx.runner, "_refresh_live_candles", refresh)
    monkeypatch.setattr(ctx.runner, "_demo_live_candle", lambda *_args: {
        "time": int((now // 60_000 * 60_000) / 1000),
        "open": 99.0, "close": next(prices), "high": 102.0,
        "low": 98.0, "volume": float(refreshes), "timeframe": "1m",
    })
    monkeypatch.setattr(ctx.runner, "_demo_backtest_window", lambda *_args: {
        "metrics": {"starting_balance": 10_000}, "trades": [],
        "equity_curve": [], "daily_balance": [],
    })
    monkeypatch.setattr(ctx, "write_report", lambda *_args: "")
    published = []
    original_set_results = ctx.sessions.set_results

    def capture(session_id, results, status="finished"):
        if status == "running":
            published.append(dict(results.get("live") or {}))
        return original_set_results(session_id, results, status=status)

    monkeypatch.setattr(ctx.sessions, "set_results", capture)
    sleeps = 0

    def stop_after_three_ticks(_seconds):
        nonlocal sleeps
        sleeps += 1
        if sleeps == 3:
            ctx.runner._canceled.add(sid)

    monkeypatch.setattr(time, "sleep", stop_after_three_ticks)

    ctx.runner._run_demo_live(sid, state)

    assert [item["tick"] for item in published] == [1, 2, 3]
    assert [item.get("price") for item in published[:2]] == [100.0, 101.0]
    assert all(item["poll_seconds"] == 1 for item in published)
    assert published[0]["candle"]["tick_time"] == int((now // 60_000 * 60_000) / 1000)
    assert published[-1]["error"] == "RuntimeError: temporary market feed failure"


def test_dashboard_backtest_runs_and_exports_end_to_end(tmp_path: Path):
    _seed_candles(tmp_path)
    client = TestClient(create_app(str(tmp_path)))
    assert client.patch("/api/config", json={"warm_up_candles": 0}).status_code == 200
    assert client.post("/api/strategies", json={"name": "DashboardTrade", "content": STRATEGY}).status_code == 200

    created = client.post("/api/sessions/backtest", json={
        "strategy": "DashboardTrade", "exchange": "Binance Perpetual Futures", "symbol": "BTC-USDT",
        "timeframe": "1m", "start_date": "2024-01-01", "finish_date": "2024-01-02",
        "title": "Dashboard parity run", "description": "Capture this strategy snapshot.",
    })
    assert created.status_code == 200
    session = _wait_for_session(client, created.json()["session_id"])
    assert session["status"] == "finished", session
    assert session["results"]["metrics"]["total"] >= 1
    assert session["results"]["benchmark"]["return_percentage"] > 0
    assert Path(session["results"]["charts_folder"]).is_dir()
    assert session["notes_metadata"]["title"] == "Dashboard parity run"
    assert session["notes_metadata"]["description"] == "Capture this strategy snapshot."
    assert session["notes_metadata"]["strategy_codes"]

    listed = client.get("/api/sessions/backtest?query=DashboardTrade").json()
    assert listed["total"] == 1
    updated = client.patch(
        f"/api/session/{session['session_id']}",
        json={"notes": "checked in dashboard", "title": "Reviewed parity run"}).json()
    assert updated["notes"] == "checked in dashboard"
    assert updated["notes_metadata"]["title"] == "Reviewed parity run"
    assert updated["notes_metadata"]["description"] == "checked in dashboard"
    assert client.get(f"/api/session/{session['session_id']}/export?format=json").status_code == 200
    csv = client.get(f"/api/session/{session['session_id']}/export?format=csv")
    assert csv.status_code == 200 and "entry_price" in csv.text
    assert client.get(f"/reports/{session['session_id']}").status_code == 200


def test_dashboard_multiroute_backtest_runs_end_to_end(tmp_path: Path):
    _seed_candles(tmp_path, ("BTC-USDT", "ETH-USDT", "SOL-USDT"))
    client = TestClient(create_app(str(tmp_path)))
    assert client.patch("/api/config", json={"warm_up_candles": 0}).status_code == 200
    assert client.post("/api/strategies", json={
        "name": "DashboardTrade", "content": STRATEGY,
    }).status_code == 200

    created = client.post("/api/sessions/backtest", json={
        "exchange": "Binance Perpetual Futures",
        "routes": [
            {"strategy": "DashboardTrade", "symbol": "BTC-USDT", "timeframe": "1m"},
            {"strategy": "DashboardTrade", "symbol": "ETH-USDT", "timeframe": "1m"},
        ],
        "data_routes": [{"symbol": "SOL-USDT", "timeframe": "15m"}],
        "start_date": "2024-01-01", "finish_date": "2024-01-02",
        "export_chart": False,
    })
    assert created.status_code == 200, created.json()
    session = _wait_for_session(client, created.json()["session_id"])
    assert session["status"] == "finished", session
    assert session["results"]["metrics"]["total"] == 2
    assert len(session["state"]["routes"]) == 2
    assert session["state"]["data_routes"][0]["symbol"] == "SOL-USDT"


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
    registered = asyncio.run(mcp.list_tools())
    names = {tool.name for tool in registered}
    assert len(names) == 58
    assert {
        "create_strategy", "read_strategy", "write_strategy", "import_candles",
        "create_backtest_draft", "run_backtest", "get_backtest_session",
        "create_significance_test_draft", "run_significance_test",
        "create_monte_carlo_draft", "run_monte_carlo",
        "create_optimization_draft", "run_optimization", "list_indicators",
        "get_indicator_details", "get_config", "update_config",
    }.issubset(names)
    schemas = {tool.name: tool.inputSchema["properties"] for tool in registered}
    assert {"routes", "data_routes", "cpu_cores", "pipeline_type", "pipeline_params"}.issubset(
        schemas["create_monte_carlo_draft"])
    assert {"routes", "data_routes", "random_seed", "cpu_cores"}.issubset(
        schemas["create_significance_test_draft"])
