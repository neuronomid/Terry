"""FastAPI dashboard for the local Terry research engine.

The dashboard deliberately calls the same Context/Runner services used by MCP.  It is a
browser client for local research only: no exchange credentials or live-trading controls are
implemented here.
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .. import indicators as ta
from .. import helpers as jh
from ..context import TerryContext, set_context
from ..data.binance import EXCHANGES
from ..loader import load_strategy_class, strategy_exists
from ..sessions.db import TERMINAL, VALID_KINDS
from ..version import __version__

_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,79}$")
_STATIC = Path(__file__).with_name("static")


def _error(status: int, message: str, code: str = "invalid_request") -> HTTPException:
    return HTTPException(status_code=status, detail={"error": code, "message": message})


def _require_name(name: str) -> str:
    if not _NAME.fullmatch(name or ""):
        raise _error(422, "Strategy names must start with a letter and contain only letters, numbers, or underscores.")
    return name


def _date(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise _error(422, f"{label} is required and must use YYYY-MM-DD.")
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise _error(422, f"{label} must use YYYY-MM-DD.") from exc
    return value


def _validation_error(ctx: TerryContext, name: str) -> str | None:
    try:
        load_strategy_class(name, ctx.strategies_dir)
    except Exception as exc:  # a saved draft should remain editable even when incomplete
        return f"{type(exc).__name__}: {exc}"
    return None


def _strategy_path(ctx: TerryContext, name: str) -> Path:
    _require_name(name)
    return Path(ctx.strategies_dir, name, "__init__.py")


def _strategy_template(name: str) -> str:
    return f'''from terry.strategies import Strategy
import terry.indicators as ta
from terry import utils


class {name}(Strategy):
    def should_long(self):
        return ta.sma(self.candles, 10) > ta.sma(self.candles, 30)

    def should_short(self):
        return ta.sma(self.candles, 10) < ta.sma(self.candles, 30)

    def go_long(self):
        qty = utils.size_to_qty(self.available_margin * 0.5, self.price, fee_rate=self.fee_rate)
        self.buy = qty, self.price

    def go_short(self):
        qty = utils.size_to_qty(self.available_margin * 0.5, self.price, fee_rate=self.fee_rate)
        self.sell = qty, self.price
'''


def _session_payload(session: dict[str, Any]) -> dict[str, Any]:
    result = _clean_json(dict(session))
    result["session_id"] = result.pop("id")
    result["dashboard_url"] = (result.get("results") or {}).get("dashboard_url", "")
    return result


def _clean_json(value: Any) -> Any:
    """Make engine results safe for strict JSON clients (NaN/Inf become null)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    return value


def _session_list(ctx: TerryContext, kind: str, limit: int, offset: int, query: str | None,
                  status: str | None) -> tuple[list[dict[str, Any]], int]:
    rows = ctx.sessions.list(kind, 1_000)
    query = (query or "").strip().lower()
    if query:
        rows = [row for row in rows if query in " ".join(map(str, (
            row["id"], row["state"].get("strategy", ""), row["state"].get("symbol", ""), row["notes"] or ""))).lower()]
    if status:
        rows = [row for row in rows if row["status"] == status]
    total = len(rows)
    return [_session_payload(row) for row in rows[offset:offset + limit]], total


def _base_state(ctx: TerryContext, payload: dict[str, Any]) -> dict[str, Any]:
    strategy = _require_name(str(payload.get("strategy", "")))
    if not strategy_exists(ctx.strategies_dir, strategy):
        raise _error(404, f'Strategy "{strategy}" does not exist.', "strategy_not_found")
    exchange = payload.get("exchange") or ctx.config.get()["exchange"]
    if exchange not in EXCHANGES:
        raise _error(422, f"Unknown exchange: {exchange}.")
    start_date = _date(payload.get("start_date"), "start_date")
    finish_date = _date(payload.get("finish_date"), "finish_date")
    if jh.date_to_timestamp(finish_date) <= jh.date_to_timestamp(start_date):
        raise _error(422, "finish_date must be after start_date.")
    timeframe = payload.get("timeframe", "4h")
    try:
        jh.timeframe_to_one_minutes(timeframe)
    except Exception as exc:
        raise _error(422, f"Unsupported timeframe: {timeframe}.") from exc
    state = {
        "strategy": strategy,
        "symbol": str(payload.get("symbol") or "BTC-USDT").upper(),
        "exchange": exchange,
        "timeframe": timeframe,
        "start_date": start_date,
        "finish_date": finish_date,
    }
    overrides = payload.get("config")
    if overrides is not None:
        if not isinstance(overrides, dict):
            raise _error(422, "config must be a JSON object.")
        state["config"] = overrides
    if "hyperparameters" in payload:
        state["hyperparameters"] = payload["hyperparameters"]
    return state


def _new_session(ctx: TerryContext, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    if kind not in VALID_KINDS:
        raise _error(404, "Unknown research mode.", "unknown_mode")
    state = _base_state(ctx, payload)
    if kind == "significance_test":
        simulations = int(payload.get("n_simulations", 2_000))
        if simulations < 2_000:
            raise _error(422, "Rule Significance Test requires at least 2,000 simulations.")
        state.update({"n_simulations": simulations, "hypothesis": str(payload.get("hypothesis", "")),
                      "rationale": str(payload.get("rationale", ""))})
    elif kind == "monte_carlo":
        scenarios = int(payload.get("num_scenarios", 200))
        if scenarios < 1:
            raise _error(422, "num_scenarios must be positive.")
        state.update({"num_scenarios": scenarios, "run_candles": bool(payload.get("run_candles", True)),
                      "run_trades": bool(payload.get("run_trades", False))})
    elif kind == "optimization":
        trials = int(payload.get("n_trials", 100))
        split = float(payload.get("train_test_split", 0.75))
        if trials < 1 or not 0.1 < split < 0.9:
            raise _error(422, "n_trials must be positive and train_test_split must be between 0.1 and 0.9.")
        state.update({"objective": str(payload.get("objective", "sharpe_ratio")), "n_trials": trials,
                      "train_test_split": split})
    sid = ctx.sessions.create(kind, state, notes=str(payload.get("notes", "")))
    session = ctx.sessions.get(sid)
    if payload.get("start", True):
        ctx.runner.run(sid)
        session = ctx.sessions.get(sid)
    return _session_payload(session)


def create_app(project_root: str | None = None) -> FastAPI:
    """Create the local dashboard app, isolated to one Terry project root."""
    ctx = set_context(TerryContext(project_root))
    password = os.environ.get("TERRY_DASHBOARD_PASSWORD", "")
    tokens: set[str] = set()

    app = FastAPI(title="Terry Dashboard", version=__version__)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                       allow_methods=["*"], allow_headers=["*"])

    @app.exception_handler(HTTPException)
    async def handle_http_error(_, exc: HTTPException):
        detail = exc.detail if isinstance(exc.detail, dict) else {"error": "request_failed", "message": str(exc.detail)}
        return JSONResponse(status_code=exc.status_code, content=detail)

    def auth(authorization: str | None = Header(default=None), token: str | None = Query(default=None)):
        if not password:
            return
        bearer = (authorization or "").removeprefix("Bearer ")
        if token not in tokens and bearer not in tokens:
            raise _error(401, "Sign in to continue.", "unauthorized")

    @app.post("/api/auth/login")
    def login(payload: dict[str, Any]):
        if password and not secrets.compare_digest(str(payload.get("password", "")), password):
            raise _error(401, "Incorrect password.", "invalid_credentials")
        token = secrets.token_urlsafe(32)
        tokens.add(token)
        return {"auth_token": token, "auth_required": bool(password)}

    @app.get("/api/status")
    def status(_: None = Depends(auth)):
        return {
            "name": "Terry", "version": __version__, "auth_required": bool(password),
            "project_root": ctx.project_root, "indicators_available": len(ta.__all__),
            "supported_exchanges": list(EXCHANGES), "datasets": ctx.candle_db.existing(),
            "sessions": {kind: len(ctx.sessions.list(kind, 1_000)) for kind in VALID_KINDS},
            "live_trading": {"available": False, "reason": "Live and paper trading are out of scope for Terry."},
        }

    @app.get("/api/strategies")
    def list_strategies(_: None = Depends(auth)):
        items = []
        for entry in sorted(Path(ctx.strategies_dir).iterdir(), key=lambda p: p.name.lower()):
            path = entry / "__init__.py"
            if entry.is_dir() and path.exists() and _NAME.fullmatch(entry.name):
                items.append({"name": entry.name, "updated_at": int(path.stat().st_mtime * 1000),
                              "validation_error": _validation_error(ctx, entry.name)})
        return {"strategies": items}

    @app.post("/api/strategies")
    def create_strategy(payload: dict[str, Any], _: None = Depends(auth)):
        name = _require_name(str(payload.get("name", "")))
        path = _strategy_path(ctx, name)
        if path.exists():
            raise _error(409, f'Strategy "{name}" already exists.', "exists")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(payload.get("content") or _strategy_template(name)), encoding="utf-8")
        return {"status": "created", "name": name, "validation_error": _validation_error(ctx, name)}

    @app.get("/api/strategies/{name}")
    def read_strategy(name: str, _: None = Depends(auth)):
        path = _strategy_path(ctx, name)
        if not path.exists():
            raise _error(404, f'Strategy "{name}" was not found.', "not_found")
        return {"name": name, "content": path.read_text(encoding="utf-8"), "validation_error": _validation_error(ctx, name)}

    @app.put("/api/strategies/{name}")
    def save_strategy(name: str, payload: dict[str, Any], _: None = Depends(auth)):
        path = _strategy_path(ctx, name)
        content = payload.get("content")
        if not isinstance(content, str) or not content.strip():
            raise _error(422, "Strategy content cannot be empty.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return {"status": "written", "name": name, "validation_error": _validation_error(ctx, name)}

    @app.post("/api/strategies/{name}/fork")
    def fork_strategy(name: str, payload: dict[str, Any], _: None = Depends(auth)):
        source = _strategy_path(ctx, name)
        if not source.exists():
            raise _error(404, f'Strategy "{name}" was not found.', "not_found")
        new_name = _require_name(str(payload.get("name", "")))
        target = _strategy_path(ctx, new_name)
        if target.exists():
            raise _error(409, f'Strategy "{new_name}" already exists.', "exists")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source.read_text(encoding="utf-8").replace(f"class {name}", f"class {new_name}", 1), encoding="utf-8")
        return {"status": "forked", "name": new_name, "validation_error": _validation_error(ctx, new_name)}

    @app.delete("/api/strategies/{name}")
    def delete_strategy(name: str, _: None = Depends(auth)):
        path = _strategy_path(ctx, name)
        if not path.exists():
            raise _error(404, f'Strategy "{name}" was not found.', "not_found")
        path.unlink()
        try:
            path.parent.rmdir()
        except OSError:
            pass
        return {"status": "deleted", "name": name}

    @app.get("/api/candles")
    def list_candles(_: None = Depends(auth)):
        return {"existing": ctx.candle_db.existing(), "exchanges": list(EXCHANGES)}

    @app.post("/api/candles/import")
    def import_candles(payload: dict[str, Any], _: None = Depends(auth)):
        exchange = payload.get("exchange")
        if exchange not in EXCHANGES:
            raise _error(422, f"Unknown exchange: {exchange}.")
        symbol = str(payload.get("symbol", "")).upper()
        if not re.fullmatch(r"[A-Z0-9]+-[A-Z0-9]+", symbol):
            raise _error(422, "symbol must use BASE-QUOTE format, for example BTC-USDT.")
        start = _date(payload.get("start_date"), "start_date")
        finish = payload.get("finish_date")
        if finish:
            finish = _date(finish, "finish_date")
        try:
            import_id = ctx.importer.start_import(exchange, symbol, start, finish)
        except ValueError as exc:
            raise _error(422, str(exc)) from exc
        return {"status": "started", "import_id": import_id}

    @app.get("/api/candles/import/{import_id}")
    def candle_import_status(import_id: str, _: None = Depends(auth)):
        result = ctx.importer.get_status(import_id)
        if result["status"] == "not_found":
            raise _error(404, "Candle import was not found.", "not_found")
        return result

    @app.post("/api/candles/import/{import_id}/cancel")
    def cancel_candle_import(import_id: str, _: None = Depends(auth)):
        if not ctx.importer.cancel(import_id):
            raise _error(404, "Candle import was not found.", "not_found")
        return {"status": "canceled", "import_id": import_id}

    @app.delete("/api/candles/{exchange}/{symbol}")
    def delete_candles(exchange: str, symbol: str, _: None = Depends(auth)):
        ctx.candle_db.delete(exchange, symbol)
        return {"status": "deleted", "exchange": exchange, "symbol": symbol}

    @app.get("/api/config")
    def get_config(_: None = Depends(auth)):
        return {"config": ctx.config.get()}

    @app.patch("/api/config")
    def update_config(payload: dict[str, Any], _: None = Depends(auth)):
        allowed = {"exchange", "starting_balance", "fee", "type", "futures_leverage",
                   "futures_leverage_mode", "quote_asset", "warm_up_candles", "optimization",
                   "monte_carlo", "significance_test"}
        unknown = set(payload) - allowed
        if unknown:
            raise _error(422, f"Unknown configuration key(s): {', '.join(sorted(unknown))}.")
        if "exchange" in payload and payload["exchange"] not in EXCHANGES:
            raise _error(422, f"Unknown exchange: {payload['exchange']}.")
        for key in ("starting_balance", "fee", "futures_leverage", "warm_up_candles"):
            if key in payload and (not isinstance(payload[key], (int, float)) or isinstance(payload[key], bool) or payload[key] < 0):
                raise _error(422, f"{key} must be a non-negative number.")
        if payload.get("type") not in (None, "spot", "futures"):
            raise _error(422, "type must be either spot or futures.")
        return {"status": "updated", "config": ctx.config.update(payload)}

    @app.get("/api/indicators")
    def indicators(query: str = "", _: None = Depends(auth)):
        names = sorted(name for name in ta.__all__ if query.lower() in name.lower())
        return {"count": len(names), "indicators": names}

    @app.get("/api/indicators/{name}")
    def indicator(name: str, _: None = Depends(auth)):
        import inspect
        fn = getattr(ta, name, None)
        if fn is None or not callable(fn):
            raise _error(404, f'Indicator "{name}" was not found.', "not_found")
        return {"name": name, "signature": f"{name}{inspect.signature(fn)}", "doc": (fn.__doc__ or "").strip()}

    @app.get("/api/sessions/{kind}")
    def list_sessions(kind: str, limit: int = 50, offset: int = 0, query: str | None = None,
                      status: str | None = None, _: None = Depends(auth)):
        if kind not in VALID_KINDS:
            raise _error(404, "Unknown research mode.", "unknown_mode")
        if not 1 <= limit <= 100 or offset < 0:
            raise _error(422, "limit must be 1–100 and offset cannot be negative.")
        if status and status not in {"draft", "running", *TERMINAL}:
            raise _error(422, "Unknown session status.")
        sessions, total = _session_list(ctx, kind, limit, offset, query, status)
        return {"sessions": sessions, "count": len(sessions), "total": total}

    @app.post("/api/sessions/{kind}")
    def create_session(kind: str, payload: dict[str, Any], _: None = Depends(auth)):
        return _new_session(ctx, kind, payload)

    @app.get("/api/session/{session_id}")
    def get_session(session_id: str, _: None = Depends(auth)):
        session = ctx.sessions.get(session_id)
        if session is None:
            raise _error(404, "Session was not found.", "not_found")
        return _session_payload(session)

    @app.post("/api/session/{session_id}/run")
    def run_session(session_id: str, _: None = Depends(auth)):
        session = ctx.sessions.get(session_id)
        if session is None:
            raise _error(404, "Session was not found.", "not_found")
        ctx.runner.run(session_id)
        return _session_payload(ctx.sessions.get(session_id))

    @app.post("/api/session/{session_id}/cancel")
    def cancel_session(session_id: str, _: None = Depends(auth)):
        session = ctx.sessions.get(session_id)
        if session is None:
            raise _error(404, "Session was not found.", "not_found")
        ctx.runner.cancel(session_id)
        return _session_payload(ctx.sessions.get(session_id))

    @app.patch("/api/session/{session_id}")
    def update_session(session_id: str, payload: dict[str, Any], _: None = Depends(auth)):
        session = ctx.sessions.get(session_id)
        if session is None:
            raise _error(404, "Session was not found.", "not_found")
        if "notes" in payload:
            ctx.sessions.update_notes(session_id, str(payload["notes"]))
        return _session_payload(ctx.sessions.get(session_id))

    @app.delete("/api/session/{session_id}")
    def delete_session(session_id: str, _: None = Depends(auth)):
        if ctx.sessions.get(session_id) is None:
            raise _error(404, "Session was not found.", "not_found")
        ctx.sessions.delete(session_id)
        return {"status": "deleted", "session_id": session_id}

    @app.get("/api/session/{session_id}/export")
    def export_session(session_id: str, format: str = "json", _: None = Depends(auth)):
        session = ctx.sessions.get(session_id)
        if session is None:
            raise _error(404, "Session was not found.", "not_found")
        if format == "json":
            return JSONResponse(_session_payload(session), headers={"Content-Disposition": f'attachment; filename="{session_id}.json"'})
        if format != "csv":
            raise _error(422, "format must be json or csv.")
        trades = (session.get("results") or {}).get("trades", [])
        output = io.StringIO()
        if trades:
            writer = csv.DictWriter(output, fieldnames=sorted({key for trade in trades for key in trade}))
            writer.writeheader()
            writer.writerows(trades)
        return Response(output.getvalue(), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{session_id}-trades.csv"'})

    @app.get("/api/live")
    def live_unavailable(_: None = Depends(auth)):
        raise _error(501, "Live and paper trading are not implemented in Terry for safety.", "not_available")

    @app.get("/reports/{session_id}")
    def report(session_id: str, _: None = Depends(auth)):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", session_id):
            raise _error(404, "Report was not found.", "not_found")
        report_path = Path(ctx.reports_dir, f"{session_id}.html")
        if not report_path.is_file():
            raise _error(404, "Report is not available yet.", "not_found")
        return FileResponse(report_path)

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        return FileResponse(_STATIC / "favicon.svg", media_type="image/svg+xml")

    @app.get("/")
    def dashboard_index():
        return FileResponse(_STATIC / "index.html")

    app.mount("/", StaticFiles(directory=_STATIC), name="dashboard-static")
    return app


def run(port: int = 9020, host: str = "127.0.0.1", project_root: str | None = None) -> None:
    """Run the dashboard with Uvicorn."""
    import uvicorn
    uvicorn.run(create_app(project_root), host=host, port=port, log_level="info")
