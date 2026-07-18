"""FastAPI dashboard for the local Terry research engine.

The dashboard deliberately calls the same Context/Runner services used by MCP.  It is a
browser client for local research only: no exchange credentials or live-trading controls are
implemented here.
"""
from __future__ import annotations

import csv
import io
import math
import os
import re
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import Cookie, Depends, FastAPI, Header, HTTPException, Response
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
_SYMBOL = re.compile(r"^[A-Z0-9]+-[A-Z0-9]+$")
_STATIC = Path(__file__).with_name("static")
_AUTH_COOKIE = "terry_dashboard_session"
_OBJECTIVES = {
    "sharpe", "sharpe_ratio", "sortino", "sortino_ratio", "calmar", "calmar_ratio",
    "omega", "omega_ratio", "serenity", "serenity_index", "smart sharpe",
    "smart sortino", "net_profit_percentage",
}
_SESSION_KINDS = ("backtest", "optimization", "monte_carlo", "significance_test")
_ENGINE_CONFIG_KEYS = {
    "starting_balance", "fee", "type", "futures_leverage", "futures_leverage_mode",
    "quote_asset", "warm_up_candles",
}


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


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    """Parse an API integer without accepting booleans, fractions, or ambiguous text."""
    if isinstance(value, bool):
        raise _error(422, f"{label} must be an integer of at least {minimum}.")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _error(422, f"{label} must be an integer of at least {minimum}.") from exc
    if isinstance(value, float) and not value.is_integer():
        raise _error(422, f"{label} must be an integer of at least {minimum}.")
    if isinstance(value, str) and not re.fullmatch(r"[+-]?\d+", value.strip()):
        raise _error(422, f"{label} must be an integer of at least {minimum}.")
    if parsed < minimum:
        raise _error(422, f"{label} must be an integer of at least {minimum}.")
    return parsed


def _number(value: Any, label: str, minimum: float = 0, *, exclusive: bool = False,
            maximum: float | None = None) -> float:
    if isinstance(value, bool):
        raise _error(422, f"{label} must be a finite number.")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _error(422, f"{label} must be a finite number.") from exc
    if not math.isfinite(parsed) or (parsed <= minimum if exclusive else parsed < minimum):
        comparator = "greater than" if exclusive else "at least"
        raise _error(422, f"{label} must be {comparator} {minimum}.")
    if maximum is not None and parsed > maximum:
        raise _error(422, f"{label} must be no greater than {maximum}.")
    return parsed


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise _error(422, f"{label} must be true or false.")
    return value


def _symbol(value: Any) -> str:
    symbol = str(value or "BTC-USDT").upper()
    if not _SYMBOL.fullmatch(symbol):
        raise _error(422, "symbol must use BASE-QUOTE format, for example BTC-USDT.")
    return symbol


def _validate_engine_config(payload: dict[str, Any]) -> dict[str, Any]:
    unknown = set(payload) - _ENGINE_CONFIG_KEYS
    if unknown:
        raise _error(422, f"Unknown engine configuration key(s): {', '.join(sorted(unknown))}.")
    out = dict(payload)
    if "starting_balance" in out:
        out["starting_balance"] = _number(out["starting_balance"], "starting_balance", exclusive=True)
    if "fee" in out:
        out["fee"] = _number(out["fee"], "fee", maximum=1)
    if "futures_leverage" in out:
        out["futures_leverage"] = _number(out["futures_leverage"], "futures_leverage", 1)
    if "warm_up_candles" in out:
        out["warm_up_candles"] = _integer(out["warm_up_candles"], "warm_up_candles")
    if out.get("type") not in (None, "spot", "futures"):
        raise _error(422, "type must be either spot or futures.")
    if out.get("futures_leverage_mode") not in (None, "cross", "isolated"):
        raise _error(422, "futures_leverage_mode must be either cross or isolated.")
    if "quote_asset" in out:
        quote = str(out["quote_asset"]).upper()
        if not re.fullmatch(r"[A-Z0-9]{2,12}", quote):
            raise _error(422, "quote_asset must contain 2–12 letters or numbers.")
        out["quote_asset"] = quote
    return out


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
    if isinstance(result.get("results"), dict):
        # Dedicated MCP retrieval exposes downsampled Monte Carlo curves. Avoid
        # transferring every scenario on normal dashboard polling/history calls.
        result["results"].pop("equity_curves", None)
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
    if not (query or "").strip() and not status:
        rows = ctx.sessions.list(kind, limit, offset)
        return [_session_payload(row) for row in rows], ctx.sessions.count(kind)
    rows = ctx.sessions.list(kind, None)
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
        "symbol": _symbol(payload.get("symbol")),
        "exchange": exchange,
        "timeframe": timeframe,
        "start_date": start_date,
        "finish_date": finish_date,
    }
    overrides = payload.get("config")
    if overrides is not None:
        if not isinstance(overrides, dict):
            raise _error(422, "config must be a JSON object.")
        state["config"] = _validate_engine_config(overrides)
    if "hyperparameters" in payload:
        if not isinstance(payload["hyperparameters"], dict):
            raise _error(422, "hyperparameters must be a JSON object.")
        state["hyperparameters"] = payload["hyperparameters"]
    return state


def _new_session(ctx: TerryContext, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    if kind not in VALID_KINDS:
        raise _error(404, "Unknown research mode.", "unknown_mode")
    allowed = {"strategy", "exchange", "symbol", "timeframe", "start_date", "finish_date",
               "config", "hyperparameters", "notes", "start"}
    allowed |= {
        "significance_test": {"n_simulations", "hypothesis", "rationale"},
        "monte_carlo": {"num_scenarios", "run_candles", "run_trades"},
        "optimization": {"n_trials", "train_test_split", "objective"},
        "backtest": {"debug_mode", "export_csv", "export_json", "export_chart",
                     "export_tradingview", "fast_mode", "benchmark"},
    }[kind]
    unknown = set(payload) - allowed
    if unknown:
        raise _error(422, f"Unknown session field(s): {', '.join(sorted(unknown))}.")
    for field in ("notes", "hypothesis", "rationale"):
        if field in payload and payload[field] is not None and not isinstance(payload[field], str):
            raise _error(422, f"{field} must be text.")
    notes = payload.get("notes") or ""
    if len(notes) > 20_000:
        raise _error(422, "notes cannot exceed 20,000 characters.")
    state = _base_state(ctx, payload)
    if kind == "significance_test":
        simulations = _integer(payload.get("n_simulations", 2_000), "n_simulations", 2_000)
        state.update({"n_simulations": simulations, "hypothesis": str(payload.get("hypothesis") or ""),
                      "rationale": str(payload.get("rationale") or "")})
    elif kind == "monte_carlo":
        scenarios = _integer(payload.get("num_scenarios", 200), "num_scenarios", 1)
        run_candles = _boolean(payload.get("run_candles", True), "run_candles")
        run_trades = _boolean(payload.get("run_trades", False), "run_trades")
        if not run_candles and not run_trades:
            raise _error(422, "Enable candle resampling, trade-order shuffling, or both.")
        state.update({"num_scenarios": scenarios, "run_candles": run_candles,
                      "run_trades": run_trades})
    elif kind == "optimization":
        trials = _integer(payload.get("n_trials", 100), "n_trials", 1)
        split = _number(payload.get("train_test_split", 0.75), "train_test_split", 0.1, exclusive=True)
        if split >= 0.9:
            raise _error(422, "train_test_split must be greater than 0.1 and less than 0.9.")
        objective = str(payload.get("objective", "sharpe_ratio"))
        if objective not in _OBJECTIVES:
            raise _error(422, f"objective must be one of: {', '.join(sorted(_OBJECTIVES))}.")
        state.update({"objective": objective, "n_trials": trials,
                      "train_test_split": split})
    elif kind == "backtest":
        for field, default in {
            "debug_mode": False, "export_csv": False, "export_json": False,
            "export_chart": True, "export_tradingview": False,
            "fast_mode": True, "benchmark": True,
        }.items():
            state[field] = _boolean(payload.get(field, default), field)
    start = _boolean(payload.get("start", True), "start")
    sid = ctx.sessions.create(kind, state, notes=notes)
    session = ctx.sessions.get(sid)
    if start:
        ctx.runner.run(sid)
        session = ctx.sessions.get(sid)
    return _session_payload(session)


def create_app(project_root: str | None = None) -> FastAPI:
    """Create the local dashboard app, isolated to one Terry project root."""
    ctx = set_context(TerryContext(project_root))
    password = os.environ.get("TERRY_DASHBOARD_PASSWORD", "")
    tokens: set[str] = set()

    app = FastAPI(title="Terry Dashboard", version=__version__)

    @app.middleware("http")
    async def security_headers(request, call_next):
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; "
            "frame-ancestors 'none'; form-action 'self'"
        )
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(HTTPException)
    async def handle_http_error(_, exc: HTTPException):
        detail = exc.detail if isinstance(exc.detail, dict) else {"error": "request_failed", "message": str(exc.detail)}
        return JSONResponse(status_code=exc.status_code, content=detail)

    def request_token(authorization: str | None, session_cookie: str | None) -> str | None:
        bearer = (authorization or "").removeprefix("Bearer ")
        if bearer in tokens:
            return bearer
        return session_cookie if session_cookie in tokens else None

    def auth(authorization: str | None = Header(default=None),
             session_cookie: str | None = Cookie(default=None, alias=_AUTH_COOKIE)) -> str | None:
        if not password:
            return None
        token = request_token(authorization, session_cookie)
        if token is None:
            raise _error(401, "Sign in to continue.", "unauthorized")
        return token

    @app.get("/api/auth/status")
    def auth_status(authorization: str | None = Header(default=None),
                    session_cookie: str | None = Cookie(default=None, alias=_AUTH_COOKIE)):
        return {"auth_required": bool(password),
                "authenticated": not password or request_token(authorization, session_cookie) is not None}

    @app.post("/api/auth/login")
    def login(payload: dict[str, Any], response: Response):
        if password and not secrets.compare_digest(str(payload.get("password", "")), password):
            raise _error(401, "Incorrect password.", "invalid_credentials")
        if not password:
            return {"auth_token": "", "auth_required": False}
        token = secrets.token_urlsafe(32)
        tokens.add(token)
        response.set_cookie(_AUTH_COOKIE, token, httponly=True, samesite="strict", path="/")
        return {"auth_token": token, "auth_required": bool(password)}

    @app.post("/api/auth/logout")
    def logout(response: Response, token: str | None = Depends(auth)):
        if token:
            tokens.discard(token)
        response.delete_cookie(_AUTH_COOKIE, path="/", samesite="strict")
        return {"status": "signed_out"}

    @app.get("/api/status")
    def status(_: None = Depends(auth)):
        return {
            "name": "Terry", "version": __version__, "auth_required": bool(password),
            "project_root": ctx.project_root, "indicators_available": len(ta.__all__),
            "supported_exchanges": list(EXCHANGES), "datasets": ctx.candle_db.existing(),
            "sessions": {kind: ctx.sessions.count(kind) for kind in _SESSION_KINDS},
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
        content = payload.get("content")
        if content is not None and not isinstance(content, str):
            raise _error(422, "Strategy content must be text.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content or _strategy_template(name), encoding="utf-8")
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
        if not path.exists():
            raise _error(404, f'Strategy "{name}" was not found.', "not_found")
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
        symbol = _symbol(payload.get("symbol"))
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
        if exchange not in EXCHANGES:
            raise _error(422, f"Unknown exchange: {exchange}.")
        symbol = _symbol(symbol)
        if ctx.candle_db.coverage(exchange, symbol) is None:
            raise _error(404, "Candle dataset was not found.", "not_found")
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
        scalar = _validate_engine_config({key: value for key, value in payload.items()
                                          if key in _ENGINE_CONFIG_KEYS})
        validated = {**payload, **scalar}
        nested_specs = {
            "optimization": {"objective", "n_trials", "train_test_split"},
            "monte_carlo": {"num_scenarios", "run_candles", "run_trades"},
            "significance_test": {"n_simulations"},
        }
        for key, keys in nested_specs.items():
            if key in validated:
                value = validated[key]
                if not isinstance(value, dict):
                    raise _error(422, f"{key} must be a JSON object.")
                nested_unknown = set(value) - keys
                if nested_unknown:
                    raise _error(422, f"Unknown {key} key(s): {', '.join(sorted(nested_unknown))}.")
        if "optimization" in validated:
            value = dict(validated["optimization"])
            if "objective" in value and value["objective"] not in _OBJECTIVES:
                raise _error(422, f"objective must be one of: {', '.join(sorted(_OBJECTIVES))}.")
            if "n_trials" in value:
                value["n_trials"] = _integer(value["n_trials"], "n_trials", 1)
            if "train_test_split" in value:
                value["train_test_split"] = _number(value["train_test_split"], "train_test_split", 0.1,
                                                       exclusive=True)
                if value["train_test_split"] >= 0.9:
                    raise _error(422, "train_test_split must be greater than 0.1 and less than 0.9.")
            validated["optimization"] = value
        if "monte_carlo" in validated:
            value = dict(validated["monte_carlo"])
            if "num_scenarios" in value:
                value["num_scenarios"] = _integer(value["num_scenarios"], "num_scenarios", 1)
            for key in ("run_candles", "run_trades"):
                if key in value:
                    value[key] = _boolean(value[key], key)
            effective = {**ctx.config.get()["monte_carlo"], **value}
            if not effective["run_candles"] and not effective["run_trades"]:
                raise _error(422, "Enable candle resampling, trade-order shuffling, or both.")
            validated["monte_carlo"] = value
        if "significance_test" in validated:
            value = dict(validated["significance_test"])
            if "n_simulations" in value:
                value["n_simulations"] = _integer(value["n_simulations"], "n_simulations", 2_000)
            validated["significance_test"] = value
        return {"status": "updated", "config": ctx.config.update(validated)}

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
        if session["status"] == "running":
            raise _error(409, "Session is already running.", "already_running")
        started = ctx.runner.run(session_id)
        if started.get("error") == "worker_active":
            raise _error(409, "The previous worker is still stopping; try again shortly.", "worker_active")
        return _session_payload(ctx.sessions.get(session_id))

    @app.post("/api/session/{session_id}/cancel")
    def cancel_session(session_id: str, _: None = Depends(auth)):
        session = ctx.sessions.get(session_id)
        if session is None:
            raise _error(404, "Session was not found.", "not_found")
        if session["status"] != "running":
            raise _error(409, "Only a running session can be canceled.", "not_running")
        ctx.runner.cancel(session_id)
        return _session_payload(ctx.sessions.get(session_id))

    @app.patch("/api/session/{session_id}")
    def update_session(session_id: str, payload: dict[str, Any], _: None = Depends(auth)):
        session = ctx.sessions.get(session_id)
        if session is None:
            raise _error(404, "Session was not found.", "not_found")
        unknown = set(payload) - {"notes"}
        if unknown:
            raise _error(422, f"Unknown session field(s): {', '.join(sorted(unknown))}.")
        if "notes" in payload:
            notes = str(payload["notes"] or "")
            if len(notes) > 20_000:
                raise _error(422, "notes cannot exceed 20,000 characters.")
            ctx.sessions.update_notes(session_id, notes)
        return _session_payload(ctx.sessions.get(session_id))

    @app.delete("/api/session/{session_id}")
    def delete_session(session_id: str, _: None = Depends(auth)):
        session = ctx.sessions.get(session_id)
        if session is None:
            raise _error(404, "Session was not found.", "not_found")
        if session["status"] == "running":
            raise _error(409, "Cancel the running session before deleting it.", "session_running")
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
