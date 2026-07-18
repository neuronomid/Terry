"""Shared helpers for session-based MCP tools (backtest / significance / monte_carlo / optimization)."""
import json
import os
import re
from datetime import datetime
from pathlib import Path

from ... import helpers as jh
from ...context import get_context

TERMINAL = {"finished", "stopped", "terminated", "canceled"}


def default_cpu_cores():
    """Match Jesse MCP's conservative local default: all-but-one, capped at four."""
    return max(1, min((os.cpu_count() or 2) - 1, 4))


def _default_dates(start_date, finish_date):
    if finish_date is None:
        # yesterday (avoid future-date errors)
        finish_date = jh.timestamp_to_date(jh.today_to_timestamp() - 86_400_000)
    if start_date is None:
        # default: ~1 year before finish
        finish_ts = jh.date_to_timestamp(finish_date)
        start_date = jh.timestamp_to_date(finish_ts - 365 * 86_400_000)
    return start_date, finish_date


def build_base_state(strategy, symbol, timeframe, exchange, start_date, finish_date, config_json):
    ctx = get_context()
    cfg = ctx.config.get()
    exchange = exchange or cfg["exchange"]
    from ...data.binance import EXCHANGES
    if exchange not in EXCHANGES:
        return None, f"❌ Unknown exchange: {exchange}."
    symbol = str(symbol or "BTC-USDT").upper()
    if not re.fullmatch(r"[A-Z0-9]+-[A-Z0-9]+", symbol):
        return None, "❌ symbol must use BASE-QUOTE format, for example BTC-USDT."
    timeframe = timeframe or "4h"
    try:
        jh.timeframe_to_one_minutes(timeframe)
    except ValueError as exc:
        return None, f"❌ {exc}"
    start_date, finish_date = _default_dates(start_date, finish_date)
    try:
        if jh.date_to_timestamp(finish_date) <= jh.date_to_timestamp(start_date):
            return None, "❌ finish_date must be after start_date."
    except (TypeError, ValueError) as exc:
        return None, f"❌ Dates must use YYYY-MM-DD: {exc}"
    state = {
        "strategy": strategy, "symbol": symbol, "timeframe": timeframe,
        "exchange": exchange, "start_date": start_date, "finish_date": finish_date,
    }
    if config_json:
        try:
            state["config"] = json.loads(config_json) if isinstance(config_json, str) else config_json
        except json.JSONDecodeError as e:
            return None, f"❌ Invalid config JSON: {e}"
        if not isinstance(state["config"], dict):
            return None, "❌ config must be a JSON object."
    return state, None


def parse_json_list(value, field):
    """Parse an MCP JSON-array argument while also accepting an already-decoded list."""
    if value is None:
        return [], None
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError as e:
        return None, f"❌ Invalid {field} JSON: {e}"
    if not isinstance(parsed, list):
        return None, f"❌ {field} must be a JSON array."
    return parsed, None


def build_routes_state(strategy, symbol, timeframe, exchange, start_date, finish_date,
                       config_json, routes_json=None, data_routes_json=None):
    """Build a state from either Terry's shorthand route or Jesse-style route arrays."""
    routes, err = parse_json_list(routes_json, "routes")
    if err:
        return None, err
    data_routes, err = parse_json_list(data_routes_json, "data_routes")
    if err:
        return None, err
    if routes:
        required = {"strategy", "symbol", "timeframe"}
        for index, route in enumerate(routes):
            if not isinstance(route, dict) or not required.issubset(route):
                return None, (f"❌ routes[{index}] must contain strategy, symbol, and "
                              "timeframe.")
            if not isinstance(route["strategy"], str) or not re.fullmatch(
                    r"[A-Za-z][A-Za-z0-9_]{0,79}", route["strategy"]):
                return None, f"❌ routes[{index}].strategy is not a valid strategy name."
        first = routes[0]
        strategy = first["strategy"]
        symbol = first["symbol"]
        timeframe = first["timeframe"]
        exchange = exchange or first.get("exchange")
    for index, route in enumerate(data_routes):
        if not isinstance(route, dict) or not {"symbol", "timeframe"}.issubset(route):
            return None, (f"❌ data_routes[{index}] must contain symbol and "
                          "timeframe.")
    if not isinstance(strategy, str) or not re.fullmatch(
            r"[A-Za-z][A-Za-z0-9_]{0,79}", strategy):
        return None, "❌ A valid strategy name is required when routes is empty."
    state, err = build_base_state(strategy, symbol, timeframe, exchange,
                                  start_date, finish_date, config_json)
    if err:
        return None, err
    if routes:
        normalized_routes = []
        pairs = set()
        for index, route in enumerate(routes):
            normalized = {**route, "exchange": route.get("exchange") or state["exchange"],
                          "symbol": str(route["symbol"]).upper()}
            if normalized["exchange"] != state["exchange"]:
                return None, "❌ All routes in one run must use the selected exchange."
            if not re.fullmatch(r"[A-Z0-9]+-[A-Z0-9]+", normalized["symbol"]):
                return None, f"❌ routes[{index}].symbol must use BASE-QUOTE format."
            try:
                jh.timeframe_to_one_minutes(normalized["timeframe"])
            except ValueError as exc:
                return None, f"❌ routes[{index}]: {exc}"
            pair = (normalized["exchange"], normalized["symbol"])
            if pair in pairs:
                return None, "❌ Two trading routes cannot use the same exchange-symbol pair."
            pairs.add(pair)
            normalized_routes.append(normalized)
        state["routes"] = normalized_routes
    if data_routes:
        normalized_data_routes = []
        for index, route in enumerate(data_routes):
            normalized = {**route, "exchange": route.get("exchange") or state["exchange"],
                          "symbol": str(route["symbol"]).upper()}
            if normalized["exchange"] != state["exchange"]:
                return None, "❌ All routes in one run must use the selected exchange."
            if not re.fullmatch(r"[A-Z0-9]+-[A-Z0-9]+", normalized["symbol"]):
                return None, f"❌ data_routes[{index}].symbol must use BASE-QUOTE format."
            try:
                jh.timeframe_to_one_minutes(normalized["timeframe"])
            except ValueError as exc:
                return None, f"❌ data_routes[{index}]: {exc}"
            normalized_data_routes.append(normalized)
        state["data_routes"] = normalized_data_routes
    return state, None


def _error(code, message, **extra):
    return {"status": "error", "error": code, "message": message, **extra}


def _kind_label(kind):
    return {
        "backtest": "Backtest",
        "significance_test": "Significance test",
        "monte_carlo": "Monte Carlo",
        "optimization": "Optimization",
    }[kind]


def _dashboard_url(session_id):
    path = Path(get_context().reports_dir, f"{session_id}.html").resolve()
    return path.as_uri()


def _routes_for_state(state):
    return state.get("routes") or [{
        "exchange": state["exchange"], "strategy": state["strategy"],
        "symbol": state["symbol"], "timeframe": state["timeframe"],
    }]


def _draft_form(kind, state, session_id):
    """Build the mode-specific form persisted by Jesse's dashboard stores."""
    shared = {
        "exchange": state["exchange"], "routes": _routes_for_state(state),
        "data_routes": state.get("data_routes", []),
    }
    if kind == "optimization":
        config = state.get("config") or {}
        return {
            "id": session_id or state.get("id"), **shared,
            "training_start_date": state.get("training_start_date", state.get("start_date")),
            "training_finish_date": state.get("training_finish_date", state.get("finish_date")),
            "testing_start_date": state.get("testing_start_date"),
            "testing_finish_date": state.get("testing_finish_date"),
            "optimal_total": state.get("optimal_total", 50),
            "fast_mode": state.get("fast_mode", True),
            "cpu_cores": state.get("cpu_cores", default_cpu_cores()),
            "objective_function": state.get("objective_function", "sharpe"),
            "trials": state.get("trials", state.get("n_trials", 200)),
            "best_candidates_count": state.get("best_candidates_count", 20),
            "warm_up_candles": config.get("warm_up_candles", 210),
        }
    form = {
        **shared, "start_date": state["start_date"],
        "finish_date": state["finish_date"],
    }
    if kind == "backtest":
        form.update({key: state.get(key, default) for key, default in {
            "debug_mode": False, "export_csv": False, "export_json": False,
            "export_chart": True, "export_tradingview": False,
            "fast_mode": True, "benchmark": True,
        }.items()})
    elif kind == "significance_test":
        form.update({
            "id": session_id or state.get("id"),
            "n_simulations": state.get("n_simulations", 2000),
            "random_seed": state.get("random_seed"),
        })
    elif kind == "monte_carlo":
        form.update({
            "id": session_id or state.get("id"),
            "num_scenarios": state.get("num_scenarios", 200),
            "run_trades": state.get("run_trades", False),
            "run_candles": state.get("run_candles", True),
            "fast_mode": state.get("fast_mode", True),
            "cpu_cores": state.get("cpu_cores", default_cpu_cores()),
            "pipeline_type": state.get("pipeline_type", "moving_block_bootstrap"),
            "pipeline_params": state.get("pipeline_params") or {"batch_size": 10_080},
        })
    return form


def _draft_results(kind, state, results):
    finished = bool(results)
    if kind in {"significance_test", "monte_carlo"}:
        payload = {"alert": {"message": "", "type": ""}}
    elif kind == "optimization":
        payload = {
            "showResults": finished, "executing": False, "logsModal": False,
            "status": "", "progressbar": {
                "current": 100 if finished else 0,
                "estimated_remaining_seconds": 0,
            },
            "routes_info": [], "best_candidates": [], "metrics": [],
            "generalInfo": [], "selectedObjectiveMetric": "", "infoLogs": "",
            "info": [], "exception": {"error": "", "traceback": ""},
            "alert": {"message": "", "type": ""},
        }
    else:
        routes = _routes_for_state(state)
        first = routes[0] if routes else {}
        payload = {
            "showResults": finished, "executing": False, "logsModal": False,
            "progressbar": {"current": 100 if finished else 0,
                            "estimated_remaining_seconds": 0},
            "routes_info": [[
                {"value": route.get("symbol", ""), "style": ""},
                {"value": route.get("timeframe", ""), "style": ""},
                {"value": route.get("strategy", ""), "style": ""},
            ] for route in routes],
            "metrics": {}, "hyperparameters": [],
            "generalInfo": {"title": None, "description": None},
            "infoLogs": "", "exception": {"error": "", "traceback": ""},
            "charts": {"equity_curve": []},
            "selectedRoute": {key: first.get(key, "")
                              for key in ("symbol", "timeframe", "strategy")},
            "alert": {"message": "", "type": ""}, "info": [], "trades": [],
        }
    if results:
        payload["result"] = results
    return payload


def _draft_state(kind, state, results=None, session_id=None):
    """Expose Jesse's dashboard state shape while storing Terry's flat state."""
    return {
        "form": _draft_form(kind, state, session_id),
        "results": _draft_results(kind, state, results),
    }


def _flat_state(value):
    if not isinstance(value, dict):
        raise ValueError("state must be a JSON object")
    if "form" in value:
        if not isinstance(value["form"], dict):
            raise ValueError("state.form must be a JSON object")
        return {key: val for key, val in value["form"].items() if key != "id"}
    return value


def _strategy_codes(state):
    ctx = get_context()
    codes = {}
    for route in _routes_for_state(state):
        path = Path(ctx.strategies_dir, route["strategy"], "__init__.py")
        try:
            source = path.read_text(encoding="utf-8")
        except OSError:
            continue
        codes[f'{route.get("exchange", state["exchange"])}-{route["symbol"]}'] = source
    return codes


def _default_title(kind, state):
    route = _routes_for_state(state)[0]
    prefix = {
        "backtest": "MCP Backtest", "significance_test": "MCP Rule Significance Test",
        "monte_carlo": "MCP Monte Carlo", "optimization": "MCP Optimization",
    }[kind]
    route_count = len(_routes_for_state(state))
    suffix = f" +{route_count - 1}" if route_count > 1 and kind != "significance_test" else ""
    return (f'{prefix}: {route["strategy"]} on {route["symbol"]} '
            f'{route["timeframe"]}{suffix}')


def _normalize_title(kind, title, state):
    if not title:
        return _default_title(kind, state)
    title = title.strip()
    lowered = title.lower()
    already_prefixed = "mcp" in lowered and (
        kind == "backtest" or
        (kind == "significance_test" and "significance" in lowered) or
        (kind == "monte_carlo" and "monte" in lowered) or
        (kind == "optimization" and "optim" in lowered)
    )
    if already_prefixed:
        return title
    prefix = {
        "backtest": "MCP Backtest", "significance_test": "MCP Rule Significance Test",
        "monte_carlo": "MCP Monte Carlo", "optimization": "MCP Optimization",
    }[kind]
    return f"{prefix}: {title}"


def _default_description(kind, state):
    start_date = (state.get("training_start_date") or state.get("start_date"))
    finish_date = (state.get("testing_finish_date") or state.get("finish_date"))
    return (f'{_kind_label(kind)} for {len(_routes_for_state(state))} trading route(s) '
            f'from {start_date} to {finish_date} on '
            f'{state["exchange"]}.')


def create_draft(kind, state, notes="", title=None, description=None):
    ctx = get_context()
    # validate every trading strategy exists on disk
    from ...loader import strategy_exists
    strategy_names = {route["strategy"] for route in state.get("routes", [])}
    strategy_names.add(state["strategy"])
    for strategy_name in strategy_names:
        if not strategy_exists(ctx.strategies_dir, strategy_name):
            return _error(
                "strategy_not_found",
                f'Strategy "{strategy_name}" not found. Create it first with create_strategy().')
    codes = _strategy_codes(state)
    metadata = {
        "title": _normalize_title(kind, title, state),
        "description": description or notes or _default_description(kind, state),
        "strategy_codes": codes,
    }
    note_text = metadata["description"]
    sid = ctx.sessions.create(
        kind, state, notes=note_text, notes_metadata=metadata)
    payload = {
        "status": "success", "session_status": "draft", "session_id": sid,
        "draft_state": _draft_state(kind, state, session_id=sid), "state": state,
        "notes": {
            "title": metadata["title"], "description": metadata["description"],
            "strategy_code_keys": list(codes),
            "strategy_codes_captured": len(codes),
        },
        "dashboard_url": _dashboard_url(sid),
        "message": f"{_kind_label(kind)} draft created with ID: {sid}",
    }
    if kind == "backtest":
        payload["backtest_id"] = sid
    return payload


def update_draft(kind, session_id, state_str):
    ctx = get_context()
    session = ctx.sessions.get(session_id)
    if session is None:
        return _error("not_found", f"Session {session_id} not found", session_id=session_id)
    if session["kind"] != kind:
        return _error("wrong_kind", f"Session {session_id} is a {session['kind']}.")
    try:
        new_state = json.loads(state_str) if isinstance(state_str, str) else state_str
    except json.JSONDecodeError as e:
        return _error("Invalid JSON format", "Failed to parse state JSON", details=str(e))
    try:
        new_state = _flat_state(new_state)
    except ValueError as exc:
        return _error("invalid_state", str(exc))
    merged = {**session["state"], **new_state}
    try:
        ctx.sessions.update_state(session_id, merged)
    except ValueError as e:
        return _error("not_draft", str(e))
    payload = {
        "status": "success", "session_status": "draft", "session_id": session_id,
        "draft_state": _draft_state(kind, merged, session_id=session_id), "state": merged,
        "message": f"{_kind_label(kind)} draft updated successfully",
    }
    if kind == "backtest":
        payload["backtest_id"] = session_id
    return payload


def update_notes(session_id, title=None, description=None, strategy_codes=None, notes=None):
    ctx = get_context()
    session = ctx.sessions.get(session_id)
    if session is None:
        return _error("not_found", f"Session {session_id} not found", session_id=session_id)
    codes = None
    if strategy_codes is not None:
        try:
            codes = json.loads(strategy_codes) if isinstance(strategy_codes, str) else strategy_codes
        except json.JSONDecodeError as exc:
            return _error("Invalid JSON format", "Failed to parse strategy_codes JSON",
                          details=str(exc))
        if not isinstance(codes, dict):
            return _error("invalid_strategy_codes",
                          "strategy_codes must be a JSON object string")
    metadata = dict(session.get("notes_metadata") or {})
    if title is not None:
        metadata["title"] = title
    if description is not None:
        metadata["description"] = description
    if codes is not None:
        metadata["strategy_codes"] = codes
    if notes is not None and description is None:
        metadata["description"] = notes
    note_text = metadata.get("description", session.get("notes") or "")
    ctx.sessions.update_notes_metadata(session_id, metadata, notes=note_text)
    captured = metadata.get("strategy_codes") or {}
    return {
        "status": "success", "session_id": session_id,
        "title": metadata.get("title"), "description": metadata.get("description"),
        "strategy_code_keys": list(captured),
        "strategy_codes_captured": len(captured), "notes": note_text,
        "message": f"{_kind_label(session['kind'])} session notes updated successfully",
    }


def get_session(session_id, include_results=True):
    ctx = get_context()
    s = ctx.sessions.get(session_id)
    if s is None:
        message = f"Session {session_id} not found"
        return {"status": "error", "data": None, "error": message,
                "error_code": "not_found", "message": message,
                "session_id": session_id}
    created_at = jh.timestamp_to_time(s["created_at"]) if s["created_at"] else None
    updated_at = jh.timestamp_to_time(s["updated_at"]) if s["updated_at"] else None
    out = {
        "session_id": s["id"], "kind": s["kind"], "status": s["status"],
        "progress": s["progress"], "state": s["state"], "notes": s["notes"],
        "notes_metadata": s.get("notes_metadata") or {},
        "created_at": created_at, "updated_at": updated_at,
        "dashboard_url": _dashboard_url(s["id"]), "error": None,
        "message": f"{_kind_label(s['kind'])} session retrieved successfully",
    }
    if include_results and s["results"] is not None:
        out["results"] = s["results"]
        out["dashboard_url"] = s["results"].get("dashboard_url") or out["dashboard_url"]
    jesse_session = {
        "id": s["id"], "status": s["status"], "progress": s["progress"],
        "state": _draft_state(
            s["kind"], s["state"], s.get("results"), session_id=s["id"]),
        "results": s.get("results"), "notes": s.get("notes_metadata") or {},
        "created_at": created_at, "updated_at": updated_at,
    }
    out["data"] = {"session": jesse_session}
    return out


def _date_matches(timestamp_ms, date_filter):
    if not date_filter:
        return True
    value = datetime.fromtimestamp(timestamp_ms / 1000)
    now = datetime.now()
    if date_filter == "today":
        return value.date() == now.date()
    if date_filter == "this_week":
        return value.isocalendar()[:2] == now.isocalendar()[:2]
    if date_filter == "this_month":
        return (value.year, value.month) == (now.year, now.month)
    return False


def list_sessions(kind, limit=50, offset=0, title_search=None,
                  status_filter=None, date_filter=None):
    ctx = get_context()
    try:
        limit, offset = max(1, int(limit)), max(0, int(offset))
    except (TypeError, ValueError):
        return _error("invalid_pagination", "limit and offset must be integers")
    if date_filter not in {None, "", "today", "this_week", "this_month"}:
        return _error("invalid_date_filter",
                      "date_filter must be today, this_week, or this_month")
    rows = ctx.sessions.list(kind, None)
    query = (title_search or "").lower().strip()
    filtered = []
    for row in rows:
        metadata = row.get("notes_metadata") or {}
        title = metadata.get("title") or ""
        if query and query not in title.lower():
            continue
        if status_filter and row["status"] != status_filter:
            continue
        if not _date_matches(row["created_at"], date_filter):
            continue
        filtered.append(row)
    count = len(filtered)
    items = []
    for row in filtered[offset:offset + limit]:
        items.append({
            "id": row["id"], "session_id": row["id"], "status": row["status"],
            "progress": row["progress"], "title": (row.get("notes_metadata") or {}).get("title"),
            "symbol": row["state"].get("symbol"),
            "timeframe": row["state"].get("timeframe"),
            "strategy": row["state"].get("strategy"), "state": row["state"],
            "created_at": jh.timestamp_to_time(row["created_at"]) if row["created_at"] else None,
            "updated_at": jh.timestamp_to_time(row["updated_at"]) if row["updated_at"] else None,
        })
    return {"status": "success", "sessions": items, "count": count,
            "message": f"Retrieved {count} {_kind_label(kind).lower()} session(s)"}


def run_session(session_id, kind):
    ctx = get_context()
    s = ctx.sessions.get(session_id)
    if s is None:
        return _error("not_found", f"Session {session_id} not found", session_id=session_id)
    if s["kind"] != kind:
        return _error("wrong_kind", f"Session {session_id} is a {s['kind']}.")
    if s["status"] == "running":
        return _error("already_running", f"Session {session_id} is already running.",
                      session_id=session_id)
    if s["status"] not in ("draft", "stopped", "terminated", "canceled"):
        # allow re-running finished by resetting to draft-like start
        pass
    started = ctx.runner.run(session_id)
    if started.get("error"):
        return started
    payload = {
        "status": "started", "session_id": session_id,
        "dashboard_url": _dashboard_url(session_id),
        "message": f"{_kind_label(kind)} started. Poll get_{_poll_name(kind)}_session(session_id) until terminal.",
    }
    if kind == "backtest":
        payload["backtest_id"] = session_id
    return payload


def _poll_name(kind):
    return {"backtest": "backtest", "significance_test": "significance_test",
            "monte_carlo": "monte_carlo", "optimization": "optimization"}[kind]


def cancel_session(session_id, kind, new_status="canceled"):
    ctx = get_context()
    s = ctx.sessions.get(session_id)
    if s is None:
        return _error("not_found", f"Session {session_id} not found", session_id=session_id)
    if s["kind"] != kind:
        return _error("wrong_kind", f"Session {session_id} is a {s['kind']}.")
    if s["status"] != "running":
        return _error("not_running", f"Session {session_id} is not running.",
                      session_status=s["status"], session_id=session_id)
    result = ctx.runner.cancel(session_id, new_status)
    if result.get("error"):
        return result
    return {"status": "success", "session_status": new_status,
            "session_id": session_id,
            "message": f"{_kind_label(kind)} session {session_id} {new_status}."}


def purge_sessions(kind, days_old=None):
    ctx = get_context()
    n = ctx.sessions.purge(kind, days_old)
    return {"status": "success", "purged": n, "count": n, "kind": kind,
            "message": f"Purged {n} {_kind_label(kind).lower()} session(s)"}
