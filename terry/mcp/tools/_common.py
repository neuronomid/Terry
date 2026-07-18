"""Shared helpers for session-based MCP tools (backtest / significance / monte_carlo / optimization)."""
import json

from ... import helpers as jh
from ...context import get_context

TERMINAL = {"finished", "stopped", "terminated", "canceled"}


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
    symbol = symbol or "BTC-USDT"
    timeframe = timeframe or "4h"
    start_date, finish_date = _default_dates(start_date, finish_date)
    state = {
        "strategy": strategy, "symbol": symbol, "timeframe": timeframe,
        "exchange": exchange, "start_date": start_date, "finish_date": finish_date,
    }
    if config_json:
        try:
            state["config"] = json.loads(config_json) if isinstance(config_json, str) else config_json
        except json.JSONDecodeError as e:
            return None, f"❌ Invalid config JSON: {e}"
    return state, None


def create_draft(kind, state, notes=""):
    ctx = get_context()
    # validate strategy exists on disk
    from ...loader import strategy_exists
    if not strategy_exists(ctx.strategies_dir, state["strategy"]):
        return {"error": "strategy_not_found",
                "message": f'Strategy "{state["strategy"]}" not found. Create it first with create_strategy().'}
    sid = ctx.sessions.create(kind, state, notes=notes)
    return {"status": "draft", "session_id": sid, "state": state}


def update_draft(kind, session_id, state_str):
    ctx = get_context()
    session = ctx.sessions.get(session_id)
    if session is None:
        return {"error": "not_found", "session_id": session_id}
    if session["kind"] != kind:
        return {"error": "wrong_kind", "message": f"Session {session_id} is a {session['kind']}."}
    try:
        new_state = json.loads(state_str) if isinstance(state_str, str) else state_str
    except json.JSONDecodeError as e:
        return {"error": "invalid_json", "message": str(e)}
    merged = {**session["state"], **new_state}
    try:
        ctx.sessions.update_state(session_id, merged)
    except ValueError as e:
        return {"error": "not_draft", "message": str(e)}
    return {"status": "draft", "session_id": session_id, "state": merged}


def update_notes(session_id, notes):
    ctx = get_context()
    if ctx.sessions.get(session_id) is None:
        return {"error": "not_found", "session_id": session_id}
    ctx.sessions.update_notes(session_id, notes)
    return {"status": "ok", "session_id": session_id, "notes": notes}


def get_session(session_id, include_results=True):
    ctx = get_context()
    s = ctx.sessions.get(session_id)
    if s is None:
        return {"error": "not_found", "session_id": session_id}
    out = {
        "session_id": s["id"], "kind": s["kind"], "status": s["status"],
        "progress": s["progress"], "state": s["state"], "notes": s["notes"],
        "created_at": jh.timestamp_to_time(s["created_at"]) if s["created_at"] else None,
        "updated_at": jh.timestamp_to_time(s["updated_at"]) if s["updated_at"] else None,
    }
    if include_results and s["results"] is not None:
        out["results"] = s["results"]
        out["dashboard_url"] = s["results"].get("dashboard_url", "")
    return out


def list_sessions(kind, limit=20):
    ctx = get_context()
    rows = ctx.sessions.list(kind, limit)
    return {"sessions": [
        {"session_id": r["id"], "status": r["status"], "progress": r["progress"],
         "symbol": r["state"].get("symbol"), "timeframe": r["state"].get("timeframe"),
         "strategy": r["state"].get("strategy"),
         "created_at": jh.timestamp_to_time(r["created_at"]) if r["created_at"] else None}
        for r in rows]}


def run_session(session_id, kind):
    ctx = get_context()
    s = ctx.sessions.get(session_id)
    if s is None:
        return {"error": "not_found", "session_id": session_id}
    if s["kind"] != kind:
        return {"error": "wrong_kind", "message": f"Session {session_id} is a {s['kind']}."}
    if s["status"] == "running":
        return {"status": "running", "session_id": session_id, "message": "Already running."}
    if s["status"] not in ("draft", "stopped", "terminated", "canceled"):
        # allow re-running finished by resetting to draft-like start
        pass
    ctx.runner.run(session_id)
    return {"status": "started", "session_id": session_id,
            "message": f"{kind} started. Poll get_{_poll_name(kind)}_session(session_id) until terminal."}


def _poll_name(kind):
    return {"backtest": "backtest", "significance_test": "significance_test",
            "monte_carlo": "monte_carlo", "optimization": "optimization"}[kind]


def cancel_session(session_id, kind, new_status="canceled"):
    ctx = get_context()
    s = ctx.sessions.get(session_id)
    if s is None:
        return {"error": "not_found", "session_id": session_id}
    return ctx.runner.cancel(session_id, new_status)


def purge_sessions(kind, days_old=None):
    ctx = get_context()
    n = ctx.sessions.purge(kind, days_old)
    return {"status": "ok", "purged": n, "kind": kind}
