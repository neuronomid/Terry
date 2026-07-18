"""Local historical logger compatible with ``jesse.services.logger`` strategy calls."""
from __future__ import annotations

from .. import helpers as jh
from ..store import get_current_store


def _record(level, message):
    message = str(message)
    try:
        current = get_current_store()
    except RuntimeError:
        print(f"[{level}] {message}")
        return None
    entry = {
        "id": jh.generate_unique_id(), "session_id": current.app.session_id,
        "timestamp": jh.now_to_timestamp(), "message": message,
    }
    getattr(current.logs, level).append(entry)
    return entry


def info(msg: str, send_notification=False, webhook=None) -> None:
    del send_notification, webhook
    _record("info", msg)


def error(msg: str, send_notification=True) -> None:
    del send_notification
    info(msg)
    _record("errors", msg)


def reset() -> None:
    try:
        current = get_current_store()
    except RuntimeError:
        return
    current.logs.info.clear()
    current.logs.errors.clear()


def create_logger_file(name):
    """Retained for source compatibility; historical logs remain in memory."""
    return str(name)


def log_exchange_message(exchange, message):
    info(f"[{exchange}]: {message}")


def log_optimize_mode(message, session_id: str):
    info(f"[{session_id}]: {message}")


def log_monte_carlo(message, session_id: str):
    info(f"[{session_id}]: {message}")


def log_significance_test(message, session_id: str):
    info(f"[{session_id}]: {message}")
