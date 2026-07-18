"""Thread-local compatibility facade for ``from jesse.store import store``.

Terry's historical engine isolates state per backtest.  Jesse-authored strategies
expect a process-global ``store`` object, so this facade resolves that name to the
current thread's isolated Terry store without leaking state between worker threads.
"""
from __future__ import annotations

import threading


_local = threading.local()


def set_current_store(value):
    _local.current = value
    return value


def get_current_store():
    value = getattr(_local, "current", None)
    if value is None:
        raise RuntimeError("No Terry historical store is active in this thread")
    return value


class _StoreProxy:
    def __getattr__(self, name):
        return getattr(get_current_store(), name)

    def __setattr__(self, name, value):
        setattr(get_current_store(), name, value)

    def reset(self):
        return get_current_store().reset()


store = _StoreProxy()

__all__ = ["store", "get_current_store", "set_current_store"]
