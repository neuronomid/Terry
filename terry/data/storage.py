"""SQLite storage for 1-minute candles (zero-config, replaces Jesse's PostgreSQL)."""
import sqlite3
import threading

import numpy as np

from .. import helpers as jh

ONE_MIN_MS = 60_000


class CandleDB:
    def __init__(self, path):
        self.path = path
        self._local = threading.local()
        self._init_schema()

    def _conn(self):
        if getattr(self._local, "conn", None) is None:
            self._local.conn = sqlite3.connect(self.path, timeout=30)
            self._local.conn.execute("PRAGMA journal_mode=WAL;")
            self._local.conn.execute("PRAGMA synchronous=NORMAL;")
        return self._local.conn

    def _init_schema(self):
        conn = self._conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS candles_1m (
                exchange TEXT NOT NULL,
                symbol   TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                open  REAL NOT NULL,
                close REAL NOT NULL,
                high  REAL NOT NULL,
                low   REAL NOT NULL,
                volume REAL NOT NULL,
                PRIMARY KEY (exchange, symbol, timestamp)
            )
            """
        )
        conn.commit()

    # ------------------------------------------------------------------ write
    def store(self, exchange, symbol, rows):
        """rows: iterable of [ts, open, close, high, low, volume]. Deduplicated by PK."""
        if len(rows) == 0:
            return 0
        conn = self._conn()
        data = [(exchange, symbol, int(r[0]), float(r[1]), float(r[2]),
                 float(r[3]), float(r[4]), float(r[5])) for r in rows]
        before = conn.total_changes
        conn.executemany(
            "INSERT OR IGNORE INTO candles_1m "
            "(exchange, symbol, timestamp, open, close, high, low, volume) "
            "VALUES (?,?,?,?,?,?,?,?)", data)
        conn.commit()
        return conn.total_changes - before

    # ------------------------------------------------------------------ read
    def get(self, exchange, symbol, start_ts=None, finish_ts=None):
        conn = self._conn()
        q = "SELECT timestamp, open, close, high, low, volume FROM candles_1m WHERE exchange=? AND symbol=?"
        params = [exchange, symbol]
        if start_ts is not None:
            q += " AND timestamp >= ?"
            params.append(int(start_ts))
        if finish_ts is not None:
            q += " AND timestamp < ?"
            params.append(int(finish_ts))
        q += " ORDER BY timestamp ASC"
        cur = conn.execute(q, params)
        rows = cur.fetchall()
        if not rows:
            return np.empty((0, 6))
        return np.array(rows, dtype=float)

    def coverage(self, exchange, symbol):
        conn = self._conn()
        cur = conn.execute(
            "SELECT MIN(timestamp), MAX(timestamp), COUNT(*) FROM candles_1m WHERE exchange=? AND symbol=?",
            (exchange, symbol))
        mn, mx, cnt = cur.fetchone()
        if cnt == 0 or mn is None:
            return None
        expected = (mx - mn) // ONE_MIN_MS + 1
        return {
            "exchange": exchange, "symbol": symbol,
            "start_timestamp": int(mn), "finish_timestamp": int(mx),
            "start_date": jh.timestamp_to_date(mn), "finish_date": jh.timestamp_to_date(mx),
            "count": int(cnt), "expected": int(expected),
            "gaps": int(expected - cnt),
        }

    def existing(self):
        conn = self._conn()
        cur = conn.execute(
            "SELECT exchange, symbol, MIN(timestamp), MAX(timestamp), COUNT(*) "
            "FROM candles_1m GROUP BY exchange, symbol ORDER BY exchange, symbol")
        out = []
        for ex, sym, mn, mx, cnt in cur.fetchall():
            out.append({
                "exchange": ex, "symbol": sym,
                "start_date": jh.timestamp_to_date(mn), "finish_date": jh.timestamp_to_date(mx),
                "count": int(cnt),
            })
        return out

    def delete(self, exchange, symbol=None):
        conn = self._conn()
        if symbol is None:
            conn.execute("DELETE FROM candles_1m WHERE exchange=?", (exchange,))
        else:
            conn.execute("DELETE FROM candles_1m WHERE exchange=? AND symbol=?", (exchange, symbol))
        conn.commit()

    def as_candles_dict(self, exchange, symbol, start_ts=None, finish_ts=None):
        arr = self.get(exchange, symbol, start_ts, finish_ts)
        return {jh.key(exchange, symbol): {"exchange": exchange, "symbol": symbol, "candles": arr}}
