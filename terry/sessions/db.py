"""SQLite-backed session store for backtest / significance / monte_carlo / optimization runs."""
import json
import sqlite3
import threading
import time

from .. import helpers as jh

VALID_KINDS = {"backtest", "significance_test", "monte_carlo", "optimization"}
TERMINAL = {"finished", "stopped", "terminated", "canceled"}


class SessionStore:
    def __init__(self, path):
        self.path = path
        self._local = threading.local()
        self._init_schema()

    def _conn(self):
        if getattr(self._local, "conn", None) is None:
            self._local.conn = sqlite3.connect(self.path, timeout=30)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL;")
        return self._local.conn

    def _init_schema(self):
        self._conn().execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                status TEXT NOT NULL,
                state_json TEXT,
                results_json TEXT,
                notes TEXT,
                progress INTEGER DEFAULT 0,
                created_at INTEGER,
                updated_at INTEGER
            )
            """
        )
        self._conn().commit()

    def create(self, kind, state, notes=""):
        if kind not in VALID_KINDS:
            raise ValueError(f"Unknown session kind: {kind}")
        sid = jh.generate_unique_id()
        now = jh.now_to_timestamp()
        self._conn().execute(
            "INSERT INTO sessions (id, kind, status, state_json, results_json, notes, progress, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (sid, kind, "draft", json.dumps(state), None, notes, 0, now, now))
        self._conn().commit()
        return sid

    def update_state(self, sid, state):
        row = self.get(sid)
        if row is None:
            raise KeyError(sid)
        if row["status"] != "draft":
            raise ValueError(f"Session {sid} is not a draft (status={row['status']}).")
        self._set(sid, state_json=json.dumps(state))

    def update_notes(self, sid, notes):
        self._set(sid, notes=notes)

    def set_status(self, sid, status):
        self._set(sid, status=status)

    def set_progress(self, sid, progress):
        self._set(sid, progress=int(progress))

    def set_results(self, sid, results, status="finished"):
        self._set(sid, results_json=json.dumps(results, default=_json_default), status=status, progress=100)

    def _set(self, sid, **cols):
        cols["updated_at"] = jh.now_to_timestamp()
        keys = ", ".join(f"{k}=?" for k in cols)
        self._conn().execute(f"UPDATE sessions SET {keys} WHERE id=?", (*cols.values(), sid))
        self._conn().commit()

    def get(self, sid):
        cur = self._conn().execute("SELECT * FROM sessions WHERE id=?", (sid,))
        row = cur.fetchone()
        return self._row_to_dict(row) if row else None

    def list(self, kind=None, limit=50, offset=0):
        params = []
        query = "SELECT * FROM sessions"
        if kind:
            query += " WHERE kind=?"
            params.append(kind)
        query += " ORDER BY created_at DESC"
        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            params.extend((int(limit), int(offset)))
        elif offset:
            query += " LIMIT -1 OFFSET ?"
            params.append(int(offset))
        cur = self._conn().execute(query, params)
        return [self._row_to_dict(r) for r in cur.fetchall()]

    def count(self, kind=None):
        if kind:
            row = self._conn().execute("SELECT COUNT(*) FROM sessions WHERE kind=?", (kind,)).fetchone()
        else:
            row = self._conn().execute("SELECT COUNT(*) FROM sessions").fetchone()
        return int(row[0])

    def purge(self, kind, days_old=None):
        conn = self._conn()
        if days_old is None:
            cur = conn.execute("DELETE FROM sessions WHERE kind=?", (kind,))
        else:
            cutoff = jh.now_to_timestamp() - int(days_old) * 86_400_000
            cur = conn.execute("DELETE FROM sessions WHERE kind=? AND created_at < ?", (kind, cutoff))
        conn.commit()
        return cur.rowcount

    def delete(self, sid):
        self._conn().execute("DELETE FROM sessions WHERE id=?", (sid,))
        self._conn().commit()

    @staticmethod
    def _row_to_dict(row):
        d = dict(row)
        d["state"] = json.loads(d.pop("state_json")) if d.get("state_json") else {}
        d["results"] = json.loads(d.pop("results_json")) if d.get("results_json") else None
        return d


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    return str(o)
