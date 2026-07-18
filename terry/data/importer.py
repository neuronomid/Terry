"""Background candle importer: fetches 1m candles into the SQLite store with progress tracking."""
import threading

from .. import helpers as jh
from .binance import fetch_1m_range

ONE_MIN_MS = 60_000


class Importer:
    def __init__(self, candle_db):
        self.db = candle_db
        self._status = {}          # import_id -> status dict
        self._stop_flags = {}      # import_id -> threading.Event
        self._lock = threading.Lock()

    def start_import(self, exchange, symbol, start_date, finish_date=None):
        import_id = jh.generate_unique_id()
        start_ts = jh.date_to_timestamp(start_date)
        finish_ts = jh.date_to_timestamp(finish_date) if finish_date else jh.today_to_timestamp()
        if finish_ts <= start_ts:
            raise ValueError("finish_date must be after start_date")

        stop = threading.Event()
        with self._lock:
            self._stop_flags[import_id] = stop
            self._status[import_id] = {
                "import_id": import_id, "status": "started", "progress": 0,
                "candles_imported": 0, "exchange": exchange, "symbol": symbol,
                "start_date": start_date, "finish_date": jh.timestamp_to_date(finish_ts),
                "message": "Import started",
            }

        t = threading.Thread(
            target=self._run, args=(import_id, exchange, symbol, start_ts, finish_ts, stop),
            daemon=True)
        t.start()
        return import_id

    def _run(self, import_id, exchange, symbol, start_ts, finish_ts, stop):
        self._update(import_id, status="running", message="Fetching candles…")
        imported = {"n": 0}

        def on_progress(done_ms, total_ms):
            pct = int(min(done_ms / total_ms, 1.0) * 100)
            self._update(import_id, progress=pct)

        # fetch in windows so we can store incrementally and dedup against existing data
        try:
            cursor = start_ts
            window = 1000 * ONE_MIN_MS  # ~16.6h per API page batch
            while cursor < finish_ts and not stop.is_set():
                chunk = fetch_1m_range(
                    exchange, symbol, cursor, min(cursor + window, finish_ts),
                    on_progress=None, should_stop=stop.is_set)
                if len(chunk) == 0:
                    cursor += window
                    continue
                self.db.store(exchange, symbol, chunk)
                imported["n"] += len(chunk)
                cursor = int(chunk[-1][0]) + ONE_MIN_MS
                pct = int(min((cursor - start_ts) / max(finish_ts - start_ts, 1), 1.0) * 100)
                self._update(import_id, progress=pct, candles_imported=imported["n"])
            if stop.is_set():
                self._update(import_id, status="canceled", message="Import canceled")
            else:
                cov = self.db.coverage(exchange, symbol)
                self._update(import_id, status="finished", progress=100,
                             candles_imported=imported["n"],
                             message=f"Imported {imported['n']} candles",
                             coverage=cov)
        except Exception as e:
            self._update(import_id, status="error", message=f"{type(e).__name__}: {e}")

    def _update(self, import_id, **kwargs):
        with self._lock:
            if import_id in self._status:
                self._status[import_id].update(kwargs)

    def get_status(self, import_id):
        with self._lock:
            return dict(self._status.get(import_id, {"status": "not_found", "import_id": import_id}))

    def cancel(self, import_id):
        with self._lock:
            flag = self._stop_flags.get(import_id)
        if flag:
            flag.set()
            return True
        return False
