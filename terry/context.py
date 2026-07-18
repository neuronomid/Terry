"""
TerryContext — the shared runtime wiring (paths + services) used by the MCP tools and CLI.

A Terry "project" is just a directory containing `strategies/` and `storage/`. By default the
project root is the current working directory (or $TERRY_PROJECT).
"""
import os

from .config import Config
from .data import CandleDB, Importer
from .sessions import SessionStore, Runner
from .report import generate_report


class TerryContext:
    def __init__(self, project_root=None):
        self.project_root = os.path.abspath(
            project_root or os.environ.get("TERRY_PROJECT") or os.getcwd())
        self.strategies_dir = os.path.join(self.project_root, "strategies")
        self.storage_dir = os.path.join(self.project_root, "storage")
        self.reports_dir = os.path.join(self.storage_dir, "reports")
        os.makedirs(self.strategies_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

        self.config = Config(os.path.join(self.storage_dir, "config.json"))
        self.candle_db = CandleDB(os.path.join(self.storage_dir, "candles.db"))
        self.importer = Importer(self.candle_db)
        self.sessions = SessionStore(os.path.join(self.storage_dir, "sessions.db"))
        self.runner = Runner(self)

    def write_report(self, sid, kind, state, results):
        path = generate_report(sid, kind, state, results, self.reports_dir)
        return f"file://{path}"


# module-level singleton (initialized by the MCP server / CLI)
_CTX = None


def get_context():
    global _CTX
    if _CTX is None:
        _CTX = TerryContext()
    return _CTX


def set_context(ctx):
    global _CTX
    _CTX = ctx
    return ctx
