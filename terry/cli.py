"""Terry command-line interface: `terry serve | init | doctor | backtest | version`."""
import argparse
import os
import sys

from .version import __version__


def _cmd_serve(args):
    from .mcp.server import run
    run(port=args.port, project_root=args.project)


def _cmd_dashboard(args):
    from .dashboard import run
    run(port=args.port, host=args.host, project_root=args.project)


def _cmd_init(args):
    root = os.path.abspath(args.project or os.getcwd())
    strategies = os.path.join(root, "strategies")
    storage = os.path.join(root, "storage", "reports")
    os.makedirs(strategies, exist_ok=True)
    os.makedirs(storage, exist_ok=True)

    sample_dir = os.path.join(strategies, "SampleTrend")
    if not os.path.exists(os.path.join(sample_dir, "__init__.py")):
        os.makedirs(sample_dir, exist_ok=True)
        with open(os.path.join(sample_dir, "__init__.py"), "w") as f:
            f.write(_SAMPLE_STRATEGY)
    # copy AGENTS.md if present in the package
    print(f"Initialized Terry project at {root}")
    print(f"  strategies/  ({len(os.listdir(strategies))} strategies)")
    print("  storage/     (candles.db, sessions.db, reports/ will be created here)")
    print("\nNext: `terry serve` then connect your agent to http://localhost:9021/mcp")


def _cmd_doctor(args):
    print(f"Terry {__version__}")
    print(f"Python: {sys.version.split()[0]}")
    ok = True
    for mod in (
        "numpy", "pandas", "requests", "mcp", "scipy", "jesse_rust", "joblib",
        "sklearn", "matplotlib", "optuna", "statsmodels", "tqdm", "fastapi", "uvicorn",
    ):
        try:
            __import__(mod)
            print(f"  ✓ {mod}")
        except ImportError:
            print(f"  ✗ {mod} MISSING")
            ok = False
    from .context import TerryContext
    ctx = TerryContext(args.project)
    print(f"Project root: {ctx.project_root}")
    print(f"  strategies_dir exists: {os.path.isdir(ctx.strategies_dir)}")
    print(f"  storage_dir exists:    {os.path.isdir(ctx.storage_dir)}")
    print(f"  datasets stored:       {len(ctx.candle_db.existing())}")
    print("Status:", "OK" if ok else "MISSING DEPENDENCIES")


def _cmd_backtest(args):
    """Quick CLI backtest (bypasses MCP) — handy for testing without an agent."""
    from .context import TerryContext, set_context
    ctx = set_context(TerryContext(args.project))
    state = {"strategy": args.strategy, "symbol": args.symbol, "timeframe": args.timeframe,
             "exchange": args.exchange or ctx.config.get()["exchange"],
             "start_date": args.start_date, "finish_date": args.finish_date}
    sid = ctx.sessions.create("backtest", state)
    ctx.runner.run(sid)
    import time
    while True:
        s = ctx.sessions.get(sid)
        if s["status"] in ("finished", "stopped"):
            break
        time.sleep(0.5)
    s = ctx.sessions.get(sid)
    if s["status"] == "stopped":
        print("STOPPED:", s["results"])
        return
    m = s["results"]["metrics"]
    print(f"Trades: {m.get('total')}  Win rate: {m.get('win_rate')}  "
          f"Net %: {m.get('net_profit_percentage'):.2f}  Sharpe: {m.get('sharpe_ratio')}")
    print("Report:", s["results"].get("dashboard_url"))


def main(argv=None):
    p = argparse.ArgumentParser(prog="terry", description="Terry — local Jesse-compatible trading MCP server")
    p.add_argument("--version", action="version", version=f"Terry {__version__}")
    sub = p.add_subparsers(dest="command")

    s = sub.add_parser("serve", help="Start the MCP server")
    s.add_argument("--port", type=int, default=9021)
    s.add_argument("--project", type=str, default=None)
    s.set_defaults(func=_cmd_serve)

    dashboard = sub.add_parser("dashboard", help="Start the local browser dashboard")
    dashboard.add_argument("--port", type=int, default=9020)
    dashboard.add_argument("--host", type=str, default="127.0.0.1")
    dashboard.add_argument("--project", type=str, default=None)
    dashboard.set_defaults(func=_cmd_dashboard)

    i = sub.add_parser("init", help="Initialize a Terry project in the current directory")
    i.add_argument("--project", type=str, default=None)
    i.set_defaults(func=_cmd_init)

    d = sub.add_parser("doctor", help="Check the environment")
    d.add_argument("--project", type=str, default=None)
    d.set_defaults(func=_cmd_doctor)

    b = sub.add_parser("backtest", help="Run a quick backtest without an agent")
    b.add_argument("strategy")
    b.add_argument("--symbol", default="BTC-USDT")
    b.add_argument("--timeframe", default="4h")
    b.add_argument("--exchange", default=None)
    b.add_argument("--start-date", dest="start_date", default="2023-01-01")
    b.add_argument("--finish-date", dest="finish_date", default=None)
    b.add_argument("--project", type=str, default=None)
    b.set_defaults(func=_cmd_backtest)

    args = p.parse_args(argv)
    if not getattr(args, "func", None):
        p.print_help()
        return
    args.func(args)


_SAMPLE_STRATEGY = '''from terry.strategies import Strategy
import terry.indicators as ta
from terry import utils


class SampleTrend(Strategy):
    """A simple EMA-cross trend follower with an ATR stop/target (futures)."""

    def should_long(self) -> bool:
        return ta.ema(self.candles, 20) > ta.ema(self.candles, 50)

    def should_short(self) -> bool:
        return ta.ema(self.candles, 20) < ta.ema(self.candles, 50)

    def go_long(self):
        qty = utils.size_to_qty(self.available_margin * 0.5, self.price, fee_rate=self.fee_rate)
        self.buy = qty, self.price

    def go_short(self):
        qty = utils.size_to_qty(self.available_margin * 0.5, self.price, fee_rate=self.fee_rate)
        self.sell = qty, self.price

    def on_open_position(self, order):
        atr = ta.atr(self.candles)
        if self.is_long:
            self.stop_loss = self.position.qty, self.price - 2 * atr
            self.take_profit = self.position.qty, self.price + 4 * atr
        elif self.is_short:
            self.stop_loss = self.position.qty, self.price + 2 * atr
            self.take_profit = self.position.qty, self.price - 4 * atr

    def update_position(self):
        pass
'''


if __name__ == "__main__":
    main()
