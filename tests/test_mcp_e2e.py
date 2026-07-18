"""
End-to-end MCP test: launches the real Terry MCP server (streamable-http) as a subprocess,
connects with an MCP client, and drives the full agent workflow over the wire:

  status -> create_strategy -> import real Binance candles -> backtest -> significance test
  -> monte carlo -> optimization -> read a resource.

Run:  python tests/test_mcp_e2e.py
"""
import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

PORT = 9077
URL = f"http://localhost:{PORT}/mcp"

STRATEGY_SRC = '''from terry.strategies import Strategy
import terry.indicators as ta
from terry import utils

class E2ECross(Strategy):
    def hyperparameters(self):
        return [{"name": "fast", "type": int, "min": 5, "max": 20, "default": 10},
                {"name": "slow", "type": int, "min": 25, "max": 60, "default": 30}]
    def should_long(self):
        return ta.sma(self.candles, self.hp["fast"]) > ta.sma(self.candles, self.hp["slow"])
    def should_short(self):
        return ta.sma(self.candles, self.hp["fast"]) < ta.sma(self.candles, self.hp["slow"])
    def go_long(self):
        self.buy = utils.size_to_qty(self.available_margin * 0.5, self.price, fee_rate=self.fee_rate), self.price
    def go_short(self):
        self.sell = utils.size_to_qty(self.available_margin * 0.5, self.price, fee_rate=self.fee_rate), self.price
    def update_position(self):
        f, s = ta.sma(self.candles, self.hp["fast"]), ta.sma(self.candles, self.hp["slow"])
        if (self.is_long and f < s) or (self.is_short and f > s):
            self.liquidate()
'''

# minimal entry-only strategy for the significance test
ENTRY_SRC = '''from terry.strategies import Strategy
import terry.indicators as ta
from terry import utils

class E2EEntry(Strategy):
    def should_long(self):
        return ta.rsi(self.candles, 14) < 35
    def should_short(self):
        return ta.rsi(self.candles, 14) > 65
    def go_long(self):
        self.buy = utils.size_to_qty(self.available_margin * 0.5, self.price, fee_rate=self.fee_rate), self.price
    def go_short(self):
        self.sell = utils.size_to_qty(self.available_margin * 0.5, self.price, fee_rate=self.fee_rate), self.price
'''

EXCHANGE = "Binance Perpetual Futures"
SYMBOL = "BTC-USDT"


def _content(result):
    """Extract the JSON dict a tool returned."""
    if getattr(result, "structuredContent", None):
        sc = result.structuredContent
        return sc.get("result", sc)
    for block in result.content:
        if getattr(block, "text", None):
            try:
                return json.loads(block.text)
            except json.JSONDecodeError:
                return {"text": block.text}
    return {}


async def call(session, tool_name, **args):
    res = await session.call_tool(tool_name, args)
    data = _content(res)
    return data


async def poll(session, getter, sid, timeout=180):
    start = time.time()
    while time.time() - start < timeout:
        s = await call(session, getter, session_id=sid)
        if s.get("status") in ("finished", "stopped", "terminated", "canceled"):
            return s
        await asyncio.sleep(1.5)
    return {"status": "timeout"}


async def run():
    async with streamablehttp_client(URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print(f"[tools] {len(tools.tools)} registered")
            assert len(tools.tools) >= 55

            status = await call(session, "get_terry_status")
            print(f"[status] Terry {status['version']} | {status['indicators_available']} indicators")

            # ---- strategies
            r = await call(session, "create_strategy", name="E2ECross", content=STRATEGY_SRC)
            assert r["status"] == "success", r
            r = await call(session, "create_strategy", name="E2EEntry", content=ENTRY_SRC)
            assert r["status"] == "success", r
            r = await call(session, "read_strategy", name="E2ECross")
            assert "class E2ECross" in r["content"]
            print("[strategy] created + read back E2ECross, E2EEntry")

            # ---- config: small warmup so a short window works
            await call(session, "update_config", config=json.dumps({"warm_up_candles": 20}))

            # ---- import real candles from Binance
            imp = await call(session, "import_candles", exchange=EXCHANGE, symbol=SYMBOL,
                             start_date="2024-01-01", finish_date="2024-02-10")
            import_id = imp["import_id"]
            print(f"[import] started {import_id}; checking for progress…")
            for _ in range(120):
                st = await call(session, "get_candle_import_status", import_id=import_id)
                if st["status"] == "finished":
                    print(f"[import] finished: {st['candles_imported']} candles")
                    break
                if st["status"] == "error":
                    raise RuntimeError(f"import error: {st['message']}")
                await asyncio.sleep(1.5)
            existing = await call(session, "get_existing_candles")
            assert existing["existing"], "no candles stored"

            # ---- significance test (entry-only strategy)
            d = await call(session, "create_significance_test_draft", strategy="E2EEntry",
                           symbol=SYMBOL, timeframe="1h", exchange=EXCHANGE,
                           start_date="2024-01-05", finish_date="2024-02-09", n_simulations=2000)
            sid = d["session_id"]
            await call(session, "run_significance_test", session_id=sid)
            s = await poll(session, "get_significance_test_session", sid)
            rst = s["results"]["results"]
            print(f"[RST] status={s['status']} p_value={rst['p_value']:.4f} verdict={rst['verdict']} "
                  f"n_obs={rst['n_observations']}  report={s.get('dashboard_url','')[:60]}")
            assert s["status"] == "finished"

            # ---- backtest
            route_json = json.dumps([{
                "exchange": EXCHANGE, "strategy": "E2ECross",
                "symbol": SYMBOL, "timeframe": "1h",
            }])
            d = await call(session, "create_backtest_draft", routes=route_json,
                           exchange=EXCHANGE, start_date="2024-01-05", finish_date="2024-02-09",
                           export_csv=True, export_json=True, export_chart=True,
                           export_tradingview=True, benchmark=True)
            sid = d["session_id"]
            await call(session, "run_backtest", session_id=sid)
            s = await poll(session, "get_backtest_session", sid)
            assert s["status"] == "finished", s
            m = s["results"]["metrics"]
            print(f"[backtest] trades={m.get('total')} netpct={m.get('net_profit_percentage'):.2f} "
                  f"sharpe={m.get('sharpe_ratio')} report={s.get('dashboard_url','')[:60]}")
            assert s.get("dashboard_url", "").startswith("file://")
            assert s["results"]["csv"] and s["results"]["json"]
            assert s["results"]["tradingview"].startswith("//@version=5")
            assert "return_percentage" in s["results"]["benchmark"]
            assert os.path.isdir(s["results"]["charts_folder"])

            # ---- monte carlo (small)
            d = await call(session, "create_monte_carlo_draft", strategy="E2ECross",
                           symbol=SYMBOL, timeframe="1h", exchange=EXCHANGE,
                           start_date="2024-01-05", finish_date="2024-02-09", num_scenarios=25)
            sid = d["session_id"]
            await call(session, "run_monte_carlo", session_id=sid)
            s = await poll(session, "get_monte_carlo_session", sid, timeout=300)
            assert s["status"] == "finished", s
            verdict = s["results"]["candles"]["overfit_verdict"]
            print(f"[monte_carlo] scenarios={s['results']['candles']['num_scenarios']} overfit={verdict}")
            curves = await call(session, "get_monte_carlo_equity_curves", session_id=sid)
            assert curves["status"] == "success" and curves["candles"]["scenarios"]
            assert curves["candles"]["original"]["equity_curve"][0]["name"] == "Portfolio"

            # ---- optimization (small)
            d = await call(session, "create_optimization_draft", routes=route_json,
                           exchange=EXCHANGE,
                           training_start_date="2024-01-05",
                           training_finish_date="2024-01-25",
                           testing_start_date="2024-01-25",
                           testing_finish_date="2024-02-09",
                           objective_function="sharpe", trials=3,
                           best_candidates_count=3, warm_up_candles=20)
            sid = d["session_id"]
            await call(session, "run_optimization", session_id=sid)
            s = await poll(session, "get_optimization_session", sid, timeout=300)
            assert s["status"] == "finished", s
            best = s["results"]["best"]
            print(f"[optimize] best_hp={best['hp']} train={best['train_score']:.3f} test={best['test_score']}")
            assert s["results"]["total_trials"] == 6
            assert best["training_metrics"] and best["testing_metrics"]

            # ---- resource read
            res = await session.read_resource("terry://strategy")
            txt = res.contents[0].text
            assert "Strategy" in txt
            print(f"[resource] terry://strategy read ({len(txt)} chars)")
            opt_res = await session.read_resource("terry://optimization")
            assert "out-of-sample" in opt_res.contents[0].text

            print("\n✅ E2E MCP workflow passed end-to-end.")


def main():
    project = tempfile.mkdtemp(prefix="terry_e2e_")
    env = {**os.environ, "TERRY_PROJECT": project, "PYTHONPATH": ROOT}
    proc = subprocess.Popen(
        [sys.executable, "-m", "terry.mcp.server", "--port", str(PORT), "--project", project],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        # wait for the server to be up
        time.sleep(4)
        if proc.poll() is not None:
            print("SERVER FAILED TO START:\n", proc.stdout.read().decode())
            sys.exit(1)
        asyncio.run(run())
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(project, ignore_errors=True)


if __name__ == "__main__":
    main()
