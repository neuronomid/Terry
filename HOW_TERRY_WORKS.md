# How Terry Works — and How to Use It

**Terry** is your own local crypto trading-strategy lab, built for **research compatibility with
[Jesse](https://jesse.trade)**. It runs on this Linux machine and talks to AI agents (like Claude
Code) through **MCP** (Model Context Protocol). You (or an agent) can research an idea, write a
trading strategy, download real price history, backtest it, and stress-test it — all locally, with
no cloud account and no paid services.

> ⚠️ **Terry does not trade real money.** It only *simulates* trades on historical data (backtesting)
> so you can see how a strategy *would have* done. It never connects to your exchange account or
> places orders. Nothing here is financial advice, and a good backtest does **not** guarantee future
> profit.

---

## 1. The big picture (in plain words)

Think of Terry as three things working together:

1. **A backtesting engine** — it replays historical 1-minute candles and pretends to trade your
   strategy, tracking every order, fill, fee, and the resulting profit/loss.
2. **A set of research tools** — beyond a plain backtest, it can gather/train/deploy ML models,
   apply candle pipelines, benchmark and export results, check whether your entry rule is
   *actually better than random* (significance test), whether a good-looking result was just *luck*
   (Monte Carlo), and it can *tune* your strategy's numbers (optimization).
3. **An MCP server** — a small local web service that exposes all of the above as "tools" an AI
   agent can call. The agent presses the same buttons you would, but through a safe channel.

Everything is stored locally in a `storage/` folder using **SQLite** (a single-file database). No
PostgreSQL, no Redis, no Docker required.

```
You / an AI agent
      │  (natural language)
      ▼
  AI agent (Claude Code, Cursor, …)
      │  MCP tool calls  ──►  http://localhost:9021/mcp
      ▼
  Terry MCP server
      ├── strategies/           ← your strategy files live here
      ├── engine (backtest)     ← replays candles, simulates trades
      ├── research tools        ← significance / Monte Carlo / optimize
      └── storage/
            ├── candles.db       ← downloaded price history
            ├── sessions.db      ← every backtest/test you've run
            └── reports/*.html   ← a visual report per run
```

---

## 2. One-time setup

```bash
cd /home/omid/Documents/Projects/Terry

# 1) create a virtual environment and install the dependencies
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2) sanity check
.venv/bin/python -m terry doctor
```

You should see all dependencies ticked and `Status: OK`.

---

## 3. Start Terry and connect your agent

**Start the server** (leave it running in a terminal):

```bash
.venv/bin/python -m terry serve
# → ✓ Terry MCP Server running at http://localhost:9021/mcp
```

**Connect Claude Code** to it (in another terminal):

```bash
claude mcp add --transport http terry http://localhost:9021/mcp
```

Then inside Claude Code, run `/mcp` — you should see **terry** and its tools. That's it. The agent
will automatically read `AGENTS.md` (the house rules) from the project root.

> For Cursor/VS Code/Zed, add an MCP server of type **HTTP** pointing at
> `http://localhost:9021/mcp`. See `.mcp.json` in this folder for a ready-made example.

---

## 4. How you actually use it (just talk to the agent)

You don't call tools yourself — you ask the agent in plain English, and it uses Terry's tools. For
example:

> *"Import 1h Binance perpetual-futures candles for BTC-USDT from 2023-01-01 to today, then build a
> simple EMA-cross trend strategy, check whether the entry has a real edge, backtest it for 2023–2024,
> and tell me if it looks overfit."*

Behind the scenes the agent will:

1. `import_candles(...)` and keep checking `get_candle_import_status(...)` until it's done.
2. `create_strategy("EmaTrend", <python code>)`.
3. `create_significance_test_draft(...)` → `run_significance_test(...)` → poll → read the p-value.
4. `create_backtest_draft(...)` → `run_backtest(...)` → poll → read the metrics.
5. `create_monte_carlo_draft(...)` → `run_monte_carlo(...)` → poll → read the overfit verdict.
6. Give you the numbers **and a link to a local HTML report** you can open in your browser.

### The recommended research workflow (what a good agent follows)

```
import candles
   → write a MINIMAL strategy with just the entry rule
      → significance test the entry (is the edge real? p < 0.05?)
         → build out the full strategy (sizing, stops, targets)
            → backtest
               → iterate with small changes
                  → Monte Carlo (was it luck / overfit?)
                     → optionally optimize the parameters
                        → validate on out-of-sample data
                           → write a report
```

---

## 5. Using Terry without an agent (optional)

You can also drive it yourself from the command line:

```bash
# quick backtest of the sample strategy (after importing some candles)
.venv/bin/python -m terry backtest SampleTrend --symbol BTC-USDT --timeframe 4h \
    --start-date 2023-01-01 --finish-date 2024-01-01
```

Or in Python:

```python
from terry.research import backtest
from terry.factories import candles_from_close_prices
# … build a candles dict and a Strategy subclass, then call backtest(config, routes, [], candles)
```

---

## 6. Writing a strategy (the important idea)

A strategy is a small Python class. You implement a few methods and Terry calls them **once per
candle, after the candle closes** (so there's no cheating with future data):

```python
from terry.strategies import Strategy
import terry.indicators as ta
from terry import utils

class EmaTrend(Strategy):
    def should_long(self):                       # when to enter a long
        return ta.ema(self.candles, 20) > ta.ema(self.candles, 50)

    def should_short(self):                      # when to enter a short (futures only)
        return ta.ema(self.candles, 20) < ta.ema(self.candles, 50)

    def go_long(self):                           # how big / at what price
        qty = utils.size_to_qty(self.available_margin * 0.5, self.price, fee_rate=self.fee_rate)
        self.buy = qty, self.price               # a "market" order at the current price

    def go_short(self):
        qty = utils.size_to_qty(self.available_margin * 0.5, self.price, fee_rate=self.fee_rate)
        self.sell = qty, self.price

    def on_open_position(self, order):           # set a stop-loss and take-profit once we're in
        atr = ta.atr(self.candles)
        if self.is_long:
            self.stop_loss   = self.position.qty, self.price - 2 * atr
            self.take_profit = self.position.qty, self.price + 4 * atr
        elif self.is_short:
            self.stop_loss   = self.position.qty, self.price + 2 * atr
            self.take_profit = self.position.qty, self.price - 4 * atr

    def update_position(self):                   # runs each candle while you're in a trade
        pass
```

Key ideas:
- **Smart orders:** you set `self.buy = qty, price` and Terry figures out whether it's a market,
  limit, or stop order from the price. You never pick the type.
- **Sizing:** never hardcode a quantity — use `utils.size_to_qty(...)` or `utils.risk_to_qty(...)`.
- **Spot vs futures:** on spot you can't short and you set stops in `on_open_position`. On futures
  you can do both directions.
- Full reference: ask the agent to read the resource `terry://strategy`, or see
  `terry/mcp/resources.py`.

---

## 7. What the numbers mean (reading a backtest)

A finished backtest gives ~44 metrics. The ones to look at first:

| Metric | Meaning |
|---|---|
| `total` | number of completed trades (need enough, ~30+, for stats to mean anything) |
| `net_profit_percentage` | total return over the period |
| `win_rate` | fraction of trades that won (0–1) |
| `sharpe_ratio` | risk-adjusted return (higher is better; >1 is decent) |
| `max_drawdown` | worst peak-to-trough drop (%). Can you stomach it? |
| `expectancy` | average profit you expect per trade |

Then sanity-check it:
- **Significance test** → is the *entry rule* better than random? (`p_value < 0.05` = yes)
- **Monte Carlo** → is the result *robust* or just lucky/overfit? (`robust` vs `overfit_suspect`)

Every run also writes a **visual HTML report** to `storage/reports/<id>.html` — the tool response
includes its path as `dashboard_url`. Open it in a browser to see the equity curve and trade list.

---

## 8. The tools an agent can call (58 total, plus 12 reference resources)

- **Status/help:** `get_terry_status`, `greet_user`
- **Strategies:** `create_strategy`, `read_strategy`, `write_strategy`
- **Config:** `get_config`, `update_config`, `get_backtest_config`, `get_live_config`, `get_optimization_config`
- **Candles:** `import_candles`, `get_candle_import_status`, `cancel_candle_import`, `get_candles`,
  `get_existing_candles`, `delete_candles`, `clear_candle_cache`
- **Indicators:** `list_indicators`, `get_indicator_details`
- **Backtest:** `create_backtest_draft`, `update_backtest_draft`, `update_backtest_notes`,
  `run_backtest`, `get_backtest_session`, `get_backtest_sessions`, `cancel_backtest`, `purge_backtest_sessions`
- **Significance test:** `create_significance_test_draft`, `run_significance_test`,
  `get_significance_test_session`, … (same draft→run→poll pattern)
- **Monte Carlo:** `create_monte_carlo_draft`, `run_monte_carlo`, `get_monte_carlo_session`,
  `get_monte_carlo_equity_curves`, `resume_monte_carlo`, `terminate_monte_carlo`, …
- **Optimization:** `create_optimization_draft`, `run_optimization`, `get_optimization_session`,
  `rerun_optimization`, …

The long-running tools (`run_backtest`, `run_significance_test`, `run_monte_carlo`,
`run_optimization`) return immediately; the agent then polls the matching `get_*_session` until the
status is `finished` or `stopped`. **In Terry these are free and unlimited** (Jesse charges credits
for them; Terry does not).

Terry's open research surface is audited against Jesse 2.5.0. Live and paper exchange execution,
exchange accounts, and live notifications are not included; the full matrix is in
[JESSE_PARITY.md](JESSE_PARITY.md).

---

## 9. Where things live

```
Terry/
├── AGENTS.md                 # the rules the agent follows (auto-read)
├── HOW_TERRY_WORKS.md        # this file
├── APIS_AND_SERVICES.md      # what external services you need (spoiler: almost none)
├── TERRY_BUILD_PLAN.md       # how Terry was designed/built
├── requirements.txt
├── .mcp.json                 # example MCP client config
├── terry/                    # the Python package (engine, tools, server)
├── strategies/               # your strategies (created via the agent)
└── storage/
    ├── candles.db            # downloaded price history
    ├── sessions.db           # every run you've done
    ├── config.json           # your saved settings
    └── reports/*.html        # a visual report per run
```

---

## 10. Troubleshooting

- **"No candle data …" when backtesting** → import candles first (start ~2 months before your
  backtest start date to cover indicator warm-up), then re-run.
- **HTTP 451 on import** → that exchange is geo-blocked from your IP; choose another supported
  public driver available in your region.
- **Agent can't see the tools** → make sure `terry serve` is running and you added the MCP server with
  `--transport http` at `http://localhost:9021/mcp`.
- **Shorting error on spot** → shorting only works on futures exchanges (e.g. "Binance Perpetual
  Futures"); on spot make `should_short` return `False`.
