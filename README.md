# Terry

**Terry** is a local, self-contained MCP server for crypto-strategy research, backtesting, and
robustness analysis. It lets an AI agent (Claude Code, Cursor, …) build and test trading
strategies end-to-end on your own machine — no cloud, PostgreSQL, Redis, or paid credits. Data
comes from the same keyless public exchange APIs Jesse enables for historical backtesting.

> Terry only **simulates** trades on historical data. It does not connect to exchange accounts or
> place real orders. Past performance never guarantees future results.

## Create a Terry strategy project

Clone this repository once for each independent strategy project:

```bash
# change "my-bot" to your strategy project's name
git clone https://github.com/neuronomid/Terry.git my-bot
cd my-bot

# local settings are deliberately untracked
cp .env.example .env
```

Each cloned project contains the framework plus only the project files it
needs:

```text
├── .env.example       # copy to .env for local Docker settings
├── docker/            # Docker image definition
├── docker-compose.yml # optional container launcher
├── storage/           # local candles, sessions, logs, and HTML reports
└── strategies/        # one directory per strategy
    └── SampleTrend/
        └── __init__.py
```

`storage/` is kept empty in Git; all generated data stays local. Create a new
directory under `strategies/` for each strategy. The supplied `SampleTrend`
strategy is a working reference that you may rename or replace.

## Run locally

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m terry doctor          # check the environment
.venv/bin/python -m terry serve           # start the MCP server on :9021
.venv/bin/python -m terry dashboard       # start the local dashboard on :9020
```

Or start it with Docker:

```bash
docker compose up --build
```

The MCP endpoint is `http://localhost:9021/mcp` by default and the dashboard is at
`http://127.0.0.1:9020`. Pass `--port` to either command if a port is already in use.
The dashboard binds to localhost by default. To require a browser sign-in, set a password
when starting it:

```bash
TERRY_DASHBOARD_PASSWORD='choose-a-local-password' .venv/bin/python -m terry dashboard
```

Connect Claude Code:

```bash
claude mcp add --transport http terry http://localhost:9021/mcp
```

Then just ask your agent to import candles, build a strategy, and backtest it.

## What's inside

- **Backtesting engine** — candle-by-candle, no look-ahead, spot & futures, smart orders,
  stop-loss/take-profit, and 44 performance metrics.
- **Strategy API** — Jesse-compatible `Strategy`, order, position, and trade surfaces, including
  ML gather/deploy helpers and candle pipelines.
- **Drop-in strategy loading** — unchanged static `jesse.*` imports are translated at load time;
  `terry.testing_utils` and the bundled `terry-strategy-tests` agent skill provide deterministic
  lifecycle regression tests.
- **Research tools** — Rule Significance Test (bootstrap p-value), Monte Carlo (overfit/robustness),
  ML training, backtest exports/charts/benchmarking, and Optuna optimization with out-of-sample
  validation. CPU controls drive bounded local workers for significance, Monte Carlo, and
  optimization.
- **Research dashboard** — IDE-like strategy editor plus multi-route/data-route, pipeline,
  optimization, export, robustness, titled-session notes, and source-snapshot controls.
- **58 MCP tools + 12 resources** — tools for strategy creation, data import, configuration,
  backtesting, and analysis.
- **Free data + local storage** — 10 Jesse backtest exchanges through public REST + SQLite.

Terry targets Jesse's open research/backtest product surface. It does not implement Jesse's
separate live/paper-trading execution plugin or exchange-account management. See the exact,
versioned comparison in [JESSE_PARITY.md](JESSE_PARITY.md).

## Docs

- [HOW_TERRY_WORKS.md](HOW_TERRY_WORKS.md) — plain-language guide for you and for agents.
- [APIS_AND_SERVICES.md](APIS_AND_SERVICES.md) — what services are needed (almost none; all free).
- [TERRY_BUILD_PLAN.md](TERRY_BUILD_PLAN.md) — design & build plan.
- [JESSE_PARITY.md](JESSE_PARITY.md) — audited Jesse 2.5 compatibility matrix and limits.
- [AGENTS.md](AGENTS.md) — the canonical rules agents follow.

## Tests

```bash
.venv/bin/python -m pytest tests/test_engine.py -q   # unit tests (offline)
.venv/bin/python -m pytest -q                        # full offline suite
.venv/bin/python tests/test_mcp_e2e.py               # full MCP workflow (needs internet)
```

## License

For personal use.
