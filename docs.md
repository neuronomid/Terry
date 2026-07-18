# Terry — Agent & Usage Documentation

Single source of truth for connecting to and operating **Terry** over MCP. Terry is a local,
self-contained Jesse-compatible crypto research framework: build, backtest, and stress-test
strategies locally. Free public exchange data, SQLite storage, no cloud/keys.

- **Version 0.2.2** · **174 indicator modules** · **58 MCP tools** · **12 resources**
- **Transport:** streamable-HTTP · **URL:** `http://localhost:9021/mcp`
- Long **and** short, spot **and** futures. Simulation only — never places real orders.

## 1. Start (one command)
    ./run.sh      # venv+deps+init+serve (--port 9030 to change port)

Browser dashboard (separate local process):
    .venv/bin/python -m terry dashboard    # http://127.0.0.1:9020
Optional browser password:
    TERRY_DASHBOARD_PASSWORD='choose-a-local-password' .venv/bin/python -m terry dashboard

## 2. Connect an agent
    claude mcp add --transport http terry http://localhost:9021/mcp      # then /mcp to verify
Any MCP client: add an HTTP server at http://localhost:9021/mcp (see .mcp.json). Agents also read AGENTS.md.

## 3. Session model (draft -> run -> poll)
Backtests, significance tests, Monte Carlo, and optimization all share it:
1. create_*_draft(...) -> returns session_id (`status=success`, `session_status=draft`).
2. run_*(session_id) -> returns immediately (runs in background).
3. Poll get_*_session(session_id) until status is finished | stopped | terminated | canceled.
4. Finished -> results + dashboard_url (file:///.../storage/reports/<id>.html; always surface it).
   Stopped -> results.error / results.message (commonly missing_candles).
These runs are free/unlimited. Don't block — keep polling.

## 4. Recommended workflow
import candles (poll import status) -> write a minimal entry-only strategy -> significance-test the
entry (need p<0.05) -> build the full strategy -> backtest -> iterate small changes -> Monte Carlo
(overfit check) -> optionally optimize -> validate out-of-sample -> surface every dashboard_url.
Don't pre-check candles before a backtest: run it; if it stops missing_candles, import starting
~2 months before start_date and re-run.

## 5. Tools (58). `*` = required; other args default from get_config.

Status: get_terry_status(), greet_user(name*)

Strategies: create_strategy(name*, content*), read_strategy(name*), write_strategy(name*, content*)
— write/read strategies/<name>/__init__.py; validated on write.

Config: get_config(), update_config(config*) (partial JSON string; user-driven only),
get_backtest_config(), get_optimization_config(), get_live_config() (live not implemented).

Candles: import_candles(exchange*, symbol*, start_date*, import_id, finish_date) -> {import_id};
get_candle_import_status(import_id*) (poll to finished); cancel_candle_import(import_id*);
get_candles(exchange*, symbol*, timeframe*); get_existing_candles();
delete_candles(exchange*, symbol*); clear_candle_cache().

Indicators: list_indicators(), get_indicator_details(indicator_name*).

Backtest: create_backtest_draft(exchange, routes, data_routes, start_date, finish_date,
debug_mode, export_csv, export_json, export_chart, export_tradingview, fast_mode, benchmark,
notes fields; Terry shorthand strategy/symbol/timeframe/config is also accepted),
run_backtest(session_id*), get_backtest_session(session_id*) (-> results.metrics 44 keys,
results.trades, results.equity_curve, dashboard_url),
get_backtest_sessions(limit, offset, title_search, status_filter, date_filter),
update_backtest_draft(backtest_id*, state*),
update_backtest_notes(session_id*, title, description, strategy_codes),
cancel_backtest(session_id*), purge_backtest_sessions(days_old).

Significance test: create_significance_test_draft(exchange, routes, data_routes, start_date,
finish_date, n_simulations, random_seed, title, description, strategy_summary, hypothesis,
rationale, plus Terry shorthand strategy/symbol/timeframe/config and cpu_cores),
run_significance_test(session_id*), get_significance_test_session(session_id*)
(-> results.results = {observed_mean, annualized_return, p_value, n_simulations, n_observations,
significant, verdict}), plus get_significance_test_sessions / update_significance_test_draft /
update_significance_test_notes / cancel_significance_test / purge_significance_test_sessions.
Interpret p_value: <0.05 significant; 0.05-0.10 borderline; >0.10 hard stop (random).

Monte Carlo: create_monte_carlo_draft(exchange, routes, data_routes, start_date, finish_date,
num_scenarios, run_trades, run_candles, fast_mode, cpu_cores, pipeline_type, pipeline_params,
notes fields; Terry shorthand strategy/symbol/timeframe/config is also accepted)
(defaults num_scenarios=200, run_candles=True, run_trades=False), run_monte_carlo(session_id*),
get_monte_carlo_session(session_id*) (-> results.candles.summary_metrics + overfit_verdict,
results.trades.max_drawdown if run_trades), get_monte_carlo_equity_curves(session_id*),
get_monte_carlo_logs(session_id*), resume_monte_carlo(session_id*), plus
get_monte_carlo_sessions / update_monte_carlo_draft / update_monte_carlo_notes / cancel_monte_carlo /
terminate_monte_carlo / purge_monte_carlo_sessions. Overfit (sharpe): original>best_5 ->
overfit_suspect; original<=median -> robust; report worst_5.

Optimization: create_optimization_draft(...) accepts Terry's strategy/symbol/start/finish shorthand
or Jesse-style JSON routes/data_routes with separate training_start_date/training_finish_date and
testing_start_date/testing_finish_date. It also accepts objective_function, trials per
hyperparameter, best_candidates_count, warm_up_candles, fast_mode, and cpu_cores. The strategy
must define hyperparameters().
run_optimization(session_id*), get_optimization_session(session_id*) (-> results.best {hp,
train_score, test_score, ...} and results.candidates validated out-of-sample),
rerun_optimization(session_id*), get_optimization_logs(session_id*), plus
get_optimization_sessions / update_optimization_draft / update_optimization_notes / cancel_optimization /
terminate_optimization / purge_optimization_sessions.

Session tools return Jesse-compatible success/error actions and draft/session envelopes while
retaining Terry's top-level convenience fields. Notes persist title, description, and automatic
strategy-code snapshots. Every session-list tool supports Jesse's limit/offset and
title/status/date filters.

Draft state JSON fields: strategy, symbol, timeframe, exchange, start_date, finish_date,
config (engine overrides), and per-kind: n_simulations · num_scenarios/run_candles/run_trades ·
objective/n_trials/train_test_split · hyperparameters.

## 6. Resources
terry://strategy, terry://strategy_examples, terry://indicator, terry://position_risk,
terry://utilities, terry://backtest_management, terry://backtest_metrics, terry://candle,
terry://configuration, terry://significance_test, terry://monte_carlo,
terry://optimization.

## 7. Writing strategies
Class in strategies/<Name>/__init__.py; methods run once per candle after close (no look-ahead).

Files may use the Terry imports below or unchanged static Jesse imports such as
`from jesse.strategies import Strategy`, `import jesse.indicators as ta`, and
`from jesse import utils`; Terry translates those imports at load time.

    from terry.strategies import Strategy
    import terry.indicators as ta
    from terry import utils

    class EmaTrend(Strategy):
        def should_long(self):  return ta.ema(self.candles, 20) > ta.ema(self.candles, 50)   # required
        def should_short(self): return ta.ema(self.candles, 20) < ta.ema(self.candles, 50)   # default False; spot must be False
        def go_long(self):                                                                     # required
            self.buy  = utils.size_to_qty(self.available_margin*0.5, self.price, fee_rate=self.fee_rate), self.price
        def go_short(self):
            self.sell = utils.size_to_qty(self.available_margin*0.5, self.price, fee_rate=self.fee_rate), self.price
        def on_open_position(self, order):
            atr = ta.atr(self.candles)
            if self.is_long:  self.stop_loss = self.position.qty, self.price-2*atr; self.take_profit = self.position.qty, self.price+4*atr
            elif self.is_short: self.stop_loss = self.position.qty, self.price+2*atr; self.take_profit = self.position.qty, self.price-4*atr
        def update_position(self):  # each candle while in a position; trail stops or self.liquidate()
            pass

- Smart orders: self.buy/sell = qty, price; type inferred (equal->MARKET; BUY below->LIMIT,
  above->STOP; mirror for SELL). Lists of (qty,price) for scaling.
- Exits: self.stop_loss, self.take_profit, self.liquidate(). Spot: set stops in on_open_position
  (not go_long). Trailing: reassign self.stop_loss in update_position; read with
  self.average_stop_loss (no self.trailing_stop).
- Long+short: futures exchange ("Binance Perpetual Futures") -> both; spot ("Binance Spot") ->
  long-only (should_short must be False, else InvalidShortSellOnSpot).
- Sizing: utils.size_to_qty(size, price, precision=3, fee_rate=0),
  utils.risk_to_qty(capital, risk_pct, entry, stop, fee_rate=0) (risk_pct is a percent). Never
  hardcode qty. self.position.qty is 0 inside go_long/go_short.
- Hooks: before/after, on_open_position(order), on_close_position(order, closed_trade) (two args),
  on_increased_position, on_reduced_position, route variants.
- Execution: flat positions evaluate entries even right after a same-candle close (flips
  supported). At backtest end, open positions are force-closed at the last price and counted as
  closed trades (+ in total_open_trades/open_pl) — matches Jesse.
- Hyperparameters (for optimization): hyperparameters() -> list of
  {name, type(int|float|"categorical"), min, max, step?, options?, default}; read via self.hp[name].

## 8. Indicators (174 modules, matching Jesse 2.5's public module set)

    import terry.indicators as ta
    v = ta.sma(self.candles, 20)                    # latest scalar
    s = ta.sma(self.candles, 20, sequential=True)   # full np.ndarray
    up, mid, low = ta.bollinger_bands(self.candles, 20)   # multi-line -> named tuple

Default source close; discover via list_indicators / get_indicator_details. Full set: acosc, ad,
adosc, adx, adxr, alligator, alma, ao, apo, aroon, aroonosc, atr, avgprice, bandpass, beta,
bollinger_bands, bollinger_bands_width, bop, cc, cci, cfo, cg, chande, chop, cksp, cmo, correl,
correlation_cycle, cvi, cwma, damiani_volatmeter, dec_osc, decycler, dema, devstop, di, dm, donchian,
dpo, dti, dx, edcf, efi, ema, emd, emv, epma, er, eri, fisher, fosc, frama, fwma, gatorosc, gauss,
heikin_ashi_candles, high_pass, high_pass_2_pole, hma, hull_suit, hurst_exponent, hwma,
ichimoku_cloud, ichimoku_cloud_seq, ift_rsi, itrend, jma, jsa, kama, kaufmanstop, kdj, keltner, kst,
kurtosis, kvo, linearreg, linearreg_angle, linearreg_intercept, linearreg_slope, lrsi, ma, maaq, mab,
macd, mama, marketfi, mass, mcginley_dynamic, mean_ad, median_ad, medprice, mfi, midpoint, midprice,
minmax, mom, mwdx, natr, nma, nvi, obv, pfe, pivot, pma, ppo, pvi, pwma, qstick, reflex, rma, roc,
rocp, rocr, rocr100, roofing, rsi, rsmk, rsx, rvi, safezonestop, sar, sinwma, skew, sma, smma,
squeeze_momentum, sqwma, srsi, srwma, stc, stddev, stiffness, stoch, stochf, supersmoother,
supersmoother_3_pole, supertrend, support_resistance_with_breaks, swma, t3, tema, trange, trendflex,
trima, trix, tsf, tsi, ttm_squeeze, ttm_trend, typprice, ui, ultosc, var, vi, vidya, vlma, volume,
vosc, voss, vpci, vpt, vpwma, vwap, vwma, vwmacd, wad, waddah_attar_explosion, wclprice, wilders,
willr, wma, wt, zlema, zscore. beta/rsmk take a second candle series.

## 9. Config (get_config/update_config)
Keys: exchange, starting_balance, fee, type("futures"|"spot"), futures_leverage,
futures_leverage_mode("cross"|"isolated"), quote_asset, warm_up_candles, plus optimization,
monte_carlo, significance_test sub-objects. A session's exchange sets its market type unless
overridden in its config. Historical drivers cover Binance Spot, Binance US Spot, Binance
Perpetual Futures, Bitfinex Spot, Coinbase Spot, Bybit USDT Perpetual, Bybit USDC Perpetual,
Bybit Spot, Gate USDT Perpetual, and Kraken Pro Futures. `Binance` and `Binance USDT Perpetual`
remain aliases for older Terry projects.

## 10. Metrics (44 in results.metrics)
total, total_winning_trades, total_losing_trades, starting_balance, finishing_balance, win_rate,
win_rate_longs, win_rate_shorts, ratio_avg_win_loss, longs_count, longs_percentage,
shorts_percentage, shorts_count, fee, net_profit, net_profit_percentage, average_win, average_loss,
expectancy, expectancy_percentage, expected_net_profit_every_100_trades, average_holding_period,
average_winning_holding_period, average_losing_holding_period, gross_profit, gross_loss,
max_drawdown, max_underwater_period, annual_return, sharpe_ratio, calmar_ratio, sortino_ratio,
omega_ratio, serenity_index, total_open_trades, open_pl, winning_streak, losing_streak,
largest_losing_trade, largest_winning_trade, current_streak, avg_trades_per_day/week/month. Trade
dict: id, strategy_name, symbol, exchange, type, entry_price, exit_price, qty, size, PNL,
PNL_percentage, fee, holding_period (s), opened_at, closed_at, orders.

## 11. Data
Keyless public REST APIs from Binance, Bitfinex, Coinbase, Bybit, Gate.io, and Kraken. Terry stores
1m candles, aggregates larger timeframes on the fly, rate-limits pagination, and deduplicates
re-imports in SQLite. Exchange availability and historical depth remain subject to each provider.

## 12. Troubleshooting
- Backtest stopped missing_candles -> import (start ~2 months early), re-run.
- strategy_not_found on draft -> create_strategy first.
- InvalidShortSellOnSpot -> use a futures exchange or should_short -> False.
- not_draft on update -> only drafts are editable; make a new draft.
- Agent can't see tools -> ensure run.sh/terry serve is running at http://localhost:9021/mcp.

## 13. Fidelity vs Jesse
The audit baseline is Jesse 2.5.0 commit `fa63531cae6c09b978711dc1892285067304e2df`
(2026-07-13). Terry matches its 174 public indicator modules, 44 metric keys, all public Strategy
method/property names, 58 MCP tool names (with Terry product naming for status), and 12 resource
topics. `size_to_qty` now uses Jesse's three-fee reserve and precision floor. Research includes ML,
candle pipelines, benchmark and export formats, chart packs, multi-route/data-route backtests, and
Optuna train/test optimization. Research workers honor cpu_cores locally; the dashboard exposes
route/data-route JSON, Monte Carlo pipelines, optimization controls, and an IDE-like editor. See
[JESSE_PARITY.md](JESSE_PARITY.md) for evidence and differences.

Developer parity includes `terry.testing_utils.single_route_backtest`,
`two_routes_backtest`, and `two_data_routes_backtest`. The project-level
`terry-strategy-tests` skill is shared through `.claude/skills` and `.agents/skills`.

This is research compatibility, not total product parity: Terry deliberately uses SQLite and a
local FastAPI/JavaScript dashboard, uses bounded local threads instead of Ray, and does not contain
Jesse's separate live/paper exchange-execution plugin, account management, or live notifications.

Dashboard API requests are same-origin, use strict validation and security headers, and can be
protected by an HttpOnly same-site session cookie via `TERRY_DASHBOARD_PASSWORD`.

## 14. Not available
Live/paper trading (simulation only; the dashboard labels it unavailable). Gmail/Calendar/Drive MCP connectors are unrelated to Terry and
require the user to authorize them in their own client.

## 15. Keep this file current
Whenever Terry changes (new/renamed tool or arg, new indicator, config key, behavior), update
docs.md in the same change. Keep the header stats, section 5 tools, section 8 indicators, section 9
config, and section 10 metrics in sync.

### Changelog
- 0.2.2 — Jesse-compatible MCP schemas/envelopes, session filters and source snapshots; unchanged
  `jesse.*` strategy imports; testing helpers and shared agent skill; titled dashboard sessions.
- 0.2.1 — concurrent local research workers; Jesse fitness and ML diagnostics; isolated shared
  strategy state; route/data-route and pipeline parity in MCP/dashboard; IDE-like browser editor.
- 0.2.0 — Jesse 2.5 audit; strategy/model/util compatibility fixes; candle pipelines and ML;
  charts, benchmark, CSV/JSON/Pine exports; Optuna OOS optimization; multi-route MCP drafts;
  10 historical exchange drivers; 12th optimization resource; parity regression suite.
- 0.1.0 — 58 tools, 11 resources, 174 indicators, long+short + spot/futures
  backtesting, significance/Monte-Carlo/optimization, SQLite, Binance free data, run.sh launcher.
  Engine validated 19/19 trades vs Jesse (same-candle flips + end-of-backtest force-close added).
