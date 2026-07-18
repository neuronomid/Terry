"""Self-contained HTML report per session — Terry's local stand-in for Jesse's dashboard."""
import html
import json
import os

from .. import helpers as jh


def _fmt(v, nd=2):
    try:
        f = float(v)
        if f != f:  # NaN
            return "—"
        return f"{f:,.{nd}f}"
    except (TypeError, ValueError):
        return "—" if v is None else html.escape(str(v))


def _metrics_table(metrics):
    if not metrics:
        return "<p>No metrics.</p>"
    order = ["total", "win_rate", "net_profit", "net_profit_percentage", "starting_balance",
             "finishing_balance", "sharpe_ratio", "sortino_ratio", "calmar_ratio",
             "omega_ratio", "max_drawdown", "annual_return", "expectancy_percentage",
             "longs_count", "shorts_count", "fee", "average_holding_period",
             "gross_profit", "gross_loss", "winning_streak", "losing_streak"]
    rows = []
    for k in order:
        if k in metrics:
            rows.append(f"<tr><td>{k}</td><td>{_fmt(metrics[k])}</td></tr>")
    return "<table class='m'>" + "".join(rows) + "</table>"


def _equity_svg(equity_curve, w=760, h=220):
    if not equity_curve:
        return ""
    vals = [p["value"] for p in equity_curve]
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1
    n = len(vals)
    pts = []
    for i, v in enumerate(vals):
        x = i / max(n - 1, 1) * w
        y = h - (v - lo) / rng * h
        pts.append(f"{x:.1f},{y:.1f}")
    return (f"<svg viewBox='0 0 {w} {h}' width='100%' preserveAspectRatio='none' "
            f"class='eq'><polyline fill='none' stroke='#2dd4bf' stroke-width='2' "
            f"points='{' '.join(pts)}'/></svg>")


def _trades_table(trades, limit=100):
    if not trades:
        return "<p>No closed trades.</p>"
    head = ("<tr><th>#</th><th>type</th><th>entry</th><th>exit</th><th>qty</th>"
            "<th>PNL</th><th>PNL %</th><th>fee</th><th>held (s)</th></tr>")
    rows = []
    for i, t in enumerate(trades[:limit], 1):
        cls = "pos" if t.get("PNL", 0) >= 0 else "neg"
        rows.append(
            f"<tr class='{cls}'><td>{i}</td><td>{html.escape(str(t.get('type')))}</td>"
            f"<td>{_fmt(t.get('entry_price'))}</td><td>{_fmt(t.get('exit_price'))}</td>"
            f"<td>{_fmt(t.get('qty'),4)}</td><td>{_fmt(t.get('PNL'))}</td>"
            f"<td>{_fmt(t.get('PNL_percentage'))}</td><td>{_fmt(t.get('fee'))}</td>"
            f"<td>{_fmt(t.get('holding_period'),0)}</td></tr>")
    return "<table class='t'>" + head + "".join(rows) + "</table>"


def _body_for_kind(kind, results):
    if kind == "backtest":
        return (f"<h2>Metrics</h2>{_metrics_table(results.get('metrics'))}"
                f"<h2>Equity curve</h2>{_equity_svg(results.get('equity_curve'))}"
                f"<h2>Trades ({results.get('num_trades', 0)})</h2>"
                f"{_trades_table(results.get('trades', []))}")
    if kind == "significance_test":
        r = results.get("results", {})
        return ("<h2>Rule Significance Test</h2><table class='m'>"
                + "".join(f"<tr><td>{k}</td><td>{_fmt(v) if isinstance(v,(int,float)) else html.escape(str(v))}</td></tr>"
                          for k, v in r.items()) + "</table>")
    if kind == "monte_carlo":
        parts = ["<h2>Monte Carlo</h2>"]
        c = results.get("candles")
        if c:
            parts.append(f"<p><b>Overfit verdict:</b> {html.escape(str(c.get('overfit_verdict')))} "
                         f"({c.get('num_scenarios')} scenarios)</p>")
            parts.append("<table class='m'><tr><th>metric</th><th>original</th><th>worst_5</th>"
                         "<th>median</th><th>best_5</th></tr>")
            for k, s in c.get("summary_metrics", {}).items():
                parts.append(f"<tr><td>{k}</td><td>{_fmt(s.get('original'),3)}</td>"
                             f"<td>{_fmt(s.get('worst_5'),3)}</td><td>{_fmt(s.get('median'),3)}</td>"
                             f"<td>{_fmt(s.get('best_5'),3)}</td></tr>")
            parts.append("</table>")
        tr = results.get("trades")
        if tr and tr.get("max_drawdown"):
            parts.append("<h3>Trades MC — max drawdown distribution</h3><table class='m'>"
                         + "".join(f"<tr><td>{k}</td><td>{_fmt(v)}</td></tr>"
                                   for k, v in tr["max_drawdown"].items()) + "</table>")
        return "".join(parts)
    if kind == "optimization":
        parts = [f"<h2>Optimization — objective {html.escape(str(results.get('objective')))}</h2>"]
        best = results.get("best")
        if best:
            parts.append(f"<p><b>Best hp:</b> {html.escape(json.dumps(best['hp']))} "
                         f"| train {_fmt(best.get('train_score'),3)} "
                         f"| test {_fmt(best.get('test_score'),3)}</p>")
        parts.append("<table class='t'><tr><th>#</th><th>hp</th><th>train</th><th>test</th></tr>")
        for i, c in enumerate(results.get("candidates", []), 1):
            parts.append(f"<tr><td>{i}</td><td>{html.escape(json.dumps(c['hp']))}</td>"
                         f"<td>{_fmt(c.get('train_score'),3)}</td><td>{_fmt(c.get('test_score'),3)}</td></tr>")
        parts.append("</table>")
        return "".join(parts)
    return "<p>Unknown session kind.</p>"


TEMPLATE = """<!doctype html><html><head><meta charset="utf-8">
<title>Terry — {kind} {sid}</title>
<style>
  body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#0f172a;color:#e2e8f0;margin:0;padding:24px}}
  h1{{font-size:20px}} h2{{font-size:16px;margin-top:28px;border-bottom:1px solid #334155;padding-bottom:6px}}
  .meta{{color:#94a3b8;font-size:13px;margin-bottom:16px}}
  table{{border-collapse:collapse;width:100%;font-size:13px;margin-top:8px}}
  td,th{{padding:4px 8px;border-bottom:1px solid #1e293b;text-align:left}}
  table.m{{max-width:520px}} th{{color:#94a3b8}}
  tr.pos td{{color:#4ade80}} tr.neg td{{color:#f87171}}
  .eq{{background:#0b1120;border:1px solid #1e293b;border-radius:8px}}
  code{{background:#1e293b;padding:2px 6px;border-radius:4px}}
</style></head><body>
<h1>Terry report — {kind}</h1>
<div class="meta">Session <code>{sid}</code> · {exchange} {symbol} {timeframe} · {start_date} → {finish_date} · {status}</div>
{body}
<p class="meta" style="margin-top:32px">Generated by Terry — a local Jesse-compatible research tool. Past performance does not guarantee future results.</p>
</body></html>"""


def generate_report(sid, kind, state, results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{sid}.html")
    doc = TEMPLATE.format(
        kind=html.escape(kind), sid=html.escape(sid),
        exchange=html.escape(str(state.get("exchange", ""))),
        symbol=html.escape(str(state.get("symbol", ""))),
        timeframe=html.escape(str(state.get("timeframe", ""))),
        start_date=html.escape(str(state.get("start_date", ""))),
        finish_date=html.escape(str(state.get("finish_date", "today"))),
        status="finished",
        body=_body_for_kind(kind, results),
    )
    with open(path, "w") as f:
        f.write(doc)
    return path
