// Chart helpers for the Terry dashboard, built on the vendored TradingView
// Lightweight Charts library (window.LightweightCharts). Every chart is tracked
// so it can be disposed on navigation, and coloured to the active theme.
const LWC = () => window.LightweightCharts;
const registry = new Set();

export function disposeCharts() {
  for (const chart of registry) { try { chart.remove(); } catch (_) {} }
  registry.clear();
}
function track(chart) { registry.add(chart); return chart; }

function palette() {
  const light = document.body.classList.contains('light');
  return light ? {
    bg: '#ffffff', text: '#4d5666', grid: '#e6e8ec', border: '#d9dce1',
    up: '#16a34a', down: '#dc2626', volUp: '#16a34a55', volDown: '#dc262655',
    gold: '#c98b12', blue: '#2563eb', line: '#818CF8', crosshair: '#9aa1ad',
  } : {
    bg: 'transparent', text: '#c6cad3', grid: '#232833', border: '#2a2e38',
    up: '#4dd49b', down: '#ff6b6b', volUp: '#4dd49b40', volDown: '#ff6b6b40',
    gold: '#f9b537', blue: '#72a6ff', line: '#818CF8', crosshair: '#6b7280',
  };
}

function baseOptions(p, height) {
  return {
    height,
    layout: { background: { type: 'solid', color: p.bg }, textColor: p.text, fontSize: 11,
              fontFamily: 'Inter, ui-sans-serif, system-ui, sans-serif' },
    grid: { vertLines: { color: p.grid }, horzLines: { color: p.grid } },
    rightPriceScale: { borderColor: p.border },
    timeScale: { borderColor: p.border, timeVisible: true, secondsVisible: false },
    crosshair: { mode: 0, vertLine: { color: p.crosshair, labelBackgroundColor: p.crosshair },
                 horzLine: { color: p.crosshair, labelBackgroundColor: p.crosshair } },
    autoSize: true,
  };
}

const DASH = () => (LWC().LineStyle ? LWC().LineStyle.Dashed : 2);

// Apply a saved viewport to a live chart, or default to the most recent bars at the
// right edge. `view.follow` keeps the newest candle pinned to the right so a running
// demo scrolls in real time instead of snapping back to the start of history.
function applyLiveView(ts, view, barCount, rightPad = 4) {
  if (!barCount) return;
  const to = barCount - 1 + rightPad;
  if (view && view.follow && view.width > 0) {
    ts.setVisibleLogicalRange({ from: to - view.width, to });
  } else if (view && Number.isFinite(view.from) && Number.isFinite(view.to)) {
    ts.setVisibleLogicalRange({ from: view.from, to: view.to });
  } else {
    ts.setVisibleLogicalRange({ from: Math.max(-rightPad, to - 150), to });
  }
}

// ── Candlestick chart: price + volume + trade markers + indicator overlays ──
export function priceChart(el, data, { extraEl, live, view, onView } = {}) {
  if (!LWC() || !el) return null;
  const p = palette();
  el.innerHTML = '';
  const chart = track(LWC().createChart(el, baseOptions(p, el.clientHeight || 460)));
  const candleSeries = chart.addCandlestickSeries({
    upColor: p.up, downColor: p.down, borderUpColor: p.up, borderDownColor: p.down,
    wickUpColor: p.up, wickDownColor: p.down,
  });
  candleSeries.setData((data.candles || []).map(c => ({
    time: c.time, open: c.open, high: c.high, low: c.low, close: c.close })));

  // Volume as a bottom overlay histogram.
  const volSeries = chart.addHistogramSeries({ priceScaleId: 'vol', priceFormat: { type: 'volume' } });
  chart.priceScale('vol').applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });
  volSeries.setData((data.candles || []).map(c => ({
    time: c.time, value: c.volume, color: c.close >= c.open ? p.volUp : p.volDown })));

  // Indicator overlay lines drawn on the price scale.
  const overlays = data.overlays || {};
  const fallback = [p.gold, p.blue, '#f472b6', '#6EE7B7', '#A78BFA'];
  let ci = 0;
  for (const [title, def] of Object.entries(overlays.candle_lines || {})) {
    const s = chart.addLineSeries({ color: def.color || fallback[ci % fallback.length],
      lineWidth: 2, priceLineVisible: false, lastValueVisible: false, title });
    s.setData((def.data || []).filter(d => d.value != null));
    ci++;
  }
  for (const [title, def] of Object.entries(overlays.candle_hlines || {})) {
    candleSeries.createPriceLine({ price: def.value, color: def.color || p.text,
      lineWidth: def.line_width || 1, lineStyle: def.line_style === 'dotted' ? DASH() : 0,
      axisLabelVisible: true, title });
  }

  if (data.markers && data.markers.length) candleSeries.setMarkers(data.markers);

  // Extra sub-charts (e.g. RSI) stacked below, time-synced with the price chart.
  const extras = Object.entries(overlays.extra_charts || {});
  if (extraEl) {
    extraEl.innerHTML = '';
    for (const [name, def] of extras) {
      const wrap = document.createElement('div');
      wrap.className = 'extra-chart';
      wrap.innerHTML = `<span class="extra-chart-label">${name}</span>`;
      const host = document.createElement('div');
      host.className = 'extra-chart-host';
      wrap.appendChild(host);
      extraEl.appendChild(wrap);
      const sub = track(LWC().createChart(host, baseOptions(p, 150)));
      let si = 0;
      for (const [title, ldef] of Object.entries(def.lines || {})) {
        const s = sub.addLineSeries({ color: ldef.color || fallback[si % fallback.length],
          lineWidth: 2, priceLineVisible: false, lastValueVisible: true, title });
        s.setData((ldef.data || []).filter(d => d.value != null));
        si++;
      }
      const anchor = sub.addLineSeries({ visible: false });
      anchor.setData((data.candles || []).map(c => ({ time: c.time, value: 0 })));
      for (const [title, hdef] of Object.entries(def.hlines || {})) {
        anchor.createPriceLine({ price: hdef.value, color: hdef.color || p.text,
          lineWidth: hdef.line_width || 1, lineStyle: hdef.line_style === 'dotted' ? DASH() : 0,
          axisLabelVisible: true, title });
      }
      syncTimeScales(chart, sub);
      if (!live) sub.timeScale().fitContent();
    }
  }

  // Set the viewport last so synced sub-charts inherit it. A live demo preserves the
  // caller's saved range (and keeps following the newest candle); everything else fits all.
  const ts = chart.timeScale();
  const barCount = (data.candles || []).length;
  if (live) {
    applyLiveView(ts, view, barCount);
    if (onView) ts.subscribeVisibleLogicalRangeChange(r => { if (r) onView(r, barCount); });
  } else {
    ts.fitContent();
  }
  return chart;
}

function syncTimeScales(a, b) {
  let guard = false;
  const link = (src, dst) => src.timeScale().subscribeVisibleLogicalRangeChange(range => {
    if (guard || !range) return; guard = true;
    try { dst.timeScale().setVisibleLogicalRange(range); } catch (_) {}
    guard = false;
  });
  link(a, b); link(b, a);
}

// ── Multi-series line chart (equity curve, cumulative returns) ──
export function lineChart(el, series, { height, area } = {}) {
  if (!LWC() || !el) return null;
  const p = palette();
  el.innerHTML = '';
  const chart = track(LWC().createChart(el, baseOptions(p, height || el.clientHeight || 300)));
  const colors = [p.line, p.gold, '#fb7185', p.blue, '#f472b6', '#A78BFA'];
  (series || []).forEach((s, i) => {
    const color = s.color || colors[i % colors.length];
    const ser = (area && i === 0)
      ? chart.addAreaSeries({ lineColor: color, topColor: color + '55', bottomColor: color + '05',
          lineWidth: 2, priceLineVisible: false, title: s.name })
      : chart.addLineSeries({ color, lineWidth: i === 0 ? 2 : 1.5,
          lineStyle: i === 0 ? 0 : DASH(), priceLineVisible: false, title: s.name });
    ser.setData((s.data || []).filter(d => d.value != null && isFinite(d.value)));
  });
  chart.timeScale().fitContent();
  return chart;
}

// ── Area chart for the underwater / drawdown plot ──
export function areaChart(el, points, { color, height } = {}) {
  if (!LWC() || !el) return null;
  const p = palette();
  color = color || p.down;
  el.innerHTML = '';
  const chart = track(LWC().createChart(el, baseOptions(p, height || 220)));
  const s = chart.addAreaSeries({ lineColor: color, topColor: color + '10',
    bottomColor: color + '55', lineWidth: 1, invertFilledArea: true, priceLineVisible: false });
  s.setData((points || []).filter(d => d.value != null && isFinite(d.value)));
  chart.timeScale().fitContent();
  return chart;
}

// ── Monte Carlo fan chart: faint scenario curves + bold original ──
export function montecarloChart(el, { original, scenarios }, { height } = {}) {
  if (!LWC() || !el) return null;
  const p = palette();
  el.innerHTML = '';
  const chart = track(LWC().createChart(el, baseOptions(p, height || 360)));
  const norm = curve => {
    if (!curve) return null;
    const raw = Array.isArray(curve.equity_curve) && curve.equity_curve[0]?.data
      ? curve.equity_curve[0].data : (curve.data || curve);
    return (raw || []).map((d, i) => ({
      time: Number.isFinite(d.time) && d.time > 1e6 ? Math.floor(d.time > 1e11 ? d.time / 1000 : d.time) : i,
      value: d.value })).filter(d => d.value != null && isFinite(d.value));
  };
  // Reindex scenarios onto sequential integers so overlapping equity paths align.
  const asIndexed = pts => pts.map((d, i) => ({ time: i, value: d.value }));
  (scenarios || []).slice(0, 300).forEach(sc => {
    const pts = norm(sc); if (!pts || pts.length < 2) return;
    const s = chart.addLineSeries({ color: p.line + '22', lineWidth: 1,
      priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
    s.setData(asIndexed(pts));
  });
  const op = norm(original);
  if (op && op.length) {
    const s = chart.addLineSeries({ color: p.gold, lineWidth: 3, priceLineVisible: false,
      lastValueVisible: true, title: 'Original' });
    s.setData(asIndexed(op));
  }
  chart.applyOptions({ timeScale: { timeVisible: false, secondsVisible: false } });
  chart.timeScale().fitContent();
  return chart;
}
