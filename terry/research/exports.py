"""Backtest exports used by the pure research API and dashboard downloads."""

from __future__ import annotations

import csv
import io
import json


def trades_csv(trades: list[dict]) -> str:
    if not trades:
        return ""
    fields = [key for key in trades[0] if key != "orders"]
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(trades)
    return stream.getvalue()


def trades_json(trades: list[dict]) -> str:
    return json.dumps(trades, indent=2, allow_nan=True)


def tradingview_pine(trades: list[dict], title: str = "Terry Backtest") -> str:
    """Generate a self-contained Pine v5 overlay of trade entries and exits."""
    lines = [
        "//@version=5",
        f'indicator("{title.replace(chr(34), chr(39))}", overlay=true, max_labels_count=500)',
    ]
    for index, trade in enumerate(trades[:250]):
        side = trade.get("type", "trade")
        lines.extend([
            f"if time == {int(trade['opened_at'])}",
            f'    label.new(bar_index, {float(trade["entry_price"])}, "{side} #{index + 1}", '
            'style=label.style_label_up, color=color.new(color.blue, 0), textcolor=color.white)',
            f"if time == {int(trade['closed_at'])}",
            f'    label.new(bar_index, {float(trade["exit_price"])}, "exit #{index + 1}", '
            'style=label.style_label_down, color=color.new(color.orange, 0), textcolor=color.white)',
        ])
    return "\n".join(lines) + "\n"
