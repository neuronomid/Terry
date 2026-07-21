"""FastAPI dashboard for the local Terry research engine.

The dashboard deliberately calls the same Context/Runner services used by MCP.  It is a
browser client for local research only: no exchange credentials or live-trading controls are
implemented here.
"""
from __future__ import annotations

import base64
import bisect
import csv
import io
import json
import math
import os
import re
import secrets
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Cookie, Depends, FastAPI, Header, HTTPException, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .. import indicators as ta
from .. import helpers as jh
from ..context import TerryContext, set_context
from ..data.binance import EXCHANGES
from ..loader import load_strategy_class, strategy_exists
from ..sessions.db import TERMINAL, VALID_KINDS
from ..version import __version__

_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,79}$")
_SYMBOL = re.compile(r"^[A-Z0-9]+-[A-Z0-9]+$")
_STATIC = Path(__file__).with_name("static")
_AUTH_COOKIE = "terry_dashboard_session"
_OBJECTIVES = {
    "sharpe", "sharpe_ratio", "sortino", "sortino_ratio", "calmar", "calmar_ratio",
    "omega", "omega_ratio", "serenity", "serenity_index", "smart sharpe",
    "smart sortino", "net_profit_percentage",
}
_SESSION_KINDS = ("backtest", "demo", "optimization", "monte_carlo", "significance_test")
_PIPELINE_TYPES = {"moving_block_bootstrap", "gaussian_noise", "gaussian_resampler"}
_ENGINE_CONFIG_KEYS = {
    "starting_balance", "fee", "type", "futures_leverage", "futures_leverage_mode",
    "quote_asset", "warm_up_candles",
}


def _error(status: int, message: str, code: str = "invalid_request") -> HTTPException:
    return HTTPException(status_code=status, detail={"error": code, "message": message})


def _short_qty(qty: Any) -> str:
    try:
        return f"{abs(float(qty)):g}"
    except (TypeError, ValueError):
        return ""


def _require_name(name: str) -> str:
    if not _NAME.fullmatch(name or ""):
        raise _error(422, "Strategy names must start with a letter and contain only letters, numbers, or underscores.")
    return name


def _date(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise _error(422, f"{label} is required and must use YYYY-MM-DD.")
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise _error(422, f"{label} must use YYYY-MM-DD.") from exc
    return value


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    """Parse an API integer without accepting booleans, fractions, or ambiguous text."""
    if isinstance(value, bool):
        raise _error(422, f"{label} must be an integer of at least {minimum}.")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _error(422, f"{label} must be an integer of at least {minimum}.") from exc
    if isinstance(value, float) and not value.is_integer():
        raise _error(422, f"{label} must be an integer of at least {minimum}.")
    if isinstance(value, str) and not re.fullmatch(r"[+-]?\d+", value.strip()):
        raise _error(422, f"{label} must be an integer of at least {minimum}.")
    if parsed < minimum:
        raise _error(422, f"{label} must be an integer of at least {minimum}.")
    return parsed


def _number(value: Any, label: str, minimum: float = 0, *, exclusive: bool = False,
            maximum: float | None = None) -> float:
    if isinstance(value, bool):
        raise _error(422, f"{label} must be a finite number.")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _error(422, f"{label} must be a finite number.") from exc
    if not math.isfinite(parsed) or (parsed <= minimum if exclusive else parsed < minimum):
        comparator = "greater than" if exclusive else "at least"
        raise _error(422, f"{label} must be {comparator} {minimum}.")
    if maximum is not None and parsed > maximum:
        raise _error(422, f"{label} must be no greater than {maximum}.")
    return parsed


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise _error(422, f"{label} must be true or false.")
    return value


def _symbol(value: Any) -> str:
    symbol = str(value or "BTC-USDT").upper()
    if not _SYMBOL.fullmatch(symbol):
        raise _error(422, "symbol must use BASE-QUOTE format, for example BTC-USDT.")
    return symbol


def _timeframe(value: Any) -> str:
    timeframe = str(value or "4h")
    try:
        jh.timeframe_to_one_minutes(timeframe)
    except Exception as exc:
        raise _error(422, f"Unsupported timeframe: {timeframe}.") from exc
    return timeframe


def _routes(ctx: TerryContext, value: Any, exchange: str, *, trading: bool) -> list[dict]:
    label = "routes" if trading else "data_routes"
    if value is None:
        return []
    if not isinstance(value, list):
        raise _error(422, f"{label} must be a JSON array.")
    output = []
    allowed = {"exchange", "symbol", "timeframe"} | ({"strategy"} if trading else set())
    required = {"symbol", "timeframe"} | ({"strategy"} if trading else set())
    for index, route in enumerate(value):
        if not isinstance(route, dict) or not required.issubset(route):
            fields = ", ".join(sorted(required))
            raise _error(422, f"{label}[{index}] must be an object containing {fields}.")
        unknown = set(route) - allowed
        if unknown:
            raise _error(422, f"Unknown {label}[{index}] field(s): {', '.join(sorted(unknown))}.")
        route_exchange = str(route.get("exchange") or exchange)
        if route_exchange not in EXCHANGES:
            raise _error(422, f"Unknown exchange in {label}[{index}]: {route_exchange}.")
        if route_exchange != exchange:
            raise _error(422, "All routes in one research run must use the selected exchange.")
        normalized = {
            "exchange": route_exchange,
            "symbol": _symbol(route["symbol"]),
            "timeframe": _timeframe(route["timeframe"]),
        }
        if trading:
            strategy = _require_name(str(route["strategy"]))
            if not strategy_exists(ctx.strategies_dir, strategy):
                raise _error(404, f'Strategy "{strategy}" does not exist.',
                             "strategy_not_found")
            normalized["strategy"] = strategy
        output.append(normalized)
    return output


def _validate_engine_config(payload: dict[str, Any]) -> dict[str, Any]:
    unknown = set(payload) - _ENGINE_CONFIG_KEYS
    if unknown:
        raise _error(422, f"Unknown engine configuration key(s): {', '.join(sorted(unknown))}.")
    out = dict(payload)
    if "starting_balance" in out:
        out["starting_balance"] = _number(out["starting_balance"], "starting_balance", exclusive=True)
    if "fee" in out:
        out["fee"] = _number(out["fee"], "fee", maximum=1)
    if "futures_leverage" in out:
        out["futures_leverage"] = _number(out["futures_leverage"], "futures_leverage", 1)
    if "warm_up_candles" in out:
        out["warm_up_candles"] = _integer(out["warm_up_candles"], "warm_up_candles")
    if out.get("type") not in (None, "spot", "futures"):
        raise _error(422, "type must be either spot or futures.")
    if out.get("futures_leverage_mode") not in (None, "cross", "isolated"):
        raise _error(422, "futures_leverage_mode must be either cross or isolated.")
    if "quote_asset" in out:
        quote = str(out["quote_asset"]).upper()
        if not re.fullmatch(r"[A-Z0-9]{2,12}", quote):
            raise _error(422, "quote_asset must contain 2–12 letters or numbers.")
        out["quote_asset"] = quote
    return out


def _validation_error(ctx: TerryContext, name: str) -> str | None:
    try:
        load_strategy_class(name, ctx.strategies_dir)
    except Exception as exc:  # a saved draft should remain editable even when incomplete
        return f"{type(exc).__name__}: {exc}"
    return None


def _strategy_path(ctx: TerryContext, name: str) -> Path:
    _require_name(name)
    return Path(ctx.strategies_dir, name, "__init__.py")


def _strategy_template(name: str) -> str:
    return f'''from terry.strategies import Strategy
import terry.indicators as ta
from terry import utils


class {name}(Strategy):
    def should_long(self):
        return ta.sma(self.candles, 10) > ta.sma(self.candles, 30)

    def should_short(self):
        return ta.sma(self.candles, 10) < ta.sma(self.candles, 30)

    def go_long(self):
        qty = utils.size_to_qty(self.available_margin * 0.5, self.price, fee_rate=self.fee_rate)
        self.buy = qty, self.price

    def go_short(self):
        qty = utils.size_to_qty(self.available_margin * 0.5, self.price, fee_rate=self.fee_rate)
        self.sell = qty, self.price
'''


_STRATEGY_BUNDLE_VERSION = 1
_MAX_BUNDLE_BYTES = 8 * 1024 * 1024            # compressed upload cap
_MAX_BUNDLE_UNCOMPRESSED = 32 * 1024 * 1024    # extracted cap (zip-bomb guard)
_MAX_BUNDLE_FILES = 400
_BUNDLE_SKIP_DIRS = {"__pycache__"}


def _iter_strategy_files(folder: Path):
    """Yield (relative_posix_path, absolute_path) for each real file in a strategy
    folder, skipping caches and compiled artifacts."""
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(folder)
        if any(part in _BUNDLE_SKIP_DIRS for part in rel.parts):
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        yield rel.as_posix(), path


def _build_strategy_bundle(ctx: TerryContext, name: str) -> bytes:
    """Zip a strategy folder into a portable bundle: a manifest plus every source
    file so importing it into another Terry project recreates the strategy in place."""
    folder = Path(ctx.strategies_dir, name)
    files = list(_iter_strategy_files(folder))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        manifest = {
            "terry_strategy_bundle": _STRATEGY_BUNDLE_VERSION,
            "name": name,
            "terry_version": __version__,
            "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "files": [rel for rel, _ in files],
        }
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))
        for rel, path in files:
            archive.write(path, f"files/{rel}")
    return buffer.getvalue()


def _clean_rel_parts(rel: str) -> list[str] | None:
    """Split an uploaded relative path into safe components, or None if it should be
    skipped (empty, traversal, cache/compiled artifact, OS junk)."""
    rel = (rel or "").replace("\\", "/").strip().lstrip("/")
    parts = [p for p in rel.split("/") if p not in ("", ".")]
    if not parts or any(p == ".." for p in parts):
        return None
    if any(p in _BUNDLE_SKIP_DIRS for p in parts):
        return None
    leaf = parts[-1]
    if leaf.endswith((".pyc", ".pyo")) or leaf in {".DS_Store", "Thumbs.db"}:
        return None
    return parts


def _safe_bundle_target(base: Path, rel: str) -> Path:
    """Resolve an entry under `base`, rejecting absolute paths and traversal."""
    parts = _clean_rel_parts(rel)
    if not parts:
        raise _error(422, "The strategy upload contains an invalid file path.")
    base_resolved = base.resolve()
    candidate = base_resolved.joinpath(*parts).resolve()
    if candidate != base_resolved and base_resolved not in candidate.parents:
        raise _error(422, "The strategy upload contains an unsafe file path.")
    return candidate


def _locate_strategy_root(files: dict[str, bytes]) -> tuple[str | None, dict[str, bytes]]:
    """Normalise an arbitrary set of uploaded files (posix relpaths → bytes) into a strategy
    payload rooted at the folder holding __init__.py.

    Accepts, in order of preference: a Terry export bundle (manifest.json + files/…), a plain
    zip/folder of the strategy directory (…/Name/__init__.py), or the bare strategy contents
    (__init__.py at the top).  Returns (name_hint, payload) where payload keys are relative to
    the strategy root; name_hint is the enclosing folder name or bundle manifest name if known.
    """
    norm: dict[str, bytes] = {}
    for rel, data in files.items():
        parts = _clean_rel_parts(rel)
        if parts is not None:
            norm["/".join(parts)] = data
    if not norm:
        return None, {}

    # Case A — a Terry export bundle: manifest.json marker + a files/ payload beside it.
    for key, data in list(norm.items()):
        if key.rsplit("/", 1)[-1] != "manifest.json":
            continue
        try:
            manifest = json.loads(data)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(manifest, dict) or "terry_strategy_bundle" not in manifest:
            continue
        prefix = key[: -len("manifest.json")] + "files/"
        payload = {k[len(prefix):]: v for k, v in norm.items()
                   if k.startswith(prefix) and k != prefix}
        if payload:
            hint = str(manifest.get("name") or "") or None
            return hint, payload

    # Case B — locate the shallowest __init__.py and treat its directory as the root.
    init_dirs = sorted(
        (k.rsplit("/", 1)[0] if "/" in k else "" for k in norm
         if k.rsplit("/", 1)[-1] == "__init__.py"),
        key=lambda d: (d.count("/") + 1) if d else 0)
    if not init_dirs:
        return None, {}
    root = init_dirs[0]
    prefix = (root + "/") if root else ""
    payload = {k[len(prefix):]: v for k, v in norm.items() if k.startswith(prefix)}
    name_hint = root.rsplit("/", 1)[-1] if root else None
    return name_hint, payload


def _install_strategy(ctx: TerryContext, name: str, original: str | None,
                      payload: dict[str, bytes], overwrite: bool) -> tuple[str, list[str]]:
    """Write a normalised strategy payload into strategies/<name>/ atomically."""
    if not payload:
        raise _error(422, "No strategy files were found in the upload.")
    if len(payload) > _MAX_BUNDLE_FILES:
        raise _error(422, "The strategy upload contains too many files.")
    if sum(len(v) for v in payload.values()) > _MAX_BUNDLE_UNCOMPRESSED:
        raise _error(422, "The strategy upload expands to too much data.")
    if "__init__.py" not in payload:
        raise _error(422, "The strategy upload has no __init__.py entry point.")
    target_dir = Path(ctx.strategies_dir, name)
    if target_dir.exists() and not overwrite:
        raise _error(409, f'Strategy "{name}" already exists. Enable overwrite to replace it.',
                     "exists")
    # Stage into a temp folder first so a bad upload never leaves a half-written strategy.
    staging_root = Path(tempfile.mkdtemp(dir=ctx.strategies_dir))
    written: list[str] = []
    try:
        staging_dir = staging_root / "payload"
        staging_dir.mkdir()
        for rel, data in payload.items():
            dest = _safe_bundle_target(staging_dir, rel)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            written.append(dest.relative_to(staging_dir).as_posix())
        # Keep folder name and class name consistent when importing under a new name.
        if name != original and original:
            init_path = staging_dir / "__init__.py"
            text = init_path.read_text(encoding="utf-8")
            init_path.write_text(
                re.sub(rf"class\s+{re.escape(original)}\b", f"class {name}", text, count=1),
                encoding="utf-8")
        if target_dir.exists():
            shutil.rmtree(target_dir)
        os.replace(staging_dir, target_dir)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
    return name, sorted(written)


def _files_from_zip(raw: bytes) -> dict[str, bytes]:
    """Read a zip upload into a {relpath: bytes} map, guarding size and count."""
    if len(raw) > _MAX_BUNDLE_BYTES:
        raise _error(422, "The strategy upload is too large.")
    try:
        archive = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile as exc:
        raise _error(422, "The uploaded file is not a valid .zip archive.") from exc
    with archive:
        entries = [item for item in archive.infolist() if not item.is_dir()]
        if len(entries) > _MAX_BUNDLE_FILES:
            raise _error(422, "The strategy upload contains too many files.")
        if sum(item.file_size for item in entries) > _MAX_BUNDLE_UNCOMPRESSED:
            raise _error(422, "The strategy upload expands to too much data.")
        files: dict[str, bytes] = {}
        for item in entries:
            files[item.filename] = archive.read(item)
    return files


def _extract_strategy_bundle(ctx: TerryContext, files: dict[str, bytes],
                             name_override: str | None, overwrite: bool) -> tuple[str, list[str]]:
    """Normalise + install an uploaded strategy (zip bundle, plain zip, or folder)."""
    original, payload = _locate_strategy_root(files)
    if not payload:
        raise _error(422, "No strategy folder (with an __init__.py) was found in the upload.")
    name = _require_name(name_override or original or "")
    return _install_strategy(ctx, name, original, payload, overwrite)


def _session_payload(session: dict[str, Any]) -> dict[str, Any]:
    result = _clean_json(dict(session))
    if isinstance(result.get("results"), dict):
        # Dedicated MCP retrieval exposes downsampled Monte Carlo curves. Avoid
        # transferring every scenario on normal dashboard polling/history calls.
        result["results"].pop("equity_curves", None)
        # Per-candle indicator overlays are served lazily by the candles endpoint.
        result["results"].pop("chart_data", None)
    result["session_id"] = result.pop("id")
    result["dashboard_url"] = (result.get("results") or {}).get("dashboard_url", "")
    return result


def _session_notes_metadata(ctx: TerryContext, kind: str, state: dict[str, Any],
                            title: str | None, description: str | None) -> dict[str, Any]:
    routes = state.get("routes") or [{
        "exchange": state["exchange"], "strategy": state["strategy"],
        "symbol": state["symbol"], "timeframe": state["timeframe"],
    }]
    primary = routes[0]
    default_title = (
        f'{kind.replace("_", " ").title()}: {primary["strategy"]} on '
        f'{primary["symbol"]} {primary["timeframe"]}')
    default_description = (
        f'{kind.replace("_", " ").title()} for {len(routes)} trading route(s) '
        f'from {state["start_date"]} to {state["finish_date"]} on {state["exchange"]}.')
    codes = {}
    for route in routes:
        path = _strategy_path(ctx, route["strategy"])
        try:
            codes[f'{route["exchange"]}-{route["symbol"]}'] = path.read_text(encoding="utf-8")
        except OSError:
            continue
    return {"title": title or default_title,
            "description": description or default_description,
            "strategy_codes": codes}


def _clean_json(value: Any) -> Any:
    """Make engine results safe for strict JSON clients (NaN/Inf become null)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    return value


def _session_list(ctx: TerryContext, kind: str, limit: int, offset: int, query: str | None,
                  status: str | None) -> tuple[list[dict[str, Any]], int]:
    if not (query or "").strip() and not status:
        rows = ctx.sessions.list(kind, limit, offset)
        return [_session_payload(row) for row in rows], ctx.sessions.count(kind)
    rows = ctx.sessions.list(kind, None)
    query = (query or "").strip().lower()
    if query:
        rows = [row for row in rows if query in " ".join(map(str, (
            row["id"], row["state"].get("strategy", ""), row["state"].get("symbol", ""),
            (row.get("notes_metadata") or {}).get("title", ""), row["notes"] or ""))).lower()]
    if status:
        rows = [row for row in rows if row["status"] == status]
    total = len(rows)
    return [_session_payload(row) for row in rows[offset:offset + limit]], total


def _base_state(ctx: TerryContext, payload: dict[str, Any]) -> dict[str, Any]:
    raw_routes = payload.get("routes")
    first_route = (raw_routes[0] if isinstance(raw_routes, list) and raw_routes
                   and isinstance(raw_routes[0], dict) else {})
    exchange = payload.get("exchange") or first_route.get("exchange") or ctx.config.get()["exchange"]
    if exchange not in EXCHANGES:
        raise _error(422, f"Unknown exchange: {exchange}.")
    routes = _routes(ctx, raw_routes, exchange, trading=True)
    data_routes = _routes(ctx, payload.get("data_routes"), exchange, trading=False)
    if routes:
        primary = routes[0]
        strategy = primary["strategy"]
        symbol = primary["symbol"]
        timeframe = primary["timeframe"]
    else:
        strategy = _require_name(str(payload.get("strategy", "")))
        if not strategy_exists(ctx.strategies_dir, strategy):
            raise _error(404, f'Strategy "{strategy}" does not exist.', "strategy_not_found")
        symbol = _symbol(payload.get("symbol"))
        timeframe = _timeframe(payload.get("timeframe", "4h"))
    start_date = _date(payload.get("start_date"), "start_date")
    finish_date = _date(payload.get("finish_date"), "finish_date")
    if jh.date_to_timestamp(finish_date) <= jh.date_to_timestamp(start_date):
        raise _error(422, "finish_date must be after start_date.")
    state = {
        "strategy": strategy,
        "symbol": symbol,
        "exchange": exchange,
        "timeframe": timeframe,
        "start_date": start_date,
        "finish_date": finish_date,
    }
    if routes:
        pairs = [(route["exchange"], route["symbol"]) for route in routes]
        if len(pairs) != len(set(pairs)):
            raise _error(422, "Two trading routes cannot use the same exchange-symbol pair.")
        state["routes"] = routes
    if data_routes:
        state["data_routes"] = data_routes
    overrides = payload.get("config")
    if overrides is not None:
        if not isinstance(overrides, dict):
            raise _error(422, "config must be a JSON object.")
        state["config"] = _validate_engine_config(overrides)
    if "hyperparameters" in payload:
        if not isinstance(payload["hyperparameters"], dict):
            raise _error(422, "hyperparameters must be a JSON object.")
        state["hyperparameters"] = payload["hyperparameters"]
    return state


def _new_session(ctx: TerryContext, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    if kind not in VALID_KINDS:
        raise _error(404, "Unknown research mode.", "unknown_mode")
    allowed = {"strategy", "exchange", "symbol", "timeframe", "routes", "data_routes",
               "start_date", "finish_date", "config", "hyperparameters", "notes",
               "title", "description",
               "cpu_cores", "start"}
    allowed |= {
        "significance_test": {"n_simulations", "random_seed", "hypothesis", "rationale"},
        "monte_carlo": {"num_scenarios", "run_candles", "run_trades", "fast_mode",
                        "pipeline_type", "pipeline_params"},
        "optimization": {"n_trials", "train_test_split", "objective", "optimal_total",
                         "best_candidates_count"},
        "backtest": {"debug_mode", "export_csv", "export_json", "export_chart",
                     "export_tradingview", "fast_mode", "benchmark"},
        "demo": {"debug_mode", "export_csv", "export_json", "export_chart",
                 "export_tradingview", "fast_mode", "benchmark", "lookback_days"},
    }[kind]
    unknown = set(payload) - allowed
    if unknown:
        raise _error(422, f"Unknown session field(s): {', '.join(sorted(unknown))}.")
    for field in ("notes", "title", "description", "hypothesis", "rationale"):
        if field in payload and payload[field] is not None and not isinstance(payload[field], str):
            raise _error(422, f"{field} must be text.")
    title = payload.get("title") or ""
    if len(title) > 160:
        raise _error(422, "title cannot exceed 160 characters.")
    notes = payload.get("description") or payload.get("notes") or ""
    if len(notes) > 20_000:
        raise _error(422, "notes cannot exceed 20,000 characters.")
    if kind == "demo":
        # A live demo trades a rolling [now - lookback, now] window; synthesize the
        # initial date range so the shared validation/state plumbing still applies.
        lookback = _integer(payload.get("lookback_days", 14), "lookback_days", 1)
        if lookback > 365:
            raise _error(422, "lookback_days cannot exceed 365.")
        now = jh.today_to_timestamp()
        payload = {**payload,
                   "start_date": jh.timestamp_to_date(now - lookback * 86_400_000),
                   "finish_date": jh.timestamp_to_date(now)}
    state = _base_state(ctx, payload)
    if kind == "significance_test" and len(state.get("routes") or [state]) != 1:
        raise _error(422, "Rule significance testing requires exactly one trading route.")
    if "cpu_cores" in payload:
        state["cpu_cores"] = _integer(payload["cpu_cores"], "cpu_cores", 1)
    if kind == "significance_test":
        simulations = _integer(payload.get("n_simulations", 2_000), "n_simulations", 2_000)
        state.update({"n_simulations": simulations, "hypothesis": str(payload.get("hypothesis") or ""),
                      "rationale": str(payload.get("rationale") or "")})
        if "random_seed" in payload:
            state["random_seed"] = _integer(payload["random_seed"], "random_seed")
    elif kind == "monte_carlo":
        scenarios = _integer(payload.get("num_scenarios", 200), "num_scenarios", 1)
        run_candles = _boolean(payload.get("run_candles", True), "run_candles")
        run_trades = _boolean(payload.get("run_trades", False), "run_trades")
        if not run_candles and not run_trades:
            raise _error(422, "Enable candle resampling, trade-order shuffling, or both.")
        pipeline_type = str(payload.get("pipeline_type", "moving_block_bootstrap"))
        if pipeline_type not in _PIPELINE_TYPES:
            raise _error(422, f"pipeline_type must be one of: {', '.join(sorted(_PIPELINE_TYPES))}.")
        pipeline_params = payload.get("pipeline_params")
        if pipeline_params is None:
            pipeline_params = {}
        if not isinstance(pipeline_params, dict):
            raise _error(422, "pipeline_params must be a JSON object.")
        pipeline_params = dict(pipeline_params)
        pipeline_params.setdefault("batch_size", 10_080)
        pipeline_params["batch_size"] = _integer(
            pipeline_params["batch_size"], "pipeline_params.batch_size", 2)
        if pipeline_type == "gaussian_noise":
            pipeline_params.setdefault("close_sigma", 0.001)
            pipeline_params.setdefault("high_sigma", 0.0001)
            pipeline_params.setdefault("low_sigma", 0.0001)
            for field in ("close_sigma", "high_sigma", "low_sigma"):
                pipeline_params[field] = _number(
                    pipeline_params[field], f"pipeline_params.{field}")
        if pipeline_type == "gaussian_resampler" and pipeline_params.get("sigma") is not None:
            pipeline_params["sigma"] = _number(
                pipeline_params["sigma"], "pipeline_params.sigma")
        state.update({"num_scenarios": scenarios, "run_candles": run_candles,
                      "run_trades": run_trades,
                      "fast_mode": _boolean(payload.get("fast_mode", True), "fast_mode"),
                      "pipeline_type": pipeline_type,
                      "pipeline_params": pipeline_params})
    elif kind == "optimization":
        trials = _integer(payload.get("n_trials", 100), "n_trials", 1)
        split = _number(payload.get("train_test_split", 0.75), "train_test_split", 0.1, exclusive=True)
        if split >= 0.9:
            raise _error(422, "train_test_split must be greater than 0.1 and less than 0.9.")
        objective = str(payload.get("objective", "sharpe_ratio"))
        if objective not in _OBJECTIVES:
            raise _error(422, f"objective must be one of: {', '.join(sorted(_OBJECTIVES))}.")
        state.update({"objective": objective, "n_trials": trials,
                      "train_test_split": split,
                      "optimal_total": _integer(payload.get("optimal_total", 200),
                                                "optimal_total", 2),
                      "best_candidates_count": _integer(
                          payload.get("best_candidates_count", 20),
                          "best_candidates_count", 1)})
    elif kind in ("backtest", "demo"):
        for field, default in {
            "debug_mode": False, "export_csv": False, "export_json": False,
            "export_chart": True, "export_tradingview": False,
            "fast_mode": True, "benchmark": True,
        }.items():
            state[field] = _boolean(payload.get(field, default), field)
        if kind == "demo":
            state["lookback_days"] = _integer(payload.get("lookback_days", 14), "lookback_days", 1)
    start = _boolean(payload.get("start", True), "start")
    metadata = _session_notes_metadata(ctx, kind, state, title or None, notes or None)
    sid = ctx.sessions.create(
        kind, state, notes=metadata["description"], notes_metadata=metadata)
    session = ctx.sessions.get(sid)
    if start:
        ctx.runner.run(sid)
        session = ctx.sessions.get(sid)
    return _session_payload(session)


def create_app(project_root: str | None = None) -> FastAPI:
    """Create the local dashboard app, isolated to one Terry project root."""
    ctx = set_context(TerryContext(project_root))
    password = os.environ.get("TERRY_DASHBOARD_PASSWORD", "")
    tokens: set[str] = set()

    app = FastAPI(title="Terry Dashboard", version=__version__)

    @app.middleware("http")
    async def security_headers(request, call_next):
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; "
            "frame-ancestors 'none'; form-action 'self'"
        )
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(HTTPException)
    async def handle_http_error(_, exc: HTTPException):
        detail = exc.detail if isinstance(exc.detail, dict) else {"error": "request_failed", "message": str(exc.detail)}
        return JSONResponse(status_code=exc.status_code, content=detail)

    def request_token(authorization: str | None, session_cookie: str | None) -> str | None:
        bearer = (authorization or "").removeprefix("Bearer ")
        if bearer in tokens:
            return bearer
        return session_cookie if session_cookie in tokens else None

    def auth(authorization: str | None = Header(default=None),
             session_cookie: str | None = Cookie(default=None, alias=_AUTH_COOKIE)) -> str | None:
        if not password:
            return None
        token = request_token(authorization, session_cookie)
        if token is None:
            raise _error(401, "Sign in to continue.", "unauthorized")
        return token

    @app.get("/api/auth/status")
    def auth_status(authorization: str | None = Header(default=None),
                    session_cookie: str | None = Cookie(default=None, alias=_AUTH_COOKIE)):
        return {"auth_required": bool(password),
                "authenticated": not password or request_token(authorization, session_cookie) is not None}

    @app.post("/api/auth/login")
    def login(payload: dict[str, Any], response: Response):
        if password and not secrets.compare_digest(str(payload.get("password", "")), password):
            raise _error(401, "Incorrect password.", "invalid_credentials")
        if not password:
            return {"auth_token": "", "auth_required": False}
        token = secrets.token_urlsafe(32)
        tokens.add(token)
        response.set_cookie(_AUTH_COOKIE, token, httponly=True, samesite="strict", path="/")
        return {"auth_token": token, "auth_required": bool(password)}

    @app.post("/api/auth/logout")
    def logout(response: Response, token: str | None = Depends(auth)):
        if token:
            tokens.discard(token)
        response.delete_cookie(_AUTH_COOKIE, path="/", samesite="strict")
        return {"status": "signed_out"}

    @app.get("/api/status")
    def status(_: None = Depends(auth)):
        return {
            "name": "Terry", "version": __version__, "auth_required": bool(password),
            "project_root": ctx.project_root, "indicators_available": len(ta.__all__),
            "supported_exchanges": list(EXCHANGES), "datasets": ctx.candle_db.existing(),
            "sessions": {kind: ctx.sessions.count(kind) for kind in _SESSION_KINDS},
            "live_trading": {"available": False, "reason": "Live and paper trading are out of scope for Terry."},
        }

    @app.get("/api/strategies")
    def list_strategies(_: None = Depends(auth)):
        items = []
        for entry in sorted(Path(ctx.strategies_dir).iterdir(), key=lambda p: p.name.lower()):
            path = entry / "__init__.py"
            if entry.is_dir() and path.exists() and _NAME.fullmatch(entry.name):
                items.append({"name": entry.name, "updated_at": int(path.stat().st_mtime * 1000),
                              "validation_error": _validation_error(ctx, entry.name)})
        return {"strategies": items}

    @app.post("/api/strategies")
    def create_strategy(payload: dict[str, Any], _: None = Depends(auth)):
        name = _require_name(str(payload.get("name", "")))
        path = _strategy_path(ctx, name)
        if path.exists():
            raise _error(409, f'Strategy "{name}" already exists.', "exists")
        content = payload.get("content")
        if content is not None and not isinstance(content, str):
            raise _error(422, "Strategy content must be text.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content or _strategy_template(name), encoding="utf-8")
        return {"status": "created", "name": name, "validation_error": _validation_error(ctx, name)}

    @app.get("/api/strategies/{name}")
    def read_strategy(name: str, _: None = Depends(auth)):
        path = _strategy_path(ctx, name)
        if not path.exists():
            raise _error(404, f'Strategy "{name}" was not found.', "not_found")
        return {"name": name, "content": path.read_text(encoding="utf-8"), "validation_error": _validation_error(ctx, name)}

    @app.put("/api/strategies/{name}")
    def save_strategy(name: str, payload: dict[str, Any], _: None = Depends(auth)):
        path = _strategy_path(ctx, name)
        if not path.exists():
            raise _error(404, f'Strategy "{name}" was not found.', "not_found")
        content = payload.get("content")
        if not isinstance(content, str) or not content.strip():
            raise _error(422, "Strategy content cannot be empty.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return {"status": "written", "name": name, "validation_error": _validation_error(ctx, name)}

    @app.post("/api/strategies/{name}/fork")
    def fork_strategy(name: str, payload: dict[str, Any], _: None = Depends(auth)):
        source = _strategy_path(ctx, name)
        if not source.exists():
            raise _error(404, f'Strategy "{name}" was not found.', "not_found")
        new_name = _require_name(str(payload.get("name", "")))
        target = _strategy_path(ctx, new_name)
        if target.exists():
            raise _error(409, f'Strategy "{new_name}" already exists.', "exists")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source.read_text(encoding="utf-8").replace(f"class {name}", f"class {new_name}", 1), encoding="utf-8")
        return {"status": "forked", "name": new_name, "validation_error": _validation_error(ctx, new_name)}

    @app.get("/api/strategies/{name}/export")
    def export_strategy(name: str, _: None = Depends(auth)):
        _require_name(name)
        if not strategy_exists(ctx.strategies_dir, name):
            raise _error(404, f'Strategy "{name}" was not found.', "not_found")
        data = _build_strategy_bundle(ctx, name)
        return Response(data, media_type="application/zip", headers={
            "Content-Disposition": f'attachment; filename="{name}.terry-strategy.zip"'})

    @app.post("/api/strategies/import")
    def import_strategy(payload: dict[str, Any], _: None = Depends(auth)):
        def _decode(value: Any, what: str) -> bytes:
            if not isinstance(value, str):
                raise _error(422, f"{what} must be base64 text.")
            text = value.strip()
            # tolerate a data: URL prefix the browser's FileReader may include
            if text.startswith("data:") and "," in text:
                text = text.split(",", 1)[1]
            try:
                return base64.b64decode(text, validate=True)
            except ValueError as exc:  # binascii.Error subclasses ValueError
                raise _error(422, f"{what} could not be decoded.") from exc

        # Collect uploaded files: either a single zip in `data`, or a `files` list of
        # {path, data} entries (a whole folder / multi-file selection from the browser).
        raw_files: dict[str, bytes] = {}
        entries = payload.get("files")
        if entries is not None:
            if not isinstance(entries, list) or not entries:
                raise _error(422, "Provide the strategy files as a non-empty 'files' list.")
            if len(entries) > _MAX_BUNDLE_FILES:
                raise _error(422, "The strategy upload contains too many files.")
            for entry in entries:
                if not isinstance(entry, dict):
                    raise _error(422, "Each file entry must include a path and data.")
                path = entry.get("path") or entry.get("name")
                if not isinstance(path, str) or not path.strip():
                    raise _error(422, "Each file entry must include a path.")
                raw_files[path] = _decode(entry.get("data"), "The strategy file")
        elif payload.get("data") is not None:
            raw = _decode(payload.get("data"), "The strategy bundle")
            raw_files = _files_from_zip(raw)
        else:
            raise _error(422, "Provide either a zip in 'data' or a 'files' list.")

        name_override = payload.get("name")
        if name_override is not None:
            if not isinstance(name_override, str):
                raise _error(422, "name must be text.")
            name_override = name_override.strip() or None
        overwrite = payload.get("overwrite", False)
        if not isinstance(overwrite, bool):
            raise _error(422, "overwrite must be true or false.")
        name, files = _extract_strategy_bundle(ctx, raw_files, name_override, overwrite)
        return {"status": "imported", "name": name, "files": files,
                "validation_error": _validation_error(ctx, name)}

    @app.delete("/api/strategies/{name}")
    def delete_strategy(name: str, _: None = Depends(auth)):
        path = _strategy_path(ctx, name)
        if not path.exists():
            raise _error(404, f'Strategy "{name}" was not found.', "not_found")
        path.unlink()
        try:
            path.parent.rmdir()
        except OSError:
            pass
        return {"status": "deleted", "name": name}

    @app.get("/api/candles")
    def list_candles(_: None = Depends(auth)):
        return {"existing": ctx.candle_db.existing(), "exchanges": list(EXCHANGES),
                "active_imports": ctx.importer.active_imports()}

    @app.post("/api/candles/import")
    def import_candles(payload: dict[str, Any], _: None = Depends(auth)):
        exchange = payload.get("exchange")
        if exchange not in EXCHANGES:
            raise _error(422, f"Unknown exchange: {exchange}.")
        symbol = _symbol(payload.get("symbol"))
        start = _date(payload.get("start_date"), "start_date")
        finish = payload.get("finish_date")
        if finish:
            finish = _date(finish, "finish_date")
        try:
            import_id = ctx.importer.start_import(exchange, symbol, start, finish)
        except ValueError as exc:
            raise _error(422, str(exc)) from exc
        return {"status": "started", "import_id": import_id}

    @app.get("/api/candles/import/{import_id}")
    def candle_import_status(import_id: str, _: None = Depends(auth)):
        result = ctx.importer.get_status(import_id)
        if result["status"] == "not_found":
            raise _error(404, "Candle import was not found.", "not_found")
        return result

    @app.post("/api/candles/import/{import_id}/cancel")
    def cancel_candle_import(import_id: str, _: None = Depends(auth)):
        if not ctx.importer.cancel(import_id):
            raise _error(404, "Candle import was not found.", "not_found")
        return {"status": "canceled", "import_id": import_id}

    @app.delete("/api/candles/{exchange}/{symbol}")
    def delete_candles(exchange: str, symbol: str, _: None = Depends(auth)):
        if exchange not in EXCHANGES:
            raise _error(422, f"Unknown exchange: {exchange}.")
        symbol = _symbol(symbol)
        if ctx.candle_db.coverage(exchange, symbol) is None:
            raise _error(404, "Candle dataset was not found.", "not_found")
        ctx.candle_db.delete(exchange, symbol)
        return {"status": "deleted", "exchange": exchange, "symbol": symbol}

    @app.get("/api/config")
    def get_config(_: None = Depends(auth)):
        return {"config": ctx.config.get()}

    @app.patch("/api/config")
    def update_config(payload: dict[str, Any], _: None = Depends(auth)):
        allowed = {"exchange", "starting_balance", "fee", "type", "futures_leverage",
                   "futures_leverage_mode", "quote_asset", "warm_up_candles", "optimization",
                   "monte_carlo", "significance_test"}
        unknown = set(payload) - allowed
        if unknown:
            raise _error(422, f"Unknown configuration key(s): {', '.join(sorted(unknown))}.")
        if "exchange" in payload and payload["exchange"] not in EXCHANGES:
            raise _error(422, f"Unknown exchange: {payload['exchange']}.")
        scalar = _validate_engine_config({key: value for key, value in payload.items()
                                          if key in _ENGINE_CONFIG_KEYS})
        validated = {**payload, **scalar}
        nested_specs = {
            "optimization": {"objective", "n_trials", "train_test_split"},
            "monte_carlo": {"num_scenarios", "run_candles", "run_trades"},
            "significance_test": {"n_simulations"},
        }
        for key, keys in nested_specs.items():
            if key in validated:
                value = validated[key]
                if not isinstance(value, dict):
                    raise _error(422, f"{key} must be a JSON object.")
                nested_unknown = set(value) - keys
                if nested_unknown:
                    raise _error(422, f"Unknown {key} key(s): {', '.join(sorted(nested_unknown))}.")
        if "optimization" in validated:
            value = dict(validated["optimization"])
            if "objective" in value and value["objective"] not in _OBJECTIVES:
                raise _error(422, f"objective must be one of: {', '.join(sorted(_OBJECTIVES))}.")
            if "n_trials" in value:
                value["n_trials"] = _integer(value["n_trials"], "n_trials", 1)
            if "train_test_split" in value:
                value["train_test_split"] = _number(value["train_test_split"], "train_test_split", 0.1,
                                                       exclusive=True)
                if value["train_test_split"] >= 0.9:
                    raise _error(422, "train_test_split must be greater than 0.1 and less than 0.9.")
            validated["optimization"] = value
        if "monte_carlo" in validated:
            value = dict(validated["monte_carlo"])
            if "num_scenarios" in value:
                value["num_scenarios"] = _integer(value["num_scenarios"], "num_scenarios", 1)
            for key in ("run_candles", "run_trades"):
                if key in value:
                    value[key] = _boolean(value[key], key)
            effective = {**ctx.config.get()["monte_carlo"], **value}
            if not effective["run_candles"] and not effective["run_trades"]:
                raise _error(422, "Enable candle resampling, trade-order shuffling, or both.")
            validated["monte_carlo"] = value
        if "significance_test" in validated:
            value = dict(validated["significance_test"])
            if "n_simulations" in value:
                value["n_simulations"] = _integer(value["n_simulations"], "n_simulations", 2_000)
            validated["significance_test"] = value
        return {"status": "updated", "config": ctx.config.update(validated)}

    @app.get("/api/indicators")
    def indicators(query: str = "", _: None = Depends(auth)):
        names = sorted(name for name in ta.__all__ if query.lower() in name.lower())
        return {"count": len(names), "indicators": names}

    @app.get("/api/indicators/{name}")
    def indicator(name: str, _: None = Depends(auth)):
        import inspect
        fn = getattr(ta, name, None)
        if fn is None or not callable(fn):
            raise _error(404, f'Indicator "{name}" was not found.', "not_found")
        return {"name": name, "signature": f"{name}{inspect.signature(fn)}", "doc": (fn.__doc__ or "").strip()}

    @app.get("/api/sessions/{kind}")
    def list_sessions(kind: str, limit: int = 50, offset: int = 0, query: str | None = None,
                      status: str | None = None, _: None = Depends(auth)):
        if kind not in VALID_KINDS:
            raise _error(404, "Unknown research mode.", "unknown_mode")
        if not 1 <= limit <= 100 or offset < 0:
            raise _error(422, "limit must be 1–100 and offset cannot be negative.")
        if status and status not in {"draft", "running", *TERMINAL}:
            raise _error(422, "Unknown session status.")
        sessions, total = _session_list(ctx, kind, limit, offset, query, status)
        return {"sessions": sessions, "count": len(sessions), "total": total}

    @app.post("/api/sessions/{kind}")
    def create_session(kind: str, payload: dict[str, Any], _: None = Depends(auth)):
        return _new_session(ctx, kind, payload)

    @app.get("/api/session/{session_id}")
    def get_session(session_id: str, _: None = Depends(auth)):
        session = ctx.sessions.get(session_id)
        if session is None:
            raise _error(404, "Session was not found.", "not_found")
        return _session_payload(session)

    @app.post("/api/session/{session_id}/run")
    def run_session(session_id: str, _: None = Depends(auth)):
        session = ctx.sessions.get(session_id)
        if session is None:
            raise _error(404, "Session was not found.", "not_found")
        if session["status"] == "running":
            raise _error(409, "Session is already running.", "already_running")
        started = ctx.runner.run(session_id)
        if started.get("error") == "worker_active":
            raise _error(409, "The previous worker is still stopping; try again shortly.", "worker_active")
        return _session_payload(ctx.sessions.get(session_id))

    @app.post("/api/session/{session_id}/cancel")
    def cancel_session(session_id: str, _: None = Depends(auth)):
        session = ctx.sessions.get(session_id)
        if session is None:
            raise _error(404, "Session was not found.", "not_found")
        if session["status"] != "running":
            raise _error(409, "Only a running session can be canceled.", "not_running")
        ctx.runner.cancel(session_id)
        return _session_payload(ctx.sessions.get(session_id))

    @app.patch("/api/session/{session_id}")
    def update_session(session_id: str, payload: dict[str, Any], _: None = Depends(auth)):
        session = ctx.sessions.get(session_id)
        if session is None:
            raise _error(404, "Session was not found.", "not_found")
        unknown = set(payload) - {"notes", "title", "description"}
        if unknown:
            raise _error(422, f"Unknown session field(s): {', '.join(sorted(unknown))}.")
        if "title" in payload and len(str(payload["title"] or "")) > 160:
            raise _error(422, "title cannot exceed 160 characters.")
        note_value = payload.get("description", payload.get("notes"))
        if note_value is not None:
            notes = str(note_value or "")
            if len(notes) > 20_000:
                raise _error(422, "notes cannot exceed 20,000 characters.")
        else:
            notes = session["notes"] or ""
        metadata = dict(session.get("notes_metadata") or {})
        if "title" in payload:
            metadata["title"] = str(payload["title"] or "")
        if note_value is not None:
            metadata["description"] = notes
        ctx.sessions.update_notes_metadata(session_id, metadata, notes=notes)
        return _session_payload(ctx.sessions.get(session_id))

    @app.delete("/api/session/{session_id}")
    def delete_session(session_id: str, _: None = Depends(auth)):
        session = ctx.sessions.get(session_id)
        if session is None:
            raise _error(404, "Session was not found.", "not_found")
        if session["status"] == "running":
            raise _error(409, "Cancel the running session before deleting it.", "session_running")
        ctx.sessions.delete(session_id)
        return {"status": "deleted", "session_id": session_id}

    @app.get("/api/session/{session_id}/export")
    def export_session(session_id: str, format: str = "json", _: None = Depends(auth)):
        session = ctx.sessions.get(session_id)
        if session is None:
            raise _error(404, "Session was not found.", "not_found")
        if format == "json":
            return JSONResponse(_session_payload(session), headers={"Content-Disposition": f'attachment; filename="{session_id}.json"'})
        if format != "csv":
            raise _error(422, "format must be json or csv.")
        trades = (session.get("results") or {}).get("trades", [])
        output = io.StringIO()
        if trades:
            writer = csv.DictWriter(output, fieldnames=sorted({key for trade in trades for key in trade}))
            writer.writeheader()
            writer.writerows(trades)
        return Response(output.getvalue(), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{session_id}-trades.csv"'})

    @app.get("/api/session/{session_id}/candles")
    def session_candles(session_id: str, route: int = 0, _: None = Depends(auth)):
        """Serve OHLC candles + trade markers + indicator overlays for one route of a
        finished backtest so the dashboard can draw Jesse's candlestick/positions chart."""
        session = ctx.sessions.get(session_id)
        if session is None:
            raise _error(404, "Session was not found.", "not_found")
        if session["kind"] not in ("backtest", "demo"):
            raise _error(422, "Candlestick charts are only available for backtests and demos.")
        state = session["state"]
        routes = state.get("routes") or [{
            "exchange": state["exchange"], "symbol": state["symbol"],
            "timeframe": state["timeframe"], "strategy": state["strategy"]}]
        if route < 0 or route >= len(routes):
            raise _error(422, "route index is out of range.")
        r = routes[route]
        exchange, symbol, timeframe = r["exchange"], r["symbol"], r["timeframe"]
        results = session.get("results") or {}
        live = results.get("live") if isinstance(results.get("live"), dict) else {}
        start_ts = (int(live["window_start_ts"])
                    if session["kind"] == "demo" and live.get("window_start_ts") is not None
                    else jh.date_to_timestamp(state["start_date"]))
        # A live demo's window advances to the present, so chart up to "now".
        finish_ts = (jh.now_to_timestamp(force_fresh=True) if session["kind"] == "demo"
                     else jh.date_to_timestamp(state["finish_date"]))
        raw = ctx.candle_db.get(exchange, symbol, start_ts, finish_ts)
        from ..engine.candle_store import aggregate_candles
        agg = aggregate_candles(raw, timeframe)
        step = max(1, len(agg) // 12000 or 1)  # keep the payload lightweight-charts-friendly
        agg = agg[::step]
        candles = [{"time": int(c[0] / 1000), "open": c[1], "high": c[3],
                    "low": c[4], "close": c[2], "volume": c[5]} for c in agg]
        # The strategy only evaluates closed candles, while the chart also shows the selected
        # timeframe's still-forming candle (the same behavior traders expect from TradingView).
        live_candle = live.get("candle") if session["kind"] == "demo" else None
        if isinstance(live_candle, dict) and live_candle.get("time") is not None:
            candle = {key: live_candle.get(key) for key in
                      ("time", "open", "high", "low", "close", "volume")}
            if candles and candles[-1]["time"] == candle["time"]:
                candles[-1] = candle
            elif not candles or candles[-1]["time"] < candle["time"]:
                candles.append(candle)
        candle_times = [c["time"] for c in candles]

        def _snap(order_ms):
            time = int(order_ms / 1000)
            if not candle_times:
                return time
            idx = bisect.bisect_right(candle_times, time) - 1
            return candle_times[max(0, idx)]

        is_demo = session["kind"] == "demo"
        markers = []
        trade_lines = []
        for trade in results.get("trades") or []:
            if trade.get("symbol") not in (None, symbol):
                continue
            # A demo's still-open position is only "closed" by the engine's terminal
            # force-close; show it as a live open entry rather than a misleading Close.
            live_open = is_demo and trade.get("is_open_at_end")
            for order in trade.get("orders") or []:
                if not order.get("executed_at"):
                    continue
                buy = order.get("side") == "buy"
                reduce_only = order.get("reduce_only")
                if live_open and reduce_only:
                    continue  # skip the synthetic close of a position that's still open
                if live_open:
                    text = f"● Open {'Long' if buy else 'Short'} {_short_qty(order.get('qty'))}"
                else:
                    text = (f"{'Close' if reduce_only else 'Buy' if buy else 'Sell'} "
                            f"{_short_qty(order.get('qty'))}")
                markers.append({
                    "time": _snap(order["executed_at"]),
                    "position": "belowBar" if buy else "aboveBar",
                    "color": "#f9b537" if live_open else ("#4dd49b" if buy else "#ff6b6b"),
                    "shape": "arrowUp" if buy else "arrowDown",
                    "text": text,
                })
            opened_at = trade.get("opened_at")
            entry_price = trade.get("entry_price")
            end_time = ((live_candle or {}).get("time") if live_open
                        else (_snap(trade["closed_at"]) if trade.get("closed_at") else None))
            end_price = ((live_candle or {}).get("close") if live_open
                         else trade.get("exit_price"))
            if opened_at and entry_price is not None and end_time is not None and end_price is not None:
                trade_lines.append({
                    "id": trade.get("id"), "side": trade.get("type"),
                    "open_time": _snap(opened_at), "close_time": end_time,
                    "entry_price": entry_price, "exit_price": end_price,
                    "is_open": bool(live_open),
                })
        markers.sort(key=lambda m: m["time"])
        overlays = (results.get("chart_data") or {}).get(jh.key(exchange, symbol, timeframe))
        return {
            "route": {"exchange": exchange, "symbol": symbol, "timeframe": timeframe,
                      "strategy": r.get("strategy")},
            "routes": [{"exchange": rr["exchange"], "symbol": rr["symbol"],
                        "timeframe": rr["timeframe"], "strategy": rr.get("strategy")}
                       for rr in routes],
            "candles": candles, "markers": markers, "trade_lines": trade_lines[-300:],
            "overlays": _clean_json(overlays) if overlays else None,
        }

    @app.get("/api/session/{session_id}/monte-carlo-curves")
    def session_mc_curves(session_id: str, _: None = Depends(auth)):
        """Serve per-scenario equity curves for the Monte Carlo fan chart."""
        session = ctx.sessions.get(session_id)
        if session is None:
            raise _error(404, "Session was not found.", "not_found")
        curves = (session.get("results") or {}).get("equity_curves") or {}
        return {"candles": _clean_json(curves.get("candles")),
                "trades": _clean_json(curves.get("trades"))}

    @app.get("/api/live")
    def live_unavailable(_: None = Depends(auth)):
        raise _error(501, "Live and paper trading are not implemented in Terry for safety.", "not_available")

    @app.get("/reports/{session_id}")
    def report(session_id: str, _: None = Depends(auth)):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", session_id):
            raise _error(404, "Report was not found.", "not_found")
        report_path = Path(ctx.reports_dir, f"{session_id}.html")
        if not report_path.is_file():
            raise _error(404, "Report is not available yet.", "not_found")
        return FileResponse(report_path)

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        return FileResponse(_STATIC / "favicon.svg", media_type="image/svg+xml")

    @app.get("/")
    def dashboard_index():
        return FileResponse(_STATIC / "index.html")

    app.mount("/", StaticFiles(directory=_STATIC), name="dashboard-static")
    return app


def run(port: int = 9020, host: str = "127.0.0.1", project_root: str | None = None) -> None:
    """Run the dashboard with Uvicorn."""
    import uvicorn
    url = f"http://{host}:{port}"
    print(f"\n  ✓ Terry Dashboard running at {url}\n")
    uvicorn.run(create_app(project_root), host=host, port=port, log_level="info")
