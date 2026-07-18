"""Monte Carlo tools (draft → run → poll), candles + trades modes."""
import json
import math

from . import _common as c
from ...context import get_context

_EXAMPLE_ROUTES = ('[{"exchange": "Binance Perpetual Futures", '
                   '"strategy": "ExampleStrategy", "symbol": "BTC-USDT", '
                   '"timeframe": "4h"}]')
_PIPELINES = {"moving_block_bootstrap", "gaussian_noise", "gaussian_resampler"}


def register_monte_carlo_tools(mcp):
    @mcp.tool()
    def create_monte_carlo_draft(
            exchange: str = "Binance Perpetual Futures", routes: str = _EXAMPLE_ROUTES,
            data_routes: str = "[]", start_date: str = "2024-01-01",
            finish_date: str = "2024-03-01", num_scenarios: int = 200,
            run_trades: bool = False, run_candles: bool = True,
            fast_mode: bool = True, cpu_cores: int = None,
            pipeline_type: str = "moving_block_bootstrap",
            pipeline_params: str = None, title: str = None,
            description: str = None, strategy_summary: str = None,
            hypothesis: str = None, rationale: str = None, *,
            strategy: str = None, symbol: str = None, timeframe: str = None,
            config: str = None) -> dict:
        """Create a Monte Carlo robustness draft.

        Default = candles mode (`run_candles=True, run_trades=False`), 200 scenarios — answers
        "is this backtest overfit/lucky?". Only enable run_trades on explicit request.
        Then call run_monte_carlo(session_id).
        """
        route_input = None if strategy is not None and routes == _EXAMPLE_ROUTES else routes
        state, err = c.build_routes_state(
            strategy, symbol, timeframe, exchange, start_date, finish_date,
            config, route_input, data_routes)
        if err:
            return {"error": "invalid_config", "message": err}
        if int(num_scenarios) < 1:
            return {"error": "invalid_config", "message":
                    "num_scenarios must be at least 1."}
        if cpu_cores is not None and int(cpu_cores) < 1:
            return {"error": "invalid_config", "message":
                    "cpu_cores must be an integer greater than 0."}
        if not run_candles and not run_trades:
            return {"error": "invalid_config", "message":
                    "At least one Monte Carlo type must be selected."}
        try:
            params = json.loads(pipeline_params) if pipeline_params else {}
        except json.JSONDecodeError as exc:
            return {"error": "invalid_config", "message":
                    f"Invalid pipeline_params JSON: {exc}"}
        if not isinstance(params, dict):
            return {"error": "invalid_config", "message":
                    "pipeline_params must be a JSON object."}
        if pipeline_type not in _PIPELINES:
            return {"error": "invalid_config", "message":
                    f"pipeline_type must be one of: {', '.join(sorted(_PIPELINES))}."}
        params.setdefault("batch_size", 10_080)
        try:
            params["batch_size"] = int(params["batch_size"])
            if params["batch_size"] < 2:
                raise ValueError("batch_size must be at least 2")
        except (TypeError, ValueError) as exc:
            return {"error": "invalid_config", "message":
                    f"Invalid pipeline batch_size: {exc}"}
        if pipeline_type == "gaussian_noise":
            params.setdefault("close_sigma", 0.001)
            params.setdefault("high_sigma", 0.0001)
            params.setdefault("low_sigma", 0.0001)
            try:
                for field in ("close_sigma", "high_sigma", "low_sigma"):
                    params[field] = float(params[field])
                    if not math.isfinite(params[field]) or params[field] < 0:
                        raise ValueError(f"{field} cannot be negative")
            except (TypeError, ValueError) as exc:
                return {"error": "invalid_config", "message":
                        f"Invalid Gaussian noise parameters: {exc}"}
        if pipeline_type == "gaussian_resampler" and params.get("sigma") is not None:
            try:
                params["sigma"] = float(params["sigma"])
                if not math.isfinite(params["sigma"]) or params["sigma"] < 0:
                    raise ValueError("sigma cannot be negative")
            except (TypeError, ValueError) as exc:
                return {"error": "invalid_config", "message":
                        f"Invalid Gaussian resampler parameters: {exc}"}
        state.update({"num_scenarios": int(num_scenarios),
                      "run_candles": bool(run_candles), "run_trades": bool(run_trades),
                      "fast_mode": bool(fast_mode),
                      "cpu_cores": (int(cpu_cores) if cpu_cores is not None
                                    else c.default_cpu_cores()),
                      "pipeline_type": pipeline_type,
                      "pipeline_params": params})
        notes = "\n\n".join(filter(None, [title, description, strategy_summary,
                                           hypothesis, rationale]))
        return c.create_draft("monte_carlo", state, notes=notes)

    @mcp.tool()
    def update_monte_carlo_draft(session_id: str, state: str) -> dict:
        """Update a Monte Carlo draft's state (JSON string)."""
        return c.update_draft("monte_carlo", session_id, state)

    @mcp.tool()
    def update_monte_carlo_notes(session_id: str, notes: str) -> dict:
        """Attach or update notes on a Monte Carlo session."""
        return c.update_notes(session_id, notes)

    @mcp.tool()
    def get_monte_carlo_session(session_id: str) -> dict:
        """Get a Monte Carlo session's status and (when finished) summary_metrics + verdict."""
        payload = c.get_session(session_id)
        if isinstance(payload.get("results"), dict):
            payload["results"].pop("equity_curves", None)
        return payload

    @mcp.tool()
    def get_monte_carlo_sessions(limit: int = 20) -> dict:
        """List recent Monte Carlo sessions."""
        return c.list_sessions("monte_carlo", limit)

    @mcp.tool()
    def get_monte_carlo_equity_curves(session_id: str) -> dict:
        """Return per-scenario Portfolio equity curves for custom analysis or charts."""
        s = get_context().sessions.get(session_id)
        if s is None:
            return {"status": "error", "error": "not_found",
                    "message": f"Session {session_id} not found"}
        curves = (s.get("results") or {}).get("equity_curves")
        if not curves:
            return {"status": "error", "error": "no_equity_curves",
                    "message": "The session has no Monte Carlo equity curves."}
        return {"status": "success", "session_id": session_id,
                "trades": curves.get("trades"), "candles": curves.get("candles"),
                "message": "Equity curves retrieved successfully"}

    @mcp.tool()
    def get_monte_carlo_logs(session_id: str) -> dict:
        """Return diagnostic info for a Monte Carlo session (status + any error)."""
        s = get_context().sessions.get(session_id)
        if s is None:
            return {"status": "error", "error": "not_found",
                    "message": f"Session {session_id} not found"}
        result = s.get("results") or {}
        error = result.get("message") or result.get("error") or ""
        return {"status": "success", "session_id": session_id,
                "logs": str(error), "session_status": s["status"],
                "message": "Monte Carlo diagnostics retrieved successfully"}

    @mcp.tool()
    def run_monte_carlo(session_id: str) -> dict:
        """Run a Monte Carlo draft (returns immediately). Poll until terminal (may take a while)."""
        return c.run_session(session_id, "monte_carlo")

    @mcp.tool()
    def resume_monte_carlo(session_id: str) -> dict:
        """Resume a stopped/terminated Monte Carlo session by re-running it."""
        return c.run_session(session_id, "monte_carlo")

    @mcp.tool()
    def cancel_monte_carlo(session_id: str) -> dict:
        """Cancel a running Monte Carlo session."""
        return c.cancel_session(session_id, "monte_carlo")

    @mcp.tool()
    def terminate_monte_carlo(session_id: str) -> dict:
        """Force-terminate a Monte Carlo session."""
        return c.cancel_session(session_id, "monte_carlo", new_status="terminated")

    @mcp.tool()
    def purge_monte_carlo_sessions(days_old: int = None) -> dict:
        """Delete Monte Carlo sessions (older than days_old, or all if omitted)."""
        return c.purge_sessions("monte_carlo", days_old)
