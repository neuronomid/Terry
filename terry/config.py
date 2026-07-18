"""Terry configuration — persisted as JSON in the project's storage dir."""
import json
import os

DEFAULT_CONFIG = {
    "exchange": "Binance Perpetual Futures",
    "starting_balance": 10000.0,
    "fee": 0.001,                     # 0.1% (matches Jesse's default)
    "type": "futures",                # 'futures' | 'spot'
    "futures_leverage": 2,
    "futures_leverage_mode": "cross", # 'cross' | 'isolated'
    "quote_asset": "USDT",
    "warm_up_candles": 210,           # route-timeframe candles reserved for indicator warmup
    "optimization": {
        "objective": "sharpe_ratio",
        "n_trials": 100,
        "train_test_split": 0.75,
    },
    "monte_carlo": {
        "num_scenarios": 200,
        "run_candles": True,
        "run_trades": False,
    },
    "significance_test": {
        "n_simulations": 2000,
    },
}


class Config:
    def __init__(self, path):
        self.path = path
        self.data = json.loads(json.dumps(DEFAULT_CONFIG))
        self.load()

    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    saved = json.load(f)
                self._deep_merge(self.data, saved)
            except Exception:
                pass

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

    def _deep_merge(self, base, override):
        for k, v in override.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                self._deep_merge(base[k], v)
            else:
                base[k] = v

    def update(self, partial: dict):
        self._deep_merge(self.data, partial)
        self.save()
        return self.data

    def get(self):
        return self.data

    # ---- typed views used by the engine ----
    def engine_config(self, overrides=None):
        cfg = {
            "exchange": self.data["exchange"],
            "starting_balance": self.data["starting_balance"],
            "fee": self.data["fee"],
            "type": self.data["type"],
            "futures_leverage": self.data["futures_leverage"],
            "futures_leverage_mode": self.data["futures_leverage_mode"],
            "quote_asset": self.data["quote_asset"],
            "warm_up_candles": self.data["warm_up_candles"],
        }
        if overrides:
            cfg.update({k: v for k, v in overrides.items() if v is not None})
        return cfg

    def backtest_config(self):
        return self.engine_config()

    def optimization_config(self):
        return {**self.engine_config(), **self.data["optimization"]}

    def live_config(self):
        return {
            **self.engine_config(),
            "note": "Live trading is not implemented in Terry (out of scope for safety).",
        }
