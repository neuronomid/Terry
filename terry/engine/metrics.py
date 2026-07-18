"""
Backtest performance metrics. Formulas ported from Jesse's services/metrics.py so the
output dict has the same keys and definitions (validated against a real Jesse run).
"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd


def _prepare_returns(returns, rf=0.0, periods=365):
    if rf != 0:
        returns = returns - (rf / periods)
    if isinstance(returns, pd.DataFrame):
        returns = returns[returns.columns[0]]
    return returns


def sharpe_ratio(returns, rf=0.0, periods=365, annualize=True):
    returns = _prepare_returns(returns, rf, periods)
    divisor = returns.std(ddof=1)
    if divisor == 0:
        return 0.0
    res = returns.mean() / divisor
    if annualize:
        res = res * np.sqrt(periods)
    return float(res)


def sortino_ratio(returns, rf=0, periods=365, annualize=True):
    returns = _prepare_returns(returns, rf, periods)
    downside = np.sqrt((returns[returns < 0] ** 2).sum() / len(returns))
    if downside == 0:
        return float(np.inf if returns.mean() > 0 else -np.inf)
    res = returns.mean() / downside
    if annualize:
        res = res * np.sqrt(periods)
    return float(res)


def max_drawdown(returns):
    prices = (returns + 1).cumprod()
    return float((prices / prices.expanding(min_periods=0).max()).min() - 1)


def cagr(returns, rf=0.0, periods=365):
    returns = _prepare_returns(returns, rf)
    last_value = (1 + returns).prod()
    days = (returns.index[-1] - returns.index[0]).days
    years = float(days) / 365
    if years == 0:
        return 0.0
    ratio = np.clip(last_value, 1e-10, 1e10)
    with np.errstate(over="ignore", under="ignore"):
        return float(ratio ** (1 / years) - 1)


def calmar_ratio(returns):
    returns = _prepare_returns(returns)
    c = cagr(returns)
    cum = (1 + returns).cumprod()
    rolling_max = cum.expanding(min_periods=1).max()
    dd = cum / rolling_max - 1
    max_dd = abs(dd.min())
    return float(c / max_dd) if max_dd != 0 else 0.0


def omega_ratio(returns, rf=0.0, required_return=0.0, periods=365):
    returns = _prepare_returns(returns, rf, periods)
    thr = (1 + required_return) ** (1.0 / periods) - 1
    less = returns - thr
    numer = less[less > 0.0].sum()
    denom = -1.0 * less[less < 0.0].sum()
    return float(numer / denom) if denom > 0.0 else np.nan


def _to_drawdown_series(returns):
    prices = (1 + returns).cumprod()
    dd = prices / np.maximum.accumulate(prices) - 1.0
    return dd.replace([np.inf, -np.inf, -0], 0)


def _cvar(returns, confidence=0.95):
    if len(returns) < 2:
        return 0
    s = np.sort(returns)
    idx = int((1 - confidence) * len(s))
    if idx == 0:
        return s[0] if len(s) else 0
    c = s[:idx].mean()
    return c if ~np.isnan(c) else 0


def _ulcer_index(returns):
    dd = _to_drawdown_series(returns)
    return np.sqrt(np.divide((dd ** 2).sum(), returns.shape[0] - 1))


def serenity_index(returns, rf=0):
    dd = _to_drawdown_series(returns)
    std = returns.std()
    if std == 0:
        return 0.0
    pitfall = -_cvar(dd) / std
    ui = _ulcer_index(returns)
    denom = ui * pitfall
    if denom == 0:
        return 0.0
    return float((returns.sum() - rf) / denom)


def calculate_max_underwater_period(daily_balance):
    if len(daily_balance) < 2:
        return 0
    max_period, current_peak, peak_idx = 0, daily_balance[0], 0
    for i in range(1, len(daily_balance)):
        b = daily_balance[i]
        if b >= current_peak:
            current_peak, peak_idx = b, i
        else:
            max_period = max(max_period, i - peak_idx)
    return max_period


def _safe(value, convert=float):
    try:
        if isinstance(value, pd.Series):
            value = value.iloc[0]
        if value is None:
            return np.nan
        if isinstance(value, float) and np.isnan(value):
            return np.nan
        return convert(value)
    except BaseException:
        return np.nan


def trades_metrics(trades_list, daily_balance, starting_balance, current_balance,
                   starting_time, ending_time, total_open_trades=0, open_pl=0.0):
    """Compute the full metrics dict (mirrors Jesse's metrics.trades())."""
    if not trades_list:
        return {"total": 0, "win_rate": 0, "net_profit_percentage": 0}

    df = pd.DataFrame.from_records([t.to_dict() for t in trades_list])
    total_completed = len(df)
    winning = df.loc[df["PNL"] > 0]
    losing = df.loc[df["PNL"] < 0]
    total_winning = len(winning)
    total_losing = len(losing)

    arr = df["PNL"].to_numpy()
    pos = np.clip(arr, 0, 1).astype(bool).cumsum()
    neg = np.clip(arr, -1, 0).astype(bool).cumsum()
    streak = np.where(arr >= 0, pos - np.maximum.accumulate(np.where(arr <= 0, pos, 0)),
                      -neg + np.maximum.accumulate(np.where(arr >= 0, neg, 0)))
    losing_streak = 0 if streak.min() > 0 else abs(int(streak.min()))
    winning_streak = max(int(streak.max()), 0)

    largest_losing = 0 if total_losing == 0 else losing["PNL"].min()
    largest_winning = 0 if total_winning == 0 else winning["PNL"].max()
    win_rate = 0 if total_winning == 0 else total_winning / (total_losing + total_winning)

    def _wr(t):
        w = df.loc[(df["type"] == t) & (df["PNL"] > 0)]
        l = df.loc[(df["type"] == t) & (df["PNL"] < 0)]
        return len(w) / (len(w) + len(l)) if (len(w) + len(l)) > 0 else 0

    longs_count = int(len(df.loc[df["type"] == "long"]))
    shorts_count = int(len(df.loc[df["type"] == "short"]))
    longs_pct = longs_count / (longs_count + shorts_count) * 100 if (longs_count + shorts_count) else 0
    fee = df["fee"].sum()
    net_profit = df["PNL"].sum()
    net_profit_pct = (net_profit / starting_balance) * 100
    average_win = winning["PNL"].mean()
    average_loss = abs(losing["PNL"].mean())
    ratio_avg_win_loss = average_win / average_loss if average_loss else np.nan
    expectancy = (0 if np.isnan(average_win) else average_win) * win_rate - (
        0 if np.isnan(average_loss) else average_loss) * (1 - win_rate)
    expectancy_pct = (expectancy / starting_balance) * 100
    gross_profit = winning["PNL"].sum()
    gross_loss = losing["PNL"].sum()

    start_dt = datetime.fromtimestamp(starting_time / 1000, tz=timezone.utc)
    date_index = pd.date_range(start=start_dt, periods=len(daily_balance))
    daily_return = pd.Series(daily_balance, index=date_index).pct_change(1)

    if ending_time and starting_time:
        duration_days = (ending_time - starting_time) / DAY_MS_
        avg_per_day = total_completed / duration_days if duration_days > 0 else 0
    else:
        avg_per_day = 0

    enough = len(daily_return) >= 2
    max_dd = np.nan if not enough else max_drawdown(daily_return) * 100
    annual = np.nan if not enough else cagr(daily_return) * 100
    sharpe = np.nan if not enough else sharpe_ratio(daily_return)
    calmar = np.nan if not enough else calmar_ratio(daily_return)
    sortino = np.nan if not enough else sortino_ratio(daily_return)
    omega = np.nan if not enough else omega_ratio(daily_return)
    serenity = np.nan if not enough else serenity_index(daily_return)
    max_uw = np.nan if len(daily_balance) < 2 else calculate_max_underwater_period(daily_balance)

    return {
        "total": _safe(total_completed, int),
        "total_winning_trades": _safe(total_winning, int),
        "total_losing_trades": _safe(total_losing, int),
        "starting_balance": _safe(starting_balance),
        "finishing_balance": _safe(current_balance),
        "win_rate": _safe(win_rate),
        "win_rate_longs": _safe(_wr("long")),
        "win_rate_shorts": _safe(_wr("short")),
        "ratio_avg_win_loss": _safe(ratio_avg_win_loss),
        "longs_count": _safe(longs_count, int),
        "longs_percentage": _safe(longs_pct),
        "shorts_percentage": _safe(100 - longs_pct),
        "shorts_count": _safe(shorts_count, int),
        "fee": _safe(fee),
        "net_profit": _safe(net_profit),
        "net_profit_percentage": _safe(net_profit_pct),
        "average_win": _safe(average_win),
        "average_loss": _safe(average_loss),
        "expectancy": _safe(expectancy),
        "expectancy_percentage": _safe(expectancy_pct),
        "expected_net_profit_every_100_trades": _safe(expectancy_pct * 100),
        "average_holding_period": _safe(df["holding_period"].mean()),
        "average_winning_holding_period": _safe(winning["holding_period"].mean()),
        "average_losing_holding_period": _safe(losing["holding_period"].mean()),
        "gross_profit": _safe(gross_profit),
        "gross_loss": _safe(gross_loss),
        "max_drawdown": _safe(max_dd),
        "max_underwater_period": _safe(max_uw),
        "annual_return": _safe(annual),
        "sharpe_ratio": _safe(sharpe),
        "calmar_ratio": _safe(calmar),
        "sortino_ratio": _safe(sortino),
        "omega_ratio": _safe(omega),
        "serenity_index": _safe(serenity),
        "total_open_trades": _safe(total_open_trades, int),
        "open_pl": _safe(open_pl),
        "winning_streak": _safe(winning_streak, int),
        "losing_streak": _safe(losing_streak, int),
        "largest_losing_trade": _safe(largest_losing),
        "largest_winning_trade": _safe(largest_winning),
        "current_streak": _safe(int(streak[-1]), int),
        "avg_trades_per_day": _safe(avg_per_day),
        "avg_trades_per_week": _safe(avg_per_day * 7),
        "avg_trades_per_month": _safe(avg_per_day * 30.44),
    }


DAY_MS_ = 86_400_000
