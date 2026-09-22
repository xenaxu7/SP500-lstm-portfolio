"""Monthly-rebalanced equal-weight portfolios, turnover, costs and metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
RISK_FREE = 0.02


def month_ends(index: pd.DatetimeIndex, start: str, end: str) -> list[pd.Timestamp]:
    """Last trading day of each month between start and end (inclusive)."""
    idx = index[(index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))]
    return list(idx.to_series().groupby(idx.to_period("M")).last())


def run_schedule(returns: pd.DataFrame, holdings: dict[pd.Timestamp, list[str]],
                 cost_bps: float = 0.0) -> tuple[pd.Series, pd.Series]:
    """Hold each equal-weight basket from the day after its rebalance date to the
    next rebalance date, letting weights drift in between.

    holdings: {rebalance_date: tickers}. Returns (daily portfolio returns net of
    cost, turnover per rebalance). Turnover is sum |target - drifted weight|,
    i.e. dollars traded per dollar of portfolio; cost = turnover * cost_bps.
    """
    dates = sorted(holdings)
    out, turns = [], {}
    w_prev = pd.Series(dtype=float)
    for i, d in enumerate(dates):
        nxt = dates[i + 1] if i + 1 < len(dates) else returns.index[-1]
        period = returns.loc[(returns.index > d) & (returns.index <= nxt)]
        if period.empty:
            continue
        names = [t for t in holdings[d] if t in returns.columns]
        target = pd.Series(1.0 / len(names), index=names)
        turnover = float((target.sub(w_prev, fill_value=0.0)).abs().sum())
        turns[d] = turnover
        w = target.copy()
        daily = []
        for day, r in period[names].fillna(0.0).iterrows():
            pr = float((w * r).sum())
            if not daily:  # charge the rebalance on the first day of the period
                pr -= turnover * cost_bps / 1e4
            daily.append((day, pr))
            w = w * (1 + r)
            w = w / w.sum()
        out.append(pd.Series(dict(daily)))
        w_prev = w
    return pd.concat(out).sort_index(), pd.Series(turns)


def metrics(daily: pd.Series) -> dict:
    """Total/annualised return, annualised vol, Sharpe (2% rf), max drawdown."""
    daily = daily.dropna()
    curve = (1 + daily).cumprod()
    total = float(curve.iloc[-1] - 1)
    years = len(daily) / TRADING_DAYS
    ann = (1 + total) ** (1 / years) - 1
    vol = float(daily.std(ddof=1) * np.sqrt(TRADING_DAYS))
    dd = float((curve / curve.cummax() - 1).min())
    return {
        "total_return": total,
        "annual_return": ann,
        "annual_vol": vol,
        "sharpe": (ann - RISK_FREE) / vol if vol > 0 else float("nan"),
        "max_drawdown": dd,
    }


def drawdown_series(daily: pd.Series) -> pd.Series:
    curve = (1 + daily.dropna()).cumprod()
    return curve / curve.cummax() - 1


def information_coefficient(pred: pd.Series, realised: pd.Series) -> float:
    """Spearman rank correlation between predicted and realised returns."""
    joined = pd.concat([pred, realised], axis=1, join="inner").dropna()
    if len(joined) < 10:
        return float("nan")
    return float(joined.iloc[:, 0].rank().corr(joined.iloc[:, 1].rank()))


def random_baskets(universe_by_date: dict[pd.Timestamp, list[str]], n: int,
                   draws: int, seed: int) -> list[dict[pd.Timestamp, list[str]]]:
    """`draws` random schedules, each picking `n` names at random on every
    rebalance date. Used to see where the LSTM basket sits versus luck."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(draws):
        out.append({d: list(rng.choice(u, size=min(n, len(u)), replace=False))
                    for d, u in universe_by_date.items()})
    return out
