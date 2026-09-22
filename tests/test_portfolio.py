import numpy as np
import pandas as pd
import pytest

from sp500lstm.model import Scaler, make_windows
from sp500lstm.portfolio import (information_coefficient, metrics, month_ends,
                                 random_baskets, run_schedule)


def _returns(n_days=60, tickers=("A", "B", "C", "D"), seed=1):
    idx = pd.bdate_range("2024-01-01", periods=n_days)
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(0.0005, 0.01, (n_days, len(tickers))), index=idx, columns=list(tickers))


def test_make_windows_shapes_and_alignment():
    p = np.arange(100, dtype="float32")
    X, y = make_windows(p, window=60, horizon=21)
    assert X.shape == (20, 60, 1) and y.shape == (20,)
    # first window is prices 0..59, target is price 59+21 = 80
    assert X[0, -1, 0] == 59 and y[0] == 80
    # horizon=1 reproduces the original next-day target
    X1, y1 = make_windows(p, window=60, horizon=1)
    assert y1[0] == 60 and len(y1) == 40


def test_make_windows_too_short_returns_empty():
    X, y = make_windows(np.arange(50.0), window=60, horizon=21)
    assert len(X) == 0 and len(y) == 0


def test_scaler_roundtrip():
    s = Scaler.fit(np.array([10.0, 20.0, 30.0]))
    assert s.transform([10.0, 30.0]).tolist() == [0.0, 1.0]
    assert np.allclose(s.inverse(s.transform([12.5, 27.0])), [12.5, 27.0])


def test_month_ends_picks_last_trading_day():
    idx = pd.bdate_range("2024-01-01", "2024-03-31")
    me = month_ends(idx, "2024-01-01", "2024-03-31")
    assert [d.strftime("%Y-%m-%d") for d in me] == ["2024-01-31", "2024-02-29", "2024-03-29"]


def test_run_schedule_matches_hand_calculation():
    r = _returns(n_days=6)
    d0, d3 = r.index[0], r.index[3]
    hold = {d0: ["A", "B"], d3: ["A", "B"]}
    daily, turn = run_schedule(r, hold, cost_bps=0.0)
    # first day after d0: equal weight of A and B
    assert daily.iloc[0] == pytest.approx(r.loc[r.index[1], ["A", "B"]].mean())
    # initial turnover is 100% (from cash); second rebalance only trades the drift
    assert turn.iloc[0] == pytest.approx(1.0)
    assert 0.0 <= turn.iloc[1] < 0.1
    assert len(daily) == 5  # days 1..5


def test_costs_reduce_return_by_turnover_times_bps():
    r = _returns(n_days=10)
    hold = {r.index[0]: ["A", "B", "C"]}
    gross, turn = run_schedule(r, hold, 0.0)
    net, _ = run_schedule(r, hold, 20.0)
    assert (gross - net).iloc[0] == pytest.approx(turn.iloc[0] * 20 / 1e4)
    assert np.allclose((gross - net).iloc[1:], 0.0)


def test_metrics_on_constant_return():
    daily = pd.Series([0.001] * 252)
    m = metrics(daily)
    assert m["total_return"] == pytest.approx(1.001 ** 252 - 1)
    assert m["annual_return"] == pytest.approx(m["total_return"])
    assert m["annual_vol"] == pytest.approx(0.0)
    assert m["max_drawdown"] == 0.0


def test_max_drawdown():
    daily = pd.Series([0.10, -0.50, 0.20])
    assert metrics(daily)["max_drawdown"] == pytest.approx(-0.5)


def test_information_coefficient_perfect_and_inverted():
    pred = pd.Series({"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8, "I": 9, "J": 10})
    assert information_coefficient(pred, pred * 3) == pytest.approx(1.0)
    assert information_coefficient(pred, -pred) == pytest.approx(-1.0)
    assert np.isnan(information_coefficient(pred.iloc[:5], pred.iloc[:5]))  # too few names


def test_random_baskets_are_reproducible_and_sized():
    uni = {pd.Timestamp("2024-01-31"): list("ABCDEFGH"), pd.Timestamp("2024-02-29"): list("ABCDEFGH")}
    a = random_baskets(uni, n=3, draws=4, seed=7)
    b = random_baskets(uni, n=3, draws=4, seed=7)
    assert a == b and len(a) == 4
    assert all(len(v) == 3 and len(set(v)) == 3 for sched in a for v in sched.values())
