"""Walk-forward LSTM stock selection on the S&P 500.

Every month-end the top `--top` stocks by predicted one-month return are held
equal-weight for the next month. Models are retrained every `--retrain-months`
months on all data up to that date, so no prediction uses future prices.

    python backtest.py                      # full run (hours on a laptop CPU, see README)
    python backtest.py --max-tickers 12 --epochs 1 --draws 50 --out results_smoke   # smoke test
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sp500lstm.data import THEMATIC_30, load_prices, sp500_constituents  # noqa: E402
from sp500lstm.model import StockLSTM  # noqa: E402
from sp500lstm.portfolio import (drawdown_series, information_coefficient,  # noqa: E402
                                 metrics, month_ends, random_baskets, run_schedule)

ROOT = Path(__file__).resolve().parent


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-start", default="2013-07-01")
    p.add_argument("--test-start", default="2023-07-01")
    p.add_argument("--test-end", default="2024-12-31")
    p.add_argument("--top", type=int, default=30)
    p.add_argument("--window", type=int, default=60)
    p.add_argument("--horizon", type=int, default=21, help="forecast horizon in trading days")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--retrain-months", type=int, default=6)
    p.add_argument("--min-history", type=int, default=160, help="min price points to train a model")
    p.add_argument("--max-tickers", type=int, default=None, help="subset for smoke tests")
    p.add_argument("--draws", type=int, default=500, help="random 30-stock baskets for the luck test")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--universe", choices=["pit", "current"], default="pit",
                   help="pit: drop names whose index 'Date added' is after the rebalance date; current: today's list as is")
    p.add_argument("--shard", default=None, help="k/n: fit only every n-th ticker starting at k (parallel runs)")
    p.add_argument("--fit-only", action="store_true", help="stop after writing predictions (used by shards)")
    p.add_argument("--out", default="results")
    return p.parse_args()


def read_checkpoints(out: Path) -> pd.DataFrame:
    """All prediction checkpoints in `out`; a secondary output directory (for
    example --out results_current) also picks up the committed ones in results/."""
    files = sorted(set(out.glob("predictions*.csv")) | set((ROOT / "results").glob("predictions*.csv")))
    if not files:
        return pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "ticker": pd.Series(dtype=str), "pred": pd.Series(dtype=float)})
    return pd.concat([pd.read_csv(f, parse_dates=["date"]) for f in files]).drop_duplicates(["date", "ticker"], keep="last")


def predictions(prices: pd.DataFrame, rebal: list[pd.Timestamp], a, out: Path) -> pd.DataFrame:
    """Long table (date, ticker, pred), checkpointed per retrain block.

    With --shard k/n only every n-th ticker is fitted and written to its own
    checkpoint file, so several processes can share the work; a final run
    without --shard picks up all checkpoint files.
    """
    k_sh, n_sh = (map(int, a.shard.split("/")) if a.shard else (0, 1))
    tickers = list(prices.columns)[k_sh::n_sh]
    ckpt = out / (f"predictions_shard{k_sh}.csv" if a.shard else "predictions.csv")
    done = read_checkpoints(out)
    mine = done[done.ticker.isin(tickers)]
    blocks = [rebal[i:i + a.retrain_months] for i in range(0, len(rebal), a.retrain_months)]
    rows = [pd.read_csv(ckpt, parse_dates=["date"])] if ckpt.exists() else []
    for block in blocks:
        if not mine.empty and set(block) <= set(mine["date"]):
            print(f"block {block[0].date()} already in checkpoint, skipping")
            continue
        T = block[0]
        hist = prices.loc[:T]
        t0, recs, n_fit = time.time(), [], 0
        for k, tk in enumerate(tickers, 1):
            s = hist[tk].dropna()
            if len(s) < a.min_history:
                continue
            m = StockLSTM(a.window, a.horizon, a.epochs, seed=a.seed).fit(s.values)
            n_fit += 1
            dates, wins = [], []
            for d in block:
                w = prices.loc[:d, tk].dropna().iloc[-a.window:]
                if len(w) == a.window and w.index[-1] == d:
                    dates.append(d)
                    wins.append(w.values)
            if wins:
                recs += [(d, tk, float(p)) for d, p in zip(dates, m.predict_returns(np.array(wins)))]
            m.close()
            if k % 25 == 0 or k == len(tickers):
                print(f"  retrain {T.date()}: {k}/{len(tickers)} tickers, {n_fit} fitted, {time.time()-t0:,.0f}s", flush=True)
        blk = pd.DataFrame(recs, columns=["date", "ticker", "pred"])
        rows.append(blk)
        pd.concat(rows).to_csv(ckpt, index=False)
        print(f"retrain {T.date()}: {n_fit} models, {len(blk)} predictions, {time.time()-t0:,.0f}s", flush=True)
    allp = read_checkpoints(out)
    allp["pred"] = allp["pred"].astype(float)
    return allp


def main():
    a = parse()
    out = ROOT / a.out
    out.mkdir(exist_ok=True)
    members = sp500_constituents(ROOT / "data" / "constituents.csv")
    tickers = members["ticker"].tolist()
    added = members.set_index("ticker")["date_added"]
    if a.max_tickers:
        tickers = tickers[: a.max_tickers]
    cache = ROOT / "data" / (f"prices_{a.max_tickers}.parquet" if a.max_tickers else "prices.parquet")
    dl_end = (pd.Timestamp(a.test_end) + pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    prices = load_prices(tickers + ["SPY"], a.train_start, dl_end, cache)
    prices = prices.loc[: a.test_end].ffill(limit=5)
    spy = prices.pop("SPY")
    print(f"{prices.shape[1]} tickers, {prices.index[0].date()} to {prices.index[-1].date()}")

    # rebalance on the last trading day of the month before each test month
    first = pd.Timestamp(a.test_start) - pd.offsets.MonthBegin(1)
    # last rebalance is the month-end before the test month containing test_end;
    # stop at the last calendar day of the previous month so the 1st of the final
    # month cannot be picked up as a spurious "month end" when it is a trading day
    last = pd.Timestamp(a.test_end) - pd.offsets.MonthBegin(1) - pd.Timedelta(days=1)
    rebal = month_ends(prices.index, str(first), str(last))
    print(f"{len(rebal)} rebalances from {rebal[0].date()} to {rebal[-1].date()}, retrain every {a.retrain_months} months")

    preds = predictions(prices, rebal, a, out)
    if a.fit_only:
        print("fit-only run finished")
        return
    pred_wide = preds.pivot(index="date", columns="ticker", values="pred").reindex(rebal)
    pred_wide = pred_wide[[t for t in pred_wide.columns if t in prices.columns]]

    # realised return from each rebalance date to the next (or to test end)
    ends = rebal[1:] + [prices.index[-1]]
    realised = pd.DataFrame({d: prices.loc[e] / prices.loc[d] - 1 for d, e in zip(rebal, ends)}).T
    returns = prices.pct_change().loc[a.test_start: a.test_end]

    # universe at each rebalance: names with a prediction, and (pit) already in the
    # index on that date according to Wikipedia's "Date added"
    def in_index(t, d):
        return a.universe == "current" or pd.isna(added.get(t)) or added[t] <= d
    universe = {d: [t for t in pred_wide.columns if pd.notna(pred_wide.loc[d, t]) and in_index(t, d)] for d in rebal}
    current = {d: [t for t in pred_wide.columns if pd.notna(pred_wide.loc[d, t])] for d in rebal}
    excluded = sorted({t for d in rebal for t in current[d] if t not in universe[d]})

    def top(score: pd.DataFrame, uni, largest=True):
        return {d: (score.loc[d, uni[d]].dropna().nlargest(a.top) if largest else
                    score.loc[d, uni[d]].dropna().nsmallest(a.top)).index.tolist() for d in rebal}

    trail60 = pd.DataFrame({d: prices.loc[d] / prices.loc[:d].iloc[-61] - 1 for d in rebal}).T
    vol60 = pd.DataFrame({d: prices.loc[:d].iloc[-61:].pct_change().std() for d in rebal}).T
    mean60 = pd.DataFrame({d: prices.loc[:d].iloc[-60:].mean() / prices.loc[d] - 1 for d in rebal}).T
    mom = pd.DataFrame({d: prices.loc[:d].iloc[-22] / prices.loc[:d].iloc[-253] - 1 for d in rebal}).T

    lstm = top(pred_wide, universe)
    lstm_current = top(pred_wide, current)
    ew = {d: universe[d] for d in rebal}
    thematic = {d: [t for t in THEMATIC_30 if t in universe[d]] for d in rebal}
    # cheap alternatives under the same top-N equal-weight rule: 12-1 month price
    # momentum, the most volatile names, and the biggest recent losers
    momentum = top(mom, universe)
    highvol = top(vol60, universe)
    losers = top(trail60, universe, largest=False)
    spy_ret = spy.pct_change().loc[returns.index]

    rows, curves = [], {}

    def add(name, hold, cost, plot=True):
        r, turn = run_schedule(returns, hold, cost)
        m = metrics(r)
        m.update(portfolio=name, cost_bps=cost, avg_turnover=float(turn.mean()))
        rows.append(m)
        if plot:
            curves[f"{name} ({cost} bps)" if cost else name] = r
        return m

    for c in (0, 5, 20):
        add(f"LSTM top {a.top}", lstm, c)
    for c in (0, 20):
        add(f"Momentum 12-1 top {a.top}", momentum, c)
    add(f"Highest 60-day vol top {a.top}", highvol, 0)
    add(f"Biggest 60-day losers top {a.top}", losers, 0, plot=False)
    for c in (0, 5):
        add("Equal-weight universe", ew, c)
    add("Thematic 30", thematic, 0)
    m = metrics(spy_ret)
    m.update(portfolio="SPY", cost_bps=0, avg_turnover=0.0)
    rows.append(m)
    curves["SPY"] = spy_ret
    table = pd.DataFrame(rows)[["portfolio", "cost_bps", "total_return", "annual_return", "annual_vol", "sharpe", "max_drawdown", "avg_turnover"]]

    # signal quality: rank IC per rebalance, top-basket hit rate vs universe median
    ic = pd.Series({d: information_coefficient(pred_wide.loc[d, universe[d]], realised.loc[d, universe[d]]) for d in rebal})
    hit = pd.Series({d: float(realised.loc[d, lstm[d]].mean() > realised.loc[d, universe[d]].median()) for d in rebal})
    ic_stats = {"mean_ic": float(ic.mean()), "std_ic": float(ic.std(ddof=1)),
                "t_stat": float(ic.mean() / ic.std(ddof=1) * np.sqrt(len(ic))), "share_positive": float((ic > 0).mean()),
                "top_basket_beat_median_share": float(hit.mean()), "n_rebalances": len(ic)}

    # what is the model actually ranking on? correlation of the prediction with
    # simple functions of the 60-day window it was shown, and the basket's tilts
    def mean_ic(score):
        return float(np.mean([information_coefficient(pred_wide.loc[d, universe[d]], score.loc[d, universe[d]]) for d in rebal]))

    def overlap(other):
        return float(np.mean([len(set(lstm[d]) & set(other[d])) / a.top for d in rebal]))

    diag = {
        "ic_pred_vs_trailing60_return": mean_ic(trail60),
        "ic_pred_vs_mean60_over_last": mean_ic(mean60),
        "ic_pred_vs_vol60": mean_ic(vol60),
        "ic_trailing60_vs_realised": float(np.mean([information_coefficient(-trail60.loc[d, universe[d]], realised.loc[d, universe[d]]) for d in rebal])),
        "ic_vol60_vs_realised": float(np.mean([information_coefficient(vol60.loc[d, universe[d]], realised.loc[d, universe[d]]) for d in rebal])),
        "basket_trailing60_minus_universe_median": float(np.mean([trail60.loc[d, lstm[d]].mean() - trail60.loc[d, universe[d]].median() for d in rebal])),
        "basket_vol60_over_universe_median": float(np.mean([vol60.loc[d, lstm[d]].mean() / vol60.loc[d, universe[d]].median() for d in rebal])),
        "overlap_with_momentum_basket": overlap(momentum),
        "overlap_with_highvol_basket": overlap(highvol),
        "overlap_with_losers_basket": overlap(losers),
        "months_lstm_beat_ew_universe": int(sum(realised.loc[d, lstm[d]].mean() > realised.loc[d, universe[d]].mean() for d in rebal)),
    }

    # where did the LSTM basket's return come from? per-name contribution
    # (1/N x realised return, summed over the months the name was held)
    contrib = pd.Series(0.0, index=pred_wide.columns)
    for d in rebal:
        contrib[lstm[d]] += realised.loc[d, lstm[d]] / len(lstm[d])
    contrib = contrib[contrib != 0].sort_values(ascending=False)
    contrib_df = pd.DataFrame({"contribution_pp": contrib * 100,
                               "months_held": [sum(t in lstm[d] for d in rebal) for t in contrib.index],
                               "date_added_to_index": [added.get(t) for t in contrib.index]})
    contrib_df.to_csv(out / "contributions_lstm.csv")

    # the same basket built from today's list without the date-added filter
    cur = metrics(run_schedule(returns, lstm_current, 0.0)[0])
    cur_contrib = pd.Series(0.0, index=pred_wide.columns)
    for d in rebal:
        cur_contrib[lstm_current[d]] += realised.loc[d, lstm_current[d]] / len(lstm_current[d])
    universe_check = {"universe": a.universe, "excluded_tickers": excluded,
                      "n_universe_min": min(len(u) for u in universe.values()), "n_universe_max": max(len(u) for u in universe.values()),
                      "current_list_lstm_total_return": cur["total_return"], "current_list_lstm_sharpe": cur["sharpe"],
                      "current_list_contribution_of_excluded_pp": float(cur_contrib[[t for t in excluded if t in cur_contrib.index]].sum() * 100)}

    # luck test: random equal-weight baskets of the same size, gross of costs
    rand_sched = random_baskets(universe, a.top, a.draws, a.seed)
    rand = [metrics(run_schedule(returns, h, 0.0)[0]) for h in rand_sched]
    rand_tot = np.array([m["total_return"] for m in rand])
    rand_sh = np.array([m["sharpe"] for m in rand])
    rand_vol = np.array([m["annual_vol"] for m in rand])
    rand_turn = float(np.mean([run_schedule(returns, h, 0.0)[1].iloc[1:].mean() for h in rand_sched[:50]]))
    lstm_row = table.loc[(table.portfolio == f"LSTM top {a.top}") & (table.cost_bps == 0)].iloc[0]
    lstm_gross, lstm_sharpe = lstm_row["total_return"], lstm_row["sharpe"]
    luck = {"draws": int(a.draws), "random_median": float(np.median(rand_tot)), "random_p5": float(np.percentile(rand_tot, 5)),
            "random_p95": float(np.percentile(rand_tot, 95)), "lstm_percentile": float((rand_tot < lstm_gross).mean() * 100),
            "random_sharpe_median": float(np.median(rand_sh)), "random_sharpe_p95": float(np.percentile(rand_sh, 95)),
            "lstm_sharpe_percentile": float((rand_sh < lstm_sharpe).mean() * 100),
            "random_vol_median": float(np.median(rand_vol)), "random_turnover_ex_initial": rand_turn}

    # outputs
    table.to_csv(out / "results.csv", index=False)
    ic.rename("ic").to_csv(out / "monthly_ic.csv")
    pd.DataFrame({d: pd.Series(v) for d, v in lstm.items()}).T.to_csv(out / "holdings_lstm.csv")
    pd.DataFrame({"n_universe": {str(d.date()): len(u) for d, u in universe.items()}}).to_csv(out / "universe_size.csv")
    json.dump({"args": vars(a), "signal": ic_stats, "diagnostics": diag, "universe_check": universe_check, "luck": luck,
               "top_contributors_pp": contrib_df.head(10)["contribution_pp"].round(2).to_dict(),
               "n_universe": {str(d.date()): len(u) for d, u in universe.items()}},
              open(out / "summary.json", "w"), indent=2, default=str)

    fmt = table.copy()
    for c in ["total_return", "annual_return", "annual_vol", "max_drawdown", "avg_turnover"]:
        fmt[c] = fmt[c].map(lambda x: f"{x:.1%}")
    fmt["sharpe"] = fmt["sharpe"].map(lambda x: f"{x:.2f}")
    fmt.columns = ["Portfolio", "Cost (bps)", "Total return", "Ann. return", "Ann. vol", "Sharpe", "Max drawdown", "Avg. turnover / rebalance"]
    md = fmt.to_markdown(index=False)
    (out / "results.md").write_text(md + "\n\nSignal: " + json.dumps(ic_stats) + "\n\nDiagnostics: " + json.dumps(diag)
                                    + "\n\nUniverse check: " + json.dumps(universe_check, default=str)
                                    + "\n\nLuck test: " + json.dumps(luck) + "\n\nTop contributors (pp):\n"
                                    + contrib_df.head(10).to_string() + "\n")
    print("\n" + md)
    print("\nsignal:", json.dumps(ic_stats, indent=1))
    print("diagnostics:", json.dumps(diag, indent=1))
    print("universe check:", json.dumps(universe_check, indent=1, default=str))
    print("luck test:", json.dumps(luck, indent=1))
    print("top contributors:\n", contrib_df.head(10).to_string())

    plot(curves, ic, rand_tot, rand_sh, lstm_gross, lstm_sharpe, a, out)
    print("done ->", out)


def plot(curves, ic, rand_tot, rand_sh, lstm_gross, lstm_sharpe, a, out: Path):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for name, r in curves.items():
        if "(5 bps)" in name:
            continue  # nearly on top of the gross line; numbers are in the table
        ls = "--" if name == "SPY" else "-"
        ax.plot((1 + r).cumprod(), label=name, lw=1.6, ls=ls)
    ax.set_title(f"Growth of $1, {a.test_start} to {a.test_end}, monthly rebalance")
    ax.grid(alpha=.3); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out / "equity_curves.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    for name in [f"LSTM top {a.top}", f"LSTM top {a.top} (20 bps)", f"Momentum 12-1 top {a.top}", "Equal-weight universe", "SPY"]:
        if name in curves:
            ax.plot(drawdown_series(curves[name]), label=name, lw=1.4)
    ax.set_title("Drawdown from running peak"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "drawdowns.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(ic.index, ic.values, width=18, color=np.where(ic.values > 0, "tab:blue", "tab:red"))
    ax.axhline(0, color="k", lw=.8); ax.axhline(ic.mean(), color="gray", ls="--", lw=1, label=f"mean IC {ic.mean():.3f}")
    ax.set_title("Rank correlation between predicted and realised one-month return, by rebalance date")
    ax.legend(); ax.grid(alpha=.3, axis="y"); fig.tight_layout(); fig.savefig(out / "monthly_ic.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, vals, mine, lab, fmt in [(axes[0], rand_tot, lstm_gross, "total return", "{:.1%}"),
                                     (axes[1], rand_sh, lstm_sharpe, "Sharpe (2% rf)", "{:.2f}")]:
        ax.hist(vals, bins=40, color="lightgray", edgecolor="gray")
        ax.axvline(mine, color="tab:blue", lw=2, label=f"LSTM top {a.top}, gross: {fmt.format(mine)}")
        ax.axvline(np.median(vals), color="k", ls="--", lw=1, label=f"random median: {fmt.format(np.median(vals))}")
        ax.set_title(f"{lab} of {len(vals)} random {a.top}-stock baskets", fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle("Random equal-weight baskets drawn on the same dates from the same universe, gross of costs", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "random_baskets.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
