"""Constituents, price download and caching.

The constituent list is today's S&P 500 from Wikipedia together with each
company's "Date added" column. The backtest uses that column to drop names
that joined the index after a given rebalance date (see README, Limitations:
this removes inclusion bias but not the names that left the index).
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# 30 names from the v1 "thematic" comparison list, kept for continuity. The list
# was suggested by ChatGPT when v1 was written (after the test window), so it
# carries hindsight; see README.
THEMATIC_30 = [
    "NVDA", "MSFT", "AVGO", "GOOGL", "AMZN", "ADBE", "ORCL", "META",
    "CRM", "NOW", "INTC", "AMD", "QCOM", "TXN", "MU", "PLTR",
    "UNH", "MRK", "LLY", "JNJ", "ABBV", "PFE", "TMO",
    "V", "MA", "JPM", "USB", "BRK-B", "WMT", "COST",
]


def sp500_constituents(cache: Path) -> pd.DataFrame:
    """Current S&P 500 members: columns ticker (yfinance format, BRK.B -> BRK-B)
    and date_added (NaT when Wikipedia has no date). Cached as CSV so a run
    without network, or a later run, uses the same list."""
    if cache.exists():
        df = pd.read_csv(cache, parse_dates=["date_added"])
        return df
    html = requests.get(WIKI_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
    table = pd.read_html(io.StringIO(html))[0]
    df = pd.DataFrame({
        "ticker": table["Symbol"].astype(str).str.replace(".", "-", regex=False),
        "date_added": pd.to_datetime(table["Date added"].astype(str).str.slice(0, 10), errors="coerce"),
    }).drop_duplicates("ticker").sort_values("ticker").reset_index(drop=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def load_prices(tickers: list[str], start: str, end: str, cache: Path) -> pd.DataFrame:
    """Daily adjusted close, one column per ticker. Cached to parquet.

    Tickers with less than 50% of the trading days are dropped, as in v1.
    """
    if cache.exists():
        return pd.read_parquet(cache)
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      progress=False, threads=True, group_by="column")
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    close = close.dropna(axis=1, thresh=int(len(close) * 0.5)).sort_index()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    cache.parent.mkdir(parents=True, exist_ok=True)
    close.to_parquet(cache)
    return close
