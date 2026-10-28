"""Download OHLCV data from Yahoo Finance with a local CSV cache.

The cache makes the pipeline reproducible and lets you work offline:
once a ticker/period is downloaded it is stored under ``data/`` and reused
unless ``refresh=True`` is passed.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance >= 0.2 returns MultiIndex columns like ``('Close', 'AAPL')``.

    This flattens them back to simple names (Close, High, Low, Open, Volume).
    """
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [str(level[0]) if isinstance(level, tuple) else str(level)
                      for level in df.columns]
    return df


def cache_path(ticker: str, start: str, end: str, cache_dir: str | Path = "data") -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{ticker.upper()}_{start}_{end}.csv"


def save_cache(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)


def load_cache(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


def load_data(
    ticker: str,
    start: str,
    end: str,
    cache_dir: str | Path = "data",
    refresh: bool = False,
) -> pd.DataFrame:
    """Download OHLCV for *ticker* between start/end (or use the CSV cache).

    Returns a DataFrame indexed by date with columns
    Open / High / Low / Close / Volume.
    """
    path = cache_path(ticker, start, end, cache_dir)
    if path.exists() and not refresh:
        return load_cache(path)

    import yfinance as yf  # lazy import: cache-only runs need no network

    raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if raw is None or raw.empty:
        raise ValueError(f"No data returned for {ticker} ({start} -> {end}). "
                         "Check the ticker symbol and your connection.")
    df = flatten_columns(raw)

    # A healthy sanity check on the expected schema
    missing = {"Open", "High", "Low", "Close", "Volume"} - set(df.columns)
    if missing:
        raise ValueError(f"Downloaded data is missing columns: {missing}")

    save_cache(df, path)
    return df
