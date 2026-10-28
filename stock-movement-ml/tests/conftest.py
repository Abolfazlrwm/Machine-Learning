"""Shared fixtures: synthetic OHLCV data so tests never touch the network."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import FEATURE_COLUMNS, calculate_technical_indicators, create_target  # noqa: E402


def make_ohlcv(n_days: int = 400, seed: int = 7, drift: float = 0.0004) -> pd.DataFrame:
    """Seeded geometric random walk with volume -> realistic-ish OHLCV frame."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    ret = rng.normal(drift, 0.015, size=n_days)
    close = 100.0 * np.exp(np.cumsum(ret))

    open_ = close * (1 + rng.normal(0, 0.004, size=n_days))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, size=n_days)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, size=n_days)))
    volume = rng.integers(1_000_000, 5_000_000, size=n_days).astype(float)

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


@pytest.fixture(scope="session")
def ohlcv() -> pd.DataFrame:
    return make_ohlcv()


@pytest.fixture(scope="session")
def features_df(ohlcv: pd.DataFrame) -> pd.DataFrame:
    return create_target(calculate_technical_indicators(ohlcv))


__all__ = ["make_ohlcv", "FEATURE_COLUMNS"]
