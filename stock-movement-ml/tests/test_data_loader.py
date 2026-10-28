import numpy as np
import pandas as pd

from src.data_loader import cache_path, flatten_columns, load_cache, save_cache


class TestFlattenColumns:
    def test_multiindex_flat_columns(self):
        idx = pd.bdate_range("2024-01-01", periods=3)
        arrays = [["Close", "High", "Low", "Open", "Volume"],
                  ["AAPL"] * 5]
        cols = pd.MultiIndex.from_arrays(arrays)
        df = pd.DataFrame(np.arange(15).reshape(3, 5).astype(float),
                          index=idx, columns=cols)
        flat = flatten_columns(df)
        assert list(flat.columns) == ["Close", "High", "Low", "Open", "Volume"]
        assert not isinstance(flat.columns, pd.MultiIndex)

    def test_plain_columns_untouched(self):
        df = pd.DataFrame({"Close": [1.0, 2.0]})
        pd.testing.assert_frame_equal(flatten_columns(df), df)


class TestCache:
    def test_roundtrip(self, tmp_path):
        idx = pd.bdate_range("2024-01-01", periods=5)
        df = pd.DataFrame({"Close": np.linspace(100, 110, 5)}, index=idx)
        path = tmp_path / "cache" / "AAPL_test.csv"
        save_cache(df, path)
        loaded = load_cache(path)
        np.testing.assert_allclose(loaded["Close"].values, df["Close"].values)
        assert isinstance(loaded.index, pd.DatetimeIndex)

    def test_cache_path_layout(self, tmp_path):
        path = cache_path("aapl", "2020-01-01", "2021-01-01", cache_dir=tmp_path)
        assert path.name == "AAPL_2020-01-01_2021-01-01.csv"
        assert path.parent == tmp_path
