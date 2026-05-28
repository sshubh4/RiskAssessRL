"""Tests for pipeline feature computation and schema validation."""
import pathlib
import numpy as np
import pandas as pd
import pytest
import pandera as pa


# ---------------------------------------------------------------------------
# Helpers that replicate pipeline logic without file I/O
# ---------------------------------------------------------------------------

def _generate_ohlcv(n: int = 200) -> pd.DataFrame:
    np.random.seed(0)
    close = 150.0 + np.cumsum(np.random.randn(n) * 0.8)
    close = np.clip(close, 50, 500)
    return pd.DataFrame({
        "Date": pd.date_range("2020-01-01", periods=n, freq="B"),
        "Open": close * 0.995,
        "High": close * 1.012,
        "Low": close * 0.988,
        "Close": close,
        "Volume": np.random.randint(5_000_000, 50_000_000, n).astype(float),
    })


def _apply_features(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror of pipeline/features.py logic, applied in-memory."""
    df = df.copy().set_index("Date").sort_index()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    df["MA_20"] = close.rolling(20).mean()
    df["MA_50"] = close.rolling(50).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["BB_mid"] = bb_mid
    df["BB_upper"] = bb_mid + 2 * bb_std
    df["BB_lower"] = bb_mid - 2 * bb_std

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["ATR_14"] = tr.rolling(14).mean()

    df["Support"] = low.rolling(7).min()
    df["Resistance"] = high.rolling(7).max()

    return df.dropna().reset_index()


@pytest.fixture
def featured_df():
    return _apply_features(_generate_ohlcv(200))


# ---------------------------------------------------------------------------
# Feature correctness tests
# ---------------------------------------------------------------------------

class TestFeatureComputation:
    def test_rsi_in_range(self, featured_df):
        assert (featured_df["RSI_14"] >= 0).all(), "RSI below 0"
        assert (featured_df["RSI_14"] <= 100).all(), "RSI above 100"

    def test_ma20_less_than_ma50_window(self, featured_df):
        # MA_20 reacts faster — both should exist as floats
        assert featured_df["MA_20"].notna().all()
        assert featured_df["MA_50"].notna().all()

    def test_bollinger_order(self, featured_df):
        assert (featured_df["BB_upper"] >= featured_df["BB_mid"]).all()
        assert (featured_df["BB_lower"] <= featured_df["BB_mid"]).all()

    def test_support_leq_low(self, featured_df):
        # 7-day rolling min of Low must be ≤ each day's Low
        assert (featured_df["Support"] <= featured_df["Low"] + 1e-9).all()

    def test_resistance_geq_high(self, featured_df):
        assert (featured_df["Resistance"] >= featured_df["High"] - 1e-9).all()

    def test_atr_non_negative(self, featured_df):
        assert (featured_df["ATR_14"] >= 0).all()

    def test_date_monotonic(self, featured_df):
        assert featured_df["Date"].is_monotonic_increasing

    def test_no_nulls_after_dropna(self, featured_df):
        assert featured_df.isnull().sum().sum() == 0

    def test_row_count_reduced(self):
        raw = _generate_ohlcv(200)
        feat = _apply_features(raw)
        # dropna removes the initial rolling-window rows
        assert len(feat) < len(raw)
        assert len(feat) > 100  # should still have plenty of rows


# ---------------------------------------------------------------------------
# Schema validation tests (via pipeline/validate.py logic)
# ---------------------------------------------------------------------------

def _load_schema():
    from pipeline.validate import schema
    return schema


class TestSchemaValidation:
    def test_valid_df_passes(self, featured_df):
        schema = _load_schema()
        schema.validate(featured_df, lazy=True)  # should not raise

    def test_null_in_rsi_fails(self, featured_df):
        schema = _load_schema()
        bad = featured_df.copy()
        bad.loc[bad.index[5], "RSI_14"] = np.nan
        with pytest.raises(pa.errors.SchemaErrors):
            schema.validate(bad, lazy=True)

    def test_negative_price_fails(self, featured_df):
        schema = _load_schema()
        bad = featured_df.copy()
        bad.loc[bad.index[0], "Close"] = -10.0
        with pytest.raises(pa.errors.SchemaErrors):
            schema.validate(bad, lazy=True)

    def test_rsi_out_of_range_fails(self, featured_df):
        schema = _load_schema()
        bad = featured_df.copy()
        bad.loc[bad.index[3], "RSI_14"] = 110.0  # invalid
        with pytest.raises(pa.errors.SchemaErrors):
            schema.validate(bad, lazy=True)

    def test_non_monotonic_dates_fails(self, featured_df):
        schema = _load_schema()
        bad = featured_df.copy()
        # Swap two dates to break monotonicity
        bad.loc[bad.index[10], "Date"], bad.loc[bad.index[11], "Date"] = (
            bad.loc[bad.index[11], "Date"], bad.loc[bad.index[10], "Date"]
        )
        with pytest.raises(pa.errors.SchemaErrors):
            schema.validate(bad, lazy=True)
