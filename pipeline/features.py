"""Compute technical indicators and save processed feature file."""
import pathlib
import numpy as np
import pandas as pd

RAW_DIR = pathlib.Path(__file__).parent.parent / "data" / "raw"
PROC_DIR = pathlib.Path(__file__).parent.parent / "data" / "processed"


def compute_features(ticker: str = "AAPL") -> pathlib.Path:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW_DIR / f"{ticker}.csv", index_col="Date", parse_dates=True)
    df = df.sort_index()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # Moving averages
    df["MA_20"] = close.rolling(20).mean()
    df["MA_50"] = close.rolling(50).mean()

    # RSI-14
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands (20-day, 2σ)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["BB_mid"] = bb_mid
    df["BB_upper"] = bb_mid + 2 * bb_std
    df["BB_lower"] = bb_mid - 2 * bb_std

    # ATR-14
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["ATR_14"] = tr.rolling(14).mean()

    # Support / Resistance (7-day rolling)
    df["Support"] = low.rolling(7).min()
    df["Resistance"] = high.rolling(7).max()

    df = df.dropna().reset_index()
    out = PROC_DIR / f"{ticker}_features.csv"
    df.to_csv(out, index=False)
    print(f"[features] saved {len(df)} rows → {out}")
    return out


if __name__ == "__main__":
    compute_features()
