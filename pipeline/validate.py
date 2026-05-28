"""Pandera schema validation for the processed feature CSV."""
import pathlib
import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check

PROC_DIR = pathlib.Path(__file__).parent.parent / "data" / "processed"

FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "MA_20", "MA_50", "RSI_14", "MACD", "MACD_signal",
    "BB_mid", "BB_upper", "BB_lower", "ATR_14",
    "Support", "Resistance",
]

schema = DataFrameSchema(
    columns={
        "Date": Column(pa.DateTime, nullable=False),
        "Open":  Column(float, Check.gt(0), nullable=False),
        "High":  Column(float, Check.gt(0), nullable=False),
        "Low":   Column(float, Check.gt(0), nullable=False),
        "Close": Column(float, Check.gt(0), nullable=False),
        "Volume": Column(float, Check.ge(0), nullable=False),
        "MA_20":  Column(float, nullable=False),
        "MA_50":  Column(float, nullable=False),
        "RSI_14": Column(float, Check.in_range(0, 100), nullable=False),
        "MACD":   Column(float, nullable=False),
        "MACD_signal": Column(float, nullable=False),
        "BB_mid":   Column(float, nullable=False),
        "BB_upper": Column(float, nullable=False),
        "BB_lower": Column(float, nullable=False),
        "ATR_14":   Column(float, Check.ge(0), nullable=False),
        "Support":    Column(float, nullable=False),
        "Resistance": Column(float, nullable=False),
    },
    checks=[
        Check(lambda df: df["Date"].is_monotonic_increasing,
              error="Dates must be monotonically increasing"),
    ],
    coerce=True,
)


def validate(ticker: str = "AAPL") -> bool:
    path = PROC_DIR / f"{ticker}_features.csv"
    df = pd.read_csv(path, parse_dates=["Date"])
    try:
        schema.validate(df, lazy=True)
        print(f"[validate] {ticker} schema OK ({len(df)} rows)")
        return True
    except pa.errors.SchemaErrors as exc:
        print(f"[validate] FAILED:\n{exc.failure_cases}")
        raise


if __name__ == "__main__":
    validate()
