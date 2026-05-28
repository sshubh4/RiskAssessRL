"""Download OHLCV data from Yahoo Finance and run the full pipeline."""
from __future__ import annotations
import pathlib
import concurrent.futures
import yfinance as yf
import pandas as pd

RAW_DIR = pathlib.Path(__file__).parent.parent / "data" / "raw"

TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "SPY"]


def ingest(ticker: str = "AAPL", period: str = "5y") -> pathlib.Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    out = RAW_DIR / f"{ticker}.csv"
    df.to_csv(out)
    print(f"[ingest] saved {len(df)} rows → {out}")
    return out


def download_ticker(ticker: str, period: str = "5y") -> bool:
    """Download, compute features, and validate a single ticker. Returns True on success."""
    from pipeline.features import compute_features
    from pipeline.validate import validate
    try:
        ingest(ticker, period)
        compute_features(ticker)
        validate(ticker)
        return True
    except Exception as exc:
        print(f"[pipeline] ERROR for {ticker}: {exc}")
        return False


def run_all(tickers: list[str] = TICKERS, period: str = "5y",
            max_workers: int = 5) -> dict[str, bool]:
    """Run the full pipeline for all tickers in parallel via threading."""
    results: dict[str, bool] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(download_ticker, t, period): t for t in tickers}
        for future in concurrent.futures.as_completed(futures):
            ticker = futures[future]
            try:
                results[ticker] = future.result()
            except Exception as exc:
                print(f"[pipeline] FATAL for {ticker}: {exc}")
                results[ticker] = False
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default=None,
                        help="Single ticker to download (omit for all)")
    parser.add_argument("--period", default="5y")
    args = parser.parse_args()

    if args.ticker:
        download_ticker(args.ticker, args.period)
    else:
        results = run_all(period=args.period)
        ok = [t for t, v in results.items() if v]
        fail = [t for t, v in results.items() if not v]
        print(f"\n[pipeline] Done — OK: {ok}  Failed: {fail}")
