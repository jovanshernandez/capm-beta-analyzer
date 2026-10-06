"""Load and validate a wide CSV of prices: one date column, one column per security."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Bundled sample: daily closes for seven stocks and the S&P 500, Nov 2013 to Aug 2020.
DEFAULT_DATASET = Path(__file__).resolve().parent.parent / "data" / "stocks_dataset.csv"


class DataError(ValueError):
    """Raised when the input file cannot be used for a CAPM regression."""


def load_prices(
    path: str | Path = DEFAULT_DATASET,
    *,
    date_column: str = "Date",
    benchmark: str = "sp500",
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Return a date-indexed frame of positive prices with the benchmark as the last column."""
    path = Path(path)
    if not path.is_file():
        raise DataError(f"price file not found: {path}")

    raw = pd.read_csv(path)
    if date_column not in raw.columns:
        raise DataError(f"date column {date_column!r} not in {list(raw.columns)}")
    if benchmark not in raw.columns:
        raise DataError(f"benchmark column {benchmark!r} not in {list(raw.columns)}")

    try:
        dates = pd.to_datetime(raw.pop(date_column))
    except (ValueError, TypeError) as exc:
        raise DataError(f"could not parse dates in {date_column!r}: {exc}") from exc

    prices = raw.apply(pd.to_numeric, errors="coerce").set_index(dates).sort_index()
    prices.index.name = "date"

    if prices.index.has_duplicates:
        raise DataError("duplicate dates in price file")

    assets = [c for c in prices.columns if c != benchmark] if tickers is None else list(tickers)
    missing = [t for t in assets if t not in prices.columns]
    if missing:
        raise DataError(f"tickers not in file: {missing}")
    if benchmark in assets:
        raise DataError("benchmark cannot also be an asset")
    if not assets:
        raise DataError("no asset columns to analyze")

    prices = prices[assets + [benchmark]].dropna(how="any")
    if len(prices) < 3:
        raise DataError("need at least 3 complete rows of prices")
    if (prices <= 0).any().any():
        raise DataError("prices must be positive")
    return prices
