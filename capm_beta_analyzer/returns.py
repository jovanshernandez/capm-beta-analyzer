"""Periodic returns from prices."""

from __future__ import annotations

import numpy as np
import pandas as pd

METHODS = ("simple", "log")


def compute_returns(prices: pd.DataFrame, method: str = "simple") -> pd.DataFrame:
    """Simple returns P_t / P_{t-1} - 1, or log returns ln(P_t / P_{t-1}). First row dropped."""
    if method == "simple":
        returns = prices.pct_change()
    elif method == "log":
        returns = np.log(prices / prices.shift(1))
    else:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    return returns.iloc[1:]


def periodic_rate(annual_rate: float, periods_per_year: int, method: str = "simple") -> float:
    """Convert an annual rate to the per-period rate matching the return method."""
    if method == "log":
        return float(np.log1p(annual_rate) / periods_per_year)
    return float((1.0 + annual_rate) ** (1.0 / periods_per_year) - 1.0)
