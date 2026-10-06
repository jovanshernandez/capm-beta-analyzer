import numpy as np
import pandas as pd
import pytest

TRUE_BETA = 1.5
TRUE_ALPHA = 0.0002  # per day


@pytest.fixture
def synthetic_returns() -> pd.DataFrame:
    """Market returns plus an asset built as alpha + 1.5 * market + noise."""
    rng = np.random.default_rng(42)
    n = 2000
    market = rng.normal(0.0004, 0.01, n)
    asset = TRUE_ALPHA + TRUE_BETA * market + rng.normal(0, 0.005, n)
    defensive = 0.5 * market + rng.normal(0, 0.004, n)
    idx = pd.bdate_range("2018-01-01", periods=n)
    return pd.DataFrame({"HIGH": asset, "LOW": defensive, "MKT": market}, index=idx)


@pytest.fixture
def synthetic_csv(tmp_path, synthetic_returns):
    """Prices compounded from the synthetic returns, written as a wide CSV."""
    prices = 100 * (1 + synthetic_returns).cumprod()
    prices.index.name = "Date"
    path = tmp_path / "prices.csv"
    prices.reset_index().assign(Date=lambda d: d["Date"].dt.strftime("%Y-%m-%d")).to_csv(
        path, index=False
    )
    return path
