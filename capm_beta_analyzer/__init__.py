"""CAPM beta, alpha and expected-return estimation from daily price data."""

from capm_beta_analyzer.capm import (
    CapmResult,
    analyze,
    capm_expected_return,
    ols_fit,
    rolling_beta,
)
from capm_beta_analyzer.data import load_prices
from capm_beta_analyzer.returns import compute_returns

__all__ = [
    "CapmResult",
    "analyze",
    "capm_expected_return",
    "compute_returns",
    "load_prices",
    "ols_fit",
    "rolling_beta",
]
__version__ = "0.2.0"
