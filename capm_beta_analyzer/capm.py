"""CAPM regression: r_i - r_f = alpha + beta * (r_m - r_f) + e."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from capm_beta_analyzer.returns import periodic_rate


@dataclass(frozen=True)
class OlsFit:
    slope: float
    intercept: float
    slope_se: float
    intercept_se: float
    r_squared: float
    n: int

    @property
    def slope_t(self) -> float:
        return self.slope / self.slope_se if self.slope_se > 0 else float("inf")

    @property
    def intercept_t(self) -> float:
        return self.intercept / self.intercept_se if self.intercept_se > 0 else float("inf")


@dataclass(frozen=True)
class CapmResult:
    ticker: str
    beta: float
    beta_se: float
    beta_t: float
    alpha_daily: float
    alpha_annual: float
    alpha_t: float
    r_squared: float
    annual_return: float
    annual_volatility: float
    expected_return: float
    observations: int

    def as_dict(self) -> dict:
        return asdict(self)


def ols_fit(x: np.ndarray, y: np.ndarray) -> OlsFit:
    """Single-regressor OLS with classical (homoskedastic) standard errors."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if n != y.size:
        raise ValueError("x and y must have the same length")
    if n < 3:
        raise ValueError("need at least 3 observations")
    x_mean, y_mean = x.mean(), y.mean()
    sxx = np.sum((x - x_mean) ** 2)
    if sxx == 0:
        raise ValueError("regressor has zero variance")
    slope = np.sum((x - x_mean) * (y - y_mean)) / sxx
    intercept = y_mean - slope * x_mean
    resid = y - (intercept + slope * x)
    ssr = np.sum(resid**2)
    sst = np.sum((y - y_mean) ** 2)
    sigma2 = ssr / (n - 2)
    return OlsFit(
        slope=float(slope),
        intercept=float(intercept),
        slope_se=float(np.sqrt(sigma2 / sxx)),
        intercept_se=float(np.sqrt(sigma2 * (1.0 / n + x_mean**2 / sxx))),
        r_squared=float(1.0 - ssr / sst) if sst > 0 else 1.0,
        n=n,
    )


def capm_expected_return(beta: float, risk_free: float, market_premium: float) -> float:
    """E[R_i] = r_f + beta * (E[R_m] - r_f)."""
    return risk_free + beta * market_premium


def realized_market_premium(
    returns: pd.DataFrame,
    benchmark: str,
    risk_free: float,
    periods_per_year: int = 252,
    method: str = "simple",
) -> float:
    """Annualized mean benchmark excess return over the sample."""
    rf = periodic_rate(risk_free, periods_per_year, method)
    return float((returns[benchmark] - rf).mean() * periods_per_year)


def analyze(
    returns: pd.DataFrame,
    benchmark: str,
    *,
    risk_free: float = 0.0,
    market_premium: float | None = None,
    periods_per_year: int = 252,
    method: str = "simple",
) -> tuple[list[CapmResult], float]:
    """Regress each asset's excess return on the benchmark's.

    `risk_free` is an annual rate. If `market_premium` is None the realized
    sample premium is used. Annualized figures are arithmetic (daily x periods).
    Returns the per-asset results and the market premium that was applied.
    """
    rf = periodic_rate(risk_free, periods_per_year, method)
    excess = returns - rf
    market = excess[benchmark].to_numpy()
    if market_premium is None:
        market_premium = realized_market_premium(
            returns, benchmark, risk_free, periods_per_year, method
        )

    results = []
    for ticker in (c for c in returns.columns if c != benchmark):
        fit = ols_fit(market, excess[ticker].to_numpy())
        results.append(
            CapmResult(
                ticker=ticker,
                beta=fit.slope,
                beta_se=fit.slope_se,
                beta_t=fit.slope_t,
                alpha_daily=fit.intercept,
                alpha_annual=fit.intercept * periods_per_year,
                alpha_t=fit.intercept_t,
                r_squared=fit.r_squared,
                annual_return=float(returns[ticker].mean() * periods_per_year),
                annual_volatility=float(returns[ticker].std(ddof=1) * np.sqrt(periods_per_year)),
                expected_return=capm_expected_return(fit.slope, risk_free, market_premium),
                observations=fit.n,
            )
        )
    return results, float(market_premium)


def rolling_beta(returns: pd.DataFrame, ticker: str, benchmark: str, window: int) -> pd.Series:
    """Beta over a trailing window: Cov(r_i, r_m) / Var(r_m).

    A constant risk-free rate shifts both series equally, so it does not change beta.
    """
    if window < 3:
        raise ValueError("rolling window must be at least 3")
    if window > len(returns):
        raise ValueError(f"rolling window {window} exceeds {len(returns)} observations")
    cov = returns[ticker].rolling(window).cov(returns[benchmark])
    var = returns[benchmark].rolling(window).var()
    return (cov / var).dropna().rename(f"{ticker} rolling beta ({window}d)")
