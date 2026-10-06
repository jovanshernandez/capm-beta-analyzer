import numpy as np
import pandas as pd
import pytest
from conftest import TRUE_ALPHA, TRUE_BETA

from capm_beta_analyzer import analyze, capm_expected_return, compute_returns, ols_fit, rolling_beta
from capm_beta_analyzer.returns import periodic_rate


def test_ols_matches_numpy_polyfit():
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    y = 0.3 + 2.0 * x + rng.normal(scale=0.5, size=500)
    fit = ols_fit(x, y)
    slope, intercept = np.polyfit(x, y, 1)
    assert fit.slope == pytest.approx(slope)
    assert fit.intercept == pytest.approx(intercept)
    assert fit.r_squared == pytest.approx(np.corrcoef(x, y)[0, 1] ** 2)


def test_ols_perfect_fit_has_unit_r_squared():
    x = np.arange(10.0)
    fit = ols_fit(x, 3 * x + 1)
    assert fit.slope == pytest.approx(3)
    assert fit.intercept == pytest.approx(1)
    assert fit.r_squared == pytest.approx(1)


def test_ols_rejects_constant_regressor():
    with pytest.raises(ValueError, match="zero variance"):
        ols_fit(np.ones(5), np.arange(5.0))


def test_analyze_recovers_known_beta(synthetic_returns):
    results, _ = analyze(synthetic_returns, "MKT")
    by = {r.ticker: r for r in results}
    high = by["HIGH"]
    # Recovered within a few standard errors of the true value.
    assert abs(high.beta - TRUE_BETA) < 4 * high.beta_se
    assert high.beta_se < 0.02
    assert high.beta_t > 50
    assert high.alpha_daily == pytest.approx(TRUE_ALPHA, abs=4e-4)
    assert high.alpha_annual == pytest.approx(high.alpha_daily * 252)
    assert 0.8 < high.r_squared < 0.95
    assert by["LOW"].beta == pytest.approx(0.5, abs=0.03)
    assert high.observations == len(synthetic_returns)


def test_beta_does_not_depend_on_risk_free(synthetic_returns):
    base, _ = analyze(synthetic_returns, "MKT", risk_free=0.0)
    shifted, _ = analyze(synthetic_returns, "MKT", risk_free=0.05)
    assert shifted[0].beta == pytest.approx(base[0].beta)
    assert shifted[0].alpha_daily != pytest.approx(base[0].alpha_daily)


def test_expected_return_uses_given_premium(synthetic_returns):
    results, premium = analyze(synthetic_returns, "MKT", risk_free=0.03, market_premium=0.06)
    assert premium == 0.06
    for r in results:
        assert r.expected_return == pytest.approx(0.03 + r.beta * 0.06)


def test_realized_premium_decomposes_mean_excess_return(synthetic_returns):
    # OLS identity: mean excess return = alpha + beta * mean market excess return.
    results, premium = analyze(synthetic_returns, "MKT", risk_free=0.02)
    rf = periodic_rate(0.02, 252)
    for r in results:
        mean_excess = (synthetic_returns[r.ticker] - rf).mean() * 252
        assert mean_excess == pytest.approx(r.alpha_annual + r.beta * premium)


def test_capm_expected_return():
    assert capm_expected_return(1.2, 0.04, 0.05) == pytest.approx(0.10)


def test_rolling_beta_tracks_regime_change():
    rng = np.random.default_rng(7)
    market = rng.normal(0, 0.01, 600)
    beta = np.where(np.arange(600) < 300, 0.5, 2.0)
    asset = beta * market + rng.normal(0, 0.002, 600)
    df = pd.DataFrame({"A": asset, "M": market})
    rb = rolling_beta(df, "A", "M", window=60)
    assert len(rb) == 600 - 59
    assert rb.iloc[200] == pytest.approx(0.5, abs=0.1)
    assert rb.iloc[-1] == pytest.approx(2.0, abs=0.1)


def test_rolling_window_validation(synthetic_returns):
    with pytest.raises(ValueError):
        rolling_beta(synthetic_returns, "HIGH", "MKT", window=10_000)


def test_simple_and_log_returns():
    prices = pd.DataFrame({"A": [100.0, 110.0, 99.0]})
    simple = compute_returns(prices, "simple")["A"].to_numpy()
    log = compute_returns(prices, "log")["A"].to_numpy()
    assert simple == pytest.approx([0.10, -0.10])
    assert log == pytest.approx(np.log([1.1, 0.9]))
    with pytest.raises(ValueError):
        compute_returns(prices, "weird")


def test_periodic_rate_compounds_back_to_annual():
    daily = periodic_rate(0.05, 252)
    assert (1 + daily) ** 252 == pytest.approx(1.05)
    assert periodic_rate(0.05, 252, "log") * 252 == pytest.approx(np.log(1.05))
