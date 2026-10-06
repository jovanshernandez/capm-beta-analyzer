"""Command-line entry point: `capm-beta`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from capm_beta_analyzer.capm import CapmResult, analyze, rolling_beta
from capm_beta_analyzer.data import DEFAULT_DATASET, DataError, load_prices
from capm_beta_analyzer.returns import METHODS, compute_returns, periodic_rate


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="capm-beta",
        description="Estimate CAPM beta, alpha and expected return for each security in a price CSV.",
    )
    p.add_argument("--csv", type=Path, default=DEFAULT_DATASET,
                   help="wide CSV of prices: a date column plus one column per security "
                        "(default: bundled sample dataset)")
    p.add_argument("--date-column", default="Date", help="name of the date column (default: Date)")
    p.add_argument("--benchmark", default="sp500", help="market benchmark column (default: sp500)")
    p.add_argument("--tickers", nargs="+", metavar="TICKER",
                   help="securities to analyze (default: every non-benchmark column)")
    p.add_argument("--returns", choices=METHODS, default="simple",
                   help="return definition (default: simple)")
    p.add_argument("--risk-free", type=float, default=0.0, metavar="RATE",
                   help="annual risk-free rate as a decimal, e.g. 0.04 (default: 0)")
    p.add_argument("--market-premium", type=float, default=None, metavar="RATE",
                   help="annual market risk premium for CAPM expected return "
                        "(default: realized premium in the sample)")
    p.add_argument("--periods-per-year", type=int, default=252,
                   help="observations per year, used to annualize (default: 252)")
    p.add_argument("--sort", choices=["beta", "alpha", "r2", "ticker"], default="beta",
                   help="table sort order (default: beta, descending)")
    p.add_argument("--report", type=Path, metavar="PATH", help="write results to this CSV file")
    p.add_argument("--charts-dir", type=Path, metavar="DIR",
                   help="write PNG charts (security market line, regression, rolling beta) here")
    p.add_argument("--chart-ticker", metavar="TICKER",
                   help="ticker for the regression and rolling-beta charts (default: highest beta)")
    p.add_argument("--rolling-window", type=int, default=126, metavar="N",
                   help="trailing window for rolling beta, in observations (default: 126)")
    return p


def _sorted(results: list[CapmResult], key: str) -> list[CapmResult]:
    if key == "ticker":
        return sorted(results, key=lambda r: r.ticker)
    attr = {"beta": "beta", "alpha": "alpha_annual", "r2": "r_squared"}[key]
    return sorted(results, key=lambda r: getattr(r, attr), reverse=True)


def render_table(results: list[CapmResult], benchmark: str) -> Table:
    table = Table(title=f"CAPM regression vs {benchmark}", title_justify="left",
                  header_style="bold cyan")
    table.add_column("Ticker", style="bold")
    for name in ("Beta", "Std err", "t(beta)", "Alpha/yr", "t(alpha)", "R²",
                 "Return/yr", "Vol/yr", "CAPM E[R]"):
        table.add_column(name, justify="right")
    for r in results:
        alpha_style = "green" if r.alpha_annual > 0 else "red"
        table.add_row(
            r.ticker,
            f"{r.beta:.3f}",
            f"{r.beta_se:.3f}",
            f"{r.beta_t:.1f}",
            f"[{alpha_style}]{r.alpha_annual:+.2%}[/]",
            f"{r.alpha_t:+.2f}",
            f"{r.r_squared:.3f}",
            f"{r.annual_return:+.2%}",
            f"{r.annual_volatility:.2%}",
            f"{r.expected_return:.2%}",
        )
    return table


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    console = Console(highlight=False)

    try:
        prices = load_prices(args.csv, date_column=args.date_column,
                             benchmark=args.benchmark, tickers=args.tickers)
        returns = compute_returns(prices, args.returns)
        results, premium = analyze(
            returns, args.benchmark, risk_free=args.risk_free,
            market_premium=args.market_premium, periods_per_year=args.periods_per_year,
            method=args.returns,
        )
    except (DataError, ValueError) as exc:
        console.print(f"[red]error:[/] {exc}")
        return 2

    results = _sorted(results, args.sort)
    start, end = returns.index[0].date(), returns.index[-1].date()
    source = "premium (realized)" if args.market_premium is None else "premium (given)"
    console.print(
        f"[dim]{args.csv.name}[/]  {start} to {end}  "
        f"{len(returns)} {args.returns} returns  "
        f"r_f {args.risk_free:.2%}  {source} {premium:.2%}"
    )
    console.print(render_table(results, args.benchmark))
    console.print("[dim]Alpha and returns annualized arithmetically "
                  f"(daily x {args.periods_per_year}); t-stats use OLS standard errors.[/]")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([r.as_dict() for r in results]).to_csv(args.report, index=False)
        console.print(f"report  {args.report}", soft_wrap=True)

    if args.charts_dir:
        from capm_beta_analyzer import charts  # matplotlib import only when needed

        by_ticker = {r.ticker: r for r in results}
        ticker = args.chart_ticker or max(results, key=lambda r: r.beta).ticker
        if ticker not in by_ticker:
            console.print(f"[red]error:[/] chart ticker {ticker!r} was not analyzed")
            return 2
        try:
            rolling = rolling_beta(returns, ticker, args.benchmark, args.rolling_window)
        except ValueError as exc:
            console.print(f"[red]error:[/] {exc}")
            return 2
        excess = returns - periodic_rate(args.risk_free, args.periods_per_year, args.returns)
        out = args.charts_dir
        written = [
            charts.security_market_line_chart(results, args.risk_free, premium,
                                              out / "security_market_line.png"),
            charts.regression_chart(excess, by_ticker[ticker], args.benchmark,
                                    out / f"regression_{ticker}.png"),
            charts.rolling_beta_chart(rolling, by_ticker[ticker].beta, ticker,
                                      out / f"rolling_beta_{ticker}.png"),
        ]
        for path in written:
            console.print(f"chart   {path}", soft_wrap=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
