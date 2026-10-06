"""PNG charts rendered with matplotlib's non-interactive Agg backend."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

from capm_beta_analyzer.capm import CapmResult  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"
SERIES = "#2a78d6"
ACCENT = "#eb6834"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_MUTED,
        "axes.titlecolor": INK,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "font.size": 10,
        "legend.frameon": False,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    }
)


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def regression_chart(
    excess: pd.DataFrame, result: CapmResult, benchmark: str, path: Path
) -> Path:
    """Scatter of daily excess returns (asset vs benchmark) with the fitted CAPM line."""
    x = excess[benchmark].to_numpy()
    y = excess[result.ticker].to_numpy()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(x, y, s=9, color=SERIES, alpha=0.35, linewidths=0, label="Daily observations")
    xs = np.linspace(x.min(), x.max(), 2)
    ax.plot(xs, result.alpha_daily + result.beta * xs, color=ACCENT, lw=2, label="OLS fit")
    ax.axhline(0, color=INK_MUTED, lw=0.6)
    ax.axvline(0, color=INK_MUTED, lw=0.6)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel(f"{benchmark} daily excess return")
    ax.set_ylabel(f"{result.ticker} daily excess return")
    ax.set_title(f"{result.ticker} vs {benchmark}: CAPM regression")
    ax.text(
        0.02,
        0.97,
        f"beta  {result.beta:.2f}  (se {result.beta_se:.3f})\n"
        f"alpha {result.alpha_annual:+.1%} / yr  (t {result.alpha_t:.2f})\n"
        f"R²    {result.r_squared:.2f}   n = {result.observations}",
        transform=ax.transAxes,
        va="top",
        family="monospace",
        color=INK,
        bbox={"boxstyle": "round,pad=0.5", "fc": "white", "ec": GRID},
    )
    ax.legend(loc="lower right")
    return _save(fig, path)


def security_market_line_chart(
    results: list[CapmResult], risk_free: float, market_premium: float, path: Path
) -> Path:
    """Realized annual return vs beta for each asset, against the security market line."""
    betas = np.array([r.beta for r in results])
    realized = np.array([r.annual_return for r in results])
    fig, ax = plt.subplots(figsize=(8, 5.5))
    lo, hi = min(0.0, betas.min() - 0.1), betas.max() + 0.2
    xs = np.linspace(lo, hi, 2)
    ax.plot(xs, risk_free + market_premium * xs, color=ACCENT, lw=2, label="Security market line")
    ax.scatter([1.0], [risk_free + market_premium], s=70, marker="D", color=INK_MUTED,
               zorder=3, label="Market (beta = 1)")
    ax.scatter(betas, realized, s=64, color=SERIES, edgecolors=SURFACE, linewidths=2,
               zorder=4, label="Realized return")
    for r in results:
        ax.annotate(r.ticker, (r.beta, r.annual_return), xytext=(7, 4),
                    textcoords="offset points", color=INK, fontsize=10)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Beta vs benchmark")
    ax.set_ylabel("Annualized return (arithmetic)")
    ax.set_title("Security market line")
    ax.text(0.99, 0.02,
            f"r_f = {risk_free:.2%}   market premium = {market_premium:.2%}\n"
            "points above the line earned positive alpha",
            transform=ax.transAxes, ha="right", va="bottom", color=INK_MUTED, fontsize=9)
    ax.legend(loc="upper left")
    return _save(fig, path)


def rolling_beta_chart(series: pd.Series, full_beta: float, ticker: str, path: Path) -> Path:
    """Trailing-window beta over time with the full-sample beta for reference."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(series.index, series.to_numpy(), color=SERIES, lw=1.5, label=series.name)
    ax.axhline(full_beta, color=ACCENT, lw=1.5, ls="--", label=f"Full-sample beta {full_beta:.2f}")
    ax.axhline(1.0, color=INK_MUTED, lw=1, ls=":", label="Beta = 1")
    ax.set_ylabel("Beta")
    ax.set_title(f"{ticker} rolling beta")
    ax.legend(loc="upper left", ncols=3)
    ax.margins(x=0.01)
    lo = min(series.min(), 1.0)
    hi = max(series.max(), full_beta, 1.0)
    ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.2 * (hi - lo))  # headroom for the legend
    return _save(fig, path)
