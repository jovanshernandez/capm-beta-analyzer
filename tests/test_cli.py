import pandas as pd
from conftest import TRUE_BETA

from capm_beta_analyzer.cli import main


def test_cli_runs_on_bundled_data(capsys):
    assert main(["--risk-free", "0.02"]) == 0
    out = capsys.readouterr().out
    assert "CAPM regression vs sp500" in out
    for ticker in ("TSLA", "NFLX", "T"):
        assert ticker in out


def test_cli_writes_report_and_charts(tmp_path, synthetic_csv):
    report = tmp_path / "out" / "report.csv"
    charts = tmp_path / "charts"
    code = main([
        "--csv", str(synthetic_csv), "--benchmark", "MKT", "--report", str(report),
        "--charts-dir", str(charts), "--chart-ticker", "HIGH", "--rolling-window", "60",
    ])
    assert code == 0
    df = pd.read_csv(report).set_index("ticker")
    assert abs(df.loc["HIGH", "beta"] - TRUE_BETA) < 0.05
    for name in ("security_market_line.png", "regression_HIGH.png", "rolling_beta_HIGH.png"):
        path = charts / name
        assert path.stat().st_size > 10_000
        assert path.read_bytes()[:4] == b"\x89PNG"


def test_cli_reports_bad_input(capsys):
    assert main(["--benchmark", "SPY"]) == 2
    assert "benchmark column" in capsys.readouterr().out
