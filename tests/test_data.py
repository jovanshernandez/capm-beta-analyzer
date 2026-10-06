import pytest

from capm_beta_analyzer.data import DEFAULT_DATASET, DataError, load_prices


def test_bundled_dataset_loads():
    prices = load_prices()
    assert prices.columns[-1] == "sp500"
    assert {"TSLA", "FB", "T"} <= set(prices.columns)
    assert prices.index.is_monotonic_increasing
    assert len(prices) > 1500


def test_ticker_subset_keeps_benchmark_last():
    prices = load_prices(DEFAULT_DATASET, tickers=["TSLA", "T"])
    assert list(prices.columns) == ["TSLA", "T", "sp500"]


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"benchmark": "SPY"}, "benchmark column"),
        ({"date_column": "when"}, "date column"),
        ({"tickers": ["NOPE"]}, "not in file"),
        ({"tickers": ["sp500"]}, "cannot also be an asset"),
    ],
)
def test_bad_arguments_raise(kwargs, message):
    with pytest.raises(DataError, match=message):
        load_prices(DEFAULT_DATASET, **kwargs)


def test_missing_file_raises(tmp_path):
    with pytest.raises(DataError, match="not found"):
        load_prices(tmp_path / "missing.csv")


def test_non_positive_prices_rejected(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("Date,A,sp500\n2020-01-01,1,10\n2020-01-02,0,11\n2020-01-03,2,12\n")
    with pytest.raises(DataError, match="positive"):
        load_prices(path)


def test_incomplete_rows_dropped(tmp_path):
    path = tmp_path / "gaps.csv"
    path.write_text(
        "Date,A,sp500\n2020-01-03,2,12\n2020-01-01,1,10\n2020-01-02,,11\n"
        "2020-01-06,3,13\n2020-01-07,4,14\n"
    )
    prices = load_prices(path)
    assert len(prices) == 4
    assert prices.index.is_monotonic_increasing
