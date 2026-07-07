import numpy as np

from prediction_stk.backtest import backtest_series


def test_backtest_returns_metrics_per_model():
    t = np.linspace(0, 20 * np.pi, 300)
    series = 100 + np.sin(t) + 0.05 * t
    res = backtest_series(series, forecast_length=10, models=("arima", "linear"),
                          n_splits=3, min_train=60)
    for model in ("arima", "linear", "naive"):
        assert model in res
        assert res[model]["n"] >= 1
        assert res[model]["rmse"] == res[model]["rmse"]  # not NaN
        assert 0.0 <= res[model]["directional_accuracy"] <= 1.0


def test_linear_has_positive_skill_on_clean_trend():
    # A pure upward trend should let the linear model beat the naive baseline.
    series = np.arange(300, dtype=float) * 0.5 + 10
    res = backtest_series(series, forecast_length=10, models=("linear",),
                          n_splits=3, min_train=60)
    assert res["linear"]["skill"] > 0
    assert res["linear"]["directional_accuracy"] == 1.0


def test_short_series_returns_nan_without_crashing():
    res = backtest_series(np.arange(30, dtype=float), forecast_length=20,
                          models=("linear",), min_train=80)
    assert res["linear"]["n"] == 0
