import numpy as np
import pandas as pd
import pytest

from prediction_stk.backtest import forecast_all
from prediction_stk.news import (
    Headline,
    NewsConfig,
    build_news_features,
    news_exog_for_symbol,
    project_news_features,
    assert_no_lookahead,
)


def _index(n=60, freq="15min"):
    return pd.date_range("2026-07-01", periods=n, freq=freq)


def _headline(ts, score):
    return Headline(published=pd.Timestamp(ts), text=f"news {score}", score=score,
                    confidence=0.9)


def test_features_are_causal_no_future_leak():
    idx = _index()
    # One headline in the middle, one after the last bar.
    headlines = [
        _headline(idx[20], 0.8),
        _headline(idx[-1] + pd.Timedelta(days=2), -0.9),  # future — must not leak
    ]
    feats = build_news_features(headlines, idx)
    # Bars before the mid headline see nothing; the future headline never shows.
    assert feats["sentiment"].iloc[:20].abs().sum() == 0.0
    assert feats["sentiment"].iloc[21] > 0
    assert (feats["sentiment"] < 0).sum() == 0  # the -0.9 future item never appears
    assert_no_lookahead(headlines, idx)  # explicit guard passes


def test_sentiment_decays_after_headline():
    idx = _index()
    feats = build_news_features([_headline(idx[10], 1.0)], idx, half_life_bars=5.0)
    at_event = feats["sentiment"].iloc[10]
    later = feats["sentiment"].iloc[10 + 5]  # one half-life later
    assert at_event == pytest.approx(1.0, abs=1e-6)
    assert later == pytest.approx(0.5, abs=1e-2)


def test_empty_headlines_give_neutral_features():
    idx = _index()
    feats = build_news_features([], idx)
    assert feats.to_numpy().sum() == 0.0


def test_projection_decays_toward_zero():
    idx = _index()
    feats = build_news_features([_headline(idx[-1], 1.0)], idx, half_life_bars=4.0)
    proj = project_news_features(feats, horizon=8, half_life_bars=4.0)
    assert proj.shape == (8, 3)
    assert proj[0, 0] > proj[-1, 0]  # decays over the horizon
    assert proj[3, 0] == pytest.approx(feats["sentiment"].iloc[-1] * 0.5, abs=1e-2)


def test_arimax_accepts_exog_and_changes_forecast():
    rng = np.random.default_rng(0)
    y = np.cumsum(rng.normal(0, 1, 120)) + 100
    exog = rng.normal(0, 1, (120, 3))
    exog_future = np.zeros((10, 3))
    with_exog = forecast_all(y, 10, models=("arima", "linear"),
                             exog=exog, exog_future=exog_future)
    plain = forecast_all(y, 10, models=("arima", "linear"))
    for m in ("arima", "linear"):
        assert with_exog[m] is not None and len(with_exog[m]) == 10
        assert plain[m] is not None


def test_disabled_news_returns_none():
    idx = _index()
    assert news_exog_for_symbol("SERV", idx, NewsConfig(enabled=False)) is None
