import pandas as pd
import pytest

from prediction_stk.horizon import (
    PROFILES,
    forecast_index,
    get_profile,
    history_window,
)


def test_week_profile_forecasts_five_working_days_skipping_weekend():
    # 2026-07-07 is a Tuesday; the next 5 working days skip Sat/Sun 11-12.
    idx = forecast_index(pd.Timestamp("2026-07-07"), get_profile("week"))
    assert list(idx.date.astype(str)) == [
        "2026-07-08", "2026-07-09", "2026-07-10", "2026-07-13", "2026-07-14",
    ]
    assert not any(ts.weekday() >= 5 for ts in idx)


def test_intraday_profile_uses_continuous_minute_bars():
    idx = forecast_index(pd.Timestamp("2026-07-07 18:00"), get_profile("intraday"))
    assert len(idx) == PROFILES["intraday"].steps
    assert (idx[1] - idx[0]) == pd.Timedelta(minutes=15)


def test_unknown_profile_raises():
    with pytest.raises(ValueError):
        get_profile("decade")


def test_business_history_window_returns_tail():
    closes = pd.Series(
        range(200),
        index=pd.bdate_range("2025-01-01", periods=200),
        dtype=float,
    )
    hist = history_window(closes, get_profile("week"))
    assert len(hist) <= len(closes)
    assert hist.iloc[-1] == closes.iloc[-1]
