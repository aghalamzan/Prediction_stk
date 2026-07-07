"""Horizon profiles for running forecasts at different time scales.

A :class:`HorizonProfile` bundles everything that differs between an intraday
5-hour forecast and a week-long daily forecast: the data interval/lookback, how
many steps to project, and whether the future index follows a continuous
intraday clock or the business-day calendar (Mon–Fri, weekends skipped).

The forecasting models themselves are horizon-agnostic — they just consume a
1-D series and emit ``steps`` values — so switching horizon is entirely a matter
of picking a profile.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class HorizonProfile:
    """One forecasting time scale.

    Attributes:
        name: Identifier, e.g. ``"week"``.
        interval: Yahoo bar interval, e.g. ``"15m"`` or ``"1d"``.
        lookback: Yahoo history range, e.g. ``"5d"`` or ``"6mo"``.
        steps: Forecast length in bars.
        calendar: ``"continuous"`` (intraday clock) or ``"business"``
            (skip weekends, one bar per working day).
        bar_minutes: Bar width in minutes for continuous calendars; ``None``
            for daily/business calendars.
        label: Human-readable horizon description for chart titles.
    """

    name: str
    interval: str
    lookback: str
    steps: int
    calendar: str = "continuous"
    bar_minutes: Optional[int] = None
    label: str = ""


PROFILES = {
    # ~5 trading hours ahead on 15-minute bars.
    "intraday": HorizonProfile(
        name="intraday", interval="15m", lookback="5d", steps=20,
        calendar="continuous", bar_minutes=15, label="next ~5 hours",
    ),
    # One trading week ahead on daily bars (5 working days, weekends skipped).
    "week": HorizonProfile(
        name="week", interval="1d", lookback="6mo", steps=5,
        calendar="business", bar_minutes=None, label="next 5 working days",
    ),
    # ~One trading month ahead on daily bars.
    "month": HorizonProfile(
        name="month", interval="1d", lookback="2y", steps=21,
        calendar="business", bar_minutes=None, label="next ~21 working days",
    ),
}


def get_profile(name: str) -> HorizonProfile:
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(
            f"Unknown horizon profile {name!r}; choose from {sorted(PROFILES)}"
        )


def forecast_index(last_ts: pd.Timestamp, profile: HorizonProfile) -> pd.DatetimeIndex:
    """Build the future timestamp index for a profile's forecast.

    Business calendars use :func:`pandas.bdate_range`, which emits Monday–Friday
    dates and skips weekends automatically.
    """
    last_ts = pd.Timestamp(last_ts)
    if profile.calendar == "business":
        return pd.bdate_range(
            start=last_ts + pd.offsets.BDay(1),
            periods=profile.steps,
        )
    minutes = profile.bar_minutes or 15
    return pd.date_range(
        last_ts + pd.Timedelta(minutes=minutes),
        periods=profile.steps,
        freq=f"{minutes}min",
    )


def history_window(closes: pd.Series, profile: HorizonProfile) -> pd.Series:
    """Slice the recent history to plot as context for the forecast.

    Intraday profiles favour the current session's bars (falling back to the
    last ~32 bars early in a session); business profiles show a fixed tail of
    daily closes scaled to the forecast length.
    """
    if profile.calendar == "business":
        span = max(profile.steps * 8, 30)
        return closes.tail(min(len(closes), span))

    session = closes.index[-1].date()
    day = closes.index.normalize() == pd.Timestamp(session)
    session_closes = closes.loc[day]
    if len(session_closes) >= 8:
        return session_closes
    return closes.tail(min(len(closes), 32))
