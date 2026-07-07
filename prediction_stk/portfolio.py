"""Position-aware risk and decision logic.

Turns a model forecast plus a real holding into: (1) risk metrics over the
forecast horizon, (2) the dollar risk/impact of each candidate decision, and
(3) a single recommended action under a configurable risk posture.

Nothing here is financial advice. The numbers are model- and volatility-derived
estimates meant to make the trade-offs explicit, not to be acted on blindly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

TRADING_MINUTES_PER_DAY = 390.0
Z95 = 1.6448536269514722  # one-sided 95% normal quantile


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass
class Position:
    symbol: str
    shares: float
    cost_basis: float  # average cost per share

    @classmethod
    def from_unrealized(cls, symbol: str, shares: float, current_price: float,
                        unrealized_pl: float) -> "Position":
        """Build a position from shares + current unrealized P&L (loss negative)."""
        cost = current_price - unrealized_pl / shares
        return cls(symbol=symbol, shares=shares, cost_basis=cost)

    def market_value(self, price: float) -> float:
        return self.shares * price

    def unrealized_pl(self, price: float) -> float:
        return self.shares * (price - self.cost_basis)


@dataclass
class RiskMetrics:
    horizon_days: float
    horizon_sigma: float          # fractional stdev of return over horizon
    expected_return: float        # model-implied fractional return over horizon
    expected_end_price: float
    model_dispersion: float       # stdev across model end-prices / price
    prob_further_loss: float      # P(return < 0) over horizon
    var95: float                  # 95% dollar loss over horizon on the position
    expected_pl: float            # model expected dollar P&L over horizon
    suggested_stop: float         # suggested stop-loss price level
    risk_to_stop: float           # dollars at risk from current price to stop


@dataclass
class DecisionOption:
    action: str
    description: str
    realized_pl: float            # dollars locked in now (0 if none)
    residual_var95: float         # 95% dollar loss still exposed after the action
    expected_pl: float            # model expected dollar P&L on retained exposure
    note: str = ""


@dataclass
class Recommendation:
    action: str
    rationale: str
    conviction: float             # signed z-score of expected move vs horizon sigma
    confidence: float             # 0..1, from model agreement + backtest accuracy
    risk: RiskMetrics
    options: List[DecisionOption]


def _horizon_days(forecast_length: int, bar_minutes: int) -> float:
    return (forecast_length * bar_minutes) / TRADING_MINUTES_PER_DAY


def compute_risk(
    position: Position,
    current_price: float,
    daily_vol: float,
    forecast_ends: Sequence[float],
    forecast_length: int,
    bar_minutes: int = 15,
    stop_k: float = 1.5,
) -> RiskMetrics:
    """Aggregate per-model horizon-end forecasts into position risk metrics.

    ``forecast_ends`` is the list of each model's predicted price at the end of
    the horizon (e.g. arima/linear/transformer). ``daily_vol`` is the fractional
    daily return stdev (from recent daily closes).
    """
    ends = [float(e) for e in forecast_ends if e is not None and math.isfinite(e)]
    hd = _horizon_days(forecast_length, bar_minutes)
    horizon_sigma = daily_vol * math.sqrt(max(hd, 1e-9))

    if ends:
        mean_end = sum(ends) / len(ends)
        dispersion = (
            math.sqrt(sum((e - mean_end) ** 2 for e in ends) / len(ends)) / current_price
            if len(ends) > 1 else 0.0
        )
    else:
        mean_end = current_price
        dispersion = 0.0

    expected_return = mean_end / current_price - 1.0
    # Total uncertainty blends market vol with cross-model disagreement.
    total_sigma = math.sqrt(horizon_sigma ** 2 + dispersion ** 2) or 1e-9

    prob_further_loss = normal_cdf((0.0 - expected_return) / total_sigma)
    position_value = position.market_value(current_price)
    var95 = max(0.0, (Z95 * total_sigma - expected_return)) * position_value
    expected_pl = position.shares * (mean_end - current_price)

    suggested_stop = current_price * (1.0 - stop_k * horizon_sigma)
    risk_to_stop = position.shares * (current_price - suggested_stop)

    return RiskMetrics(
        horizon_days=hd,
        horizon_sigma=horizon_sigma,
        expected_return=expected_return,
        expected_end_price=mean_end,
        model_dispersion=dispersion,
        prob_further_loss=prob_further_loss,
        var95=var95,
        expected_pl=expected_pl,
        suggested_stop=suggested_stop,
        risk_to_stop=risk_to_stop,
    )


def build_options(position: Position, current_price: float, risk: RiskMetrics,
                  add_fraction: float = 0.1) -> List[DecisionOption]:
    """Dollar risk/impact for each candidate decision (what the user asked for)."""
    value = position.market_value(current_price)
    unreal = position.unrealized_pl(current_price)
    var_per_value = risk.var95 / value if value else 0.0

    options = [
        DecisionOption(
            action="hold",
            description="Keep the full position",
            realized_pl=0.0,
            residual_var95=risk.var95,
            expected_pl=risk.expected_pl,
            note=f"{risk.prob_further_loss*100:.0f}% modelled chance of further loss over the horizon",
        ),
        DecisionOption(
            action="trim_50",
            description="Sell half now",
            realized_pl=unreal * 0.5,
            residual_var95=risk.var95 * 0.5,
            expected_pl=risk.expected_pl * 0.5,
            note="Halves both downside and upside; locks in half the current P&L",
        ),
        DecisionOption(
            action="sell_all",
            description="Close the position",
            realized_pl=unreal,
            residual_var95=0.0,
            expected_pl=0.0,
            note="Removes all future risk; realizes the current loss/gain in full",
        ),
        DecisionOption(
            action="add",
            description=f"Add {add_fraction*100:.0f}% more shares",
            realized_pl=0.0,
            residual_var95=risk.var95 * (1.0 + add_fraction),
            expected_pl=risk.expected_pl * (1.0 + add_fraction),
            note=f"Raises exposure and 95% VaR to ~${risk.var95*(1+add_fraction):,.0f}; lowers average cost",
        ),
    ]
    return options


def recommend(
    position: Position,
    current_price: float,
    risk: RiskMetrics,
    directional_accuracy: float,
    model_agreement: float,
    posture: str = "balanced",
) -> Recommendation:
    """Choose an action given risk + signal quality under a risk posture.

    ``directional_accuracy`` (0..1) comes from the backtest; ``model_agreement``
    (0..1) is the fraction of models pointing the same direction. ``conviction``
    is the expected move measured in horizon-sigma units.
    """
    conviction = risk.expected_return / risk.horizon_sigma if risk.horizon_sigma else 0.0
    confidence = 0.5 * max(0.0, min(1.0, directional_accuracy)) + 0.5 * max(0.0, min(1.0, model_agreement))

    # Posture-specific thresholds.
    presets = {
        "preservation": dict(conf_min=0.50, bear_z=-0.35, strong_bear_z=-0.8, bull_z=1.1, loss_gate=0.55),
        "balanced":     dict(conf_min=0.55, bear_z=-0.60, strong_bear_z=-1.1, bull_z=0.9, loss_gate=0.60),
        "accumulate":   dict(conf_min=0.55, bear_z=-0.90, strong_bear_z=-1.5, bull_z=0.6, loss_gate=0.66),
    }
    p = presets.get(posture, presets["balanced"])

    action = "hold"
    if confidence < p["conf_min"] or abs(conviction) < 0.3:
        rationale = (
            f"Signal too weak/uncertain (conviction {conviction:+.2f}σ, "
            f"confidence {confidence:.0%}). Default to hold."
        )
    elif conviction <= p["bear_z"] and risk.prob_further_loss >= p["loss_gate"]:
        # Clearly bearish AND elevated downside -> reduce risk.
        if conviction <= p["strong_bear_z"]:
            action = "sell_all"
            rationale = (
                f"Model is strongly bearish (conviction {conviction:.2f}σ) with "
                f"{risk.prob_further_loss:.0%} chance of further loss — cut the position."
            )
        else:
            action = "trim_50"
            rationale = (
                f"Model is bearish (conviction {conviction:.2f}σ) with elevated downside "
                f"({risk.prob_further_loss:.0%} loss probability) — reduce exposure."
            )
    elif conviction >= p["bull_z"] and risk.prob_further_loss <= (1 - p["loss_gate"]):
        action = "add" if posture == "accumulate" else "hold"
        rationale = (
            f"Model is constructive (conviction {conviction:.2f}σ, confidence {confidence:.0%}). "
            + ("Averaging down is justified." if action == "add"
               else "Hold; not bearish enough to sell, and balanced posture avoids chasing.")
        )
    else:
        rationale = (
            f"Mixed picture (conviction {conviction:+.2f}σ, {risk.prob_further_loss:.0%} loss "
            f"probability). No clear edge — hold."
        )

    return Recommendation(
        action=action,
        rationale=rationale,
        conviction=conviction,
        confidence=confidence,
        risk=risk,
        options=build_options(position, current_price, risk),
    )
