import math

from prediction_stk.portfolio import (
    Position, compute_risk, build_options, recommend, normal_cdf,
)


def test_position_from_unrealized_recovers_cost_basis():
    pos = Position.from_unrealized("SERV", shares=3761, current_price=6.42,
                                   unrealized_pl=-3500)
    assert math.isclose(pos.cost_basis, 6.42 + 3500 / 3761, rel_tol=1e-9)
    assert math.isclose(pos.unrealized_pl(6.42), -3500, rel_tol=1e-6)


def test_normal_cdf_symmetry():
    assert math.isclose(normal_cdf(0.0), 0.5, abs_tol=1e-12)
    assert math.isclose(normal_cdf(1.0) + normal_cdf(-1.0), 1.0, abs_tol=1e-12)


def _risk(expected_end):
    pos = Position("SERV", 3761, cost_basis=7.35)
    return pos, compute_risk(pos, current_price=6.42, daily_vol=0.0475,
                             forecast_ends=[expected_end], forecast_length=20)


def test_bearish_forecast_raises_loss_probability():
    _, up = _risk(6.80)     # bullish
    _, down = _risk(6.00)   # bearish
    assert down.prob_further_loss > up.prob_further_loss
    assert down.expected_pl < 0 < up.expected_pl


def test_options_cover_all_actions_with_sell_zeroing_risk():
    pos, risk = _risk(6.30)
    opts = {o.action: o for o in build_options(pos, 6.42, risk)}
    assert set(opts) == {"hold", "trim_50", "sell_all", "add"}
    assert opts["sell_all"].residual_var95 == 0.0
    assert opts["trim_50"].residual_var95 == risk.var95 * 0.5


def test_balanced_holds_on_weak_signal():
    pos, risk = _risk(6.44)  # tiny move
    rec = recommend(pos, 6.42, risk, directional_accuracy=0.5,
                    model_agreement=0.5, posture="balanced")
    assert rec.action == "hold"


def test_balanced_sells_on_strong_bearish_high_confidence():
    pos, risk = _risk(5.60)  # ~ -13% over horizon, strongly bearish
    rec = recommend(pos, 6.42, risk, directional_accuracy=0.9,
                    model_agreement=1.0, posture="balanced")
    assert rec.action in {"sell_all", "trim_50"}
