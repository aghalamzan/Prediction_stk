"""Daily market-open advisor for a single holding.

Fetches live intraday + daily data for one symbol, runs the model ensemble
(ARIMA / linear / GPU transformer), backtests it for confidence, and produces a
position-aware recommendation with the dollar risk of each decision.

Intended to be run once at the market open. Output is a decision-support
report, not financial advice.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
import requests

from .data import StockDataFetcher
from .models import ModelManager
from .backtest import backtest_series, forecast_all
from .portfolio import Position, compute_risk, recommend


@dataclass
class DailyResult:
    symbol: str
    current_price: float
    daily_vol: float
    per_model_end: Dict[str, float]
    backtest: Dict[str, Dict[str, float]]
    recommendation: object  # portfolio.Recommendation
    position: Position


class DailyAdvisor:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.forecast_length = self.config.get("forecast_length", 20)
        self.bar_minutes = self.config.get("bar_minutes", 15)
        self.posture = self.config.get("posture", "balanced")
        self.headers = self.config.get("headers", {"User-Agent": "Mozilla/5.0"})
        self.api_url = self.config.get(
            "api_url", "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        )
        # Pull enough intraday history to fit + backtest (Yahoo caps 15m at 60d).
        fetch_config = dict(self.config)
        fetch_config.setdefault("interval", f"{self.bar_minutes}m")
        fetch_config.setdefault("range", "60d")
        self.fetcher = StockDataFetcher(fetch_config)
        # Speed-tuned transformer defaults: large batches keep the once-a-day run
        # fast (few-thousand training windows fit in a handful of GPU steps).
        tconf = dict(self.config.get("transformer", {}))
        tconf.setdefault("epochs", 120)
        tconf.setdefault("batch_size", 512)
        self.config["transformer"] = tconf
        self.model_manager = ModelManager(self.config)

    # -- data --------------------------------------------------------------
    def _intraday_closes(self, symbol: str) -> pd.Series:
        # Data is already fetched at the target bar interval (trading hours only);
        # resampling onto a continuous grid would inject flat overnight fills, so
        # just drop gaps and keep the real bars.
        df = self.fetcher.fetch_symbol(symbol)
        return df["close"].dropna()

    def _daily_history(self, symbol: str):
        url = self.api_url.format(symbol=symbol)
        resp = requests.get(url, params={"interval": "1d", "range": "1mo"},
                            headers=self.headers, timeout=15)
        resp.raise_for_status()
        result = resp.json().get("chart", {}).get("result", [None])[0]
        if result is None:
            raise ValueError(f"No daily data for {symbol}")
        meta = result.get("meta", {})
        closes = [c for c in result["indicators"]["quote"][0]["close"] if c is not None]
        rets = [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes))]
        daily_vol = statistics.pstdev(rets) if len(rets) > 1 else 0.0
        current = meta.get("regularMarketPrice") or (closes[-1] if closes else None)
        return float(current), float(daily_vol)

    # -- modelling ---------------------------------------------------------
    def _forecast_ends(self, closes: pd.Series) -> Dict[str, float]:
        fmodels = self.config.get("forecast_models", ("arima", "linear", "transformer"))
        if not self.model_manager.use_transformer:
            fmodels = tuple(m for m in fmodels if m != "transformer")
        preds = forecast_all(
            closes.values,
            forecast_length=self.forecast_length,
            models=fmodels,
            transformer_config=self.model_manager.transformer_config,
            device=self.model_manager.device,
        )
        out: Dict[str, float] = {}
        for model_name, arr in preds.items():
            if arr is None:
                continue
            arr = np.asarray(arr, dtype=float)
            if arr.size and np.isfinite(arr[-1]):
                out[model_name] = float(arr[-1])
        return out

    # -- orchestration -----------------------------------------------------
    def run(self, symbol: str, shares: float,
            unrealized_pl: Optional[float] = None,
            cost_basis: Optional[float] = None,
            posture: Optional[str] = None) -> DailyResult:
        posture = posture or self.posture
        current_price, daily_vol = self._daily_history(symbol)
        closes = self._intraday_closes(symbol)

        per_model_end = self._forecast_ends(closes)

        # Backtest the cheap models by default; refitting the transformer per
        # fold is slow, so it is opt-in via config["backtest_models"].
        bt_models = self.config.get("backtest_models", ("arima", "linear"))
        bt = backtest_series(
            closes.values,
            forecast_length=self.forecast_length,
            models=bt_models,
            n_splits=self.config.get("backtest_splits", 4),
            transformer_config=self.model_manager.transformer_config,
            device=self.model_manager.device,
        )

        if cost_basis is not None:
            position = Position(symbol=symbol, shares=shares, cost_basis=cost_basis)
        elif unrealized_pl is not None:
            position = Position.from_unrealized(symbol, shares, current_price, unrealized_pl)
        else:
            position = Position(symbol=symbol, shares=shares, cost_basis=current_price)

        risk = compute_risk(
            position, current_price, daily_vol,
            forecast_ends=list(per_model_end.values()),
            forecast_length=self.forecast_length,
            bar_minutes=self.bar_minutes,
        )

        # Confidence inputs: mean directional accuracy of real models + agreement.
        real = [m for m in per_model_end if m != "naive"]
        accs = [bt[m]["directional_accuracy"] for m in real
                if m in bt and not np.isnan(bt[m]["directional_accuracy"])]
        directional_accuracy = float(np.mean(accs)) if accs else 0.5
        if per_model_end:
            up = sum(1 for e in per_model_end.values() if e > current_price)
            down = len(per_model_end) - up
            model_agreement = max(up, down) / len(per_model_end)
        else:
            model_agreement = 0.0

        rec = recommend(position, current_price, risk, directional_accuracy,
                        model_agreement, posture=posture)

        return DailyResult(
            symbol=symbol, current_price=current_price, daily_vol=daily_vol,
            per_model_end=per_model_end, backtest=bt, recommendation=rec,
            position=position,
        )


def format_report(result: DailyResult) -> str:
    r = result.recommendation
    risk = r.risk
    pos = result.position
    price = result.current_price
    lines = []
    lines.append(f"=== Daily recommendation: {result.symbol} ===")
    lines.append(f"Price ${price:,.2f} | daily vol {result.daily_vol*100:.2f}% | "
                 f"horizon {risk.horizon_days*6.5:.1f}h (~{risk.horizon_days:.2f} trading day)")
    lines.append(f"Position: {pos.shares:,.0f} sh @ avg ${pos.cost_basis:,.2f} = "
                 f"${pos.market_value(price):,.0f} | unrealized ${pos.unrealized_pl(price):,.0f}")
    lines.append("")
    lines.append(">>> RECOMMENDED: " + r.action.upper().replace("_", " "))
    lines.append("    " + r.rationale)
    lines.append(f"    conviction {r.conviction:+.2f}σ | confidence {r.confidence:.0%} | "
                 f"P(further loss) {risk.prob_further_loss:.0%}")
    lines.append("")
    lines.append("Model forecasts (end of horizon):")
    for m, end in result.per_model_end.items():
        acc = result.backtest.get(m, {}).get("directional_accuracy", float("nan"))
        acc_s = f"{acc:.0%}" if acc == acc else "n/a"  # NaN check
        lines.append(f"  {m:<12} ${end:,.2f} ({(end/price-1)*100:+.2f}%)  dir-acc {acc_s}")
    lines.append(f"  ensemble     ${risk.expected_end_price:,.2f} "
                 f"({risk.expected_return*100:+.2f}%)")
    lines.append("")
    lines.append("Risk of each decision:")
    lines.append(f"  {'action':<10}{'realized P&L':>14}{'residual VaR95':>16}{'exp. P&L':>12}")
    for opt in r.options:
        lines.append(f"  {opt.action:<10}{opt.realized_pl:>14,.0f}"
                     f"{opt.residual_var95:>16,.0f}{opt.expected_pl:>12,.0f}")
    lines.append("")
    lines.append(f"Suggested stop-loss: ${risk.suggested_stop:,.2f} "
                 f"(risk to stop ${risk.risk_to_stop:,.0f})")
    lines.append("")
    lines.append("Not financial advice — model estimates on volatile intraday data.")
    return "\n".join(lines)
