"""Generate forecast charts across horizons (intraday, weekly, monthly)."""
from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .backtest import backtest_series, forecast_all
from .data import DEFAULT_SYMBOLS, SECTOR_SYMBOLS, StockDataFetcher
from .horizon import HorizonProfile, forecast_index, get_profile, history_window
from .models import ModelManager
from .news import NewsConfig, news_exog_for_symbol, project_news_features

MODEL_COLORS = {
    "arima": "#e45756",
    "linear": "#4c78a8",
    "transformer": "#54a24b",
}


def default_symbols() -> List[str]:
    sector = sum(SECTOR_SYMBOLS.values(), [])
    seen: set[str] = set()
    out: List[str] = []
    for sym in list(DEFAULT_SYMBOLS) + sector:
        if sym not in seen:
            seen.add(sym)
            out.append(sym)
    return out


def _break_time_gaps(series: pd.Series, gap_factor: float = 2.0) -> Tuple[list, list]:
    """Return x/y for plotting with NaN inserted across large time gaps.

    Prevents matplotlib from drawing a straight diagonal across breaks such as
    the overnight gap between two trading sessions.
    """
    idx = series.index
    if len(idx) < 3:
        return list(idx), list(series.values)

    diffs = idx.to_series().diff().dropna()
    threshold = diffs.median() * gap_factor
    xs: list = [idx[0]]
    ys: list = [series.iloc[0]]
    for i in range(1, len(idx)):
        if idx[i] - idx[i - 1] > threshold:
            xs.append(idx[i - 1] + (idx[i] - idx[i - 1]) / 2)
            ys.append(np.nan)
        xs.append(idx[i])
        ys.append(series.iloc[i])
    return xs, ys


def compute_confidence_bands(
    predictions: Dict[str, np.ndarray],
    backtest_rmse: Optional[float] = None,
    percentile: float = 10.0,
    z: float = 1.645,
) -> Dict[str, np.ndarray]:
    """Build per-step bands from model spread, widened by backtest residual error."""
    stack = np.stack(list(predictions.values()))
    mean = stack.mean(axis=0)
    low = np.percentile(stack, percentile, axis=0)
    high = np.percentile(stack, 100.0 - percentile, axis=0)

    if backtest_rmse is not None and np.isfinite(backtest_rmse) and backtest_rmse > 0:
        horizon = len(mean)
        scale = np.sqrt(np.arange(1, horizon + 1) / horizon)
        margin = z * backtest_rmse * scale
        low = np.minimum(low, mean - margin)
        high = np.maximum(high, mean + margin)

    return {"mean": mean, "low": low, "high": high}


def _backtest_rmse(closes: np.ndarray, forecast_length: int, models: Sequence[str]) -> Optional[float]:
    bt = backtest_series(
        closes,
        forecast_length=forecast_length,
        models=tuple(m for m in models if m != "transformer"),
        n_splits=3,
    )
    rmses = [bt[m]["rmse"] for m in bt if m != "naive" and np.isfinite(bt[m]["rmse"])]
    return float(np.mean(rmses)) if rmses else None


def forecast_symbol(
    symbol: str,
    fetcher: StockDataFetcher,
    model_manager: ModelManager,
    profile: HorizonProfile,
    models: Sequence[str],
    news_config: Optional[NewsConfig] = None,
) -> Optional[Dict]:
    forecast_length = profile.steps
    closes = fetcher.fetch_symbol(symbol)["close"].dropna()
    if len(closes) < max(24, forecast_length * 2):
        return None

    session = closes.index[-1].date()
    hist = history_window(closes, profile)

    exog = exog_future = None
    if news_config is not None and news_config.enabled:
        feats = news_exog_for_symbol(symbol, closes.index, news_config)
        if feats is not None and feats.to_numpy().any():
            exog = feats.to_numpy(dtype=float)
            exog_future = project_news_features(
                feats, forecast_length, news_config.half_life_bars
            )

    preds = forecast_all(
        closes.values,
        forecast_length=forecast_length,
        models=models,
        transformer_config=model_manager.transformer_config,
        device=model_manager.device,
        exog=exog,
        exog_future=exog_future,
    )
    valid = {k: np.asarray(v, dtype=float) for k, v in preds.items() if v is not None}
    if not valid:
        return None

    forecast_idx = forecast_index(hist.index[-1], profile)
    bands = compute_confidence_bands(
        valid,
        backtest_rmse=_backtest_rmse(closes.values, forecast_length, models),
    )
    return {
        "symbol": symbol,
        "session": session,
        "profile": profile,
        "current_price": float(hist.iloc[-1]),
        "history": hist,
        "forecast_index": forecast_idx,
        "predictions": valid,
        "bands": bands,
        "ensemble_end": float(bands["mean"][-1]),
        "band_low_end": float(bands["low"][-1]),
        "band_high_end": float(bands["high"][-1]),
    }


def plot_predictions(
    results: Sequence[Dict],
    output_path: Path,
    title_date: Optional[date] = None,
) -> Path:
    n = len(results)
    if n == 0:
        raise ValueError("No forecast results to plot")

    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.6 * rows), squeeze=False)
    session = title_date or results[0]["session"]
    profile = results[0].get("profile")
    horizon_label = f" ({profile.label})" if profile and profile.label else ""
    fig.suptitle(
        f"Forecasts{horizon_label} — from {session.isoformat()}",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    for idx, result in enumerate(results):
        ax = axes[idx // cols][idx % cols]
        history = result["history"]
        forecast_idx = result["forecast_index"]
        current = result["current_price"]

        hist_x, hist_y = _break_time_gaps(history)
        ax.plot(hist_x, hist_y, color="#333333", linewidth=2, label="actual")
        ax.axhline(current, color="#999999", linestyle=":", linewidth=1)

        bands = result["bands"]
        anchor_x = [history.index[-1], forecast_idx[0]]
        anchor_y = [history.iloc[-1], bands["mean"][0]]
        ax.fill_between(
            forecast_idx,
            bands["low"],
            bands["high"],
            color="#6b6ecf",
            alpha=0.22,
            label="80% confidence",
        )
        ax.plot(
            anchor_x + list(forecast_idx),
            anchor_y + list(bands["mean"]),
            color="#6b6ecf",
            linewidth=2.2,
            linestyle="--",
            label=f"ensemble → ${bands['mean'][-1]:,.2f}",
        )

        for model_name, path in result["predictions"].items():
            color = MODEL_COLORS.get(model_name, None)
            joined_x = [history.index[-1], forecast_idx[0]]
            joined_y = [history.iloc[-1], path[0]]
            ax.plot(
                joined_x + list(forecast_idx),
                joined_y + list(path),
                color=color,
                linewidth=1.8,
                label=f"{model_name} → ${path[-1]:,.2f}",
            )

        ens = result["ensemble_end"]
        pct = (ens / current - 1) * 100
        band_lo = result["band_low_end"]
        band_hi = result["band_high_end"]
        ax.set_title(
            f"{result['symbol']}  ${current:,.2f}  →  ${ens:,.2f} ({pct:+.2f}%)\n"
            f"80% band ${band_lo:,.2f} – ${band_hi:,.2f}"
        )
        ax.tick_params(axis="x", rotation=25, labelsize=8)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, loc="upper left")

    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def run_forecast(
    symbols: Optional[Sequence[str]] = None,
    profile: str | HorizonProfile = "intraday",
    output_dir: Optional[str | Path] = "outputs",
    steps: Optional[int] = None,
    prefer_gpu: bool = True,
    use_transformer: bool = True,
    news_config: Optional[NewsConfig] = None,
) -> Tuple[List[Dict], Optional[Path]]:
    """Run the model ensemble for a horizon profile and optionally chart it.

    Args:
        symbols: Tickers to forecast; defaults to the mega-cap + sector set.
        profile: A :class:`HorizonProfile` or its name (``"intraday"``,
            ``"week"``, ``"month"``) selecting bar interval, lookback, calendar.
        output_dir: Directory for the PNG chart; pass ``None`` to skip plotting.
        steps: Override the profile's forecast length (bars).
        prefer_gpu / use_transformer: Model toggles.

    Returns:
        ``(results, chart_path)`` where ``chart_path`` is ``None`` when
        ``output_dir`` is ``None``.
    """
    profile = get_profile(profile) if isinstance(profile, str) else profile
    if steps is not None:
        profile = replace(profile, steps=steps)
    symbols = list(symbols or default_symbols())
    config = {
        "interval": profile.interval,
        "range": profile.lookback,
        "forecast_length": profile.steps,
        "prefer_gpu": prefer_gpu,
        "use_transformer": use_transformer,
        "transformer": {"epochs": 80, "batch_size": 512},
    }
    fetcher = StockDataFetcher(config)
    model_manager = ModelManager(config)
    models = ("arima", "linear", "transformer") if use_transformer else ("arima", "linear")

    results: List[Dict] = []
    for symbol in symbols:
        item = forecast_symbol(
            symbol, fetcher, model_manager, profile, models, news_config
        )
        if item is not None:
            results.append(item)

    if not results:
        raise RuntimeError("No symbols produced forecasts — check market data availability.")

    chart_path: Optional[Path] = None
    if output_dir is not None:
        session = results[0]["session"]
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        chart_path = Path(output_dir) / f"predictions_{profile.name}_{session.isoformat()}_{stamp}.png"
        plot_predictions(results, chart_path, title_date=session)
    return results, chart_path


# Backwards-compatible alias for the original intraday entry point.
def run_visualization(
    symbols: Optional[Sequence[str]] = None,
    output_dir: str | Path = "outputs",
    forecast_length: int = 20,
    prefer_gpu: bool = True,
    use_transformer: bool = True,
) -> Tuple[List[Dict], Path]:
    results, chart_path = run_forecast(
        symbols=symbols,
        profile="intraday",
        output_dir=output_dir,
        steps=forecast_length,
        prefer_gpu=prefer_gpu,
        use_transformer=use_transformer,
    )
    return results, chart_path


def format_summary(results: Sequence[Dict]) -> str:
    lines = [f"=== Forecast summary ({results[0]['session'].isoformat()}) ===", ""]
    lines.append(f"{'symbol':<8}{'current':>12}{'ensemble':>12}{'80% band':>24}{'change':>10}")
    for r in sorted(results, key=lambda x: x["symbol"]):
        cur = r["current_price"]
        ens = r["ensemble_end"]
        band = f"${r['band_low_end']:,.2f}–${r['band_high_end']:,.2f}"
        lines.append(
            f"{r['symbol']:<8}{f'${cur:,.2f}':>12}{f'${ens:,.2f}':>12}{band:>24}"
            f"{(ens / cur - 1) * 100:>9.2f}%"
        )
    lines.append("")
    lines.append("Per-model end-of-horizon prices:")
    for r in sorted(results, key=lambda x: x["symbol"]):
        parts = [f"{m}: ${p[-1]:,.2f}" for m, p in r["predictions"].items()]
        lines.append(f"  {r['symbol']}: " + " | ".join(parts))
    return "\n".join(lines)
