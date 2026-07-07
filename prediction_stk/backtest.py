"""Walk-forward (expanding-window) holdout evaluation.

Unlike ``Analyzer.review_predictions`` (which compared a forecast against the
tail of the *training* data), this refits each model on an expanding window and
scores its forecast against genuinely unseen future values. Reports error and
directional accuracy per model, plus skill relative to a naive last-value
baseline — the directional accuracy feeds the daily recommendation's confidence.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA

from .transformer import TransformerForecaster


def _fit_predict_arima(train: np.ndarray, horizon: int) -> Optional[np.ndarray]:
    try:
        fit = ARIMA(train, order=(1, 1, 1)).fit()
        return np.asarray(fit.forecast(steps=horizon))
    except Exception:
        return None


def _fit_predict_linear(train: np.ndarray, horizon: int) -> Optional[np.ndarray]:
    try:
        X = np.arange(len(train)).reshape(-1, 1)
        pipe = Pipeline([("scale", StandardScaler()), ("lr", LinearRegression())])
        pipe.fit(X, train)
        future = np.arange(len(train), len(train) + horizon).reshape(-1, 1)
        return pipe.predict(future)
    except Exception:
        return None


def _fit_predict_transformer(train: np.ndarray, horizon: int, config, device) -> Optional[np.ndarray]:
    try:
        fc = TransformerForecaster(forecast_length=horizon, device=device, **(config or {}))
        if fc.fit(train) is None:
            return None
        return fc.forecast()
    except Exception:
        return None


def forecast_all(
    series: Sequence[float],
    forecast_length: int,
    models: Sequence[str] = ("arima", "linear", "transformer"),
    transformer_config: Optional[dict] = None,
    device=None,
) -> Dict[str, Optional[np.ndarray]]:
    """Fit each model on the full series and return its forecast array.

    Shares the exact fit/predict paths used by :func:`backtest_series`, so a
    model's live forecast is generated the same way it was scored.
    """
    y = np.asarray(series, dtype=float).ravel()
    out: Dict[str, Optional[np.ndarray]] = {}
    if "arima" in models:
        out["arima"] = _fit_predict_arima(y, forecast_length)
    if "linear" in models:
        out["linear"] = _fit_predict_linear(y, forecast_length)
    if "transformer" in models:
        out["transformer"] = _fit_predict_transformer(y, forecast_length, transformer_config, device)
    return out


def backtest_series(
    series: Sequence[float],
    forecast_length: int = 20,
    models: Sequence[str] = ("arima", "linear", "transformer"),
    n_splits: int = 4,
    min_train: int = 80,
    transformer_config: Optional[dict] = None,
    device=None,
) -> Dict[str, Dict[str, float]]:
    """Expanding-window backtest. Returns per-model metrics + a ``naive`` baseline.

    Metrics per model: ``mae``, ``rmse``, ``directional_accuracy`` (did the sign of
    the predicted end-of-horizon move match reality), ``skill`` (1 - rmse/naive_rmse),
    and ``n`` (folds scored).
    """
    y = np.asarray(series, dtype=float).ravel()
    n = len(y)
    horizon = forecast_length

    acc: Dict[str, Dict[str, list]] = {
        m: {"abs": [], "sq": [], "dir": []} for m in list(models) + ["naive"]
    }

    last_train_end = n - horizon
    if last_train_end < min_train:
        return {m: {"mae": float("nan"), "rmse": float("nan"),
                    "directional_accuracy": float("nan"), "skill": float("nan"), "n": 0}
                for m in list(models) + ["naive"]}

    split_points = np.linspace(min_train, last_train_end, num=n_splits, dtype=int)
    split_points = sorted(set(int(s) for s in split_points))

    for k in split_points:
        train, actual = y[:k], y[k:k + horizon]
        if len(actual) < horizon:
            continue
        true_move = actual[-1] - train[-1]

        preds = {"naive": np.full(horizon, train[-1], dtype=float)}
        if "arima" in models:
            preds["arima"] = _fit_predict_arima(train, horizon)
        if "linear" in models:
            preds["linear"] = _fit_predict_linear(train, horizon)
        if "transformer" in models:
            preds["transformer"] = _fit_predict_transformer(train, horizon, transformer_config, device)

        for m, pred in preds.items():
            if pred is None or len(pred) < horizon or not np.isfinite(pred).all():
                continue
            err = pred - actual
            acc[m]["abs"].append(np.mean(np.abs(err)))
            acc[m]["sq"].append(np.mean(err ** 2))
            pred_move = pred[-1] - train[-1]
            acc[m]["dir"].append(1.0 if np.sign(pred_move) == np.sign(true_move) else 0.0)

    naive_rmse = float(np.sqrt(np.mean(acc["naive"]["sq"]))) if acc["naive"]["sq"] else float("nan")

    results: Dict[str, Dict[str, float]] = {}
    for m, d in acc.items():
        if not d["sq"]:
            results[m] = {"mae": float("nan"), "rmse": float("nan"),
                          "directional_accuracy": float("nan"), "skill": float("nan"), "n": 0}
            continue
        rmse = float(np.sqrt(np.mean(d["sq"])))
        results[m] = {
            "mae": float(np.mean(d["abs"])),
            "rmse": rmse,
            "directional_accuracy": float(np.mean(d["dir"])) if d["dir"] else float("nan"),
            "skill": (1.0 - rmse / naive_rmse) if naive_rmse and np.isfinite(naive_rmse) else float("nan"),
            "n": len(d["sq"]),
        }
    return results
