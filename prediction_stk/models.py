import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from typing import Dict, Any

from .transformer import TransformerForecaster, resolve_device

class ModelManager:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.forecast_length = self.config.get("forecast_length", 20)
        self.use_transformer = self.config.get("use_transformer", True)
        self.transformer_config = self.config.get("transformer", {})
        self.prefer_gpu = self.config.get("prefer_gpu", True)
        self.device = resolve_device(self.prefer_gpu)

    def fit_models(self, data: pd.DataFrame) -> Dict[str, Any]:
        models = {}
        aggregated = data.resample("15min").last().ffill()
        for col in aggregated.columns:
            series = aggregated[col].dropna()
            if series.empty:
                continue
            arima_fit = self._fit_arima(series)
            lin_fit = self._fit_linear_regression(series)
            model_set = {
                "arima": arima_fit,
                "linear": lin_fit,
            }
            if self.use_transformer:
                model_set["transformer"] = self._fit_transformer(series)
            models[col] = model_set
        return models

    def _fit_transformer(self, series: pd.Series):
        try:
            forecaster = TransformerForecaster(
                forecast_length=self.forecast_length,
                device=self.device,
                **self.transformer_config,
            )
            return forecaster.fit(series.values)
        except Exception:
            return None

    def _fit_arima(self, series: pd.Series):
        try:
            model = ARIMA(series, order=(1, 1, 1))
            fit_result = model.fit()
            return fit_result
        except Exception:
            return None

    def _fit_linear_regression(self, series: pd.Series):
        try:
            X = np.arange(len(series)).reshape(-1, 1)
            y = series.values
            pipeline = Pipeline([
                ("scale", StandardScaler()),
                ("lr", LinearRegression())
            ])
            pipeline.fit(X, y)
            return pipeline
        except Exception:
            return None

    def predict(self, data: pd.DataFrame, models: Dict[str, Any]) -> Dict[str, Any]:
        predictions = {}
        aggregated = data.resample("15min").last().ffill()
        for col, model_set in models.items():
            predictions[col] = {
                "arima": self._predict_arima(aggregated[col], model_set.get("arima")),
                "linear": self._predict_linear(aggregated[col], model_set.get("linear")),
            }
            if model_set.get("transformer") is not None:
                predictions[col]["transformer"] = model_set["transformer"].forecast()
        return predictions

    def _predict_arima(self, series: pd.Series, fit_result):
        if fit_result is None:
            return None
        start = len(series)
        end = start + self.forecast_length - 1
        forecast = fit_result.predict(start=start, end=end)
        return forecast

    def _predict_linear(self, series: pd.Series, pipeline):
        if pipeline is None:
            return None
        start = len(series)
        X_future = np.arange(start, start + self.forecast_length).reshape(-1, 1)
        return pipeline.predict(X_future)
