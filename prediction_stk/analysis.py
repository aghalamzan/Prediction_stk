import numpy as np
import pandas as pd
from scipy.stats import pearsonr

class Analyzer:
    def __init__(self, config=None):
        self.config = config or {}

    def compute_cross_correlation(self, data: pd.DataFrame) -> pd.DataFrame:
        closes = data.filter(regex="^close_")
        corr = closes.corr()
        return corr

    def review_predictions(self, data: pd.DataFrame, predictions: dict) -> dict:
        review = {}
        aggregated = data.resample("15min").last().ffill()
        for model_name, cols in predictions.items():
            for col, pred in cols.items():
                if pred is None or col not in aggregated.columns:
                    continue
                truth = aggregated[col].dropna()
                if truth.empty:
                    continue
                aligned = np.asarray(pred)
                mse = np.mean((aligned - truth.iloc[-len(aligned):].values) ** 2) if len(truth) >= len(aligned) else None
                review.setdefault(col, {})[model_name] = {
                    "mse": mse,
                    "last_value": truth.iloc[-1],
                    "pred_mean": np.mean(pred) if len(pred) else None,
                    "std": np.std(pred) if len(pred) else None,
                }
        return review

    def compute_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        closes = data.filter(regex="^close_")
        indicators = pd.DataFrame(index=closes.index)
        for col in closes.columns:
            indicators[f"{col}_ma_5"] = closes[col].rolling(5).mean()
            indicators[f"{col}_ma_15"] = closes[col].rolling(15).mean()
            indicators[f"{col}_rsi_14"] = self._rsi(closes[col], 14)
        return indicators

    def _rsi(self, series: pd.Series, window: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(window).mean()
        loss = -delta.clip(upper=0).rolling(window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
