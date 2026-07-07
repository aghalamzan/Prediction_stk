import numpy as np
import pandas as pd
from typing import Dict, Any

class AgentManager:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.forecast_length = self.config.get("forecast_length", 20)

    def predict(self, data: pd.DataFrame, models: Dict[str, Any]) -> Dict[str, Any]:
        agent_outputs = {}
        for model_type in ["arima", "linear", "transformer"]:
            agent_outputs[model_type] = {}
            for col, model_set in models.items():
                predictor = model_set.get(model_type)
                if predictor is None:
                    continue
                if model_type == "arima":
                    agent_outputs[model_type][col] = predictor.predict(start=len(data), end=len(data) + self.forecast_length - 1)
                elif model_type == "linear":
                    X_future = np.arange(len(data), len(data) + self.forecast_length).reshape(-1, 1)
                    agent_outputs[model_type][col] = predictor.predict(X_future)
                else:
                    agent_outputs[model_type][col] = predictor.forecast()
        return agent_outputs

    def confidence_intervals(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        intervals = {}
        for model_type, cols in predictions.items():
            intervals[model_type] = {}
            for col, pred in cols.items():
                arr = np.asarray(pred)
                intervals[model_type][col] = {
                    "low": np.percentile(arr, 5),
                    "high": np.percentile(arr, 95),
                    "mean": np.mean(arr),
                    "std": np.std(arr),
                }
        return intervals
