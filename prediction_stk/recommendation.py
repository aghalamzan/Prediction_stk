from typing import Dict, Any

class RecommendationEngine:
    def __init__(self, config: Dict = None):
        self.config = config or {}

    def generate(self, data: Any, predictions: Dict[str, Any], review: Dict[str, Any]) -> Dict[str, Any]:
        recommendations = {}
        for model_type, cols in predictions.items():
            recommendations[model_type] = {}
            for col, pred in cols.items():
                if pred is None:
                    continue
                mean_pred = float(pred.mean())
                last = float(data[col].iloc[-1]) if col in data.columns else float(data.filter(like=col).iloc[-1, 0])
                if mean_pred > last * 1.01:
                    action = "buy"
                elif mean_pred < last * 0.99:
                    action = "sell"
                else:
                    action = "hold"
                recommendations[model_type][col] = {
                    "action": action,
                    "predicted_mean": mean_pred,
                    "confidence_interval": {
                        "low": float(pred.min()),
                        "high": float(pred.max()),
                    },
                    "review": review.get(col, {}).get(model_type, {}),
                }
        return recommendations
