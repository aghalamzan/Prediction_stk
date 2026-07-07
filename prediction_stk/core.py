from .data import StockDataFetcher
from .models import ModelManager
from .analysis import Analyzer
from .agents import AgentManager
from .recommendation import RecommendationEngine

class StockPipeline:
    def __init__(self, config=None):
        self.fetcher = StockDataFetcher(config=config)
        self.model_manager = ModelManager(config=config)
        self.analyzer = Analyzer(config=config)
        self.agent_manager = AgentManager(config=config)
        self.recommender = RecommendationEngine(config=config)

    def run(self):
        raw_data = self.fetcher.fetch_batch()
        processed = self.fetcher.process_raw(raw_data)
        cross_corr = self.analyzer.compute_cross_correlation(processed)
        fitted = self.model_manager.fit_models(processed)
        predictions = self.agent_manager.predict(processed, fitted)
        review = self.analyzer.review_predictions(processed, predictions)
        recommendations = self.recommender.generate(processed, predictions, review)
        return {
            "data": processed,
            "cross_correlation": cross_corr,
            "models": fitted,
            "predictions": predictions,
            "review": review,
            "recommendations": recommendations,
        }
