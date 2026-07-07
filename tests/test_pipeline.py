import pytest
from prediction_stk import StockPipeline


def test_pipeline_runs_without_error(monkeypatch):
    class DummyFetcher:
        def fetch_batch(self):
            import pandas as pd
            idx = pd.date_range("2026-01-01", periods=20, freq="1min")
            return {"AAPL": pd.DataFrame({"open": range(20), "high": range(20), "low": range(20), "close": range(20), "volume": range(20)}, index=idx)}
        def process_raw(self, raw_data):
            return raw_data["AAPL"].rename(columns={"close": "close_AAPL", "volume": "volume_AAPL"})
    pipeline = StockPipeline()
    pipeline.fetcher = DummyFetcher()
    result = pipeline.run()
    assert "data" in result
    assert "predictions" in result
