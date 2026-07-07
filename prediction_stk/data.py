import asyncio
import datetime
import pandas as pd
import requests
from typing import Dict, List, Optional

DEFAULT_SYMBOLS = ["NVDA", "MSFT", "META", "GOOGL", "AAPL", "AMZN"]
SECTOR_SYMBOLS = {
    "semiconductor": ["NVDA"],
    "energy": ["XOM", "CVX"]
}

class StockDataFetcher:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.symbols = self.config.get("symbols", DEFAULT_SYMBOLS)
        self.sector_symbols = self.config.get("sector_symbols", SECTOR_SYMBOLS)
        self.api_url = self.config.get("api_url", "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}")
        self.interval = self.config.get("interval", "1m")
        self.range = self.config.get("range", "1d")
        self.headers = self.config.get("headers", {"User-Agent": "Mozilla/5.0"})

    def fetch_batch(self) -> Dict[str, pd.DataFrame]:
        results = {}
        for symbol in self.symbols + sum(self.sector_symbols.values(), []):
            if symbol in results:
                continue
            results[symbol] = self.fetch_symbol(symbol)
        return results

    def fetch_symbol(self, symbol: str) -> pd.DataFrame:
        url = self.api_url.format(symbol=symbol)
        params = {"interval": self.interval, "range": self.range}
        resp = requests.get(url, params=params, headers=self.headers, timeout=15)
        resp.raise_for_status()
        data = resp.json().get("chart", {}).get("result", [None])[0]
        if data is None:
            raise ValueError(f"No data returned for {symbol}")
        timestamps = data["timestamp"]
        indicators = data["indicators"]["quote"][0]
        df = pd.DataFrame({
            "open": indicators["open"],
            "high": indicators["high"],
            "low": indicators["low"],
            "close": indicators["close"],
            "volume": indicators["volume"],
        }, index=pd.to_datetime(timestamps, unit="s"))
        df.index.name = "datetime"
        return df

    def process_raw(self, raw_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        frames = []
        for symbol, df in raw_data.items():
            df = df.resample("1min").ffill().tail(300)
            df = df.assign(symbol=symbol)
            df = df[["symbol", "close", "volume"]]
            frames.append(df)
        merged = pd.concat(frames)
        pivot = merged.reset_index().pivot(index="datetime", columns="symbol", values=["close", "volume"])
        pivot.columns = [f"{field}_{symbol}" for field, symbol in pivot.columns]
        return pivot

    def aggregate_15m(self, data: pd.DataFrame) -> pd.DataFrame:
        aggregated = data.resample("15min").agg({col: "last" if col.startswith("close_") else "sum" for col in data.columns})
        return aggregated
