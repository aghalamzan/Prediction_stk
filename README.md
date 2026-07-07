# Prediction Stk

A Python library to build stock prediction models with cross-correlation between major technology stocks, semiconductor, and energy sectors.

## Features

- Online stock data fetch API
- Per-minute data ingestion with 15-minute modeling granularity
- Cross-correlation analysis for server robotics, Nvidia, Microsoft, Meta, Google, Apple, Amazon
- Classical metrics and traditional stock model checks
- Transformer-style time-series prediction
- Multi-agent prediction and critic system
- Buy/sell recommendation with confidence intervals

## Install

```bash
pip install .
```

## Usage

```python
from prediction_stk import StockPipeline
pipeline = StockPipeline()
pipeline.run()
```

### Horizon profiles (intraday / week / month)

`run_forecast` is a callable component that runs the model ensemble at a chosen
time scale. Each **profile** sets the bar interval, lookback, forecast length,
and calendar. The `week` and `month` profiles use daily bars and a **business-day**
forecast index, so weekends are skipped automatically.

```python
from prediction_stk import run_forecast, PROFILES

# One trading week ahead (5 working days) for SERV, no chart:
results, _ = run_forecast(["SERV"], profile="week", output_dir=None)

# Override a profile's length, e.g. 10 working days:
results, chart = run_forecast(["SERV"], profile="week", steps=10)

print(sorted(PROFILES))  # ['intraday', 'month', 'week']
```

From the CLI:

```bash
prediction_stk forecast --profile week --symbols SERV,NVDA   # 5 working days
prediction_stk forecast --profile month --symbols SERV       # ~21 working days
prediction_stk visualize --symbols SERV                      # intraday (~5h)
# options: --horizon N (override bars) | --no-chart | --cpu | --no-transformer
```

### News-informed forecasts (Phase 1)

Add `--news` to fold news sentiment into the forecast as an exogenous input.
The pipeline fetches company headlines from **Finnhub**, scores each with
**Claude (Haiku)** in `[-1, 1]`, decays them onto the price bars, and feeds the
result to **ARIMAX** and the **linear** model (the transformer stays univariate
for now).

```bash
export FINNHUB_API_KEY=...      # real publish timestamps -> honest backtests
export ANTHROPIC_API_KEY=...    # or use an `ant auth login` profile
prediction_stk forecast --profile week --symbols SERV --news
# --news-half-life N  tunes how fast a headline's influence fades (in bars)
```

Everything degrades gracefully: a missing key, a missing `anthropic` SDK, or a
network error logs a warning and yields neutral (zero) features, so forecasting
never breaks. Features are **strictly causal** — a bar only sees headlines
published at or before it — and Claude scores are cached on disk (keyed by
headline hash) so repeated backtests don't re-pay the API.

```python
from prediction_stk import run_forecast, NewsConfig
results, _ = run_forecast(["SERV"], profile="week", output_dir=None,
                          news_config=NewsConfig(enabled=True, half_life_bars=8))
```

### GPU transformer

The transformer forecaster trains and predicts on CUDA automatically when a GPU
is available, alongside the ARIMA and linear models. Configure it via the
pipeline config:

```python
config = {
    "forecast_length": 20,
    "use_transformer": True,   # set False to skip the torch model
    "prefer_gpu": True,        # set False to force CPU
    "transformer": {"window": 24, "epochs": 200, "d_model": 64},
}
pipeline = StockPipeline(config=config)
```

Series with fewer than `window + forecast_length` points fall back to `None`
gracefully, so the pipeline never fails on short data.

### Daily position-aware recommendation

Fetch live data for one holding, run the model ensemble, backtest it, and print
a recommendation with the dollar risk of each decision:

```bash
prediction_stk daily --symbol SERV --shares 3761 --unrealized-pl -3500
# options: --cost-basis 7.38 | --horizon 20 | --posture balanced|preservation|accumulate
#          --cpu (force CPU) | --no-transformer
```

The report shows the recommended action, model forecasts with backtested
directional accuracy, a risk table (realized P&L / residual VaR95 / expected
P&L for hold, trim, sell, add), and a suggested stop-loss.

**Run it automatically at the US market open (14:30 UK).** Add to your crontab
(`crontab -e`); the `30 14` fires 14:30 local time on weekdays:

```cron
30 14 * * 1-5 cd /path/to/Prediction_stk && /path/to/python -m prediction_stk.cli \
  daily --symbol SERV --shares 3761 --unrealized-pl -3500 >> ~/serv_daily.log 2>&1
```

Make sure your machine's timezone is Europe/London (or adjust the hour), and
remember US market holidays aren't filtered out.

> Not financial advice. Forecasts come from small statistical/transformer models
> on volatile intraday data; treat the output as one structured input, not a
> trading signal.
