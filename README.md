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
