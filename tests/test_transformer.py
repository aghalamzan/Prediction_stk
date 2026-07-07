import numpy as np
import pytest
import torch

from prediction_stk import TransformerForecaster, resolve_device


def _sine_series(n=400):
    t = np.linspace(0, 40 * np.pi, n)
    return (np.sin(t) + 0.1 * t).astype(np.float32)


def test_forecaster_trains_and_predicts():
    series = _sine_series()
    fc = TransformerForecaster(forecast_length=20, window=48, epochs=30, seed=0)
    fitted = fc.fit(series)
    assert fitted is not None
    out = fitted.forecast()
    assert out.shape == (20,)
    assert np.isfinite(out).all()


def test_forecaster_returns_none_on_short_series():
    # Fewer points than forecast_length + min_window cannot form a window.
    fc = TransformerForecaster(forecast_length=20, window=48, min_window=4, epochs=5)
    assert fc.fit(np.arange(10, dtype=np.float32)) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA GPU")
def test_forecaster_runs_on_gpu():
    device = resolve_device(prefer_gpu=True)
    assert device.type == "cuda"
    series = _sine_series()
    fc = TransformerForecaster(forecast_length=20, window=48, epochs=30, device=device)
    fitted = fc.fit(series)
    assert fitted is not None
    # Model parameters must actually live on the GPU.
    assert next(fitted.model.parameters()).is_cuda
    out = fitted.forecast()
    assert out.shape == (20,) and np.isfinite(out).all()


def test_forecaster_beats_naive_on_trend():
    # On a smooth trending signal the transformer should track the continuation
    # better than repeating the last observed value.
    series = _sine_series(500)
    train, future = series[:-20], series[-20:]
    fc = TransformerForecaster(forecast_length=20, window=48, epochs=80, seed=0)
    fitted = fc.fit(train)
    assert fitted is not None
    pred = fitted.forecast()
    naive = np.full(20, train[-1], dtype=np.float32)
    model_mse = float(np.mean((pred - future) ** 2))
    naive_mse = float(np.mean((naive - future) ** 2))
    assert model_mse < naive_mse
