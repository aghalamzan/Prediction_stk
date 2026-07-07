"""PyTorch time-series transformer forecaster.

Implements the "Transformer-style time-series prediction" advertised in the
README. The model is a standard Transformer encoder that consumes a window of
past (normalized) closes and directly regresses the next ``forecast_length``
values. Training and inference run on CUDA when available (e.g. the RTX A6000),
falling back to CPU otherwise.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn


def resolve_device(prefer_gpu: bool = True) -> torch.device:
    """Return the CUDA device when available and requested, else CPU."""
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class PositionalEncoding(nn.Module):
    """Classic sinusoidal positional encoding (batch_first)."""

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TimeSeriesTransformer(nn.Module):
    """Encoder-only transformer that maps a window to a multi-step forecast."""

    def __init__(
        self,
        forecast_length: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.forecast_length = forecast_length
        self.input_proj = nn.Linear(1, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, forecast_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, 1) -> (batch, forecast_length)
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.encoder(h)
        last = h[:, -1, :]
        return self.head(last)


class TransformerForecaster:
    """Fit/predict wrapper handling normalization, windowing, and training.

    ``fit`` returns ``self`` on success or ``None`` when the series is too short
    to form a single training window, mirroring the graceful degradation used by
    the ARIMA/linear models so the pipeline never crashes on sparse data.
    """

    def __init__(
        self,
        forecast_length: int = 20,
        window: int = 24,
        min_window: int = 4,
        epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 128,
        prefer_gpu: bool = True,
        device: Optional[torch.device] = None,
        seed: int = 0,
        model_kwargs: Optional[Dict] = None,
    ):
        self.forecast_length = forecast_length
        self.window = window
        self.min_window = min_window
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device or resolve_device(prefer_gpu)
        self.seed = seed
        self.model_kwargs = model_kwargs or {}

        self.model: Optional[TimeSeriesTransformer] = None
        self.mean_: float = 0.0
        self.std_: float = 1.0
        self._last_window: Optional[np.ndarray] = None
        self._fit_window: int = window

    def _make_windows(self, series: np.ndarray, window: int):
        n = len(series)
        n_samples = n - window - self.forecast_length + 1
        X = np.empty((n_samples, window, 1), dtype=np.float32)
        y = np.empty((n_samples, self.forecast_length), dtype=np.float32)
        for i in range(n_samples):
            X[i, :, 0] = series[i : i + window]
            y[i] = series[i + window : i + window + self.forecast_length]
        return X, y

    def fit(self, series) -> Optional["TransformerForecaster"]:
        series = np.asarray(series, dtype=np.float32).ravel()
        n = len(series)

        # Adapt the window down for short series; bail if we cannot form one
        # complete (window -> forecast_length) training pair.
        window = min(self.window, n - self.forecast_length)
        if window < self.min_window:
            return None

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.mean_ = float(series.mean())
        self.std_ = float(series.std()) or 1.0
        norm = (series - self.mean_) / self.std_

        X_np, y_np = self._make_windows(norm, window)
        if len(X_np) == 0:
            return None

        X = torch.from_numpy(X_np).to(self.device)
        y = torch.from_numpy(y_np).to(self.device)

        model = TimeSeriesTransformer(
            forecast_length=self.forecast_length, **self.model_kwargs
        ).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        model.train()
        n_samples = len(X)
        for _ in range(self.epochs):
            perm = torch.randperm(n_samples, device=self.device)
            for start in range(0, n_samples, self.batch_size):
                idx = perm[start : start + self.batch_size]
                optimizer.zero_grad()
                pred = model(X[idx])
                loss = loss_fn(pred, y[idx])
                loss.backward()
                optimizer.step()

        model.eval()
        self.model = model
        self._fit_window = window
        self._last_window = norm[-window:].copy()
        return self

    @torch.no_grad()
    def forecast(self, n: Optional[int] = None) -> np.ndarray:
        """Return the next ``forecast_length`` predicted values (denormalized)."""
        if self.model is None or self._last_window is None:
            raise RuntimeError("TransformerForecaster must be fit before forecasting")
        x = torch.from_numpy(
            self._last_window.reshape(1, self._fit_window, 1).astype(np.float32)
        ).to(self.device)
        out = self.model(x).cpu().numpy().ravel()
        denorm = out * self.std_ + self.mean_
        if n is not None:
            denorm = denorm[:n]
        return denorm

    # Convenience alias so downstream code can treat it like other predictors.
    def predict(self, *args, **kwargs) -> np.ndarray:
        return self.forecast()
