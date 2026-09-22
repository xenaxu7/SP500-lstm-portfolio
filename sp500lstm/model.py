"""Per-stock LSTM: 60 scaled prices in, scaled price `horizon` days ahead out.

Same architecture as the original script (LSTM(50) -> LSTM(50) -> Dense(25)
-> Dense(1), Adam, MSE). `horizon=1` reproduces the original next-day target;
the walk-forward backtest uses `horizon=21` so the signal matches the
one-month holding period.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def make_windows(prices: np.ndarray, window: int = 60, horizon: int = 21):
    """Return (X, y) where X[k] is `window` consecutive prices and y[k] is the
    price `horizon` steps after the last element of that window."""
    p = np.asarray(prices, dtype="float32")
    n = len(p) - window - horizon + 1
    if n <= 0:
        return np.empty((0, window, 1), "float32"), np.empty((0,), "float32")
    idx = np.arange(window)[None, :] + np.arange(n)[:, None]
    X = p[idx][..., None]
    y = p[window + horizon - 1: window + horizon - 1 + n]
    return X, y


@dataclass
class Scaler:
    """Min-max scaler to [0, 1] fitted on the training prices."""
    lo: float
    hi: float

    @classmethod
    def fit(cls, prices: np.ndarray) -> "Scaler":
        lo, hi = float(np.min(prices)), float(np.max(prices))
        return cls(lo, hi if hi > lo else lo + 1.0)

    def transform(self, x):
        return (np.asarray(x, dtype="float32") - self.lo) / (self.hi - self.lo)

    def inverse(self, x):
        return np.asarray(x, dtype="float32") * (self.hi - self.lo) + self.lo


class StockLSTM:
    """Train once on a price history, then predict a forward return from any
    60-day window. Kept deliberately small so 400+ models fit in a CPU run."""

    def __init__(self, window: int = 60, horizon: int = 21, epochs: int = 3,
                 batch_size: int = 32, seed: int = 0):
        self.window, self.horizon = window, horizon
        self.epochs, self.batch_size, self.seed = epochs, batch_size, seed
        self.model = None
        self.scaler: Scaler | None = None

    def fit(self, prices: np.ndarray) -> "StockLSTM":
        import tensorflow as tf
        from tensorflow.keras import Input, Sequential
        from tensorflow.keras.layers import LSTM, Dense

        tf.keras.utils.set_random_seed(self.seed)
        self.scaler = Scaler.fit(prices)
        X, y = make_windows(self.scaler.transform(prices), self.window, self.horizon)
        self.model = Sequential([
            Input((self.window, 1)),
            LSTM(50, return_sequences=True),
            LSTM(50),
            Dense(25),
            Dense(1),
        ])
        self.model.compile(optimizer="adam", loss="mean_squared_error")
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        return self

    def predict_returns(self, windows: np.ndarray) -> np.ndarray:
        """For each row of `windows` (shape n x window): predicted price
        `horizon` days ahead divided by the window's last price, minus 1."""
        w = np.asarray(windows, dtype="float32").reshape(-1, self.window)
        x = self.scaler.transform(w)[..., None]
        pred = self.scaler.inverse(self.model(x, training=False).numpy()[:, 0])
        return pred / w[:, -1] - 1.0

    def predict_return(self, last_window: np.ndarray) -> float:
        return float(self.predict_returns(np.asarray(last_window)[None, :])[0])

    def close(self):
        import tensorflow as tf
        tf.keras.backend.clear_session()
