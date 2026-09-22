"""One tiny end-to-end fit to make sure the Keras path runs. Slow (~5 s)."""
import numpy as np

from sp500lstm.model import StockLSTM


def test_stock_lstm_fits_and_predicts_finite_return():
    rng = np.random.default_rng(0)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400)))
    m = StockLSTM(window=60, horizon=21, epochs=1, seed=0).fit(prices)
    r = m.predict_return(prices[-60:])
    assert np.isfinite(r) and -1 < r < 5
    many = m.predict_returns(np.stack([prices[-60:], prices[-120:-60]]))
    assert many.shape == (2,) and many[0] == r
    m.close()
