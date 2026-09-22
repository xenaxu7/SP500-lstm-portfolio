# S&P 500 LSTM Portfolio Selection

Forecast next-period returns for every S&P 500 constituent with an LSTM, hold the top 30 equal-weighted, and test the result out of sample against SPY and a thematic stock list.

## What it does
1. **Data**: downloads 10 years of daily prices (2013-07-01 to 2023-06-30) for the current S&P 500 constituents with `yfinance`; tickers with too little history are dropped (430+ remain, about 1M price points). Missing data is handled before modelling.
2. **Model**: one LSTM per stock on a 60-day sliding window of scaled prices: `LSTM(50) → LSTM(50) → Dense(25) → Dense(1)`, trained with Keras. Each model predicts the next price, converted to a predicted return.
3. **Portfolio**: rank stocks by predicted return, take the top 30, equal weight.
4. **Backtest**: 18 months out of sample (2023-07-01 to 2024-12-31). Metrics: annualized return, annualized volatility, Sharpe ratio (2% risk-free), maximum drawdown. Compared with SPY and with a 30-stock thematic portfolio.

## Results
The script prints annualized return, annualized volatility, Sharpe ratio (2% risk-free) and maximum drawdown for the LSTM top-30 portfolio, the 30-stock thematic portfolio and SPY over the 18-month test window, and saves the comparison chart. A results table from the latest run will be added here.

## Run it
```bash
git clone https://github.com/xenaxu7/SP500-lstm-portfolio.git
cd SP500-lstm-portfolio
pip install -r requirements.txt
python main.py
```
Training 430+ models takes a while on CPU; the script trains in batches and prints progress.

## Limitations I know about
- Each model is trained for 3 epochs to keep runtime manageable; more epochs and a validation split would give a fairer read of the LSTM.
- A single prediction at the end of the training window drives the whole 18-month holding period; a rolling re-selection would be the natural next step.
- Constituent list is the current index, so the universe has survivorship bias.
- Equal weighting and no transaction costs.

## Stack
Python, TensorFlow/Keras, pandas, NumPy, scikit-learn (scaling), yfinance, Matplotlib
