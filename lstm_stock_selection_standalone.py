#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
S&P 500 Stock Selection using LSTM with Equal Weighting
Standalone version for Google Colab or Jupyter Notebook

This script analyzes ALL S&P 500 stocks using LSTM neural networks
and compares the performance against thematic selection and SPY benchmark.

Author: Xena Xu
Date: 2024
"""

# ============================================================================
# IMPORTS
# ============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("S&P 500 LSTM PORTFOLIO SELECTION SYSTEM")
print("="*60)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration settings"""
    # Time periods
    TESTING_END = datetime(2024, 12, 31)
    TESTING_START = datetime(2023, 7, 1)
    TRAINING_END = TESTING_START - timedelta(days=1)
    TRAINING_START = datetime(2013, 7, 1)
    
    # Model parameters
    TIME_STEP = 60
    EPOCHS = 3  # Reduced for speed
    BATCH_SIZE = 32
    
    # Portfolio parameters
    NUM_STOCKS_TO_SELECT = 30
    RISK_FREE_RATE = 0.02
    
    # Thematic stocks selection
    THEMATIC_STOCKS = [
        'NVDA', 'MSFT', 'AVGO', 'GOOGL', 'AMZN', 'ADBE', 'ORCL', 'META',
        'CRM', 'NOW', 'INTC', 'AMD', 'QCOM', 'TXN', 'MU', 'PLTR',
        'UNH', 'MRK', 'LLY', 'JNJ', 'ABBV', 'PFE', 'TMO',
        'V', 'MA', 'JPM', 'USB', 'BRK-B',
        'WMT', 'COST', 'PEP', 'PG', 'KO', 'MCD', 'HD', 'NKE',
        'XOM', 'NEE', 'HON', 'UNP', 'WM', 'NFLX', 'DIS', 'INTU', 'FTNT'
    ]

config = Config()

print(f"\nConfiguration:")
print(f"Training: {config.TRAINING_START.date()} to {config.TRAINING_END.date()}")
print(f"Testing: {config.TESTING_START.date()} to {config.TESTING_END.date()}")
print(f"Stocks to select: {config.NUM_STOCKS_TO_SELECT}")

# ============================================================================
# FUNCTIONS
# ============================================================================

def get_sp500_tickers():
    """Get S&P 500 tickers"""
    try:
        sp500_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        tickers = sp500_table[0]['Symbol'].tolist()
        tickers = [ticker.replace('.', '-') for ticker in tickers]
        print(f"✓ Fetched {len(tickers)} S&P 500 tickers from Wikipedia")
        return tickers
    except:
        print("⚠ Using fallback ticker list...")
        # Return top 100 S&P 500 stocks as fallback
        return [
            'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'TSLA', 'BRK-B',
            'GOOG', 'UNH', 'XOM', 'JPM', 'JNJ', 'V', 'PG', 'MA', 'AVGO', 'HD',
            'CVX', 'MRK', 'LLY', 'PEP', 'ABBV', 'KO', 'COST', 'WMT', 'BAC',
            'MCD', 'CRM', 'ACN', 'ADBE', 'TMO', 'CSCO', 'ABT', 'NFLX', 'PFE',
            'TMUS', 'ORCL', 'CMCSA', 'VZ', 'DHR', 'DIS', 'INTC', 'NKE', 'INTU',
            'WFC', 'TXN', 'PM', 'COP', 'UNP', 'RTX', 'AMGN', 'HON', 'BA', 'BMY',
            'QCOM', 'NEE', 'IBM', 'UPS', 'LOW', 'SPGI', 'LMT', 'DE', 'CAT', 'SBUX',
            'GE', 'AMD', 'MDT', 'PLD', 'AMAT', 'GS', 'SYK', 'BLK', 'ISRG', 'GILD',
            'ELV', 'ADI', 'CVS', 'VRTX', 'ADP', 'TJX', 'REGN', 'MDLZ', 'C', 'MS',
            'CI', 'NOW', 'SCHW', 'PGR', 'BDX', 'CB', 'EQIX', 'PANW', 'MU', 'SO',
            'ZTS', 'LRCX', 'SNPS', 'HUM', 'BSX', 'MMC', 'TT', 'ITW', 'DUK', 'SLB'
        ]

def download_stock_data(tickers, start_date, end_date):
    """Download stock data"""
    print(f"\nDownloading {len(tickers)} stocks...")
    
    data = yf.download(tickers, start=start_date, end=end_date, 
                       progress=True, threads=True, group_by='ticker')
    
    # Extract Adjusted Close prices
    close_data = pd.DataFrame()
    
    if isinstance(data.columns, pd.MultiIndex):
        for ticker in tickers:
            try:
                if ticker in data.columns.levels[0]:
                    close_data[ticker] = data[ticker]['Adj Close']
            except:
                continue
    else:
        close_data = data
    
    # Remove stocks with too many missing values
    threshold = len(close_data) * 0.5
    close_data = close_data.dropna(axis=1, thresh=threshold)
    
    print(f"✓ Downloaded {len(close_data.columns)} stocks successfully")
    return close_data

def create_lstm_model(input_shape):
    """Create LSTM model"""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def prepare_lstm_data(data, time_step=60):
    """Prepare data for LSTM"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    
    X, y = [], []
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i-time_step:i, 0])
        y.append(scaled_data[i, 0])
    
    return np.array(X), np.array(y), scaler

def train_and_predict_lstm(ticker, df_train, time_step=60, epochs=3):
    """Train LSTM and predict returns"""
    try:
        train_data = df_train[ticker].dropna().values
        
        if len(train_data) < time_step + 100:
            return None
        
        # Prepare data
        X_train, y_train, scaler = prepare_lstm_data(train_data, time_step)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        
        # Train model
        model = create_lstm_model((X_train.shape[1], 1))
        model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)
        
        # Predict
        last_60_days = train_data[-time_step:]
        last_60_days_scaled = scaler.transform(last_60_days.reshape(-1, 1))
        X_test = last_60_days_scaled.reshape((1, time_step, 1))
        
        predicted_price = model.predict(X_test, verbose=0)
        predicted_price = scaler.inverse_transform(predicted_price)
        
        current_price = train_data[-1]
        predicted_return = (predicted_price[0][0] - current_price) / current_price
        
        return predicted_return
        
    except:
        return None

def calculate_portfolio_metrics(returns):
    """Calculate portfolio metrics"""
    cumulative_returns = (1 + returns).cumprod()
    total_return = cumulative_returns.iloc[-1] - 1
    
    n_days = len(returns)
    annual_return = (1 + total_return) ** (252 / n_days) - 1
    annual_volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = (annual_return - config.RISK_FREE_RATE) / annual_volatility if annual_volatility > 0 else 0
    
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'cumulative_returns': cumulative_returns
    }

def get_equal_weighted_returns(selected_stocks, daily_returns):
    """Calculate equal-weighted portfolio returns"""
    available = [s for s in selected_stocks if s in daily_returns.columns]
    if not available:
        return None
    weights = 1.0 / len(available)
    return (daily_returns[available] * weights).sum(axis=1)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

print("\n" + "="*60)
print("STEP 1: DATA LOADING")
print("="*60)

# Get S&P 500 tickers
sp500_tickers = get_sp500_tickers()

# Download data
df_train = download_stock_data(sp500_tickers, config.TRAINING_START, config.TRAINING_END)
df_test = download_stock_data(df_train.columns.tolist(), config.TESTING_START, config.TESTING_END)

# Calculate daily returns
daily_returns_test = df_test.pct_change().dropna()

print("\n" + "="*60)
print("STEP 2: LSTM TRAINING")
print("="*60)

# Train LSTM for all stocks
print(f"Training LSTM models for {len(df_train.columns)} stocks...")
print("(This may take 10-20 minutes...)")

lstm_predictions = {}
batch_size = 50
total_stocks = len(df_train.columns)

for batch_start in range(0, total_stocks, batch_size):
    batch_end = min(batch_start + batch_size, total_stocks)
    batch_stocks = df_train.columns[batch_start:batch_end]
    
    print(f"Processing batch {batch_start//batch_size + 1}: stocks {batch_start+1}-{batch_end}")
    
    for ticker in batch_stocks:
        predicted_return = train_and_predict_lstm(ticker, df_train, epochs=config.EPOCHS)
        if predicted_return is not None:
            lstm_predictions[ticker] = predicted_return

print(f"\n✓ Trained LSTM models for {len(lstm_predictions)} stocks")

# Select top stocks
if lstm_predictions:
    lstm_df = pd.DataFrame(list(lstm_predictions.items()), columns=['Ticker', 'Predicted_Return'])
    lstm_df = lstm_df.sort_values('Predicted_Return', ascending=False)
    lstm_selected = lstm_df.head(config.NUM_STOCKS_TO_SELECT)['Ticker'].tolist()
    
    print(f"\nTop 10 LSTM predictions:")
    for i, (ticker, pred) in enumerate(lstm_df.head(10).values, 1):
        print(f"{i:2d}. {ticker:5s}: {pred:+7.2%}")
else:
    lstm_selected = list(df_train.columns[:config.NUM_STOCKS_TO_SELECT])

# Create thematic portfolio
chatgpt_selected = [s for s in config.THEMATIC_STOCKS if s in df_train.columns][:config.NUM_STOCKS_TO_SELECT]

print(f"\nThematic portfolio: {len(chatgpt_selected)} stocks selected")

print("\n" + "="*60)
print("STEP 3: PERFORMANCE EVALUATION")
print("="*60)

# Calculate portfolio returns
lstm_returns = get_equal_weighted_returns(lstm_selected, daily_returns_test)
lstm_metrics = calculate_portfolio_metrics(lstm_returns) if lstm_returns is not None else None

chatgpt_returns = get_equal_weighted_returns(chatgpt_selected, daily_returns_test)
chatgpt_metrics = calculate_portfolio_metrics(chatgpt_returns) if chatgpt_returns is not None else None

# SPY Benchmark
spy_data = yf.download('SPY', start=config.TESTING_START, end=config.TESTING_END, progress=False)
spy_prices = spy_data['Adj Close'] if 'Adj Close' in spy_data.columns else spy_data['Close']
spy_returns = spy_prices.pct_change().dropna()
spy_metrics = calculate_portfolio_metrics(spy_returns)

# Display results
print("\nPERFORMANCE COMPARISON")
print("-" * 60)

results_data = []
if lstm_metrics:
    results_data.append({
        'Portfolio': 'LSTM Selection',
        'Total Return': f"{lstm_metrics['total_return']:.2%}",
        'Annual Return': f"{lstm_metrics['annual_return']:.2%}",
        'Volatility': f"{lstm_metrics['annual_volatility']:.2%}",
        'Sharpe Ratio': f"{lstm_metrics['sharpe_ratio']:.2f}",
        'Max Drawdown': f"{lstm_metrics['max_drawdown']:.2%}"
    })

if chatgpt_metrics:
    results_data.append({
        'Portfolio': 'Thematic Selection',
        'Total Return': f"{chatgpt_metrics['total_return']:.2%}",
        'Annual Return': f"{chatgpt_metrics['annual_return']:.2%}",
        'Volatility': f"{chatgpt_metrics['annual_volatility']:.2%}",
        'Sharpe Ratio': f"{chatgpt_metrics['sharpe_ratio']:.2f}",
        'Max Drawdown': f"{chatgpt_metrics['max_drawdown']:.2%}"
    })

results_data.append({
    'Portfolio': 'S&P 500 (SPY)',
    'Total Return': f"{spy_metrics['total_return']:.2%}",
    'Annual Return': f"{spy_metrics['annual_return']:.2%}",
    'Volatility': f"{spy_metrics['annual_volatility']:.2%}",
    'Sharpe Ratio': f"{spy_metrics['sharpe_ratio']:.2f}",
    'Max Drawdown': f"{spy_metrics['max_drawdown']:.2%}"
})

results_df = pd.DataFrame(results_data)
print("\n", results_df.to_string(index=False))

# Visualization
plt.figure(figsize=(14, 7))

if lstm_metrics:
    plt.plot(lstm_metrics['cumulative_returns'].index,
             lstm_metrics['cumulative_returns'].values,
             label='LSTM Portfolio', linewidth=2, color='blue')

if chatgpt_metrics:
    plt.plot(chatgpt_metrics['cumulative_returns'].index,
             chatgpt_metrics['cumulative_returns'].values,
             label='Thematic Portfolio', linewidth=2, color='green')

plt.plot(spy_metrics['cumulative_returns'].index,
         spy_metrics['cumulative_returns'].values,
         label='S&P 500 (SPY)', linewidth=2, color='red', linestyle='--')

plt.title('Portfolio Performance Comparison', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
