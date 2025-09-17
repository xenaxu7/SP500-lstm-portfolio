# S&P 500 Stock Selection using LSTM with Equal Weighting
# Analyzing ALL S&P 500 stocks, not just 50

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf

from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import risk_matrix

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

import warnings
warnings.filterwarnings('ignore')

# Define Time Periods
testing_end = datetime(2024, 12, 31)
testing_start = datetime(2023, 7, 1)
training_end = testing_start - timedelta(days=1)
training_start = datetime(2013, 7, 1)

print(f"Data periods:")
print(f"Training period: {training_start.date()} to {training_end.date()}")
print(f"Testing period: {testing_start.date()} to {testing_end.date()}")

# Get ALL S&P 500 Stocks
try:
    sp500_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    sp500_tickers = sp500_table[0]['Symbol'].tolist()
    # Clean tickers (remove any dots for special classes)
    sp500_tickers = [ticker.replace('.', '-') for ticker in sp500_tickers]
    print(f"\nSuccessfully fetched {len(sp500_tickers)} S&P 500 tickers from Wikipedia")
except:
    print("\nCouldn't fetch from Wikipedia, using hardcoded list...")
    # Fallback: Complete S&P 500 list (as of 2024)
    sp500_tickers = [
        'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'TSLA', 'BRK-B', 'GOOG', 'UNH',
        'XOM', 'JPM', 'JNJ', 'V', 'PG', 'MA', 'AVGO', 'HD', 'CVX', 'MRK',
        'LLY', 'PEP', 'ABBV', 'KO', 'COST', 'WMT', 'BAC', 'MCD', 'CRM', 'ACN',
        'ADBE', 'TMO', 'CSCO', 'ABT', 'NFLX', 'PFE', 'TMUS', 'ORCL', 'CMCSA', 'VZ',
        'DHR', 'DIS', 'INTC', 'NKE', 'INTU', 'WFC', 'TXN', 'PM', 'COP', 'UNP',
        'RTX', 'AMGN', 'HON', 'BA', 'BMY', 'QCOM', 'NEE', 'IBM', 'UPS', 'LOW',
        'SPGI', 'LMT', 'DE', 'CAT', 'SBUX', 'GE', 'AMD', 'MDT', 'PLD', 'AMAT',
        'GS', 'SYK', 'BLK', 'ISRG', 'GILD', 'ELV', 'ADI', 'CVS', 'VRTX', 'ADP',
        'TJX', 'REGN', 'MDLZ', 'C', 'MS', 'CI', 'NOW', 'SCHW', 'PGR', 'BDX',
        'CB', 'EQIX', 'PANW', 'MU', 'SO', 'ZTS', 'LRCX', 'SNPS', 'HUM', 'BSX',
        'MMC', 'TT', 'ITW', 'DUK', 'SLB', 'AON', 'ETN', 'USB', 'CME', 'APH',
        'MCO', 'CL', 'FI', 'WM', 'FCX', 'MO', 'ICE', 'HCA', 'PNC', 'EW',
        'T', 'CSX', 'MAR', 'MCK', 'GIS', 'NSC', 'MSCI', 'KLAC', 'MSI', 'ROP',
        'CDNS', 'AXP', 'PSX', 'ORLY', 'TDG', 'MNST', 'MET', 'SHW', 'APD', 'CTAS',
        'CCI', 'EMR', 'KMB', 'FDX', 'VLO', 'OXY', 'ADSK', 'PAYX', 'CHTR', 'D',
        'SRE', 'TRV', 'NXPI', 'MCHP', 'AEP', 'TFC', 'TEL', 'PH', 'CARR', 'AZO',
        'IQV', 'STZ', 'CMI', 'ROST', 'KDP', 'IDXX', 'CPRT', 'PCAR', 'COF', 'AFL',
        'KVUE', 'AJG', 'FIS', 'EXC', 'FTNT', 'GWW', 'DXCM', 'SPG', 'ODFL', 'BK',
        'NUE', 'HSY', 'FAST', 'AMT', 'RSG', 'CSGP', 'OTIS', 'PRU', 'ANET', 'XEL',
        'YUM', 'BIIB', 'ALL', 'EA', 'SQ', 'EQR', 'KHC', 'VRSK', 'PSA', 'CTSH',
        'WBD', 'A', 'MLM', 'MTD', 'WEC', 'TROW', 'DD', 'DOV', 'PPG', 'HPQ',
        'ED', 'BKR', 'DLR', 'ALB', 'O', 'ECL', 'HES', 'ON', 'STT', 'ACGL',
        'ES', 'IFF', 'LHX', 'ETR', 'RMD', 'WTW', 'GPN', 'AWK', 'EFX', 'TSCO',
        'CBRE', 'DLTR', 'AVB', 'ILMN', 'ZBH', 'DFS', 'HLT', 'EIX', 'ROK', 'ANSS',
        'DTE', 'URI', 'CDW', 'EBAY', 'PCG', 'AEE', 'KEYS', 'VICI', 'FANG', 'WST',
        'VMC', 'CAH', 'AIG', 'IT', 'ULTA', 'LEN', 'FTV', 'CMS', 'IR', 'GPC',
        'CHD', 'DAL', 'GLW', 'NTRS', 'TRGP', 'LVS', 'SBAC', 'PEG', 'UAL', 'CINF',
        'SYF', 'MKC', 'LUV', 'WAB', 'EXR', 'PPL', 'DGX', 'EXPE', 'ALGN', 'MOH',
        'NVR', 'DHI', 'INVH', 'CLX', 'EQT', 'VRSN', 'VTR', 'WY', 'RF', 'PWR',
        'CF', 'LYB', 'BALL', 'BAX', 'TER', 'HOLX', 'GRMN', 'NTAP', 'FE', 'STLD',
        'HPE', 'K', 'BR', 'BRO', 'NDAQ', 'LW', 'TDY', 'MRO', 'CNP', 'PKI',
        'DOC', 'TRMB', 'MAA', 'OMC', 'COO', 'AKAM', 'SNA', 'CTLT', 'EXPD', 'BBY',
        'RJF', 'AMCR', 'DPZ', 'TYL', 'PTC', 'POOL', 'NI', 'PKG', 'PFG', 'J',
        'CBOE', 'KEY', 'HUBB', 'MAS', 'TXT', 'ESS', 'IEX', 'CZR', 'JBHT', 'WDC',
        'SWK', 'CFG', 'CCL', 'TSN', 'GEN', 'IRM', 'L', 'FITB', 'RE', 'MPWR',
        'KMX', 'RCL', 'EMN', 'CE', 'LDOS', 'STE', 'CAG', 'FDS', 'ZBRA', 'HIG',
        'EPAM', 'HBAN', 'TAP', 'IPG', 'VTRS', 'DRI', 'WAT', 'JKHY', 'EVRG', 'ATO',
        'HWM', 'BEN', 'UDR', 'AOS', 'NRG', 'CPT', 'MTCH', 'PEAK', 'PAYC', 'INCY',
        'CHRW', 'LNT', 'MKTX', 'SJM', 'TPR', 'KIM', 'HRL', 'LKQ', 'SWKS', 'CRL',
        'BWA', 'ALLE', 'FOXA', 'BIO', 'HSIC', 'SEDG', 'ENPH', 'CMA', 'PNW', 'REG',
        'AAL', 'BBWI', 'BF-B', 'CPB', 'WHR', 'GNRC', 'TECH', 'MGM', 'ZION', 'PNR',
        'FFIV', 'BXP', 'UHS', 'NDSN', 'FMC', 'MOS', 'ETSY', 'CDAY', 'AIZ', 'JNPR',
        'NWSA', 'QRVO', 'ROL', 'XRAY', 'APA', 'GL', 'WYNN', 'DVA', 'LNC', 'FRT',
        'IVZ', 'VFC', 'RHI', 'WBA', 'ALK', 'AAP', 'NWS', 'PARA', 'FOX', 'RL',
        'SEE', 'DVN', 'CNC', 'HII', 'CTVA', 'OGN', 'DISH', 'CEG', 'MRNA', 'PLTR'
    ]

# Number of stocks to select for portfolio
num_stocks_to_select = 30

print(f"Will analyze ALL {len(sp500_tickers)} S&P 500 stocks")
print(f"Will select top {num_stocks_to_select} stocks for portfolios")

# Download ALL S&P 500 Stock Data
print("\n" + "="*60)
print("DOWNLOADING S&P 500 DATA")
print("="*60)

def download_all_sp500_data(tickers, start, end):
    """Download data for ALL S&P 500 stocks"""
    print(f"Downloading {len(tickers)} S&P 500 stocks from {start.date()} to {end.date()}...")

    # Download all at once with progress bar
    data = yf.download(tickers, start=start, end=end, progress=True, threads=True, group_by='ticker')

    # Extract Adj Close prices
    if isinstance(data.columns, pd.MultiIndex):
        # Multi-ticker download returns MultiIndex columns
        close_data = pd.DataFrame()
        for ticker in tickers:
            try:
                if ticker in data.columns.levels[0]:
                    close_data[ticker] = data[ticker]['Adj Close']
            except:
                try:
                    close_data[ticker] = data[ticker]['Close']
                except:
                    continue
    else:
        close_data = data

    # Remove stocks with too many missing values
    threshold = len(close_data) * 0.5
    close_data = close_data.dropna(axis=1, thresh=threshold)

    print(f"✓ Successfully downloaded {len(close_data.columns)} out of {len(tickers)} stocks")
    return close_data

# Download training data for ALL S&P 500 stocks
df_train = download_all_sp500_data(sp500_tickers, training_start, training_end)

# Download testing data for the same stocks
df_test = download_all_sp500_data(df_train.columns.tolist(), testing_start, testing_end)

# Calculate daily returns
daily_returns_test = df_test.pct_change().dropna()

print(f"\nData shapes:")
print(f"Training data: {df_train.shape} (days x stocks)")
print(f"Testing data: {df_test.shape} (days x stocks)")

# LSTM Model for Stock Selection
print("\n" + "="*60)
print("LSTM MODEL TRAINING")
print("="*60)

def create_lstm_model(input_shape):
    """Create LSTM model"""
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
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
    """Train LSTM and predict returns - FASTER VERSION"""
    try:
        train_data = df_train[ticker].dropna().values

        if len(train_data) < time_step + 100:
            return None

        # Prepare data
        X_train, y_train, scaler = prepare_lstm_data(train_data, time_step)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

        # Train model with fewer epochs for speed
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

    except Exception as e:
        return None

# Train LSTM for ALL stocks
print(f"Training LSTM models for ALL {len(df_train.columns)} S&P 500 stocks...")
print("(This will take 10-20 minutes for 400+ stocks with reduced epochs...)")

lstm_predictions = {}
batch_size = 50
total_stocks = len(df_train.columns)

for batch_start in range(0, total_stocks, batch_size):
    batch_end = min(batch_start + batch_size, total_stocks)
    batch_stocks = df_train.columns[batch_start:batch_end]

    print(f"\nBatch {batch_start//batch_size + 1}/{(total_stocks-1)//batch_size + 1}: "
          f"Stocks {batch_start + 1}-{batch_end} of {total_stocks}")

    for ticker in batch_stocks:
        predicted_return = train_and_predict_lstm(ticker, df_train, epochs=3)  # Reduced epochs
        if predicted_return is not None:
            lstm_predictions[ticker] = predicted_return

print(f"\n✓ Trained LSTM models for {len(lstm_predictions)} out of {total_stocks} stocks")

# Select top stocks
if not lstm_predictions:
    print("Warning: No successful LSTM predictions")
    lstm_selected = list(df_train.columns[:num_stocks_to_select])
else:
    lstm_df = pd.DataFrame(list(lstm_predictions.items()), columns=['Ticker', 'Predicted_Return'])
    lstm_df = lstm_df.sort_values('Predicted_Return', ascending=False)
    lstm_selected = lstm_df.head(num_stocks_to_select)['Ticker'].tolist()

print(f"\nTop 10 LSTM-selected stocks from {len(lstm_predictions)} analyzed:")
for i, stock in enumerate(lstm_selected[:10], 1):
    if stock in lstm_predictions:
        print(f"{i}. {stock}: {lstm_predictions[stock]:.2%}")

# ChatGPT Selection (From thematic recommendations)
print("\n" + "="*60)
print("CHATGPT THEMATIC SELECTION")
print("="*60)

# From thematic recommendations
chatgpt_pool = [
    'NVDA', 'MSFT', 'AVGO', 'GOOGL', 'AMZN', 'ADBE', 'ORCL', 'META',
    'CRM', 'NOW', 'INTC', 'AMD', 'QCOM', 'TXN', 'MU', 'PLTR',
    'UNH', 'MRK', 'LLY', 'JNJ', 'ABBV', 'PFE', 'TMO',
    'V', 'MA', 'JPM', 'USB', 'BRK-B',
    'WMT', 'COST', 'PEP', 'PG', 'KO', 'MCD', 'HD', 'NKE',
    'XOM', 'NEE', 'HON', 'UNP', 'WM', 'NFLX', 'DIS', 'INTU', 'FTNT'
]

# Select 30 available stocks
chatgpt_selected = [s for s in chatgpt_pool if s in df_train.columns][:num_stocks_to_select]

print(f"ChatGPT portfolio ({len(chatgpt_selected)} stocks from thematic analysis)")
print("First 10:", chatgpt_selected[:10])

# Portfolio Performance Analysis
print("\n" + "="*60)
print("PORTFOLIO PERFORMANCE ANALYSIS")
print("="*60)

def calculate_portfolio_metrics(returns):
    """Calculate portfolio metrics"""
    cumulative_returns = (1 + returns).cumprod()
    total_return = cumulative_returns.iloc[-1] - 1

    n_days = len(returns)
    annual_return = (1 + total_return) ** (252 / n_days) - 1
    annual_volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = (annual_return - 0.02) / annual_volatility if annual_volatility > 0 else 0

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
    """Calculate equal-weighted returns"""
    available = [s for s in selected_stocks if s in daily_returns.columns]
    if not available:
        return None
    weights = 1.0 / len(available)
    return (daily_returns[available] * weights).sum(axis=1)

# Calculate metrics
lstm_returns = get_equal_weighted_returns(lstm_selected, daily_returns_test)
lstm_metrics = calculate_portfolio_metrics(lstm_returns) if lstm_returns is not None else None

chatgpt_returns = get_equal_weighted_returns(chatgpt_selected, daily_returns_test)
chatgpt_metrics = calculate_portfolio_metrics(chatgpt_returns) if chatgpt_returns is not None else None

# SPY Benchmark
spy_data = yf.download('SPY', start=testing_start, end=testing_end, progress=False)
if isinstance(spy_data.columns, pd.MultiIndex):
    spy_prices = spy_data['Adj Close'].squeeze()
else:
    spy_prices = spy_data['Adj Close'] if 'Adj Close' in spy_data.columns else spy_data['Close']
spy_returns = spy_prices.pct_change().dropna()
spy_metrics = calculate_portfolio_metrics(spy_returns)

# Display Results
print("\n" + "="*60)
print("PERFORMANCE COMPARISON (EQUAL-WEIGHTED)")
print("="*60)

results = []
if lstm_metrics:
    results.append({
        'Portfolio': f'LSTM ({len(lstm_predictions)} stocks analyzed)',
        'Total Return': f"{lstm_metrics['total_return']:.2%}",
        'Annual Return': f"{lstm_metrics['annual_return']:.2%}",
        'Volatility': f"{lstm_metrics['annual_volatility']:.2%}",
        'Sharpe Ratio': f"{lstm_metrics['sharpe_ratio']:.2f}",
        'Max Drawdown': f"{lstm_metrics['max_drawdown']:.2%}"
    })

if chatgpt_metrics:
    results.append({
        'Portfolio': 'ChatGPT Thematic',
        'Total Return': f"{chatgpt_metrics['total_return']:.2%}",
        'Annual Return': f"{chatgpt_metrics['annual_return']:.2%}",
        'Volatility': f"{chatgpt_metrics['annual_volatility']:.2%}",
        'Sharpe Ratio': f"{chatgpt_metrics['sharpe_ratio']:.2f}",
        'Max Drawdown': f"{chatgpt_metrics['max_drawdown']:.2%}"
    })

results.append({
    'Portfolio': 'S&P 500 (SPY)',
    'Total Return': f"{spy_metrics['total_return']:.2%}",
    'Annual Return': f"{spy_metrics['annual_return']:.2%}",
    'Volatility': f"{spy_metrics['annual_volatility']:.2%}",
    'Sharpe Ratio': f"{spy_metrics['sharpe_ratio']:.2f}",
    'Max Drawdown': f"{spy_metrics['max_drawdown']:.2%}"
})

results_df = pd.DataFrame(results)
print("\n", results_df.to_string(index=False))

# Visualization
plt.figure(figsize=(14, 7))

if lstm_metrics:
    plt.plot(lstm_metrics['cumulative_returns'].index,
             lstm_metrics['cumulative_returns'].values,
             label=f'LSTM Portfolio ({len(lstm_predictions)} stocks analyzed)',
             linewidth=2, color='blue')

if chatgpt_metrics:
    plt.plot(chatgpt_metrics['cumulative_returns'].index,
             chatgpt_metrics['cumulative_returns'].values,
             label='ChatGPT Thematic Portfolio', linewidth=2, color='green')

plt.plot(spy_metrics['cumulative_returns'].index,
         spy_metrics['cumulative_returns'].values,
         label='S&P 500 (SPY)', linewidth=2, color='red', linestyle='--')

plt.title('Portfolio Performance: LSTM (All S&P 500) vs ChatGPT Thematic', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Summary
print("\n" + "="*60)
print("ANALYSIS SUMMARY")
print("="*60)

print(f"\n📊 Key Statistics:")
print(f"• LSTM analyzed {len(lstm_predictions)} S&P 500 stocks (entire universe)")
print(f"• Selected top {num_stocks_to_select} stocks based on predicted returns")
print(f"• ChatGPT used thematic selection across sectors")
print(f"• Both portfolios equally weighted with {num_stocks_to_select} stocks")

if lstm_metrics and chatgpt_metrics:
    if lstm_metrics['total_return'] > chatgpt_metrics['total_return']:
        print(f"\n✓ LSTM outperformed by {(lstm_metrics['total_return'] - chatgpt_metrics['total_return'])*100:.2f}%")
    else:
        print(f"\n✓ ChatGPT outperformed by {(chatgpt_metrics['total_return'] - lstm_metrics['total_return'])*100:.2f}%")

print("\n" + "="*60)
print("COMPLETE")
print("="*60)
