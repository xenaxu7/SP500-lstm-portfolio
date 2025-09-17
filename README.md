# 📈 S&P 500 LSTM Portfolio Selection

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)](https://www.tensorflow.org/)
[![Status](https://img.shields.io/badge/Status-Active-success)](https://github.com/)

> *Where Statistics meets Deep Learning meets Wall Street* 🚀

## 🎯 What's This About?

Hey there! As a Stats & Econ student at UTSC who loves data-driven decision making, I built this project to combine my passion for quantitative analysis with practical financial applications. This system uses **LSTM neural networks** to analyze **ALL 500+ stocks** in the S&P 500 and build optimized portfolios that (hopefully!) beat the market.

Think of it as applying game theory to the stock market - but with deep learning doing the heavy lifting! 🧠

## ✨ Cool Features

- 🔍 **Complete Market Analysis**: Analyzes every single S&P 500 stock (not just a subset!)
- 🤖 **Deep Learning Magic**: LSTM models predict future returns with time series analysis
- 📊 **Risk Metrics Galore**: Sharpe ratio, Sortino ratio, maximum drawdown, and more
- ⚡ **Real-time Data**: Fetches latest market data using yfinance
- 🎨 **Beautiful Visualizations**: Performance charts that actually make sense

## 🏃‍♀️ Quick Start

```bash
# Clone this repo
git clone https://github.com/xenaxu7/sp500-lstm-portfolio.git
cd sp500-lstm-portfolio

# Install dependencies
pip install -r requirements.txt

# Run the magic! ✨
python main.py
```

Or if you want the standalone version (great for Google Colab):
```python
python lstm_stock_selection_standalone.py
```


## 🎓 The Tech Behind It

### LSTM Architecture
```
60 days of prices → LSTM(50) → LSTM(50) → Dense(25) → Price prediction
```

The model learns from 60 days of historical prices to predict the next move. It's like teaching the computer to recognize patterns that even seasoned traders might miss!

### Why LSTM?
Unlike traditional models, LSTMs can:
- Remember long-term patterns (perfect for market cycles!)
- Handle sequential data naturally
- Capture complex non-linear relationships

## 🎮 How It Works

1. **Data Collection** 📥
   - Downloads 10+ years of historical data for all S&P 500 stocks
   - Cleans and preprocesses everything (missing data? handled!)

2. **Model Training** 🏋️‍♀️
   - Trains individual LSTM models for each stock
   - Uses rolling windows for time series validation

3. **Portfolio Construction** 🏗️
   - Ranks stocks by predicted returns
   - Selects top 30 performers
   - Equal-weight allocation (keeping it simple!)

4. **Performance Analysis** 📈
   - Calculates risk-adjusted metrics
   - Compares against benchmarks
   - Creates beautiful visualizations

## 💡 What I Learned

As someone coming from Stats & Econ, this project taught me:
- **Theory meets Practice**: Academic concepts actually work in real markets!
- **Scale Matters**: Processing 500+ stocks is very different from toy datasets
- **Risk Management**: Returns are only half the story - volatility matters!
- **Deep Learning Power**: LSTMs can capture patterns I'd never spot manually


## 🤝 Let's Connect!

Love talking about data, markets, or strategy games? Let's chat!

**Xena Xu**
- 📧 Email: xenaxu7@gmail.com
- 💼 LinkedIn: [linkedin.com/in/xena-xu](https://www.linkedin.com/in/xena-xu/)
- 🐙 GitHub: [@xenaxu7](https://github.com/xenaxu7)
- 📍 Toronto, ON 🍁

## 📚 Resources & Inspiration

- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) - Best LSTM explanation ever!
- [PyPortfolioOpt Docs](https://pyportfolioopt.readthedocs.io/) - Portfolio optimization made easy
- My STAT302 & ECO220 courses at UTSC - Where it all started!

---

<p align="center">
  <i>"In investing, what is comfortable is rarely profitable."</i> - Robert Arnott
  <br>
  <i>(But with LSTMs, we can at least make it more scientific!)</i> 🎯
</p>

<p align="center">
  ⭐ If you found this helpful, consider giving it a star! ⭐
  <br>
  <sub>Built with 💜 and lots of ☕ by a UTSC student who should probably be studying for finals</sub>
</p>
