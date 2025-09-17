#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
S&P 500 Stock Selection using LSTM Neural Networks
Main execution script for portfolio strategy comparison

Author: Xena Xu
Date: 2024
License: MIT
"""

import warnings
warnings.filterwarnings('ignore')

from src.config import Config
from src.data_loader import SP500DataLoader
from src.lstm_model import LSTMPredictor
from src.portfolio import PortfolioConstructor
from src.evaluation import PerformanceEvaluator
from src.visualization import Visualizer

def main():
    """Main execution function"""
    
    print("="*60)
    print("S&P 500 LSTM PORTFOLIO SELECTION SYSTEM")
    print("="*60)
    
    # Initialize configuration
    config = Config()
    
    # 1. Data Loading Phase
    print("\n[1/5] Loading S&P 500 Data...")
    data_loader = SP500DataLoader(config)
    train_data, test_data, tickers = data_loader.load_all_data()
    
    # 2. LSTM Training Phase
    print("\n[2/5] Training LSTM Models...")
    lstm_predictor = LSTMPredictor(config)
    predictions = lstm_predictor.train_and_predict_all(train_data, tickers)
    
    # 3. Portfolio Construction Phase
    print("\n[3/5] Constructing Portfolios...")
    portfolio_constructor = PortfolioConstructor(config)
    
    # LSTM Portfolio
    lstm_portfolio = portfolio_constructor.create_lstm_portfolio(
        predictions, config.NUM_STOCKS_TO_SELECT
    )
    
    # Thematic Portfolio (ChatGPT selection)
    thematic_portfolio = portfolio_constructor.create_thematic_portfolio()
    
    # 4. Performance Evaluation Phase
    print("\n[4/5] Evaluating Performance...")
    evaluator = PerformanceEvaluator(config)
    
    results = evaluator.evaluate_all_portfolios(
        lstm_portfolio,
        thematic_portfolio,
        test_data
    )
    
    # 5. Visualization Phase
    print("\n[5/5] Generating Visualizations...")
    visualizer = Visualizer(config)
    visualizer.plot_performance_comparison(results)
    visualizer.save_results_to_csv(results)
    
    # Display final summary
    print("\n" + "="*60)
    print("EXECUTION COMPLETE")
    print("="*60)
    print(f"\n📊 Results Summary:")
    print(f"• LSTM Portfolio Return: {results['lstm']['total_return']:.2%}")
    print(f"• Thematic Portfolio Return: {results['thematic']['total_return']:.2%}")
    print(f"• S&P 500 Return: {results['spy']['total_return']:.2%}")
    
    print(f"\n📈 Best Performing: {results['best_performer']}")
    print(f"\n✅ Results saved to: results/")
    
    return results

if __name__ == "__main__":
    results = main()
