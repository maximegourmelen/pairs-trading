# Statistical Arbitrage Pairs Trading Strategy  

This project implements a **pairs trading strategy** based on **statistical arbitrage**. It uses historical stock price data to test for cointegration, calculate spreads, generate trading signals, backtest performance, and visualize results.  

The current example applies the strategy to **Coca-Cola (KO)** and **Becton Dickinson (BDX)** as a demonstration, but it can be extended to other asset pairs.  

---

## Features  

- 📈 **Data Retrieval**: Downloads historical stock prices from Yahoo Finance.  
- 🔗 **Cointegration Test**: Verifies statistical relationships between two assets.  
- ⚖️ **Spread & Hedge Ratio**: Calculates spread using OLS regression.  
- 🚦 **Signal Generation**: Creates long/short trading signals using z-score thresholds.  
- 💰 **Backtesting**: Simulates trading with capital allocation, transaction costs, and no look-ahead bias.  
- 📊 **Performance Analysis**: Computes return, volatility, Sharpe ratio, max drawdown, and profit factor.  
- 🎨 **Visualization**: Generates plots of prices, spreads with entry/exit points, and portfolio value.  

---

## Requirements  

Install the dependencies via `pip`:  

```bash
pip install numpy pandas yfinance statsmodels matplotlib seaborn scipy
