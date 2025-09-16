import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

# Configuration
TICKERS = ['BDX', 'KO']  # Coca-Cola and Pepsi - classic pairs example
START_DATE = '2010-01-01'
END_DATE = '2023-01-01'
Z_THRESHOLD = 1.5  # Entry/exit threshold for z-score
INITIAL_CAPITAL = 100000
TRADE_SIZE = 10000  # Capital allocated per trade

def fetch_data():
    """Download historical price data with error handling"""
    print("Downloading data...")
    try:
        # Download data with auto_adjust to get corrected prices
        data = yf.download(TICKERS, start=START_DATE, end=END_DATE, 
                           auto_adjust=True, progress=False)
        
        # Handle multi-index columns
        if isinstance(data.columns, pd.MultiIndex):
            close_prices = data['Close'].copy()
        else:
            close_prices = data['Close'].to_frame()
            close_prices.columns = [TICKERS[0]]
        
        close_prices.dropna(inplace=True)
        return close_prices
    except Exception as e:
        print(f"Error downloading data: {e}")
        raise

def test_cointegration(prices):
    """Test for cointegration between two series"""
    print("Testing for cointegration...")
    try:
        score, pvalue, _ = coint(prices[TICKERS[0]], prices[TICKERS[1]])
        print(f"Cointegration test p-value: {pvalue:.6f}")
        return pvalue < 0.05
    except Exception as e:
        print(f"Cointegration test failed: {e}")
        return False

def calculate_spread(prices):
    """Calculate spread and hedge ratio using OLS"""
    print("Calculating spread...")
    try:
        # Prepare data for regression
        y = prices[TICKERS[0]]
        X = prices[TICKERS[1]]
        X = sm.add_constant(X)
        
        model = sm.OLS(y, X)
        results = model.fit()
        hedge_ratio = results.params[1]
        spread = y - hedge_ratio * prices[TICKERS[1]]
        return spread, hedge_ratio
    except Exception as e:
        print(f"Spread calculation failed: {e}")
        raise

def generate_signals(spread):
    """Generate trading signals based on z-score"""
    print("Generating signals...")
    try:
        # Calculate z-score with rolling window
        z_spread = (spread - spread.rolling(window=30).mean()) / spread.rolling(window=30).std()
        signals = pd.DataFrame(index=spread.index)
        signals['z'] = z_spread
        signals['position_a'] = 0
        signals['position_b'] = 0
        
        # Long spread (buy A, sell B) when z < -Z_THRESHOLD
        signals.loc[signals['z'] <= -Z_THRESHOLD, 'position_a'] = 1
        signals.loc[signals['z'] <= -Z_THRESHOLD, 'position_b'] = -1
        
        # Short spread (sell A, buy B) when z > Z_THRESHOLD
        signals.loc[signals['z'] >= Z_THRESHOLD, 'position_a'] = -1
        signals.loc[signals['z'] >= Z_THRESHOLD, 'position_b'] = 1
        
        # Exit positions when z crosses zero
        signals.loc[np.sign(signals['z']).diff().ne(0), ['position_a', 'position_b']] = 0
        
        return signals
    except Exception as e:
        print(f"Signal generation failed: {e}")
        raise

def backtest(prices, signals, hedge_ratio):
    """Backtest trading strategy with transaction costs"""
    print("Backtesting strategy...")
    try:
        positions = signals[['position_a', 'position_b']].copy()
        
        # Calculate returns
        returns = pd.DataFrame(index=prices.index)
        returns['price_a'] = prices[TICKERS[0]]
        returns['price_b'] = prices[TICKERS[1]]
        returns['daily_ret_a'] = returns['price_a'].pct_change()
        returns['daily_ret_b'] = returns['price_b'].pct_change()
        
        # Shift positions to avoid look-ahead bias
        positions = positions.shift(1)
        
        # Calculate strategy returns
        strategy_returns = (
            positions['position_a'] * returns['daily_ret_a'] +
            positions['position_b'] * returns['daily_ret_b']
        )
        
        # Add transaction costs (0.1% per trade)
        trades = positions.diff().abs().sum(axis=1)
        strategy_returns -= trades * 0.0002
        
        # Calculate equity curve
        strategy_returns.dropna(inplace=True)
        cumulative_returns = (1 + strategy_returns).cumprod()
        portfolio_value = INITIAL_CAPITAL * cumulative_returns
        
        return portfolio_value, strategy_returns
    except Exception as e:
        print(f"Backtesting failed: {e}")
        raise

def analyze_performance(returns, portfolio_value):
    """Calculate performance metrics"""
    try:
        # Annualized return
        total_days = len(returns)
        total_return = portfolio_value.iloc[-1] / INITIAL_CAPITAL - 1
        annualized_return = (1 + total_return) ** (252/total_days) - 1
        
        # Annualized volatility
        annualized_vol = returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0
        
        # Max drawdown
        peak = portfolio_value.cummax()
        drawdown = (portfolio_value - peak) / peak
        max_drawdown = drawdown.min()
        
        # Profit Factor
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        profit_factor = gains / losses if losses > 0 else float('inf')
        
        print("\n===== Strategy Performance =====")
        print(f"Initial Capital: ${INITIAL_CAPITAL:,.0f}")
        print(f"Final Portfolio Value: ${portfolio_value.iloc[-1]:,.2f}")
        print(f"Total Return: {total_return:.2%}")
        print(f"Annualized Return: {annualized_return:.2%}")
        print(f"Annualized Volatility: {annualized_vol:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2%}")
        print(f"Profit Factor: {profit_factor:.2f}")
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_vol': annualized_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor
        }
    except Exception as e:
        print(f"Performance analysis failed: {e}")
        return {}

def visualize_results(prices, spread, signals, portfolio_value):
    """Create visualization of results with proper styling"""
    try:
        # Set plot style safely
        available_styles = plt.style.available
        preferred_styles = ['seaborn-darkgrid', 'seaborn', 'ggplot', 'bmh', 'dark_background']
        for style in preferred_styles:
            if style in available_styles:
                plt.style.use(style)
                break
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True)
        
        # Price plot
        axes[0].plot(prices[TICKERS[0]], label=TICKERS[0], lw=1.5)
        axes[0].plot(prices[TICKERS[1]], label=TICKERS[1], lw=1.5)
        axes[0].set_title('Stock Prices', fontsize=12)
        axes[0].set_ylabel('Price ($)', fontsize=10)
        axes[0].legend(fontsize=9)
        axes[0].grid(True, linestyle='--', alpha=0.7)
        
        # Spread plot with signals
        axes[1].plot(spread, label='Spread', color='purple', lw=1.5)
        axes[1].axhline(spread.mean(), color='gray', linestyle='--', alpha=0.7, label='Mean')
        
        # Add ±1 standard deviation lines
        std_dev = spread.std()
        axes[1].axhline(spread.mean() + Z_THRESHOLD * std_dev, color='red', 
                        linestyle='--', alpha=0.5, label=f'+{Z_THRESHOLD}σ')
        axes[1].axhline(spread.mean() - Z_THRESHOLD * std_dev, color='green', 
                        linestyle='--', alpha=0.5, label=f'-{Z_THRESHOLD}σ')
        
        # Plot signals
        long_signals = spread[signals['position_a'] > 0]
        short_signals = spread[signals['position_a'] < 0]
        axes[1].scatter(long_signals.index, long_signals, marker='^', 
                        color='green', s=60, alpha=0.9, label='Long Spread')
        axes[1].scatter(short_signals.index, short_signals, marker='v', 
                        color='red', s=60, alpha=0.9, label='Short Spread')
        
        axes[1].set_title('Spread with Trading Signals', fontsize=12)
        axes[1].set_ylabel('Spread Value', fontsize=10)
        axes[1].legend(fontsize=8, loc='upper left')
        axes[1].grid(True, linestyle='--', alpha=0.7)
        
        # Portfolio value
        axes[2].plot(portfolio_value, color='blue', lw=1.5)
        axes[2].set_title('Portfolio Value', fontsize=12)
        axes[2].set_ylabel('Value ($)', fontsize=10)
        axes[2].set_xlabel('Date', fontsize=10)
        axes[2].grid(True, linestyle='--', alpha=0.7)
        
        # Add horizontal line for initial capital
        axes[2].axhline(INITIAL_CAPITAL, color='gray', linestyle='--', alpha=0.7)
        
        # Add performance annotation
        final_value = portfolio_value.iloc[-1]
        total_return = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL
        ann_return = ((1 + total_return) ** (252/len(portfolio_value)) - 1) * 100
        
        axes[2].annotate(f'Final Value: ${final_value:,.2f}\n'
                        f'Total Return: {total_return:.1%}\n'
                        f'Annualized: {ann_return:.1f}%',
                        xy=(portfolio_value.index[-1], final_value),
                        xytext=(10, -30), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig('stat_arb_results.png', dpi=300)
        plt.show()
        return True
    except Exception as e:
        print(f"Visualization failed: {e}")
        return False

def main():
    print("Statistical Arbitrage Pairs Trading Strategy")
    print("=" * 50)
    print(f"Trading Pair: {TICKERS[0]} vs {TICKERS[1]}")
    print(f"Date Range: {START_DATE} to {END_DATE}")
    print(f"Z-Score Threshold: {Z_THRESHOLD}")
    print("=" * 50 + "\n")
    
    try:
        # Pipeline execution
        prices = fetch_data()
        print(f"Data Range: {prices.index[0].date()} to {prices.index[-1].date()}")
        
        coint_result = test_cointegration(prices)
        if not coint_result:
            print("\nWarning: Selected pairs are not cointegrated. Results may be unreliable.")
            print("Consider choosing different assets or a different time period.")
        
        spread, hedge_ratio = calculate_spread(prices)
        print(f"Hedge Ratio: {hedge_ratio:.4f} ({TICKERS[0]} / {TICKERS[1]})")
        
        signals = generate_signals(spread)
        portfolio_value, strategy_returns = backtest(prices, signals, hedge_ratio)
        
        analyze_performance(strategy_returns, portfolio_value)
        visualize_results(prices, spread, signals, portfolio_value)
        
        print("\nProcess completed successfully!")
    except Exception as e:
        print(f"\nError in main execution: {e}")

if __name__ == "__main__":
    main()