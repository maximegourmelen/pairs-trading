import yfinance as yf
import pandas as pd
import numpy as np
import random
import itertools
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm
import math

# === Configuration ===
START_DATE = '2010-01-01'
END_DATE = '2023-01-01'
NUM_SAMPLES = 450
MIN_DATA_POINTS = 1000
SEED = 42
CSV_FILE = 'all data from 2000-01-01 to 2024-01-01.csv'

# === Step 1: Load ticker list ===
TICKERS_DF = pd.read_csv(CSV_FILE)
FORTUNE_500_TICKERS = list(TICKERS_DF.transpose().index)
FORTUNE_500_TICKERS.pop(0)

random.seed(SEED)
SAMPLE_TICKERS = random.sample(FORTUNE_500_TICKERS, NUM_SAMPLES)

# === Step 2: Load & clean data ===
def fetch_data(tickers):
    print("Loading price data from CSV...")
    df = pd.read_csv(CSV_FILE, index_col=0, parse_dates=True)
    df = df[tickers]
    df = df.loc[(df.index >= START_DATE) & (df.index <= END_DATE)]
    
    # Drop columns with too many NaNs
    df = df.dropna(axis=1, thresh=MIN_DATA_POINTS)
    
    # Drop rows where any stock has NaN
    df = df.dropna()
    return df

# === Step 3: Find most cointegrated pair ===
def find_most_cointegrated_pair(data):
    print("Testing cointegration between pairs...")
    best_pvalue = 1
    best_pair = (None, None)
    best_hedge_ratio = None

    tickers = data.columns
    counter = 0
    total_combinations = math.factorial(len(tickers)) / (math.factorial(len(tickers)-2) * 2)
    for a, b in itertools.combinations(tickers, 2):
        print(f'testing {a} and {b}, progress: {round(counter/total_combinations*100, 2)} %')
        counter += 1
        series_a = data[a]
        series_b = data[b]

        if len(series_a) != len(series_b):
            continue

        # Drop rows with NaN or inf
        combined = pd.concat([series_a, series_b], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
        if combined.empty:
            continue

        series_a_clean = combined.iloc[:, 0]
        series_b_clean = combined.iloc[:, 1]

        try:
            score, pvalue, _ = coint(series_a_clean, series_b_clean)
            if pvalue < best_pvalue:
                # Calculate hedge ratio
                X = sm.add_constant(series_b_clean)
                model = sm.OLS(series_a_clean, X).fit()
                hedge_ratio = model.params[series_b_clean.name]
                best_pvalue = pvalue
                best_pair = (a, b)
                best_hedge_ratio = hedge_ratio
        except Exception as e:
            print(f"Skipping pair {a}, {b} due to error: {e}")

    return best_pair, best_pvalue, best_hedge_ratio

# === Step 4: Main ===
def main():
    print("Finding most cointegrated stock pair in Fortune 500 sample...\n")
    data = fetch_data(SAMPLE_TICKERS)
    print(f"Data shape: {data.shape}")
    best_pair, best_pvalue, hedge_ratio = find_most_cointegrated_pair(data)

    if best_pair[0] is not None:
        print("\nMost Cointegrated Pair Found:")
        print(f"Stocks: {best_pair[0]} & {best_pair[1]}")
        print(f"Cointegration p-value: {best_pvalue:.6f}")
        print(f"Hedge Ratio ({best_pair[0]} / {best_pair[1]}): {hedge_ratio:.4f}")
    else:
        print("No suitable cointegrated pair found.")

if __name__ == "__main__":
    main()
