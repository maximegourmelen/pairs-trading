import pandas as pd
import yfinance as yf
import random 

START_DATE = '2000-01-01'
END_DATE = '2024-01-01'
MIN_DATA_POINTS = 1000


TICKERS_DF = pd.read_csv("flat-ui__data-Fri Jul 04 2025.csv")

FORTUNE_500_TICKERS = TICKERS_DF['Symbol'].values.tolist()
SEED = 42


# === Download historical price data ===
def download_data(tickers):
    print(f"Downloading data for {len(tickers)} tickers...")
    data = yf.download(tickers, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)['Close']
    data = data.dropna(axis=1, thresh=MIN_DATA_POINTS)  # Drop columns with too many NaNs
    all_data_df = pd.DataFrame(data)
    all_data_df.to_csv(f'all data from {START_DATE} to {END_DATE}.csv')
    return 0


def main():
    download_data(FORTUNE_500_TICKERS)

if __name__ == "__main__":
    main()