from datetime import datetime
import pandas as pd
import numpy as np
import pandas_datareader.data as web


# -----------------------------
# Configuration
# -----------------------------
START_DATE = datetime(2015, 1, 1)
END_DATE   = datetime(2025, 12, 31)

TICKERS = {
    "SPY": "spy.us", # US equities
    "EFA": "efa.us", # Developed markets ex-US
    "AGG": "agg.us", # Bonds
    "GLD": "gld.us", # Gold
    "USO": "uso.us", # Oil
    "TLH": "tlh.us", # Long Treasuries
#    "EURUSD": "eurusd", # FX
    "BTC": "btc.us" # Crypto (fat tails)
}


# -----------------------------
# Data fetching
# -----------------------------
def fetch_prices(ticker: str) -> pd.Series:
    """
    Fetch daily prices from Stooq using pandas-datareader.
    Returns a Series indexed by date.

    Robust to column differences: prefers Close, otherwise uses the best available
    price-like column.
    """
    df = web.DataReader(ticker, "stooq", START_DATE, END_DATE)

    # Stooq returns newest -> oldest
    df = df.sort_index()

    # Common price columns seen across readers
    candidate_cols = ["Close", "Adj Close", "AdjClose", "Settlement", "Price", "Value", "Last"]

    for col in candidate_cols:
        if col in df.columns:
            s = df[col].copy()
            s.name = "Price"
            return s

    # If no known price col exists, show what we did get (debug-friendly)
    raise ValueError(f"No usable price column for {ticker}. Columns: {list(df.columns)}")



def build_price_matrix(tickers: dict) -> pd.DataFrame:
    prices = {}
    failed = {}

    for name, ticker in tickers.items():
        try:
            print(f"Fetching {name} ({ticker})...")
            prices[name] = fetch_prices(ticker)
        except Exception as e:
            failed[name] = (ticker, str(e))
            print(f"  -> SKIPPED {name} ({ticker}): {e}")

    if len(prices) < 3:
        raise RuntimeError(f"Too few series fetched successfully: {list(prices.keys())}")

    price_df = pd.DataFrame(prices).dropna(how="all")

    if failed:
        print("\nFailed tickers summary:")
        for name, (ticker, err) in failed.items():
            print(f"  {name:8s} {ticker:12s}  {err}")

    return price_df


# -----------------------------
# Returns computation
# -----------------------------
def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = np.log(prices / prices.shift(1))
    return returns.dropna()


# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":
    prices = build_price_matrix(TICKERS)
    returns = compute_log_returns(prices)

    prices.to_csv("prices.csv")
    returns.to_csv("returns.csv")

    print("\nData summary:")
    print(prices.tail())
    print("\nReturns summary:")
    print(returns.describe())

    print("\nSaved:")
    print("  prices.csv")
    print("  returns.csv")
