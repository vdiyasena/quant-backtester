import yfinance as yf
import pandas as pd
import numpy as np

def download_data(ticker, start, end):
    """
    Download adjusted OHLCV data for a given ticker from Yahoo Finance.

    Args:
        ticker: stock/ETF symbol e.g. 'SPY'
        start: start date string e.g. '2010-01-01'
        end: end date string e.g. '2024-01-01'

    Returns:
        Clean DataFrame with OHLCV data and log returns
    """
    print(f"Downloading {ticker} from {start} to {end}...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)

    # Validate data
    assert not df.empty, f"No data returned for {ticker}"
    assert df.isnull().sum().sum() == 0, "NaN values present in raw data"

    print(f"Downloaded {len(df)} rows of data")
    return df


def compute_returns(df):
    """
    Compute daily log returns from closing prices.
    Log returns are preferred over simple returns as they
    are additive across time and better behaved statistically.

    Args:
        df: DataFrame with Close column

    Returns:
        DataFrame with log_returns column added
    """
    df = df.copy()
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df = df.dropna()

    # Validate returns are sensible
    assert df['log_returns'].abs().max() < 1, "Suspiciously large returns detected"

    return df


def load_data(ticker='SPY', start='2010-01-01', end='2024-01-01'):
    """
    Main function to load and prepare data.
    This is the function you will call from your notebook.

    Args:
        ticker: stock/ETF symbol
        start: start date
        end: end date

    Returns:
        Clean DataFrame ready for feature engineering
    """
    df = download_data(ticker, start, end)
    df = compute_returns(df)
    return df
