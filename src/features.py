import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def add_moving_averages(df, windows=[20, 50]):
    """
    Add simple moving averages for given window lengths.
    All calculations use only past data — no look-ahead bias.
    """
    df = df.copy()
    for w in windows:
        df[f'sma_{w}'] = df['Close'].rolling(w).mean()
    return df


def add_momentum(df, windows=[20]):
    """
    Momentum: cumulative log return over the past N days.
    Positive momentum = upward trend, negative = downward.
    """
    df = df.copy()
    for w in windows:
        df[f'momentum_{w}'] = df['log_returns'].rolling(w).sum()
    return df


def add_volatility(df, window=20):
    """
    Rolling volatility: standard deviation of log returns.
    Used for position sizing and regime detection.
    """
    df = df.copy()
    df[f'volatility_{window}'] = df['log_returns'].rolling(window).std()
    return df


def add_all_features(df, ma_windows=[20, 50], mom_windows=[20], vol_window=20):
    """
    Master function to add all features at once.
    Call this from your notebook rather than individual functions.
    """
    df = add_moving_averages(df, ma_windows)
    df = add_momentum(df, mom_windows)
    df = add_volatility(df, vol_window)
    df = df.dropna()
    return df


def adf_test(series, name=''):
    """
    Augmented Dickey-Fuller test for mean reversion.
    Tests whether a time series is stationary (mean-reverting).

    H0: series has a unit root (not mean-reverting)
    H1: series is stationary (mean-reverting)

    p < 0.05: reject H0, series is likely mean-reverting
    p > 0.05: fail to reject H0, series is not mean-reverting

    Args:
        series: pandas Series to test
        name: label for printing

    Returns:
        dict with test results
    """
    result = adfuller(series.dropna())
    output = {
        'series': name,
        'test_statistic': round(result[0], 4),
        'p_value': round(result[1], 4),
        'is_stationary': result[1] < 0.05,
        'critical_values': result[4]
    }
    print(f"\nADF Test — {name}")
    print(f"  Test statistic : {output['test_statistic']}")
    print(f"  p-value        : {output['p_value']}")
    print(f"  Mean-reverting : {output['is_stationary']}")
    return output


def hurst_exponent(series, max_lag=100):
    """
    Hurst exponent — from Chan's Algorithmic Trading book.
    Measures the long-term memory of a time series.

    H < 0.5: mean reverting (series tends to return to mean)
    H = 0.5: random walk (no memory, unpredictable)
    H > 0.5: trending (momentum, past predicts future direction)

    Args:
        series: pandas Series of prices or returns
        max_lag: maximum lag to consider

    Returns:
        float: Hurst exponent
    """
    series = series.dropna().values
    lags = range(2, max_lag)
    tau = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]
    reg = np.polyfit(np.log(lags), np.log(tau), 1)
    hurst = reg[0]
    print(f"\nHurst Exponent: {hurst:.4f}")
    if hurst < 0.5:
        print("  Interpretation: Mean reverting")
    elif hurst > 0.5:
        print("  Interpretation: Trending")
    else:
        print("  Interpretation: Random walk")
    return hurst
