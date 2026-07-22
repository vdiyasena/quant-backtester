import pandas as pd
import numpy as np


def run_backtest(df, signal, transaction_cost=0.001):
    """
    Core backtesting engine.

    Takes a DataFrame of price/return data and a signal series,
    simulates trading and returns a results DataFrame.

    Methodology:
        - Strategy return on day T = signal(T) * log_return(T)
        - Signal on day T was generated from day T-1 prices
          (look-ahead bias already prevented in signals.py via shift(1))
        - Transaction cost applied on every position change
        - Transaction cost of 0.001 = 0.1% per trade (realistic for ETFs)

    Args:
        df: DataFrame with log_returns column
        signal: Series of signals (-1, 0, 1), already shifted
        transaction_cost: fraction of position value charged per trade

    Returns:
        DataFrame with strategy and market returns
    """
    results = df.copy()
    results['signal'] = signal

    # Drop rows where signal is NaN (first row after shift)
    results = results.dropna(subset=['signal', 'log_returns'])

    # Raw strategy returns = signal * market return
    results['strategy_returns'] = results['signal'] * results['log_returns']

    # Identify position changes — this is when we incur transaction costs
    results['position_change'] = results['signal'].diff().abs()

    # Apply transaction costs on position changes
    # A full reversal from -1 to +1 has position_change = 2
    # so cost = 2 * transaction_cost, which is correct
    results['transaction_costs'] = results['position_change'] * transaction_cost
    results['strategy_returns_net'] = (results['strategy_returns']
                                       - results['transaction_costs'])

    # Cumulative returns — use exp(cumsum) of log returns
    # This is mathematically equivalent to compounding daily returns
    results['cumulative_market'] = (results['log_returns']
                                    .cumsum()
                                    .apply(np.exp))
    results['cumulative_strategy'] = (results['strategy_returns_net']
                                      .cumsum()
                                      .apply(np.exp))

    # Number of trades (position changes)
    n_trades = int(results['position_change'].sum() / 2)
    total_costs = results['transaction_costs'].sum()

    print(f"Backtest complete:")
    print(f"  Period        : {results.index[0].date()} to "
          f"{results.index[-1].date()}")
    print(f"  Trading days  : {len(results)}")
    print(f"  Num trades    : {n_trades}")
    print(f"  Total costs   : {total_costs:.4f} ({total_costs*100:.2f}%)")

    return results


def split_in_out_sample(df, split_date='2019-01-01'):
    """
    Split data into in-sample (training) and out-of-sample (testing) periods.

    This is the most important methodological step in backtesting.
    Strategy parameters are optimised on in-sample data only.
    Final results are reported on out-of-sample data only.

    Using 2010-2018 as in-sample and 2019-2024 as out-of-sample
    gives roughly 70/30 split and ensures the test period includes
    the 2020 COVID crash and 2022 bear market — a genuine stress test.

    Args:
        df: full DataFrame
        split_date: date string to split on

    Returns:
        tuple of (in_sample_df, out_of_sample_df)
    """
    in_sample = df[df.index < split_date]
    out_of_sample = df[df.index >= split_date]

    print(f"In-sample     : {in_sample.index[0].date()} to "
          f"{in_sample.index[-1].date()} ({len(in_sample)} days)")
    print(f"Out-of-sample : {out_of_sample.index[0].date()} to "
          f"{out_of_sample.index[-1].date()} ({len(out_of_sample)} days)")

    return in_sample, out_of_sample