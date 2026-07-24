import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def annualised_return(returns, periods_per_year=252):
    """
    Annualised return from a series of daily log returns.
    
    Args:
        returns: Series of daily log returns
        periods_per_year: 252 trading days in a year
    
    Returns:
        float: annualised return
    """
    total_return = returns.sum()
    n_years = len(returns) / periods_per_year
    return np.exp(total_return / n_years) - 1


def annualised_volatility(returns, periods_per_year=252):
    """
    Annualised volatility (standard deviation of returns).
    
    Args:
        returns: Series of daily log returns
        periods_per_year: 252 trading days
    
    Returns:
        float: annualised volatility
    """
    return returns.std() * np.sqrt(periods_per_year)


def sharpe_ratio(returns, risk_free_rate=0.02, periods_per_year=252):
    """
    Sharpe ratio — risk-adjusted return.
    Measures excess return per unit of total volatility.
    
    Sharpe = (Annualised Return - Risk Free Rate) / Annualised Volatility
    
    Above 1.0 is generally considered acceptable.
    Above 2.0 is considered very good.
    Above 3.0 is exceptional and rare.
    
    Note: Sharpe assumes normally distributed returns. Given the
    fat-tailed distribution observed in SPY returns (kurtosis = 11.5),
    the Sortino ratio is a more appropriate risk measure.
    
    Args:
        returns: Series of daily log returns
        risk_free_rate: annual risk-free rate (default 2%)
        periods_per_year: 252 trading days
    
    Returns:
        float: Sharpe ratio
    """
    ann_return = annualised_return(returns, periods_per_year)
    ann_vol = annualised_volatility(returns, periods_per_year)
    
    if ann_vol == 0:
        return 0.0
    
    return (ann_return - risk_free_rate) / ann_vol


def sortino_ratio(returns, risk_free_rate=0.02, periods_per_year=252):
    """
    Sortino ratio — downside risk-adjusted return.
    Like Sharpe but only penalises downside volatility.
    
    More appropriate than Sharpe for fat-tailed return distributions
    like those observed in equity markets (kurtosis >> 3).
    
    Sortino = (Annualised Return - Risk Free Rate) / Downside Deviation
    
    Args:
        returns: Series of daily log returns
        risk_free_rate: annual risk-free rate
        periods_per_year: 252 trading days
    
    Returns:
        float: Sortino ratio
    """
    ann_return = annualised_return(returns, periods_per_year)
    
    # Downside deviation — only negative returns contribute
    downside_returns = returns[returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(periods_per_year)
    
    if downside_deviation == 0:
        return 0.0
    
    return (ann_return - risk_free_rate) / downside_deviation


def max_drawdown(cumulative_returns):
    """
    Maximum drawdown — largest peak to trough decline.
    Measures the worst case loss an investor could have experienced.
    
    Args:
        cumulative_returns: Series of cumulative returns
    
    Returns:
        float: maximum drawdown (negative number)
    """
    rolling_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    return drawdown.min()


def calmar_ratio(returns, cumulative_returns, periods_per_year=252):
    """
    Calmar ratio — return relative to maximum drawdown.
    Measures how much return you get per unit of worst-case loss.
    
    Calmar = Annualised Return / |Maximum Drawdown|
    
    Args:
        returns: Series of daily log returns
        cumulative_returns: Series of cumulative returns
        periods_per_year: 252 trading days
    
    Returns:
        float: Calmar ratio
    """
    ann_return = annualised_return(returns, periods_per_year)
    mdd = max_drawdown(cumulative_returns)
    
    if mdd == 0:
        return 0.0
    
    return ann_return / abs(mdd)


def win_rate(returns, signal):
    """
    Fraction of trading days where the strategy made money.
    
    Args:
        returns: Series of strategy returns
        signal: Series of signals
    
    Returns:
        float: win rate between 0 and 1
    """
    # Only count days where we had an active position
    active = returns[signal != 0]
    if len(active) == 0:
        return 0.0
    return (active > 0).sum() / len(active)


def compute_all_metrics(results, label='Strategy'):
    """
    Compute and print all performance metrics for a backtest result.
    
    Args:
        results: DataFrame from run_backtest()
        label: name to display
    
    Returns:
        dict of all metrics
    """
    returns = results['strategy_returns_net']
    market_returns = results['log_returns']
    cum_strategy = results['cumulative_strategy']
    cum_market = results['cumulative_market']
    signal = results['signal']

    metrics = {
        'label': label,
        
        # Returns
        'total_return': (cum_strategy.iloc[-1] - 1) * 100,
        'annualised_return': annualised_return(returns) * 100,
        'benchmark_return': (cum_market.iloc[-1] - 1) * 100,
        
        # Risk
        'annualised_volatility': annualised_volatility(returns) * 100,
        'max_drawdown': max_drawdown(cum_strategy) * 100,
        
        # Risk-adjusted
        'sharpe_ratio': sharpe_ratio(returns),
        'sortino_ratio': sortino_ratio(returns),
        'calmar_ratio': calmar_ratio(returns, cum_strategy),
        
        # Trading activity
        'win_rate': win_rate(returns, signal) * 100,
        'n_trades': int(results['position_change'].sum() / 2),
    }

    # Pretty print
    print(f"\n{'='*45}")
    print(f"  Performance Metrics — {label}")
    print(f"{'='*45}")
    print(f"  Total Return          : {metrics['total_return']:.1f}%")
    print(f"  Benchmark Return      : {metrics['benchmark_return']:.1f}%")
    print(f"  Annualised Return     : {metrics['annualised_return']:.1f}%")
    print(f"  Annualised Volatility : {metrics['annualised_volatility']:.1f}%")
    print(f"  Max Drawdown          : {metrics['max_drawdown']:.1f}%")
    print(f"  Sharpe Ratio          : {metrics['sharpe_ratio']:.3f}")
    print(f"  Sortino Ratio         : {metrics['sortino_ratio']:.3f}")
    print(f"  Calmar Ratio          : {metrics['calmar_ratio']:.3f}")
    print(f"  Win Rate              : {metrics['win_rate']:.1f}%")
    print(f"  Number of Trades      : {metrics['n_trades']}")
    print(f"{'='*45}")

    return metrics

# Defining Monte Carlo significance test for Sharpe ratio

def monte_carlo_sharpe(results, n_simulations=10000,
                        risk_free_rate=0.02, seed=42):
    """
    Monte Carlo permutation test for Sharpe ratio significance.

    Tests whether the strategy's timing skill produces a Sharpe
    ratio distinguishable from random signal timing.

    If the observed Sharpe exceeds the 95th percentile of the
    null distribution, the strategy's timing adds genuine value
    beyond random entry/exit (p < 0.05).

    Args:
        results: DataFrame from run_backtest()
        n_simulations: number of random shuffles
        risk_free_rate: annual risk-free rate
        seed: random seed for reproducibility

    Returns:
        dict with observed Sharpe, null distribution, and p-value

    Reference:
        Lopez de Prado, M. (2018). Advances in Financial Machine
        Learning. Wiley. Chapter 8.
    """
    np.random.seed(seed)

    market_returns = results['log_returns'].dropna()
    signal = results['signal'].dropna()
    transaction_cost = 0.001

    # Align index
    common_idx = market_returns.index.intersection(signal.index)
    market_returns = market_returns.loc[common_idx]
    signal = signal.loc[common_idx]

    # Observed Sharpe using actual signal
    actual_strategy = signal * market_returns
    position_changes = signal.diff().abs()
    actual_strategy -= position_changes * transaction_cost
    observed_sharpe = sharpe_ratio(actual_strategy, risk_free_rate)

    # Generate null distribution by shuffling SIGNAL
    # This preserves the same fraction of long/flat days
    # but randomises WHEN the strategy is in the market
    signal_array = signal.values
    null_sharpes = []

    for _ in range(n_simulations):
        # Shuffle signal randomly
        shuffled_signal = np.random.permutation(signal_array)
        shuffled_signal_series = pd.Series(
            shuffled_signal, index=signal.index)

        # Compute strategy returns with shuffled signal
        strat_returns = shuffled_signal_series * market_returns
        costs = shuffled_signal_series.diff().abs() * transaction_cost
        strat_returns -= costs

        sr = sharpe_ratio(strat_returns, risk_free_rate)
        if np.isfinite(sr):
            null_sharpes.append(sr)

    null_sharpes = np.array(null_sharpes)

    # p-value: fraction of simulations that beat observed Sharpe
    p_value = (null_sharpes >= observed_sharpe).mean()
    percentile = (null_sharpes < observed_sharpe).mean() * 100

    return {
        'observed_sharpe': observed_sharpe,
        'null_sharpes': null_sharpes,
        'p_value': p_value,
        'percentile': percentile,
        'mean_null': null_sharpes.mean(),
        'std_null': null_sharpes.std(),
        'significant': p_value < 0.05
    }

def plot_monte_carlo(mc_results, label='Strategy', 
                     save_dir='figures'):
    """
    Plot Monte Carlo null distribution vs observed Sharpe ratio.
    
    Args:
        mc_results: dict from monte_carlo_sharpe()
        label: strategy name
        save_dir: directory to save figure
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    null = mc_results['null_sharpes']

    null = null[np.isfinite(null)]  # Remove any NaN or inf values

    observed = mc_results['observed_sharpe']
    
    # Plot null distribution
    ax.hist(null, bins=100, color='steelblue', 
            edgecolor='white', alpha=0.7,
            label='Null distribution (shuffled returns)')
    
    # Observed Sharpe
    ax.axvline(observed, color='red', linewidth=2,
               label=f'Observed Sharpe: {observed:.3f}')
    
    # 95th percentile
    p95 = np.percentile(null, 95)
    ax.axvline(p95, color='orange', linewidth=1.5,
               linestyle='--',
               label=f'95th percentile: {p95:.3f}')
    
    # Shade significant region
    x_fill = null[null >= p95]
    ax.hist(x_fill, bins=100, color='orange', 
            alpha=0.4, edgecolor='none')
    
    ax.set_title(
        f'Monte Carlo Significance Test — {label}\n'
        f'p-value: {mc_results["p_value"]:.4f}  |  '
        f'Percentile: {mc_results["percentile"]:.1f}%  |  '
        f'Significant: {mc_results["significant"]}',
        fontsize=12)
    ax.set_xlabel('Sharpe Ratio')
    ax.set_ylabel('Frequency')
    ax.legend()
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 
                              'monte_carlo_significance.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")