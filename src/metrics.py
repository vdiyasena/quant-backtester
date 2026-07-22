import numpy as np
import pandas as pd


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