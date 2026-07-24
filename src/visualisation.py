import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
import seaborn as sns
import os


def plot_price_and_returns(df, ticker='SPY', save_dir='notebooks'):
    """
    Plot price series and daily log returns.

    References:
        Hull, J.C. (2018). Options, Futures, and Other Derivatives,
        10th ed. Pearson. Chapter 15.

    Args:
        df: DataFrame with Close and log_returns columns
        ticker: asset name for titles
        save_dir: directory to save the figure
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    df['Close'].plot(ax=ax1, color='steelblue', linewidth=0.8)
    ax1.set_title(f'{ticker} Close Price', fontsize=13)
    ax1.set_ylabel('Price ($)')

    df['log_returns'].plot(ax=ax2, color='grey',
                            linewidth=0.6, alpha=0.8)
    ax2.set_title(f'{ticker} Daily Log Returns', fontsize=13)
    ax2.set_ylabel('Log Return')
    ax2.axhline(y=0, color='red', linewidth=0.8, linestyle='--')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'price_and_returns.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")


def plot_returns_distribution(df, ticker='SPY', save_dir='notebooks'):
    """
    Plot distribution of daily log returns with key statistics.
    Highlights fat-tailed nature of financial returns.

    References:
        Hull, J.C. (2018). Options, Futures, and Other Derivatives,
        10th ed. Pearson. Chapter 15.

    Args:
        df: DataFrame with log_returns column
        ticker: asset name for title
        save_dir: directory to save the figure
    """
    os.makedirs(save_dir, exist_ok=True)

    returns = df['log_returns'].dropna()
    mean = returns.mean()
    std = returns.std()

    fig, ax = plt.subplots(figsize=(12, 5))

    returns.hist(bins=100, ax=ax, color='steelblue',
                 edgecolor='white', alpha=0.8)

    ax.axvline(mean, color='red', linestyle='--',
               linewidth=1.5, label=f'Mean: {mean:.4f}')
    ax.axvline(mean + 2*std, color='orange', linestyle='--',
               linewidth=1.2, label=f'+2σ: {mean+2*std:.4f}')
    ax.axvline(mean - 2*std, color='orange', linestyle='--',
               linewidth=1.2, label=f'-2σ: {mean-2*std:.4f}')

    ax.set_title(f'{ticker} Daily Log Return Distribution\n'
                 f'Skewness: {returns.skew():.3f}  |  '
                 f'Kurtosis: {returns.kurtosis():.3f}  |  '
                 f'Annualised Vol: {returns.std()*np.sqrt(252)*100:.1f}%',
                 fontsize=12)
    ax.set_xlabel('Log Return')
    ax.set_ylabel('Frequency')
    ax.legend()

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'returns_distribution.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")


def plot_signals(df, fast=50, slow=200, ticker='SPY',
                 save_dir='notebooks'):
    """
    Plot price with moving averages and trading signal.

    References:
        Murphy, J.J. (1999). Technical Analysis of the Financial
        Markets. New York Institute of Finance. Chapter 9.

        Chan, E. (2013). Algorithmic Trading. Wiley. Chapter 3.

    Args:
        df: DataFrame with Close, sma columns and signal
        fast: fast MA window
        slow: slow MA window
        ticker: asset name
        save_dir: directory to save the figure
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                    sharex=True)

    df['Close'].plot(ax=ax1, color='steelblue',
                     linewidth=0.8, label=f'{ticker} Price')
    df[f'sma_{fast}'].plot(ax=ax1, color='orange',
                            linewidth=1.2,
                            label=f'SMA {fast}')
    df[f'sma_{slow}'].plot(ax=ax1, color='red',
                            linewidth=1.2,
                            label=f'SMA {slow}')

    signal = df['signal']
    for i in range(1, len(signal)):
        if signal.iloc[i] == 1:
            ax1.axvspan(signal.index[i-1], signal.index[i],
                        alpha=0.1, color='green')

    ax1.set_title(f'{ticker} Price with SMA {fast}/{slow} '
                  f'and Long Periods (green shading)',
                  fontsize=12)
    ax1.set_ylabel('Price ($)')
    ax1.legend(loc='upper left')

    signal.plot(ax=ax2, color='green', linewidth=0.8)
    ax2.set_title('Trading Signal (1 = Long, 0 = Flat)',
                  fontsize=12)
    ax2.set_ylabel('Signal')
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_ylim(-0.2, 1.2)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'signals.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")


def plot_cumulative_returns(results_dict, save_dir='notebooks',
                             title='Strategy Comparison'):
    """
    Plot cumulative returns for multiple strategies on one chart.

    References:
        Chan, E. (2013). Algorithmic Trading. Wiley. Chapter 3.

    Args:
        results_dict: dict of {label: results_DataFrame}
        save_dir: directory to save the figure
        title: chart title
    """
    os.makedirs(save_dir, exist_ok=True)

    colours = ['steelblue', 'green', 'orange', 'purple', 'red']

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, (label, results) in enumerate(results_dict.items()):
        colour = colours[i % len(colours)]

        if label == 'Buy & Hold SPY':
            results['cumulative_market'].plot(
                ax=ax, label=label, color=colour,
                linewidth=2.0)
        else:
            results['cumulative_strategy'].plot(
                ax=ax, label=label, color=colour,
                linewidth=1.5)

    ax.set_title(title, fontsize=13)
    ax.set_ylabel('Cumulative Return (1.0 = starting value)')
    ax.set_xlabel('Date')
    ax.legend(loc='upper left')
    ax.axhline(y=1.0, color='black', linewidth=0.5,
               linestyle='--', alpha=0.5)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'cumulative_returns.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")


def plot_drawdown(results_dict, save_dir='notebooks',
                  title='Strategy Drawdown Comparison'):
    """
    Plot drawdown over time for multiple strategies.

    References:
        Magdon-Ismail, M. and Atiya, A. (2004). Maximum Drawdown.
        Risk Magazine, 17(10), 99-102.

    Args:
        results_dict: dict of {label: results_DataFrame}
        save_dir: directory to save the figure
        title: chart title
    """
    os.makedirs(save_dir, exist_ok=True)

    colours = ['steelblue', 'green', 'orange', 'purple']

    fig, ax = plt.subplots(figsize=(14, 5))

    for i, (label, results) in enumerate(results_dict.items()):
        colour = colours[i % len(colours)]

        if label == 'Buy & Hold SPY':
            cum = results['cumulative_market']
        else:
            cum = results['cumulative_strategy']

        rolling_max = cum.cummax()
        drawdown = (cum - rolling_max) / rolling_max

        drawdown.plot(ax=ax, label=label, color=colour,
                      linewidth=1.0, alpha=0.8)
        ax.fill_between(drawdown.index, drawdown, 0,
                         alpha=0.1, color=colour)

    ax.set_title(title, fontsize=13)
    ax.set_ylabel('Drawdown')
    ax.set_xlabel('Date')
    ax.legend(loc='lower left')
    ax.axhline(y=0, color='black', linewidth=0.5)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'drawdown.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")


def plot_monthly_returns_heatmap(results, label='Strategy',
                                  save_dir='notebooks'):
    """
    Plot monthly returns as a heatmap.
    Makes it easy to spot which months and years performed well or poorly.

    References:
        Lopez de Prado, M. (2018). Advances in Financial Machine
        Learning. Wiley. Chapter 14.

    Args:
        results: DataFrame from run_backtest()
        label: strategy name for title
        save_dir: directory to save the figure
    """
    os.makedirs(save_dir, exist_ok=True)

    monthly = (results['strategy_returns_net']
               .resample('ME')
               .sum()
               .apply(np.exp) - 1) * 100

    monthly_df = pd.DataFrame({
        'year': monthly.index.year,
        'month': monthly.index.month,
        'return': monthly.values
    })
    pivot = monthly_df.pivot(index='year',
                              columns='month',
                              values='return')
    pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May',
                     'Jun', 'Jul', 'Aug', 'Sep', 'Oct',
                     'Nov', 'Dec']

    fig, ax = plt.subplots(figsize=(14, 6))

    sns.heatmap(pivot,
                annot=True,
                fmt='.1f',
                cmap='RdYlGn',
                center=0,
                ax=ax,
                linewidths=0.3,
                cbar_kws={'label': 'Return (%)'})

    ax.set_title(f'{label} — Monthly Returns Heatmap (%)',
                 fontsize=13)
    ax.set_xlabel('Month')
    ax.set_ylabel('Year')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'monthly_returns_heatmap.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")


def plot_walk_forward_comparison(results_in, results_out,
                                  label='Golden Cross 50/200',
                                  save_dir='notebooks'):
    """
    Plot in-sample vs out-of-sample cumulative returns side by side.
    Key visualisation for walk-forward testing section.

    References:
        Chan, E. (2013). Algorithmic Trading. Wiley. Chapter 3,
        pp. 67-71.

        Lopez de Prado, M. (2018). Advances in Financial Machine
        Learning. Wiley. Chapter 7.

    Args:
        results_in: in-sample backtest results DataFrame
        results_out: out-of-sample backtest results DataFrame
        label: strategy name
        save_dir: directory to save the figure
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    results_in['cumulative_market'].plot(
        ax=ax1, label='Buy & Hold', color='steelblue',
        linewidth=1.5)
    results_in['cumulative_strategy'].plot(
        ax=ax1, label=label, color='green', linewidth=1.5)
    ax1.set_title('In-Sample (2010-2018)', fontsize=12)
    ax1.set_ylabel('Cumulative Return')
    ax1.legend()
    ax1.axhline(y=1.0, color='black', linewidth=0.5,
                linestyle='--')

    results_out['cumulative_market'].plot(
        ax=ax2, label='Buy & Hold', color='steelblue',
        linewidth=1.5)
    results_out['cumulative_strategy'].plot(
        ax=ax2, label=label, color='green', linewidth=1.5)
    ax2.set_title('Out-of-Sample (2019-2024)', fontsize=12)
    ax2.set_ylabel('Cumulative Return')
    ax2.legend()
    ax2.axhline(y=1.0, color='black', linewidth=0.5,
                linestyle='--')

    fig.suptitle(f'Walk-Forward Test: {label}', fontsize=14,
                 fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'walk_forward_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")


def plot_metrics_comparison(metrics_list, save_dir='notebooks'):
    """
    Bar chart comparing key metrics across strategies.

    References:
        Sharpe, W.F. (1994). The Sharpe Ratio.
        Journal of Portfolio Management, 21(1), 49-58.

        Sortino, F.A. and van der Meer, R. (1991). Downside Risk.
        Journal of Portfolio Management, 17(4), 27-31.

    Args:
        metrics_list: list of metrics dicts from compute_all_metrics()
        save_dir: directory to save the figure
    """
    os.makedirs(save_dir, exist_ok=True)

    labels = [m['label'] for m in metrics_list]
    sharpes = [m['sharpe_ratio'] for m in metrics_list]
    sortinos = [m['sortino_ratio'] for m in metrics_list]
    drawdowns = [abs(m['max_drawdown']) for m in metrics_list]
    ann_returns = [m['annualised_return'] for m in metrics_list]

    colours = ['steelblue', 'green', 'orange', 'purple']

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    axes[0, 0].bar(labels, sharpes, color=colours)
    axes[0, 0].set_title('Sharpe Ratio', fontsize=12)
    axes[0, 0].axhline(y=0, color='black', linewidth=0.5)
    axes[0, 0].tick_params(axis='x', rotation=15)

    axes[0, 1].bar(labels, sortinos, color=colours)
    axes[0, 1].set_title('Sortino Ratio', fontsize=12)
    axes[0, 1].axhline(y=0, color='black', linewidth=0.5)
    axes[0, 1].tick_params(axis='x', rotation=15)

    axes[1, 0].bar(labels, drawdowns, color=colours)
    axes[1, 0].set_title('Max Drawdown (absolute %)', fontsize=12)
    axes[1, 0].tick_params(axis='x', rotation=15)

    axes[1, 1].bar(labels, ann_returns, color=colours)
    axes[1, 1].set_title('Annualised Return (%)', fontsize=12)
    axes[1, 1].axhline(y=0, color='black', linewidth=0.5)
    axes[1, 1].tick_params(axis='x', rotation=15)

    fig.suptitle('Strategy Performance Comparison',
                  fontsize=14, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'metrics_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")