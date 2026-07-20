# quant-backtester

# Quantitative Trading Backtester 

A systematic backtesting framework built in Python to evaluate 
quantitative trading strategies on historical market data. 
Developed as an independent summer project to demonstrate quantitative 
research methodology, statistical analysis, and software 
engineering skills.

---

## Project Overview

This project implements a full backtesting pipeline from raw 
price data through to strategy evaluation and performance 
reporting. The framework is designed with a rigorous methodology,
taken from Ernest Chan's book "Algorithmic Trading," 
explicitly avoiding common backtesting pitfalls 
such as look-ahead bias, survivorship bias, and in-sample 
overfitting.

The primary asset under investigation is SPY (S&P 500 ETF), 
covering the period 2010–2024, providing exposure to multiple 
market regimes including bull markets, the 2020 COVID crash, 
and the 2022 bear market.

---

## Project Structure

quant-backtester/
├── data/               # Raw and processed data storage
├── src/
│   ├── data_pipeline.py    # Data download, validation, returns
│   ├── features.py         # Feature engineering, ADF test,
│   │                       # Hurst exponent
│   ├── signals.py          # Signal generation
│   ├── backtester.py       # Core backtesting engine
│   ├── metrics.py          # Performance metrics
│   └── visualisation.py    # Plotting and reporting
├── notebooks/
│   └── exploration.ipynb   # Analysis and results
├── requirements.txt
└── README.md

---

## Statistical Analysis — Asset Selection

Before implementing any strategy, SPY was tested for mean 
reversion using two established methods from Chan (2013): 
the Augmented Dickey-Fuller (ADF) test and the Hurst exponent.

### Augmented Dickey-Fuller Test

| Series | Test Statistic | p-value | Mean-Reverting |
|---|---|---|---|
| SPY Price | 0.6301 | 0.9883 | No |
| SPY Log Returns | -12.7941 | 0.0000 | Yes |

The SPY price series yields p = 0.9883, far above the 0.05 
significance threshold. We fail to reject the null hypothesis 
of a unit root — the price series is non-stationary and 
unsuitable for mean reversion strategies.

The log returns series yields p ≈ 0.0, confirming returns 
are stationary and bounce around a stable mean, consistent 
with efficient market behaviour at the daily level.

### Hurst Exponent

| Series | Hurst Exponent | Interpretation |
|---|---|---|
| SPY Price | 0.4076 | Weakly mean reverting |
| SPY Log Returns | -0.0011 | Random walk |

The Hurst exponent on the price series (H = 0.4076) is 
below 0.5 but only marginally, indicating insufficient mean 
reversion to support a profitable mean reversion strategy. 
The returns series (H ≈ 0) confirms near-random walk 
behaviour at the daily level.

### Strategy Selection

Based on this statistical analysis, a **momentum-based moving 
average crossover strategy** was selected as the appropriate 
approach for SPY. The mildly trending nature of the price 
series (confirmed by the long-run upward drift visible in 
the price chart) makes trend-following more statistically 
justified than mean reversion for this asset.

This follows the methodology outlined in Chan, E. (2013), 
*Algorithmic Trading: Winning Strategies and Their Rationale*.

---

## Data

- **Source**: Yahoo Finance via the `yfinance` Python library
- **Asset**: SPY (SPDR S&P 500 ETF Trust)
- **Period**: January 2010 — January 2024
- **Frequency**: Daily OHLCV data, adjusted for splits and 
  dividends
- **Observations**: 3,472 trading days after feature 
  engineering

### Key Data Characteristics

| Metric | Value |
|---|---|
| Mean daily log return | 0.0005 |
| Daily volatility | 0.0109 |
| Annualised volatility | 17.38% |
| Skewness | -0.7178 |
| Kurtosis | 11.51 |

The return distribution exhibits significant negative skew 
(-0.72) and excess kurtosis (11.51 vs 3.0 for a normal 
distribution), confirming fat-tailed behaviour. Large 
negative moves occur more frequently than a normal 
distribution would predict — a well-established empirical 
property of equity returns with important implications for 
risk measurement.

---

## Methodology

### Look-Ahead Bias Prevention

All signals are generated using only information available 
at the time of the decision. Specifically, signals computed 
from end-of-day prices on day T are shifted forward by one 
day and executed at day T+1. This is implemented via 
`.shift(1)` in the signal generation module and is the 
single most important correctness requirement in backtesting.

### Walk-Forward Testing

*(To be completed — in progress)*

### Transaction Costs

*(To be completed — in progress)*

---

## Results

*(To be completed as strategy is implemented)*

---

## Dependencies
