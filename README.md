Markdown

# Stock Market Volatility Modeling and Forecasting with GARCH

## Project Overview
This project demonstrates **volatility modeling and forecasting** for stock prices using the **GARCH(1,1)** model. It downloads historical stock data (default: Apple - AAPL), computes log returns, performs exploratory data analysis (EDA) to visualize volatility clustering, tests for ARCH effects, fits a GARCH model, generates forecasts, evaluates out-of-sample performance, and calculates dynamic Value at Risk (VaR) for risk management.

**Key Highlights:**
- Uses **yfinance** to download real-time historical stock data.
- Implements **GARCH(1,1)** via the `arch` library.
- Includes **Engle's ARCH-LM test** to justify conditional heteroskedasticity modeling.
- Generates **30-day volatility forecasts**.
- Compares **historical VaR** vs **dynamic GARCH-based VaR**.
- Saves all plots and processed dataset in organized folders.

## Dataset
- **Source**: Yahoo Finance (via `yfinance`)
- **Ticker**: `AAPL` (Apple Inc.) by default — easily changeable
- **Period**: January 1, 2010 – December 31, 2025
- **Output**: Daily adjusted closing prices and log returns (in percentage)

## Project Structure

.
├── volatility_plots/
│   ├── 01_eda_volatility_clustering.png     # Returns distribution, clustering, ACF
│   ├── 02_fitted_volatility.png             # GARCH fitted vs absolute returns
│   ├── 03_volatility_forecast.png           # 30-day volatility forecast
│   └── 04_forecast_evaluation.png           # Out-of-sample rolling forecast
├── data/
│   └── AAPL_historical_data_with_returns.csv # Processed dataset with prices & returns
├── volatility_modeling.py                   # Main script
└── README.md                                # This file
text

## Requirements
Install the required packages:
```bash
pip install pandas numpy matplotlib seaborn yfinance arch statsmodels

How to Run

    Save the script as volatility_modeling.py
    (Optional) Change the ticker variable to any stock symbol (e.g., 'TSLA', 'MSFT', '^GSPC' for S&P 500)
    Run the script:

Bash

python volatility_modeling.py

    Output:
        Automatically downloads data
        Creates volatility_plots/ and data/ folders
        Saves 4 high-quality plots and the processed CSV
        Prints GARCH summary, ARCH test results, and risk management insights

Key Outputs

    EDA Plot: Shows volatility clustering and leptokurtic returns distribution
    ARCH Test: Confirms presence of conditional heteroskedasticity (p-value typically << 0.05)
    GARCH Fit: Captures time-varying volatility effectively
    Forecast: 30-day ahead conditional volatility
    Evaluation: Rolling forecast vs realized volatility on test set
    VaR Comparison:
        Historical VaR (static)
        GARCH VaR (dynamic — adapts to current volatility regime)

Example Risk Insights (Typical Output)
text

==================================================
RISK MANAGEMENT INSIGHTS
==================================================
Latest Date: 2025-12-30
1-Day 95% Historical VaR : -3.21%
1-Day 95% GARCH-based VaR : -2.85%
Current Conditional Volatility : 1.73%
→ GARCH VaR adjusts dynamically to recent market conditions.
==================================================

Customization

    Change ticker to model any stock or index
    Adjust forecast_horizon for longer/shorter forecasts
    Modify train/test split for different evaluation periods

Applications

    Portfolio risk management
    Option pricing (volatility input)
    Trading strategy adjustment during high-volatility regimes
    Stress testing and capital allocation

Future Enhancements

    Implement EGARCH or TGARCH for asymmetry/leverage effects
    Compare multiple GARCH variants (GJR-GARCH, etc.)
    Multivariate GARCH (DCC-GARCH) for portfolio volatility
    Integrate with live trading signals
