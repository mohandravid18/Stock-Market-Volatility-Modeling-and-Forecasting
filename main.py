# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf  # For downloading stock data
from arch import arch_model
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_arch
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# ===================================================================
# Create separate folders for plots and data
# ===================================================================
plots_dir = 'volatility_plots'
data_dir = 'data'

os.makedirs(plots_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

print(f"Plots will be saved in: ./{plots_dir}/")
print(f"Dataset will be saved in: ./{data_dir}/\n")

# ===================================================================
# 1. Load Historical Data (AAPL as example - change ticker if needed)
# ===================================================================
ticker = 'AAPL'
print(f"Downloading historical data for {ticker}...")

data = yf.download(ticker, start='2010-01-01', end='2025-12-31', progress=False)
prices = data['Adj Close']

# ===================================================================
# 2. Compute Log Returns
# ===================================================================
returns = np.log(prices / prices.shift(1)).dropna() * 100  # Percentage returns
returns.name = 'Log_Returns_Pct'

# Combine into a clean DataFrame
df = pd.DataFrame({
    'Adj_Close': prices,
    'Log_Returns_Pct': pd.Series(returns, index=returns.index)
}).dropna()

print(f"Data loaded: {len(df)} daily observations")
print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}\n")
print(df['Log_Returns_Pct'].describe())

# ===================================================================
# Save the processed dataset to CSV
# ===================================================================
csv_filename = f"{data_dir}/{ticker}_historical_data_with_returns.csv"
df.to_csv(csv_filename)
print(f"Dataset saved as: {csv_filename}\n")

# Use only returns for modeling
returns = df['Log_Returns_Pct']

# ===================================================================
# 3. EDA: Returns Distribution and Volatility Clustering
# ===================================================================
plt.figure(figsize=(14, 8))

# Histogram + KDE
plt.subplot(2, 2, 1)
sns.histplot(returns, kde=True, bins=100, color='skyblue')
plt.title('Distribution of Log Returns')
plt.xlabel('Log Returns (%)')

# Time series of returns
plt.subplot(2, 2, 2)
returns.plot(color='blue')
plt.title('Daily Log Returns (Volatility Clustering Visible)')
plt.ylabel('Log Returns (%)')

# Absolute returns
plt.subplot(2, 2, 3)
abs_returns = np.abs(returns)
abs_returns.plot(color='purple')
plt.title('Absolute Returns (Clear Volatility Clustering)')
plt.ylabel('|Log Returns| (%)')

# ACF of squared returns
plt.subplot(2, 2, 4)
sm.graphics.tsa.plot_acf(returns**2, lags=40, color='darkred')
plt.title('ACF of Squared Returns')

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '01_eda_volatility_clustering.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved: 01_eda_volatility_clustering.png")

# ===================================================================
# 4. Test for ARCH Effects (Engle's ARCH-LM Test)
# ===================================================================
lm_stat, lm_pval, f_stat, f_pval = het_arch(returns)
print(f"\nEngle's ARCH-LM Test:")
print(f"LM Statistic: {lm_stat:.2f}, p-value: {lm_pval:.6f}")
if lm_pval < 0.05:
    print("→ Significant ARCH effects detected → GARCH modeling justified.")
else:
    print("→ No strong evidence of ARCH effects.")

# ===================================================================
# 5. Fit GARCH(1,1) Model
# ===================================================================
model = arch_model(returns, vol='Garch', p=1, q=1, mean='Zero', dist='Normal')
garch_fit = model.fit(disp='off')

print("\nGARCH(1,1) Model Summary:")
print(garch_fit.summary())

# Conditional Volatility
cond_vol = garch_fit.conditional_volatility

# Plot fitted volatility
plt.figure(figsize=(12, 6))
plt.plot(returns.index, np.abs(returns), label='Absolute Returns', alpha=0.6, color='gray')
plt.plot(cond_vol.index, cond_vol, label='GARCH(1,1) Conditional Volatility', color='red', linewidth=2)
plt.title(f'{ticker} - GARCH(1,1) Fitted Conditional Volatility')
plt.legend()
plt.ylabel('Volatility (%)')
plt.savefig(os.path.join(plots_dir, '02_fitted_volatility.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved: 02_fitted_volatility.png")

# ===================================================================
# 6. Volatility Forecasting (Next 30 Days)
# ===================================================================
forecast_horizon = 30
forecast = garch_fit.forecast(horizon=forecast_horizon)
forecast_vol = np.sqrt(forecast.variance.dropna().iloc[-1])

last_date = returns.index[-1]
forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon)

plt.figure(figsize=(12, 6))
plt.plot(cond_vol.index[-200:], cond_vol[-200:], label='Historical Conditional Volatility', color='blue')
plt.plot(forecast_dates, forecast_vol, label='30-Day GARCH Forecast', color='orange', linestyle='--', linewidth=2)
plt.title(f'{ticker} - 30-Day Volatility Forecast')
plt.legend()
plt.ylabel('Forecasted Volatility (%)')
plt.savefig(os.path.join(plots_dir, '03_volatility_forecast.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved: 03_volatility_forecast.png")

# ===================================================================
# 7. Forecast Evaluation (Rolling Window)
# ===================================================================
split = int(len(returns) * 0.8)
train, test = returns[:split], returns[split:]

rolling_model = arch_model(train, vol='Garch', p=1, q=1, mean='Zero', dist='Normal')
rolling_fit = rolling_model.fit(disp='off')
rolling_forecast = rolling_fit.forecast(horizon=len(test))
forecasted_vol = np.sqrt(rolling_forecast.variance.values[-1, :])

plt.figure(figsize=(12, 6))
plt.plot(test.index, np.abs(test), label='Realized |Returns|', alpha=0.7, color='gray')
plt.plot(test.index, forecasted_vol, label='Rolling GARCH Forecast', color='green', linewidth=2)
plt.title('Out-of-Sample Volatility Forecast Evaluation')
plt.legend()
plt.ylabel('Volatility (%)')
plt.savefig(os.path.join(plots_dir, '04_forecast_evaluation.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved: 04_forecast_evaluation.png")

# ===================================================================
# 8. Risk Management: VaR Comparison
# ===================================================================
confidence_level = 0.05
historical_var = np.percentile(returns, confidence_level * 100)
z_score = sm.stats.stattools.norm_ppf(confidence_level)
garch_var = cond_vol.iloc[-1] * z_score  # Dynamic VaR based on latest volatility

print("\n" + "="*50)
print("RISK MANAGEMENT INSIGHTS")
print("="*50)
print(f"Latest Date: {returns.index[-1].date()}")
print(f"1-Day 95% Historical VaR       : {historical_var:.2f}%")
print(f"1-Day 95% GARCH-based VaR      : {garch_var:.2f}%")
print(f"Current Conditional Volatility : {cond_vol.iloc[-1]:.2f}%")
print("→ GARCH VaR adjusts dynamically to recent market conditions.")
print("="*50)

# ===================================================================
# Final Confirmation
# ===================================================================
print(f"\nProject Complete!")
print(f"All 4 plots saved in: './{plots_dir}/'")
print(f"Processed dataset saved as: './{data_dir}/{ticker}_historical_data_with_returns.csv'")
print(f"\nYou can now use the CSV for further analysis and plots for reports/presentations.")