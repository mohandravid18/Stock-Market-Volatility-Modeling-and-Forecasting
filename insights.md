# Volatility Modeling & Risk Analysis – Insights

This project builds an end‑to‑end **volatility modeling and risk analysis pipeline** for AAPL using historical prices, GARCH(1,1), and Value at Risk (VaR).

---

## 1. Data & Preprocessing

- **Data source:** Daily historical prices for Apple (AAPL) downloaded via `yfinance` from 2010‑01‑01 to 2025‑12‑30 (4,022 trading days).  
- **Price used:** Adjusted close prices (`Adj_Close`) to account for splits/dividends.  
- **Returns:** Computed **daily log returns in percentage**:

  \[
  r_t = 100 \times \ln\left(\frac{P_t}{P_{t-1}}\right)
  \]

- The final modelling series is `Log_Returns_Pct` with no missing values.

**Key descriptive stats (returns):**

- Mean: ~0.09% per day  
- Std dev: ~1.77% per day  
- Min / Max: about −13.77% to +14.26%  

This confirms **fat tails and high variability**, typical for equity returns.

---

## 2. Volatility EDA & ARCH Effects

Generated `01_eda_volatility_clustering.png` with four panels:

1. **Histogram + KDE of returns**  
   - Distribution is **leptokurtic** (fat tails) with more extreme moves than a normal distribution.
2. **Time series of daily returns**  
   - Shows periods of calm and turbulence, indicating **volatility clustering**.
3. **Absolute returns over time**  
   - Makes volatility clustering very clear: large |returns| tend to cluster in time.
4. **ACF of squared returns**  
   - Significant autocorrelation in squared returns → persistence in volatility.

**Engle’s ARCH‑LM test:**
<img width="1179" height="781" alt="01_eda_volatility_clustering" src="https://github.com/user-attachments/assets/586784c6-6757-4833-ac1b-e6fb9eb8e829" />

- LM statistic ≈ 403.15, p‑value ≈ 0.000000  
- Interpretation: **strong and highly significant ARCH effects**, so modeling conditional heteroskedasticity (e.g., via GARCH) is justified.

---

## 3. GARCH(1,1) Volatility Model

Fitted a **GARCH(1,1)** model with zero mean and normal errors:

- Model: \( r_t = 0 + \varepsilon_t \), with  
  \[
  \varepsilon_t \sim N(0, h_t), \quad
  h_t = \omega + \alpha \varepsilon_{t-1}^2 + \beta h_{t-1}
  \]

**Estimated parameters (approx):**

- \(\omega \approx 0.1607\)  
- \(\alpha_1 \approx 0.0937\)  
- \(\beta_1 \approx 0.8546\)  

**Insights:**

- \(\alpha_1 + \beta_1 \approx 0.948\) → volatility is **highly persistent**.  
- Shocks to volatility decay slowly: after large moves, the market stays volatile for some time.  
- All parameters are statistically significant (very small p‑values).
<img width="1993" height="1054" alt="02_fitted_volatility" src="https://github.com/user-attachments/assets/cb780275-5e8d-4141-bae9-fb0a239e1dea" />

**Plot `02_fitted_volatility.png`:**

- Grey line: |returns|.  
- Red line: GARCH conditional volatility.  
- Shows that model‑estimated volatility closely tracks clusters of large absolute returns.

---

## 4. Volatility Forecasting

### 30‑Day Ahead Volatility

- Used the fitted GARCH(1,1) to forecast **30‑day conditional variance** and converted to volatility (square root).  
- Plot `03_volatility_forecast.png`:
  - Blue: recent historical conditional volatility (last ~200 days).
  - Orange dashed: 30‑<img width="1976" height="1054" alt="03_volatility_forecast" src="https://github.com/user-attachments/assets/2e37aa20-89b7-4f7d-9194-d9db6b454795" />
day GARCH volatility forecast.
  - 
- The forecast **reverts towards a long‑run volatility level**, but still depends on the latest estimated volatility:
  - After calm periods → lower forecasted volatility.
  - After turbulent periods → higher forecasted volatility.

### Out‑of‑Sample Evaluation (Rolling)

- Split data: 80% train, 20% test.  
- Fitted GARCH(1,1) on train and forecasted volatility over the test window.
- Plot `04_forecast_evaluation.png`:
  - Grey: realized |returns| (as a proxy for realized volatility).
  - Green: rolling GARCH forecasted volatility.
  - ![Uploading 04_forecast_evaluation.png…]()

- Visual result: GARCH forecasts **reasonably track broad swings** in realized volatility but, as expected, are smoother and sometimes under/over‑react to extreme spikes.

---

## 5. Risk Management & VaR

Computed **1‑day 95% Value at Risk (VaR)** using two approaches:

1. **Historical VaR:**  
   - 5th percentile of historical returns ⇒ approx **−2.71%**.  
   - Interpretation: based purely on empirical distribution of past returns.

2. **GARCH‑based VaR:**  
   - Uses latest conditional volatility and normal quantile at 5%:

     \[
     \text{VaR}_{0.95} = \sigma_{\text{t}} \cdot z_{0.05}
     \]

   - Latest conditional volatility ≈ **1.17%**.  
   - GARCH VaR ≈ **−1.92%**.

**Key insight:**

- **Historical VaR** captures empirical tail behavior, including fat tails and crisis periods.  
- **GARCH VaR** is **dynamic**: it scales risk estimates up or down depending on recent volatility.  
- When recent volatility is low, GARCH VaR < historical VaR; during stress periods, GARCH VaR would increase significantly, making it more responsive for **real‑time risk monitoring**.

Console summary:

- Latest Date: 2025‑12‑30  
- 1‑Day 95% Historical VaR: −2.71%  
- 1‑Day 95% GARCH‑based VaR: −1.92%  
- Current Conditional Volatility: 1.17%  
- Conclusion: **“GARCH VaR adjusts dynamically to recent market conditions.”**

---

## 6. What This Project Demonstrates

- Ability to **work with real market data** (AAPL) end‑to‑end: download, clean, transform, and store.  
- Understanding of **volatility clustering, ARCH effects, and GARCH family models**.  
- Practical skills in:
  - Engle’s ARCH‑LM test for validating volatility models.
  - Fitting and interpreting GARCH(1,1) parameters.
  - Volatility forecasting and out‑of‑sample evaluation.
  - Translating model output into **risk metrics** (VaR) useful for trading and risk teams.

All plots are saved to `./volatility_plots/` and the processed dataset to `./data/AAPL_historical_data_with_returns.csv` for further analysis and reporting.
