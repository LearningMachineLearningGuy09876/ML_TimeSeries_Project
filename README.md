# Time Series Forecasting — Sales with ARIMA

> End-to-end time series analysis pipeline on a sales dataset: stationarity testing with the Augmented Dickey-Fuller test, seasonal decomposition, autocorrelation analysis, and a 60-step ahead forecast using `auto_arima` — covering the full diagnostic-to-forecast workflow that precedes any time series model.

---

## Problem

Forecast future sales values from a historical time series. Time series data breaks a core assumption of most ML models — that observations are independent. Sales at time *t* are correlated with sales at *t-1*, *t-2*, and so on. ARIMA models this autocorrelation explicitly and uses it to make forecasts.

## Dataset

- **Source:** Sales dataset (4Geeks / breathecode)
- **Structure:** Date-indexed time series of sales figures
- **Target:** `sales` — a single univariate series set as the DataFrame index after `pd.to_datetime()` parsing

## Analysis Pipeline

| Step | Tool | Finding |
|---|---|---|
| Visual inspection | `sns.lineplot` | Clear upward trend; series does not revert to a mean |
| Seasonal decomposition | `seasonal_decompose` | Trend confirmed; **no meaningful seasonal component** found |
| Stationarity test | ADF test (`adfuller`) | p-value = **0.98** — fail to reject H₀ → series is **non-stationary** |
| Autocorrelation | `plot_acf` | High autocorrelation throughout, slowly declining with lag |
| Model selection | `auto_arima` (pmdarima) | Searches (p, d, q) space automatically; seasonal=False, m=7 |
| Forecast | `model.predict(60)` | 60-step ahead forecast plotted against historical data |

## Key Diagnostics

**ADF Test (Augmented Dickey-Fuller):**
- H₀: the series has a unit root (is non-stationary)
- p-value = 0.98 >> 0.05 → fail to reject H₀ → **non-stationary confirmed**
- This means the series has no stable mean to revert to — it must be differenced before fitting a standard ARIMA

**Seasonal decomposition:**
- Trend component: strong, persistent upward slope
- Seasonal component: flat — no repeating seasonal pattern in this dataset
- Residual component: random noise remaining after removing trend

**ACF plot:**
- High positive autocorrelation at all lags, gradually decreasing
- Characteristic of a non-stationary series with strong memory — consistent with ADF result

**auto_arima:**
- Automatically searches ARIMA(p, d, q) combinations using AIC minimisation
- `d` (differencing order) determined data-driven from stationarity tests
- Selects the most parsimonious model that fits the autocorrelation structure

## Key Takeaways

- **Stationarity is a prerequisite, not an assumption to skip:** ARIMA requires the series to be stationary (constant mean, constant variance). p=0.98 on the ADF test makes non-stationarity unmistakable — differencing is mandatory before fitting.
- **Seasonal decomposition is diagnostic, not just visual:** Decomposing the series confirms that the upward drift is a trend, not cyclical behaviour — ruling out seasonal ARIMA (SARIMA) and simplifying the model selection space.
- **auto_arima does the grid search that would otherwise be manual:** Choosing (p, d, q) by hand requires inspecting ACF and PACF plots and iterating. `auto_arima` automates this systematically using information criteria, following the same logic but faster.

## Tech Stack

`Python` · `statsmodels` · `pmdarima` · `pandas` · `Matplotlib` · `Seaborn`

## Run It Locally

```bash
git clone https://github.com/matthewkane-ml/ML_TimeSeries_MTK.git
cd ML_TimeSeries_MTK
pip install -r requirements.txt
jupyter notebook src/TimeSeries.ipynb
```

## What I'd Do Next

- Evaluate forecast accuracy with held-out test data using **MAE** and **RMSE** — a 60-step visual forecast is compelling, but quantified error on unseen data is what makes it credible
- Apply **log transformation** before differencing to stabilise the variance of the upward-trending series
- Compare against a **Facebook Prophet** model — Prophet handles trend changepoints and holidays automatically and often outperforms ARIMA on business sales data with irregular patterns

---

**Author:** Matthew Kane — [LinkedIn](https://www.linkedin.com/in/thomas-k-392094410/) · [GitHub portfolio](https://github.com/matthewkane-ml)
