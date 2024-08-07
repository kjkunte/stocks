import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

def find_best_arima(data, p_range, d_range, q_range):
    """
    Finds the best ARIMA model based on AIC and BIC.

    Args:
        data: Time series data.
        p_range: Range of AR orders to test.
        d_range: Range of differencing orders to test.
        q_range: Range of MA orders to test.

    Returns:
        Best ARIMA model.
    """
    best_aic = float("inf")
    best_bic = float("inf")
    best_model = None

    for p in p_range:
        for d in d_range:
            for q in q_range:
                try:
                    model = ARIMA(data, order=(p, d, q))
                    model_fit = model.fit()
                    aic = model_fit.aic
                    bic = model_fit.bic
                    if aic < best_aic:
                        best_aic = aic
                        best_model = model_fit
                    if bic < best_bic:
                        best_bic = bic
                        best_model = model_fit
                except:
                    continue

    return best_model

# Define parameters
ticker = "HDFCBANK.NS"
start_date = "2018-01-01"
end_date = "2023-01-01"
forecast_days = 20
num_simulations = 100

# Fetch data
data = yf.download(ticker, start=start_date, end=end_date)
prices = data['Close']

# Experiment with different ARIMA orders
p_range = range(0, 3)
d_range = range(0, 2)
q_range = range(0, 3)

# Find the best model
best_model = find_best_arima(prices, p_range, d_range, q_range)

# Forecast using ARIMA
arima_forecast = best_model.forecast(steps=forecast_days)
arima_pred_mean = arima_forecast

# Calculate residuals from ARIMA model
residuals = best_model.resid

# Fit GBM model on residuals
log_returns = np.log(1 + residuals.pct_change().dropna())
u = log_returns.mean()
sigma = log_returns.std()
drift = u - (0.5 * sigma**2)

# Generate GBM paths for residuals
Z = np.random.standard_normal((forecast_days, num_simulations))
daily_returns = np.exp(drift + sigma * Z)
gbm_residuals = np.zeros_like(daily_returns)
gbm_residuals[0] = residuals.iloc[-1]  # Start with the last residual value

for t in range(1, forecast_days):
    gbm_residuals[t] = gbm_residuals[t-1] * daily_returns[t]

# Combine ARIMA forecast with GBM residuals
price_paths = np.zeros_like(gbm_residuals)
for i in range(num_simulations):
    price_paths[:, i] = arima_pred_mean + gbm_residuals[:, i]

# Evaluate the forecast
# Note: Ensure you have enough data to compare forecast with actual prices
actual_prices = yf.download(ticker, start=end_date, end="2023-01-31")['Close']

# Align the lengths for evaluation
aligned_forecast = arima_pred_mean[:len(actual_prices)]
aligned_actual = actual_prices[:len(arima_pred_mean)]

mse = mean_squared_error(aligned_actual, aligned_forecast)
mape = mean_absolute_percentage_error(aligned_actual, aligned_forecast)

print("Best ARIMA Model:", best_model.summary())
print("MSE:", mse)
print("MAPE:", mape)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(prices, label='Historical Prices')
plt.plot(pd.Series(aligned_forecast, index=aligned_actual.index), label='ARIMA Forecast', color='orange')

# Plot simulated paths
forecast_index = np.arange(len(prices), len(prices) + forecast_days)
for i in range(num_simulations):
    plt.plot(forecast_index, price_paths[:, i], color='blue', alpha=0.1)

plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.title('HDFC Bank Stock Price Forecast using ARIMA and GBM')
plt.show()
