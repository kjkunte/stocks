import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
import warnings

# Ignore warnings from model fitting
warnings.filterwarnings("ignore")

# Function to find the best SARIMA model (as you already have)
def find_best_sarima(data, p_range, d_range, q_range, P_range, D_range, Q_range, m, tolerance=0.01, max_iter=50):
    # Your existing SARIMA function
    ...

# Define parameters
ticker = "HDFCBANK.NS"
start_date = "2019-06-01"
end_date = "2024-08-08"  # Set this to today's date
forecast_days = 40
num_simulations = 100

# Fetch data
data = yf.download(ticker, start=start_date, end=end_date)
prices = data['Close']

# Experiment with different SARIMA orders
p_range = range(0, 2)
d_range = range(0, 2)
q_range = range(0, 2)
P_range = range(0, 2)
D_range = range(0, 2)
Q_range = range(0, 2)
m = 12  # Monthly seasonality

# Use time series cross-validation to find the best SARIMA model
tscv = TimeSeriesSplit(n_splits=2)
best_model = None
best_score = float("inf")

for train_index, test_index in tscv.split(prices):
    train, test = prices.iloc[train_index], prices.iloc[test_index]
    model = find_best_sarima(train, p_range, d_range, q_range, P_range, D_range, Q_range, m, tolerance=0.01, max_iter=50)
    if model:
        predictions = model.forecast(steps=len(test))
        score = mean_absolute_percentage_error(test, predictions)
        if score < best_score:
            best_score = score
            best_model = model

# Forecast using SARIMA
sarima_forecast = best_model.get_forecast(steps=forecast_days)
sarima_pred_mean = sarima_forecast.predicted_mean
sarima_conf_int = sarima_forecast.conf_int()

# Calculate residuals from SARIMA model
residuals = best_model.resid

# Fit GARCH model on SARIMA residuals
garch_model = arch_model(residuals, vol='Garch', p=1, q=1)
garch_fit = garch_model.fit(disp="off")
garch_forecast = garch_fit.forecast(horizon=forecast_days)
garch_volatility = garch_forecast.variance[-1:]**0.5

# Simulate GARCH-residual paths
garch_simulated_residuals = np.zeros((forecast_days, num_simulations))

for i in range(num_simulations):
    simulated_volatility = garch_volatility * np.random.normal(size=forecast_days)
    garch_simulated_residuals[:, i] = simulated_volatility.flatten()

# Combine SARIMA forecast with GARCH volatility-adjusted residuals
price_paths = np.zeros_like(garch_simulated_residuals)
last_actual_price = prices.iloc[-1]

for i in range(num_simulations):
    price_paths[:, i] = sarima_pred_mean.values + garch_simulated_residuals[:, i]

# Clip the prices to prevent unrealistic values
price_paths = np.clip(price_paths, last_actual_price * 0.5, last_actual_price * 1.5)

# Create DataFrame for the simulated paths
simulated_data = pd.DataFrame(price_paths, index=pd.date_range(start=prices.index[-1], periods=forecast_days+1, freq='B')[1:])

# Save the simulated data to Excel
simulated_data.to_excel("simulated_paths_with_garch.xlsx")

# Evaluation cannot be performed since actual future prices are not available

print("Best SARIMA Model:", best_model.summary())

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(prices, label='Historical Prices')
forecast_index = pd.date_range(start=prices.index[-1], periods=forecast_days+1, freq='B')[1:]
plt.plot(forecast_index, sarima_pred_mean, label='SARIMA Forecast', color='orange')

# Plot the SARIMA forecast confidence intervals
plt.fill_between(forecast_index, sarima_conf_int.iloc[:, 0], sarima_conf_int.iloc[:, 1], color='pink', alpha=0.5, label='SARIMA 95% CI')

# Plot simulated paths with better visual representation
for i in range(num_simulations):
    plt.plot(forecast_index, price_paths[:, i], color='skyblue', alpha=0.3)

# Plot the range of simulated paths
plt.fill_between(forecast_index, np.min(price_paths, axis=1), np.max(price_paths, axis=1), color='lightblue', alpha=0.4, label='Simulated Path Range')

plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.title('HDFC Bank Stock Price Forecast using SARIMA and GARCH')
plt.grid(True)
plt.show()
