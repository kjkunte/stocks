import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
import warnings

# Ignore warnings from ARIMA model fitting
warnings.filterwarnings("ignore")

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
                except Exception as e:
                    print(f"ARIMA({p},{d},{q}) failed: {e}")
                    continue

    return best_model

def add_technical_indicators(data):
    """
    Adds technical indicators to the dataframe.

    Args:
        data: DataFrame with stock price data.

    Returns:
        DataFrame with additional technical indicators.
    """
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
    data['RSI'] = calculate_rsi(data['Close'])
    return data.dropna()

def calculate_rsi(series, period=14):
    """
    Calculates the Relative Strength Index (RSI) for a given time series.

    Args:
        series: Time series data.
        period: Period for RSI calculation.

    Returns:
        RSI values.
    """
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Define parameters
ticker = "HDFCBANK.NS"
start_date = "2024-01-01"
end_date = "2024-08-06"
forecast_days = 20
num_simulations = 100

# Fetch data
data = yf.download(ticker, start=start_date, end=end_date)
prices = data['Close']

# Add technical indicators
data = add_technical_indicators(data)
prices = data['Close']

# Experiment with different ARIMA orders
p_range = range(0, 3)
d_range = range(0, 2)
q_range = range(0, 3)

# Use time series cross-validation to find the best ARIMA model
tscv = TimeSeriesSplit(n_splits=5)
best_model = None
best_score = float("inf")

for train_index, test_index in tscv.split(prices):
    train, test = prices.iloc[train_index], prices.iloc[test_index]
    model = find_best_arima(train, p_range, d_range, q_range)
    predictions = model.forecast(steps=len(test))
    score = mean_absolute_percentage_error(test, predictions)
    if score < best_score:
        best_score = score
        best_model = model

# Forecast using ARIMA
arima_forecast = best_model.get_forecast(steps=forecast_days)
arima_pred_mean = arima_forecast.predicted_mean
arima_conf_int = arima_forecast.conf_int()

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
last_actual_price = prices.iloc[-1]

for i in range(num_simulations):
    price_paths[:, i] = arima_pred_mean.values + gbm_residuals[:, i]

# Evaluate the forecast
# Note: Ensure you have enough data to compare forecast with actual prices
actual_prices = yf.download(ticker, start=end_date, end="2024-09-20")['Close']

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
forecast_index = pd.date_range(start=prices.index[-1], periods=forecast_days+1, freq='B')[1:]
plt.plot(forecast_index, arima_pred_mean, label='ARIMA Forecast', color='orange')

# Plot the ARIMA forecast confidence intervals
plt.fill_between(forecast_index, arima_conf_int.iloc[:, 0], arima_conf_int.iloc[:, 1], color='pink', alpha=0.5, label='ARIMA 95% CI')

# Plot simulated paths with better visual representation
for i in range(num_simulations):
    plt.plot(forecast_index, price_paths[:, i], color='skyblue', alpha=0.3)

# Plot the range of simulated paths
plt.fill_between(forecast_index, np.min(price_paths, axis=1), np.max(price_paths, axis=1), color='lightblue', alpha=0.4, label='Simulated Path Range')

# Improve y-axis ticks and labels
y_min = min(prices.min(), arima_pred_mean.min())
y_max = max(prices.max(), arima_pred_mean.max())
y_range = np.linspace(y_min, y_max, 20)
plt.yticks(y_range, [f'{y:.2f}' for y in y_range])

plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.title('HDFC Bank Stock Price Forecast using ARIMA and GBM')
plt.grid(True)
plt.show()
