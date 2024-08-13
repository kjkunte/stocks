import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import warnings

# Ignore warnings from model fitting
warnings.filterwarnings("ignore")

# Function to find the best SARIMA model
def find_best_sarima(data, p_range, d_range, q_range, P_range, D_range, Q_range, m_values, tolerance=0.01, max_iter=50):
    best_aic = float("inf")
    best_model = None
    previous_aic = float("inf")
    iteration = 0

    for m in m_values:
        for p in p_range:
            for d in d_range:
                for q in q_range:
                    for P in P_range:
                        for D in D_range:
                            for Q in Q_range:
                                try:
                                    print(f"Trying SARIMA({p},{d},{q})x({P},{D},{Q},{m})")
                                    model = SARIMAX(data, order=(p, d, q), seasonal_order=(P, D, Q, m))
                                    model_fit = model.fit()
                                    current_aic = model_fit.aic

                                    # Check for convergence
                                    aic_change = previous_aic - current_aic
                                    if aic_change < tolerance:
                                        print(f"Convergence achieved with AIC change: {aic_change}")
                                        return best_model
                                    
                                    if current_aic < best_aic:
                                        best_aic = current_aic
                                        best_model = model_fit

                                    previous_aic = current_aic
                                    iteration += 1

                                    if iteration >= max_iter:
                                        print(f"Reached maximum iterations: {max_iter}")
                                        return best_model

                                except Exception as e:
                                    print(f"SARIMA({p},{d},{q})x({P},{D},{Q},{m}) failed: {e}")
                                    continue

    if best_model is None:
        print("No valid SARIMA model could be found.")
    
    return best_model

# Define parameters
ticker = "CL=F"  # Crude Oil Futures
start_date = "2020-12-02"
end_date = "2024-01-11"
forecast_days = 100
num_simulations = 100

# Fetch data
data = yf.download(ticker, start=start_date, end=end_date)
prices = data['Close']

# Split data into training and test set
train_size = int(len(prices) * 0.8)
train_data, test_data = prices[:train_size], prices[train_size:]

# Experiment with different SARIMA orders
p_range = range(0, 3)
d_range = range(0, 2)
q_range = range(0, 3)
P_range = range(0, 2)
D_range = range(0, 2)
Q_range = range(0, 2)
m_values = [12, 52]  # Monthly, Weekly seasonality

# Find the best SARIMA model using the training data
best_model = find_best_sarima(train_data, p_range, d_range, q_range, P_range, D_range, Q_range, m_values, tolerance=0.01, max_iter=50)

# Forecast using SARIMA
sarima_forecast = best_model.get_forecast(steps=len(test_data))
sarima_pred_mean = sarima_forecast.predicted_mean
sarima_conf_int = sarima_forecast.conf_int()

# Calculate residuals from SARIMA model
residuals = best_model.resid

# Fit GARCH model on SARIMA residuals
garch_model = arch_model(residuals, vol='Garch', p=1, q=1)
garch_fit = garch_model.fit(disp='off')
garch_forecast = garch_fit.forecast(horizon=len(test_data))
garch_volatility = garch_forecast.variance[-1:]**0.5

# Simulate GARCH-residual paths
garch_simulated_residuals = np.zeros((len(test_data), num_simulations))

for i in range(num_simulations):
    simulated_volatility = garch_volatility.values.flatten() * np.random.normal(size=len(test_data))
    garch_simulated_residuals[:, i] = simulated_volatility

# Combine SARIMA forecast with GARCH volatility-adjusted residuals
price_paths = np.zeros_like(garch_simulated_residuals)
last_actual_price = train_data.iloc[-1]

for i in range(num_simulations):
    price_paths[:, i] = sarima_pred_mean.values + garch_simulated_residuals[:, i]

# Clip the prices to prevent unrealistic values
price_paths = np.clip(price_paths, last_actual_price * 0.5, last_actual_price * 1.5)

# Create DataFrame for the simulated paths
simulated_data = pd.DataFrame(price_paths, index=test_data.index)

# Evaluate the model
mape = mean_absolute_percentage_error(test_data, sarima_pred_mean)
print(f"MAPE: {mape:.2f}%")

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(prices, label='Historical Prices')
plt.plot(test_data.index, test_data, label='Actual Prices', color='blue', alpha=0.75)
plt.plot(test_data.index, sarima_pred_mean, label='SARIMA Forecast', color='orange')
plt.fill_between(test_data.index, sarima_conf_int.iloc[:, 0], sarima_conf_int.iloc[:, 1], color='pink', alpha=0.3, label='SARIMA 95% CI')

# Plot simulated paths with better visual representation
for i in range(num_simulations):
    plt.plot(test_data.index, price_paths[:, i], color='skyblue', alpha=0.05)

plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.title('Crude Oil Futures Price Forecast using SARIMA and GARCH')
plt.grid(True)
plt.show()
