import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# Define parameters
ticker = 'HDFCBANK.NS'  # HDFC Bank ticker on NSE
start_date = '2018-01-01'
end_date = '2023-01-01'
backtest_start_date = '2022-12-01'
backtest_end_date = '2023-01-01'
num_simulations = 100  # Increase the number of simulations for a better range
num_days = 20  # 1 month (20 trading days)

# Fetch historical stock prices for model calibration
data = yf.download(ticker, start=start_date, end=end_date)
prices = data['Close']

# Calculate log returns
log_returns = np.log(1 + prices.pct_change().dropna())

# Calculate the mean and standard deviation of log returns
u = log_returns.mean()
sigma = log_returns.std()

# Calculate drift
drift = u - (0.5 * sigma**2)

# Fetch historical stock prices for backtesting
backtest_data = yf.download(ticker, start=backtest_start_date, end=backtest_end_date)
actual_prices = backtest_data['Close']

# Set the starting price for the simulation to the last actual price before the backtest period
starting_price = prices[backtest_start_date]

# Generate random variables for simulation
Z = np.random.standard_normal((num_days, num_simulations))

# Geometric Brownian Motion equation
daily_returns = np.exp(drift + sigma * Z)
price_paths = np.zeros_like(daily_returns)
price_paths[0] = starting_price

for t in range(1, num_days):
    price_paths[t] = price_paths[t-1] * daily_returns[t]

# Calculate percentiles for confidence intervals
percentiles = np.percentile(price_paths, [5, 95], axis=1)

# Plot the actual prices, simulated paths, and confidence intervals
plt.figure(figsize=(12, 6))

# Plot actual prices
plt.plot(actual_prices.values, label='Actual Price', color='black')

# Plot multiple simulated paths
for i in range(num_simulations):
    plt.plot(price_paths[:, i], color='blue', alpha=0.1)

# Plot confidence intervals
plt.fill_between(range(num_days), percentiles[0], percentiles[1], color='gray', alpha=0.3, label='90% Confidence Interval')

plt.title('HDFC Bank Stock Price: Actual vs Simulated (1 Month)')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.show()
