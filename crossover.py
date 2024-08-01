import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# Download stock data for an Indian stock (e.g., TCS)
stock_data = yf.download('TCS.NS', start='2018-01-01', end='2024-08-01')

# Calculate moving averages
stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
stock_data['SMA_200'] = stock_data['Close'].rolling(window=200).mean()

# Identify crossover patterns
buy_signals = (stock_data['SMA_50'] > stock_data['SMA_200']) & (stock_data['SMA_50'].shift(1) <= stock_data['SMA_200'].shift(1))
sell_signals = (stock_data['SMA_50'] < stock_data['SMA_200']) & (stock_data['SMA_50'].shift(1) >= stock_data['SMA_200'].shift(1))

stock_data['Buy Signal'] = np.where(buy_signals, stock_data['Close'], np.nan)
stock_data['Sell Signal'] = np.where(sell_signals, stock_data['Close'], np.nan)

# Plot data
plt.figure(figsize=(14, 7))
plt.plot(stock_data['Close'], label='Close Price', alpha=0.5)
plt.plot(stock_data['SMA_50'], label='50-day SMA', alpha=0.75)
plt.plot(stock_data['SMA_200'], label='200-day SMA', alpha=0.75)
plt.scatter(stock_data.index, stock_data['Buy Signal'], marker='^', color='g', s=100, label='Buy Signal')
plt.scatter(stock_data.index, stock_data['Sell Signal'], marker='v', color='r', s=100, label='Sell Signal')
plt.title('Stock Price with Moving Averages (TCS)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
