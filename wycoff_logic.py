import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np

# Download historical data for ICICI Bank
symbol = 'ICICIBANK.NS'  # ICICI Bank NSE symbol
data = yf.download(symbol, start="2022-01-01", end="2024-08-12")

# Calculate moving averages
data['50_MA'] = data['Close'].rolling(window=5).mean()
data['200_MA'] = data['Close'].rolling(window=20).mean()

# Initialize the Phase column as an object (string) type
data['Phase'] = np.nan
data['Phase'] = data['Phase'].astype('object')

# Wyckoff Phase Identification Using Simplified Logic
def identify_wyckoff_phases_simple(data):
    for i in range(1, len(data)):
        if data['50_MA'].iloc[i] > data['200_MA'].iloc[i]:
            if data['Close'].iloc[i] > data['200_MA'].iloc[i] and data['Close'].iloc[i] < data['50_MA'].iloc[i]:
                data.loc[data.index[i], 'Phase'] = 'Accumulation'
            elif data['Close'].iloc[i] < data['200_MA'].iloc[i]:
                data.loc[data.index[i], 'Phase'] = 'Markdown'
            elif data['Close'].iloc[i] > data['50_MA'].iloc[i]:
                data.loc[data.index[i], 'Phase'] = 'Markup'
        elif data['50_MA'].iloc[i] < data['200_MA'].iloc[i]:
            if data['Close'].iloc[i] > data['200_MA'].iloc[i] and data['Close'].iloc[i] < data['50_MA'].iloc[i]:
                data.loc[data.index[i], 'Phase'] = 'Distribution'
            elif data['Close'].iloc[i] > data['50_MA'].iloc[i]:
                data.loc[data.index[i], 'Phase'] = 'Markup'
            elif data['Close'].iloc[i] < data['200_MA'].iloc[i]:
                data.loc[data.index[i], 'Phase'] = 'Markdown'

    return data

# Apply Wyckoff Phase Identification
data = identify_wyckoff_phases_simple(data)

# Plot the data with phases
plt.figure(figsize=(14,8))

# Plot the close price with phases
plt.subplot(2, 1, 1)
plt.plot(data['Close'], label='Close Price', color='blue')
plt.plot(data['50_MA'], label='50-Day Moving Average', color='orange')
plt.plot(data['200_MA'], label='200-Day Moving Average', color='green')
for phase in data['Phase'].unique():
    if pd.notna(phase):
        phase_data = data[data['Phase'] == phase]
        plt.scatter(phase_data.index, phase_data['Close'], label=phase, s=50, alpha=0.6)
plt.title(f'{symbol} Wyckoff Method Analysis with Simplified Logic')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()

# Plot the volume
plt.subplot(2, 1, 2)
plt.bar(data.index, data['Volume'], color='blue')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.title('Volume')

plt.tight_layout()
plt.show()
