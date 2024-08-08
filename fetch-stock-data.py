import yfinance as yf

# List of stock codes
stock_codes = [
    'RELIANCE.NS', 'TATAMOTORS.NS', 'ADANIENT.NS', 'BAJFINANCE.NS', 'SBIN.NS',
    'HDFCBANK.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'AXISBANK.NS', 'INDUSINDBK.NS',
    'PIDILITIND.NS', 'VOLTAS.NS', 'L&TFH.NS', 'TATACHEM.NS', 'GODREJPROP.NS',
    'TCS.NS', 'INFY.NS', 'LT.NS', 'SUNPHARMA.NS', 'M&M.NS'
]

# Download stock data
stock_data = yf.download(stock_codes, start='2019-01-01', end='2024-08-01')

# Display data
print(stock_data.head())
