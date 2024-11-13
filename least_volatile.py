import pandas as pd
import numpy as np
import glob
import os

# Debug: Print the current working directory
print("Current working directory:", os.getcwd())

# Load all CSV files from the directory containing your stock data
file_pattern = 'market_data/*.csv'  # Path to market data folder
file_list = glob.glob(file_pattern)

# Debug: Print the list of files found
print("Files found:", file_list)

def calculate_volatility(df):
    # Calculate annualized volatility using daily returns
    daily_returns = df['close'].pct_change(fill_method=None)
    annual_volatility = np.std(daily_returns.dropna()) * np.sqrt(252)
    return annual_volatility

# Load all CSV files into a list of DataFrames
dataframes = []
for file in file_list:
    try:
        df = pd.read_csv(file)
        ticker = os.path.basename(file).replace('.csv', '')
        df['ticker'] = ticker
        dataframes.append(df)
    except Exception as e:
        print(f'Error reading {file}: {e}')

# Concatenate all DataFrames
all_data = pd.concat(dataframes, ignore_index=True)

# Convert 'date' column to datetime
all_data['date'] = pd.to_datetime(all_data['date'], format='%Y-%m-%d', utc=True, errors='coerce')

# Filter out stocks with at least 5 years of data
date_range = all_data.groupby('ticker')['date'].agg(['min', 'max'])
sufficient_data_tickers = date_range[(date_range['max'] - date_range['min']).dt.days >= 5 * 365].index

# Filter the main DataFrame
filtered_data = all_data[all_data['ticker'].isin(sufficient_data_tickers)]

# Calculate volatility for each ticker
volatility = filtered_data.groupby('ticker', group_keys=False).apply(lambda x: calculate_volatility(x))

# Sort and convert to DataFrame
sorted_volatility = volatility.sort_values()
volatility_df = sorted_volatility.to_frame(name='Volatility')

# Determine thresholds for categorization
total_tickers = len(volatility_df)
least_volatile_count = total_tickers // 3
medium_volatile_count = total_tickers // 3

# Identify the least, medium, and high volatile tickers
least_volatile_tickers = volatility_df.head(least_volatile_count).index
medium_volatile_tickers = volatility_df.iloc[least_volatile_count:least_volatile_count + medium_volatile_count].index
high_volatile_tickers = volatility_df.tail(total_tickers - least_volatile_count - medium_volatile_count).index

# Filter the original data to include all data for each category
least_volatile_data = all_data[all_data['ticker'].isin(least_volatile_tickers)]
medium_volatile_data = all_data[all_data['ticker'].isin(medium_volatile_tickers)]
high_volatile_data = all_data[all_data['ticker'].isin(high_volatile_tickers)]

# Define the output directory
output_dir = 'market_data'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Save the data to separate CSV files in the market_data folder
least_volatile_data.to_csv(os.path.join(output_dir, 'least_volatile_stocks_data.csv'), index=False)
medium_volatile_data.to_csv(os.path.join(output_dir, 'medium_volatile_stocks_data.csv'), index=False)
high_volatile_data.to_csv(os.path.join(output_dir, 'high_volatile_stocks_data.csv'), index=False)

print("Data for least volatile stocks saved to 'market_data/least_volatile_stocks_data.csv'")
print("Data for medium volatile stocks saved to 'market_data/medium_volatile_stocks_data.csv'")
print("Data for high volatile stocks saved to 'market_data/high_volatile_stocks_data.csv'")
