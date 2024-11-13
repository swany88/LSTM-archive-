# %% Check TensorFlow using my build and set environment variables
import os
import pickle
import hashlib
from datetime import datetime

# Set optimal TensorFlow environment variables
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
os.environ['OMP_NUM_THREADS'] = '16'
os.environ['KMP_BLOCKTIME'] = '0'
os.environ['KMP_SETTINGS'] = '1'
os.environ['KMP_AFFINITY'] = 'granularity=fine,verbose,compact,1,0'
os.environ['TF_ENABLE_XLA'] = '1'

import sys
sys.path.insert(0, '/home/erik/.local/lib/python3.12/site-packages')

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# TF params set in local .env file
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras_tuner import HyperModel
import time

# Set thread counts (you may want to remove or adjust these based on the environment variables set above)
# tf.config.threading.set_intra_op_parallelism_threads(8)
# tf.config.threading.set_inter_op_parallelism_threads(2)

# Check TensorFlow flags and options
print("\nTensorFlow Environment Variables:")
print(f"TF_NUM_INTRAOP_THREADS: {os.environ.get('TF_NUM_INTRAOP_THREADS')}")
print(f"TF_NUM_INTEROP_THREADS: {os.environ.get('TF_NUM_INTEROP_THREADS')}")
print(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS')}")
print(f"TF_ENABLE_ONEDNN_OPTS: {os.environ.get('TF_ENABLE_ONEDNN_OPTS')}")
print(f"TF_CPU_ENABLE_AVX512: {os.environ.get('TF_CPU_ENABLE_AVX512')}")
print(f"TF_CPU_ENABLE_AVX2: {os.environ.get('TF_CPU_ENABLE_AVX2')}")
print(f"TF_CPU_ENABLE_FMA: {os.environ.get('TF_CPU_ENABLE_FMA')}")
print(f"TF_ENABLE_XLA: {os.environ.get('TF_ENABLE_XLA')}")
print(f"TF_ENABLE_AUTO_MIXED_PRECISION: {os.environ.get('TF_ENABLE_AUTO_MIXED_PRECISION')}")

# Print current thread settings
print("\nTensorFlow Threading Info:")
print(f"Intra-op parallelism threads: {tf.config.threading.get_intra_op_parallelism_threads()}")
print(f"Inter-op parallelism threads: {tf.config.threading.get_inter_op_parallelism_threads()}")

# Print current optimizer settings
from tensorflow.python.framework import test_util
print("\nOptimizer Settings:")
print(f"oneDNN enabled: {test_util.IsMklEnabled()}")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer  # Add this line
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
import yfinance as yf
from fredapi import Fred
from scipy.fft import fft
from scipy.signal import detrend
from scipy.stats import kurtosis
from collections import defaultdict
from keras_tuner import HyperModel
from keras_tuner.tuners import BayesianOptimization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

# At the beginning of the script, right after the imports:
import time

# Start the timer
start_time = time.time()

# %% Define stock symbol, api key, functions for retrieving stock data
symbol = 'QCOM'

# FRED API key
fred_api_key = os.getenv('FRED_API_KEY')
fred = Fred(api_key=fred_api_key)

def make_tz_naive(df):
    if df.index.tzinfo is not None:
        df.index = df.index.tz_localize(None)
    return df

def get_stock_data(symbol, start_date, end_date):
    return make_tz_naive(yf.Ticker(symbol).history(start=start_date, end=end_date))

def calculate_technical_indicators(data):
    result = data.copy()
    close = result['Close']
    result['SMA_20'] = close.rolling(window=20).mean()
    result['EMA_20'] = close.ewm(span=20, adjust=False).mean()
    result['BB_Middle'] = result['SMA_20']
    bb_std = close.rolling(window=20).std()
    result['BB_Upper'] = result['BB_Middle'] + 2 * bb_std
    result['BB_Lower'] = result['BB_Middle'] - 2 * bb_std
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    result['RSI'] = 100 - (100 / (1 + rs))
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    result['MACD'] = ema_12 - ema_26
    result['MACD_Signal'] = result['MACD'].ewm(span=9, adjust=False).mean()
    result['OBV'] = (np.sign(delta) * result['Volume']).fillna(0).cumsum()

    #? New volatility indicator: Average True Range (ATR)
    tr1 = result['High'] - result['Low']
    tr2 = abs(result['High'] - result['Close'].shift())
    tr3 = abs(result['Low'] - result['Close'].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    result['ATR'] = tr.rolling(window=14).mean()

    return result

def get_economic_indicators(fred, start_date, end_date):
    fred_series = {
        'GDP': 'GDP', 'Interest_Rates': 'FEDFUNDS', 'Consumer_Confidence': 'UMCSENT',
        'Industrial_Production': 'INDPRO', 'Unemployment_Rate': 'UNRATE',
        'Retail_Sales': 'RSAFS', 'Housing_Starts': 'HOUST', 'Corporate_Profits': 'CP',
        'Inflation_Rate': 'CPIAUCSL', 'Economic_Policy_Uncertainty': 'USEPUINDXD'
    }
    fred_data = pd.DataFrame({name: fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
                              for name, series_id in fred_series.items()})
    return make_tz_naive(fred_data)

def get_market_indices(start_date, end_date):
    indices = yf.download(['^GSPC', '^VIX'], start=start_date, end=end_date)['Close']
    indices.columns = ['SP500', 'VIX']
    return make_tz_naive(indices)

def get_sector_data(symbol, start_date, end_date):
    sector_etfs = {
        'Information Technology': 'XLK', 'Health Care': 'XLV', 'Financials': 'XLF',
        'Consumer Discretionary': 'XLY', 'Communication Services': 'XLC',
        'Industrials': 'XLI', 'Consumer Staples': 'XLP', 'Energy': 'XLE',
        'Utilities': 'XLU', 'Real Estate': 'XLRE', 'Materials': 'XLB'
    }
    stock = yf.Ticker(symbol)
    sector = stock.info['sector']
    sector_etf = sector_etfs.get(sector, 'SPY')
    sector_data = yf.download(sector_etf, start=start_date, end=end_date)['Close']
    return make_tz_naive(pd.DataFrame({'Sector': sector, f'{sector}_ETF': sector_data}))

# Function to calculate hash of data
def calculate_hash(data):
    return hashlib.md5(pickle.dumps(data)).hexdigest()

# Function to save data
def save_data(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump((data, datetime.now().date()), f)

# Function to load data
def load_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Function to check if data needs to be updated
def data_needs_update(filename):
    if not os.path.exists(filename):
        return True
    saved_data, save_date = load_data(filename)
    return save_date != datetime.now().date()

# Data Collection
print("\nData Collection")
start_date, end_date = '2019-01-01', '2023-05-31'  # Example date range

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
data_filename = os.path.join(script_dir, f'stock_data_{symbol}_{start_date}_{end_date}.pkl')

# Check if we need to pull new data
if not data_needs_update(data_filename):
    print("Loading saved data...")
    cleaned_data, _ = load_data(data_filename)
else:
    print("\nFetching and processing data for the common time frame...")

    stock_data = get_stock_data(symbol, start_date, end_date)
    data_with_indicators = calculate_technical_indicators(stock_data)

    economic_data = get_economic_indicators(fred, start_date, end_date)
    market_indices = get_market_indices(start_date, end_date)
    sector_data = get_sector_data(symbol, start_date, end_date)

    # Prefix columns to avoid overlapping
    data_with_indicators = data_with_indicators.add_prefix('Stock_')
    economic_data = economic_data.add_prefix('Economic_')
    market_indices = market_indices.add_prefix('Market_')
    sector_data = sector_data.add_prefix('Sector_')

    # Join the DataFrames
    combined_data = data_with_indicators.join([
        economic_data,
        market_indices,
        sector_data,
    ])

    # Data Preprocessing nan and interpolate, weekend removal
    def preprocess_data(data):
        data = data.interpolate().ffill().bfill()
        data = data[data.index.dayofweek < 5]  # Remove weekends
        print("\nNAN values in data after preprocessing:\n", data.isna().sum())
        return data

    # Data Preprocessing
    cleaned_data = preprocess_data(combined_data)

    # Save the cleaned data
    save_data(cleaned_data, data_filename)
    print(f"Data saved to {data_filename}")

# Analyze the cleaned data
def analyze_cleaned_data(cleaned_data):
    print("\n--- Cleaned Data Analysis ---")
    print("\nData Info:")
    print(cleaned_data.info())
    
    start_date, end_date = cleaned_data.index.min(), cleaned_data.index.max()
    print(f"\nCleaned data date range: {start_date} to {end_date}")
    print(f"Total number of rows: {len(cleaned_data)}")
    print(f"Number of days between start and end date: {(end_date - start_date).days}")

    full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    missing_dates = full_date_range.difference(cleaned_data.index)
    print(f"\nNumber of missing dates: {len(missing_dates)}")
    if len(missing_dates) > 0:
        print("First few missing dates:", missing_dates[:5].tolist())

    print(f"\nNumber of NaN values in 'Stock_Close' column: {cleaned_data['Stock_Close'].isna().sum()}")

    day_counts = cleaned_data.index.dayofweek.value_counts().sort_index()
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    print("\nSummary of days of the week:")
    for day, count in day_counts.items():
        print(f"{day_names[day]}: {count}")

    weekend_data = cleaned_data[cleaned_data.index.dayofweek.isin([5, 6])]
    if not weekend_data.empty:
        print("\nWarning: Data contains weekend entries:")
        print(weekend_data)

    total_weekdays = sum(day.weekday() < 5 for day in full_date_range)
    available_weekdays = sum(day_counts[:5])
    coverage_percentage = (available_weekdays / total_weekdays) * 100
    print(f"\nPercentage of available trading days: {coverage_percentage:.2f}%")

analyze_cleaned_data(cleaned_data)

# %% Calculate Business Cycle Indicator
def calculate_cycle_score(row, data):
    score = 0
    score += 1 if row['Stock_Close'] > row['Stock_SMA_20'] else -1
    score += 1 if row['Economic_GDP'] > data['Economic_GDP'].rolling(window=20).mean().loc[row.name] else -1
    score += 1 if row['Economic_Industrial_Production'] > data['Economic_Industrial_Production'].rolling(window=20).mean().loc[row.name] else -1
    score += 1 if row['Economic_Unemployment_Rate'] < data['Economic_Unemployment_Rate'].rolling(window=20).mean().loc[row.name] else -1
    score += 1 if row['Economic_Consumer_Confidence'] > data['Economic_Consumer_Confidence'].rolling(window=20).mean().loc[row.name] else -1
    score += 1 if row['Economic_Corporate_Profits'] > data['Economic_Corporate_Profits'].rolling(window=20).mean().loc[row.name] else -1
    score += 1 if row['Market_SP500'] > data['Market_SP500'].rolling(window=20).mean().loc[row.name] else -1
    return score

def determine_business_cycle(score):
    if score >= 5:
        return 'Expansion'
    elif 2 <= score < 5:
        return 'Peak'
    elif -2 <= score < 2:
        return 'Contraction'
    else:
        return 'Trough'

def iterative_bci_assignment(data):
    data['Cycle_Score'] = data.apply(lambda row: calculate_cycle_score(row, data), axis=1)
    data['Raw_BCI'] = data['Cycle_Score'].apply(determine_business_cycle)
    
    bci_periods = []
    current_period = data['Raw_BCI'].iloc[0]
    transition_count = 0
    transition_threshold = 20  # Adjust this value as needed

    for _, row in data.iterrows():
        if row['Raw_BCI'] == current_period:
            transition_count = 0
        else:
            transition_count += 1

        if transition_count >= transition_threshold:
            if current_period == 'Expansion':
                current_period = 'Peak'
            elif current_period == 'Peak':
                current_period = 'Contraction'
            elif current_period == 'Contraction':
                current_period = 'Trough'
            else:  # Trough
                current_period = 'Expansion'
            transition_count = 0

        bci_periods.append(current_period)

    return bci_periods

# Calculate Business Cycle Indicators
print("\nCalculating Business Cycle Indicator")
cleaned_data['Cycle_Score'] = cleaned_data.apply(lambda row: calculate_cycle_score(row, cleaned_data), axis=1)
cleaned_data['Iterative_Business_Cycle'] = iterative_bci_assignment(cleaned_data)

print("\nBusiness Cycle Indicator Distribution:")
print(cleaned_data['Iterative_Business_Cycle'].value_counts(normalize=True))

# Calculate Average Timeframes for Each Cycle Phase
def calculate_average_timeframes(data):
    phase_durations = defaultdict(list)
    current_phase = data['Iterative_Business_Cycle'].iloc[0]
    phase_start = data.index[0]
    
    for date, phase in zip(data.index[1:], data['Iterative_Business_Cycle'].iloc[1:]):
        if phase != current_phase:
            duration = (date - phase_start).days
            phase_durations[current_phase].append(duration)
            current_phase = phase
            phase_start = date
    
    # Add the last phase
    duration = (data.index[-1] - phase_start).days
    phase_durations[current_phase].append(duration)
    
    average_durations = {phase: np.mean(durations) for phase, durations in phase_durations.items()}
    return average_durations

average_timeframes = calculate_average_timeframes(cleaned_data)
print("\nAverage Timeframes for Each Cycle Phase (in days):")
for phase, duration in average_timeframes.items():
    print(f"{phase}: {duration:.2f}")
avg_bc_length = sum(average_timeframes.values())
print(f"\nAverage Business Cycle Length: {avg_bc_length:.2f} days")

# %% FFT Analysis, add statistical features
def perform_enhanced_fft_analysis(data):
    print("\n--- Enhanced FFT Analysis ---")
    
    close_prices = data['Stock_Close'].values
    detrended_prices = detrend(close_prices)
    
    window = np.hanning(len(detrended_prices))
    windowed_prices = detrended_prices * window
    fft_result = fft(windowed_prices)
    frequencies = np.fft.fftfreq(len(detrended_prices), d=1)
    amplitudes = np.abs(fft_result)
    
    all_cycles = {}
    time_scales = [
        ('Long-term', 252, 1260),  # 1-5 years
        ('Medium-term', 21, 252),  # 1 month to 1 year
        ('Short-term', 2, 21)      # 2 days to 1 month
    ]
    
    for scale_name, min_period, max_period in time_scales:
        print(f"\n{scale_name} Analysis:")
        mask = (1/max_period <= np.abs(frequencies)) & (np.abs(frequencies) <= 1/min_period)
        scale_frequencies = frequencies[mask]
        scale_amplitudes = amplitudes[mask]
        
        sorted_indices = np.argsort(scale_amplitudes)[::-1]
        top_frequencies = scale_frequencies[sorted_indices[:5]]
        top_amplitudes = scale_amplitudes[sorted_indices[:5]]
        
        print("Top 5 dominant frequencies:")
        for i, (freq, amp) in enumerate(zip(top_frequencies, top_amplitudes), 1):
            period = 1 / abs(freq) if freq != 0 else np.inf
            print(f"{i}. Frequency: {freq:.6f}, Period: {period:.2f} days, Amplitude: {amp:.2f}")
            
            cycle_name = f"{scale_name.lower()}_cycle_{i}"
            all_cycles[cycle_name] = (freq, amp, period)
    
    variables_to_compare = ['Market_SP500', 'Market_VIX', 'Sector_Technology_ETF']
    for var in variables_to_compare:
        if var in data.columns:
            var_fft = fft(detrend(data[var].values) * window)
            var_amplitudes = np.abs(var_fft)
            correlation = np.corrcoef(amplitudes, var_amplitudes)[0, 1]
            print(f"\nFFT Amplitude correlation with {var}: {correlation:.4f}")
    
    return all_cycles

def add_statistical_features(data):
    print("\n--- Adding Statistical Features ---")
    window_sizes = [5, 10, 20, 50, 100]
    new_features = {}
    for window in window_sizes:
        new_features[f'rolling_mean_{window}'] = data['Stock_Close'].rolling(window=window, min_periods=1).mean()
        
        # For standard deviation, we need at least 2 points
        std = data['Stock_Close'].rolling(window=window, min_periods=2).std()
        new_features[f'rolling_std_{window}'] = std.bfill()
        
        # For skewness, we need at least 3 points
        skew = data['Stock_Close'].rolling(window=window, min_periods=3).skew()
        new_features[f'rolling_skew_{window}'] = skew.bfill()
        
        # For kurtosis, we need at least 4 points
        kurt = data['Stock_Close'].rolling(window=window, min_periods=4).apply(kurtosis)
        new_features[f'rolling_kurt_{window}'] = kurt.bfill()
        
        print(f"Added rolling statistics for window size {window}")
    
    result = pd.concat([data, pd.DataFrame(new_features, index=data.index)], axis=1)
    
    # Replace any remaining NaNs with the first valid value
    result = result.bfill().ffill()
    
    return result

all_cycles = perform_enhanced_fft_analysis(cleaned_data)
cleaned_data = add_statistical_features(cleaned_data)
print("\nCycle features and statistical features have been added to the dataframe.")

# %% train test split, transform, filter features
# First, split the data into train and test sets
train_data, test_data = train_test_split(cleaned_data, test_size=0.2, shuffle=False)

# Identify categorical and numeric columns
categorical_columns = cleaned_data.select_dtypes(include=['object', 'category']).columns
numeric_columns = cleaned_data.select_dtypes(include=['int64', 'float64']).columns

# Create a ColumnTransformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), numeric_columns),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_columns)
    ])

# Fit the preprocessor on the training data only
preprocessor.fit(train_data)

# Transform both training and test data
train_processed = preprocessor.transform(train_data)
test_processed = preprocessor.transform(test_data)

# Create DataFrames with processed data
feature_names = (numeric_columns.tolist() + 
                 preprocessor.named_transformers_['cat']
                 .get_feature_names_out(categorical_columns).tolist())
train_processed_df = pd.DataFrame(train_processed, columns=feature_names, index=train_data.index)
test_processed_df = pd.DataFrame(test_processed, columns=feature_names, index=test_data.index)

# Perform correlation analysis on training data only
def perform_correlation_analysis(data, threshold):
    print("\n--- Correlation Analysis ---")
    
    correlations = data.corr()
    
    # Correlation with target variable (assuming 'Stock_Close' is the target)
    target_correlations = correlations['Stock_Close'].sort_values(ascending=False)
    target_correlations = target_correlations.drop('Stock_Close')
    
    print("\nTop 10 Positive Correlations with Stock_Close:")
    print(target_correlations.head(10))
    print("\nTop 10 Negative Correlations with Stock_Close:")
    print(target_correlations.tail(10))
    
    # Filter features based on correlation threshold
    strong_corr = target_correlations[abs(target_correlations) >= threshold]
    weak_corr = target_correlations[abs(target_correlations) < threshold]
    
    print(f"\nFeatures with strong correlation (|r| >= {threshold}): {len(strong_corr)}")
    print(f"Features with weak correlation (|r| < {threshold}): {len(weak_corr)}")
    
    return strong_corr, weak_corr

strong_corr, weak_corr = perform_correlation_analysis(train_processed_df, threshold=0.5)

# Filter features based on correlation analysis
filtered_train_df = train_processed_df[strong_corr.index.tolist() + ['Stock_Close']]
filtered_test_df = test_processed_df[strong_corr.index.tolist() + ['Stock_Close']]

# %% LSTM model
# Prepare sequences
def prepare_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data.iloc[i:(i + sequence_length)].drop('Stock_Close', axis=1).values)
        y.append(data.iloc[i + sequence_length]['Stock_Close'])
    return np.array(X), np.array(y)

def set_tuning_params(data, feature_names, time_horizon):
    num_samples, num_features = data.shape
    num_features -= 1  # Subtract 1 to account for 'Stock_Close'

    params = {}
    
    # Hyperparameter Search Space
    params['lstm_units_min'] = min(32, num_features)
    params['lstm_units_max'] = min(256, num_features * 8)
    params['dense_units_min'] = min(16, num_features)
    params['dense_units_max'] = min(128, num_features * 4)

    # Model Architecture
    params['max_lstm_layers'] = min(5, max(1, int(np.log2(num_samples / time_horizon))))

    # Optimization Algorithm
    params['optimizers'] = ['adam', 'rmsprop', 'sgd']

    # Batch Size
    params['batch_sizes'] = [2**i for i in range(5, min(10, int(np.log2(num_samples))+1))]

    # Sequence Length
    params['sequence_length_min'] = 2 * time_horizon
    params['sequence_length_max'] = 3 * time_horizon

    # Feature Selection
    params['min_features'] = max(5, num_features // 4)
    params['max_features'] = num_features

    # Early Stopping
    params['early_stopping_patience_min'] = 5
    params['early_stopping_patience_max'] = min(50, num_samples // (10 * time_horizon))

    # Learning Rate Schedule
    params['use_lr_schedule'] = [True, False]
    params['lr_schedule_patience_min'] = 3
    params['lr_schedule_patience_max'] = 10
    params['lr_schedule_factor_min'] = 0.1
    params['lr_schedule_factor_max'] = 0.5

    # Regularization Techniques
    params['dropout_min'] = 0.0
    params['dropout_max'] = min(0.5, 1000 / num_samples)
    params['l1_min'] = 1e-6
    params['l1_max'] = 1e-3
    params['l2_min'] = 1e-6
    params['l2_max'] = 1e-3

    # Loss Function
    params['loss_functions'] = ['mse', 'mae', 'huber']

    # Learning Rate
    params['learning_rate_min'] = 1e-4
    params['learning_rate_max'] = 1e-2

    return params

# Usage:
time_horizon = 60  # Set time horizon to 60 for now

tuning_params = set_tuning_params(filtered_train_df, filtered_train_df.columns, time_horizon)

class SilentCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        pass

class ComplexLSTMHyperModel(HyperModel):
    def __init__(self, input_shape, max_trials, tuning_params, time_horizon):
        self.input_shape = input_shape
        self.max_trials = max_trials
        self.tuning_params = tuning_params
        self.time_horizon = time_horizon
        self.current_trial = 0  # Initialize current_trial here

    def build(self, hp):
        sequence_length = hp.Int('sequence_length', 
                                 min_value=self.tuning_params['sequence_length_min'], 
                                 max_value=self.tuning_params['sequence_length_max'],
                                 step=1)
        
        inputs = Input(shape=(sequence_length, self.input_shape[1]))
        
        x = inputs
        for i in range(hp.Int('num_lstm_layers', 1, self.tuning_params['max_lstm_layers'])):
            x = LSTM(units=hp.Int(f'lstm_units_{i}', 
                                  self.tuning_params['lstm_units_min'], 
                                  self.tuning_params['lstm_units_max'], 
                                  step=32),
                     return_sequences=(i < hp.Int('num_lstm_layers', 1, self.tuning_params['max_lstm_layers']) - 1),
                     kernel_regularizer=l1_l2(
                         l1=hp.Float(f'l1_{i}', self.tuning_params['l1_min'], self.tuning_params['l1_max'], sampling='LOG'),
                         l2=hp.Float(f'l2_{i}', self.tuning_params['l2_min'], self.tuning_params['l2_max'], sampling='LOG')))(x)
            x = Dropout(hp.Float(f'dropout_{i}', self.tuning_params['dropout_min'], self.tuning_params['dropout_max']))(x)
        
        x = Dense(hp.Int('dense_units', 
                         self.tuning_params['dense_units_min'], 
                         self.tuning_params['dense_units_max'], 
                         step=16), 
                  activation='relu')(x)
        
        outputs = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        optimizer_choice = hp.Choice('optimizer', self.tuning_params['optimizers'])
        learning_rate = hp.Float('learning_rate', self.tuning_params['learning_rate_min'], self.tuning_params['learning_rate_max'], sampling='LOG')

        if optimizer_choice == 'adam':
            optimizer = Adam(learning_rate=learning_rate)
        elif optimizer_choice == 'rmsprop':
            optimizer = RMSprop(learning_rate=learning_rate)
        else:
            optimizer = SGD(learning_rate=learning_rate)

        loss_function = hp.Choice('loss_function', self.tuning_params['loss_functions'])
        
        model.compile(optimizer=optimizer, loss=loss_function)
        return model

    def fit(self, hp, model, *args, **kwargs):
        self.current_trial += 1
        
        sequence_length = hp.get('sequence_length')
        
        # Prepare sequences with the current sequence length
        X, y = prepare_sequences(kwargs['x'], sequence_length)
        
        batch_size = hp.Choice('batch_size', self.tuning_params['batch_sizes'])
        
        # Early stopping setup
        early_stopping_patience_min = max(1, self.tuning_params['early_stopping_patience_min'])
        early_stopping_patience_max = max(early_stopping_patience_min + 1, self.tuning_params['early_stopping_patience_max'])
        
        early_stopping_patience = hp.Int('early_stopping_patience', 
                                         early_stopping_patience_min, 
                                         early_stopping_patience_max)
        early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True)
        
        callbacks = [early_stopping, SilentCallback()]
        
        # Learning rate schedule setup
        if hp.Boolean('use_lr_schedule'):
            lr_schedule_patience = hp.Int('lr_schedule_patience', 
                                          self.tuning_params['lr_schedule_patience_min'], 
                                          self.tuning_params['lr_schedule_patience_max'])
            lr_schedule_factor = hp.Float('lr_schedule_factor', 
                                          self.tuning_params['lr_schedule_factor_min'], 
                                          self.tuning_params['lr_schedule_factor_max'])
            lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=lr_schedule_factor, patience=lr_schedule_patience)
            callbacks.append(lr_schedule)
        
        # Update callbacks if they're provided in kwargs
        if 'callbacks' in kwargs:
            existing_callbacks = kwargs['callbacks']
            callbacks.extend([cb for cb in existing_callbacks if not isinstance(cb, (EarlyStopping, SilentCallback, ReduceLROnPlateau))])
            kwargs.pop('callbacks')
        
        trial_start_time = time.time()
        
        # Remove 'x' from kwargs to avoid duplicate argument
        kwargs.pop('x', None)
        
        result = model.fit(
            X, y,
            batch_size=batch_size,
            callbacks=callbacks,
            validation_split=0.2,
            **kwargs
        )
        
        trial_end_time = time.time()
        trial_duration = trial_end_time - trial_start_time
        print(f"Trial {self.current_trial} completed in {trial_duration:.2f} seconds")
        
        return result


# Get the current script name without the .py extension
script_name = os.path.splitext(os.path.basename(__file__))[0]
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Create a directory name based on the stock symbol
stock_dir = f"tuning_results_{symbol}"
# Create the full path for the tuning results
tuning_dir = os.path.join(script_dir, stock_dir)
# Create the directory if it doesn't exist
os.makedirs(tuning_dir, exist_ok=True)

# Set up parameters for the tuner
max_trials = 5
executions_per_trial = 1
tuner_algorithm = "BayesianOptimization"
objective = "val_loss"
tuner_objective = "minimize"  # or "maximize" if that's what you need

# Set up the tuner with the new directory and project name
input_shape = (None, filtered_train_df.shape[1] - 1)  # Subtract 1 to account for 'Stock_Close'
time_horizon = len(filtered_test_df)

tuner = BayesianOptimization(
    ComplexLSTMHyperModel(input_shape=input_shape, max_trials=max_trials, tuning_params=tuning_params, time_horizon=time_horizon),
    objective=objective,
    max_trials=max_trials,
    executions_per_trial=executions_per_trial,
    directory=tuning_dir,
    project_name=script_name
)

# Before running the tuner search
print("\n" + "="*50)
print(f"Starting {tuner_algorithm} for {symbol}")
print(f"Script: {script_name}")
print(f"Number of trials: {max_trials * executions_per_trial}")
print(f"Tuner: {tuner_algorithm}")
print(f"Objective: {tuner_objective} {objective}")
print(f"Executions per trial: {executions_per_trial}")
print(f"Results will be saved in: {os.path.join(tuning_dir, script_name)}")
print("="*50 + "\n")

# Run the tuner search
tuner.search(x=filtered_train_df,  # Pass the whole dataframe
             epochs=100,
             verbose=0)

# After tuner search
print("\n" + "="*50)
print("Hyperparameter tuning completed")
print("="*50 + "\n")

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Print best hyperparameters
best_hps = tuner.get_best_hyperparameters(1)[0]
print("Best Hyperparameters:")
for param, value in best_hps.values.items():
    print(f"{param}: {value}")
print()

# After getting the best hyperparameters and before training the best model

# Get the best early stopping patience
best_early_stopping_patience = best_hps.get('early_stopping_patience')

# Define early stopping callback with the best patience
early_stopping = EarlyStopping(monitor='val_loss', 
                               patience=best_early_stopping_patience, 
                               restore_best_weights=True)

# Get the best optimizer and learning rate
best_optimizer_choice = best_hps.get('optimizer')
best_learning_rate = best_hps.get('learning_rate')

if best_optimizer_choice == 'adam':
    best_optimizer = Adam(learning_rate=best_learning_rate)
elif best_optimizer_choice == 'rmsprop':
    best_optimizer = RMSprop(learning_rate=best_learning_rate)
else:
    best_optimizer = SGD(learning_rate=best_learning_rate)

# Recompile the best model with the best optimizer
best_model.compile(optimizer=best_optimizer, loss=best_model.loss)

# Train the best model
print("Training the best model...")
history = best_model.fit(filtered_train_df, filtered_train_df['Stock_Close'],
                         epochs=200,
                         validation_split=0.2,
                         callbacks=[early_stopping, SilentCallback()],
                         verbose=0)

# Evaluate the model
test_loss = best_model.evaluate(filtered_test_df, filtered_test_df['Stock_Close'], verbose=0)
print(f'\nTest loss: {test_loss}')

def forecast_future(model, last_known_sequence, preprocessor, strong_corr, time_horizon, best_sequence_length):
    single_step_predictions = []
    multi_step_predictions = []
    combined_predictions = []
    
    current_sequence = last_known_sequence[-best_sequence_length:].copy()
    
    for _ in range(time_horizon):
        # Single-step prediction
        single_step_input = current_sequence.reshape((1, best_sequence_length, -1))
        single_step_pred = model.predict(single_step_input, verbose=0)[0, 0]
        single_step_predictions.append(single_step_pred)
        
        # Multi-step prediction (predict next 5 steps)
        multi_step_pred = multi_step_forecast(model, current_sequence, 5, preprocessor, strong_corr, best_sequence_length)
        multi_step_predictions.append(multi_step_pred[0])
        
        # Combine predictions (you can adjust the weights)
        combined_pred = 0.7 * single_step_pred + 0.3 * multi_step_pred[0]
        combined_predictions.append(combined_pred)
        
        # Update current sequence for next iteration
        new_step = estimate_future_features(current_sequence[-1], combined_pred, strong_corr)
        current_sequence = np.vstack([current_sequence[1:], new_step])
    
    return np.array(single_step_predictions), np.array(multi_step_predictions), np.array(combined_predictions)

def multi_step_forecast(model, initial_sequence, num_steps, preprocessor, strong_corr, best_sequence_length):
    predictions = []
    current_sequence = initial_sequence[-best_sequence_length:].copy()
    
    for _ in range(num_steps):
        input_seq = current_sequence.reshape((1, best_sequence_length, -1))
        pred = model.predict(input_seq, verbose=0)
        predictions.append(pred[0, 0])
        
        new_step = estimate_future_features(current_sequence[-1], pred[0, 0], strong_corr)
        current_sequence = np.vstack([current_sequence[1:], new_step])
    
    return np.array(predictions)

def estimate_future_features(last_known_step, predicted_close, strong_corr):
    """
    Estimate future features based on the last known values and the predicted close price.
    """
    new_step = last_known_step.copy()
    new_step[strong_corr.index.get_loc('Stock_Close')] = predicted_close
    
    for feature in strong_corr.index:
        if feature != 'Stock_Close':
            # Use last known value for simplicity, could be replaced with more sophisticated forecasting
            new_step[strong_corr.index.get_loc(feature)] = last_known_step[strong_corr.index.get_loc(feature)]
    
    return new_step

# Replace the existing evaluation code with this:

# Get the best sequence length
best_sequence_length = best_hps.get('sequence_length')
print(f"Best sequence length: {best_sequence_length}")

# Prepare test data with the best sequence length
X_test, y_test = prepare_sequences(filtered_test_df, best_sequence_length)

# Forecast future values
last_known_sequence = X_test[-best_sequence_length:]  # Use the last known sequence from X_test
time_horizon = len(X_test)  # Use the same length as the test set for comparison

future_single, future_multi, future_combined = forecast_future(best_model, last_known_sequence, preprocessor, strong_corr, time_horizon, best_sequence_length)

# Get the MinMaxScaler from the ColumnTransformer
scaler = preprocessor.named_transformers_['num']

# Find the index of 'Stock_Close' in the numeric columns
stock_close_index = list(numeric_columns).index('Stock_Close')

# Inverse transform predictions and actual values
def inverse_transform_predictions(predictions):
    predictions_scaled = np.zeros((len(predictions), len(numeric_columns)))
    predictions_scaled[:, stock_close_index] = predictions.flatten()
    return scaler.inverse_transform(predictions_scaled)[:, stock_close_index]

future_single_original = inverse_transform_predictions(future_single)
future_multi_original = inverse_transform_predictions(future_multi)
future_combined_original = inverse_transform_predictions(future_combined)

y_test_scaled = np.zeros((len(y_test), len(numeric_columns)))
y_test_scaled[:, stock_close_index] = y_test.flatten()
y_test_original = scaler.inverse_transform(y_test_scaled)[:, stock_close_index]

# Calculate MAPE for both prediction methods
def calculate_mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100

single_step_mape = calculate_mape(y_test_original, future_single_original)
multi_step_mape = calculate_mape(y_test_original, future_multi_original)
combined_mape = calculate_mape(y_test_original, future_combined_original)

print(f"Single-step MAPE: {single_step_mape:.2f}%")
print(f"Multi-step MAPE: {multi_step_mape:.2f}%")
print(f"Combined MAPE: {combined_mape:.2f}%")

# Plot results
plt.figure(figsize=(14, 7))
plt.plot(y_test_original, label='Actual', color='black')
plt.plot(future_single_original, label='Single-step Predictions', color='blue')
plt.plot(future_multi_original, label='Multi-step Predictions', color='red')
plt.plot(future_combined_original, label='Combined Predictions', color='green')
plt.title('Comparison of Single-step, Multi-step, and Combined Predictions')
plt.xlabel('Time Step')
plt.ylabel('Stock Close Price')
plt.legend()
plt.show()

# At the very end of the script, after the plot:
# Calculate and print the total run time
end_time = time.time()
total_time = end_time - start_time
print(f"\nTotal run time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

# %%