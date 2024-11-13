from _1_config import *

def one_hot_encode_categorical(data):
    categorical_columns = data.select_dtypes(include=['object']).columns
    numeric_data = data.select_dtypes(exclude=['object'])
    
    if len(categorical_columns) > 0:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_data = encoder.fit_transform(data[categorical_columns])
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_columns), index=data.index)
        final_data = pd.concat([numeric_data, encoded_df], axis=1)
    else:
        final_data = numeric_data
    
    return final_data

def log_and_scale(data):
    processed_data = data.copy()
    numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
    non_binary_columns = [col for col in numeric_columns if processed_data[col].nunique() > 2]
    
    for col in non_binary_columns:
        processed_data[col] = np.sign(processed_data[col]) * np.log1p(np.abs(processed_data[col]))
    
    processed_data = processed_data.replace([np.inf, -np.inf], np.nan)
    processed_data = processed_data.interpolate()
    
    # Separate scaler for 'Close'
    close_scaler = MinMaxScaler(feature_range=(0, 1))
    processed_data['Close_Scaled'] = close_scaler.fit_transform(processed_data[['Close']])
    
    # Scale other features
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    other_columns = [col for col in non_binary_columns if col != 'Close']
    processed_data[other_columns] = feature_scaler.fit_transform(processed_data[other_columns])
    
    return processed_data, close_scaler, feature_scaler

def inv_log_scale(scaled_values, close_scaler):
    # Reshape to 2D array if it's 1D
    if scaled_values.ndim == 1:
        scaled_values = scaled_values.reshape(-1, 1)
    
    # Inverse transform the scaled values
    original_values = close_scaler.inverse_transform(scaled_values)
    
    # Apply inverse of log transformation
    return np.expm1(original_values)

def diff_and_scale(data): # difference and scale 
    processed_data = data.copy()
    numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
    non_binary_columns = [col for col in numeric_columns if processed_data[col].nunique() > 2]
    
    # Apply differencing
    for col in non_binary_columns:
        processed_data[f'{col}_diff'] = processed_data[col].diff()
    
    # Drop the first row (NaN after differencing) and original columns
    processed_data = processed_data.dropna()
    processed_data = processed_data.drop(columns=non_binary_columns)
    
    # Separate scaler for 'Close'
    close_scaler = MinMaxScaler(feature_range=(0, 1))
    processed_data['Close_Scaled'] = close_scaler.fit_transform(processed_data[['Close_diff']])
    
    # Scale other features
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    other_columns = [col for col in processed_data.columns if col.endswith('_diff') and col != 'Close_diff']
    processed_data[other_columns] = feature_scaler.fit_transform(processed_data[other_columns])
    
    return processed_data, close_scaler, feature_scaler

def inv_diff_scale(scaled_values, close_scaler, feature_scaler, original_data): # inverse scale values seperate from close (if scaled seperately)
    # Reshape to 2D array if it's 1D
    if scaled_values.ndim == 1:
        scaled_values = scaled_values.reshape(-1, 1)
    
    # Separate the close price (assumed to be the last column)
    close_scaled = scaled_values[:, -1].reshape(-1, 1)
    other_features_scaled = scaled_values[:, :-1]
    
    # Inverse transform the close price
    close_diff = close_scaler.inverse_transform(close_scaled)
    
    # Inverse transform other features
    other_features_diff = feature_scaler.inverse_transform(other_features_scaled) if feature_scaler else None
    
    # Combine the results
    diff_values = np.column_stack((other_features_diff, close_diff)) if other_features_diff is not None else close_diff
    
    # Revert the differencing
    original_values = np.zeros_like(diff_values)
    original_values[0] = original_data.iloc[-1].values + diff_values[0]
    for i in range(1, len(diff_values)):
        original_values[i] = original_values[i-1] + diff_values[i]
    
    return original_values

def inv_scale(scaled_values, scaler):   # inverse transform the scaled values
    # Reshape to 2D array if it's 1D
    if scaled_values.ndim == 1:
        scaled_values = scaled_values.reshape(-1, 1)
    
    # Inverse transform the scaled values
    original_values = scaler.inverse_transform(scaled_values)
    
    # Apply inverse of log transformation
    return np.expm1(original_values)

def perform_enhanced_fft_analysis(data):    # FFT Analysis, for identifying cyclical patterns
    print("\n--- Enhanced FFT Analysis ---")
    
    close_prices = data['Close'].values
    detrended_prices = detrend(close_prices)
    
    window = np.hanning(len(detrended_prices))
    windowed_prices = detrended_prices * window
    fft_result = fft(windowed_prices)
    frequencies = np.fft.fftfreq(len(detrended_prices), d=1)
    amplitudes = np.abs(fft_result)
    
    all_cycles = {}
    time_scales = [
        ('Long-term', 252, 1260),  # 1-5 years
        ('Medium-term', 21, 252),   # 1 month to 1 year
        ('Short-term', 2, 21)       # 2 days to 1 month
    ]
    
    for scale_name, min_period, max_period in time_scales:
        print(f"\n{scale_name} Analysis:")
        mask = (1/max_period <= np.abs(frequencies)) & (np.abs(frequencies) <= 1/min_period)
        scale_frequencies = frequencies[mask]
        scale_amplitudes = amplitudes[mask]
        
        sorted_indices = np.argsort(scale_amplitudes)[::-1]
        top_frequencies = scale_frequencies[sorted_indices[:5]]
        top_amplitudes = scale_amplitudes[sorted_indices[:5]]
        
        print(f"{'Index':<5} {'Frequency (Hz)':<20} {'Period (days)':<15} {'Amplitude':<15}")
        print("-" * 60)
        for i, (freq, amp) in enumerate(zip(top_frequencies, top_amplitudes), 1):
            period = 1 / abs(freq) if freq != 0 else np.inf
            print(f"{i:<5} {freq:<20.6f} {period:<15.2f} {amp:<15.2f}")
            
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

def print_fft_analysis(fft_cycles):
    print("\n--- Enhanced FFT Analysis Results ---")
    print(f"{'Cycle Name':<25} {'Frequency (Hz)':<20} {'Amplitude':<15} {'Period (days)':<15}")
    print("-" * 75)
    
    for cycle_name, (frequency, amplitude, period) in fft_cycles.items():
        print(f"{cycle_name:<25} {frequency:<20.6f} {amplitude:<15.2f} {period:<15.2f}")

# Load saved model to use if you want to make predictions without retraining
def load_saved_model(filepath):
    model_data = torch.load(filepath)
    
    # Initialize model with saved hyperparameters
    model = LSTMModel(
        input_size=filtered_train_df.shape[1]-1,
        hidden_size=model_data['hyperparameters']['hidden_size'],
        num_layers=model_data['hyperparameters']['num_layers'],
        dropout=model_data['hyperparameters']['dropout']
    ).to(device)
    
    # Load the saved state
    model.load_state_dict(model_data['model_state'])
    
    return model, model_data
#! implementation for finding sequence length
# # Average Timeframes for Each Business Cycle Phase (in days)
# average_timeframes = avg_BCI_timeframe(data)
# print("\nAverage Timeframes for Each Cycle Phase (in days):")
# for phase, duration in average_timeframes.items():
#     print(f"{phase}: {duration:.2f}")

# # This can be used for sequence length
# avg_bc_length = round(sum(average_timeframes.values()))
# print(f"\nAverage Business Cycle Length: {avg_bc_length:.2f} days")

# # Enhanced FFT Analysis
# FFT_cycles = perform_enhanced_fft_analysis(data)
#! END


def plot_residuals(residuals):
    plt.figure(figsize=(14, 7))
    plt.plot(residuals)
    plt.title('Residuals (Actual - Predicted)')
    plt.xlabel('Time Step')
    plt.ylabel('Residual ($)')
    plt.axhline(y=0, color='r', linestyle='--')

    rolling_mean = pd.Series(residuals).rolling(window=20).mean()
    plt.plot(rolling_mean, color='g', label='20-period Rolling Mean')

    plt.legend()
    plt.show()

def plot_acf_residuals(residuals):
    plot_acf(residuals)
    plt.title('Autocorrelation of Residuals')
    plt.show()

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

    print(f"\nNumber of NaN values in 'Close' column: {cleaned_data['Close'].isna().sum()}")

    day_counts = cleaned_data.index.dayofweek.value_counts().sort_index()
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    print("\nSummary of days of the week:")
    for day, count in day_counts.items():
        print(f"{day_names[day]}: {count}")

    weekend_data = cleaned_data[cleaned_data.index.dayofweek.isin([5, 6])]
    if not weekend_data.empty:
        print("\nWarning: Data contains weekend entries:")
        print(weekend_data)