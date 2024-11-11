from _1_config import *
from _2_fetch_data import *
from _3_utils import *
from _4_feat_engineering import *
from _5_modify_data import *
from _6_hypermodel import *
from _7_visualize import *

check_tensorflow_settings()

start_time = time.time()
symbol = 'QCOM'

# FRED API key
fred_api_key = os.getenv('FRED_API_KEY')
fred = Fred(api_key=fred_api_key)

# Data Collection
print("\nData Collection")
start_date, end_date = '2019-01-01', datetime.now().strftime('%Y-%m-%d')  # Use datetime.now() to get the current date
data = collect_data(symbol, start_date, end_date, fred)

# Preprocess and engineer features
data = engineer_features(data)

# Split the data into train and test sets, and transform into categorical and numeric columns
train_df, test_df, feature_names, preprocessor = train_test_split(data, test_size=0.2)

# Correlation Analysis
strong_corr, weak_corr = perform_correlation_analysis(train_df, threshold=0.2)

# Filter Significant Features
filtered_train_df, filtered_test_df = filter_significant_features(train_df, test_df, strong_corr)

# After filtering significant features
print("Filtered Train DataFrame:")
print(filtered_train_df.head())  # Print the first few rows
print("Columns in filtered_train_df:", filtered_train_df.columns.tolist())
print("Shape of filtered_train_df:", filtered_train_df.shape)

print("Filtered Test DataFrame:")
print(filtered_test_df.head())  # Print the first few rows
print("Columns in filtered_test_df:", filtered_test_df.columns.tolist())
print("Shape of filtered_test_df:", filtered_test_df.shape)

time_horizon = 60
tuning_params = set_tuning_params(filtered_train_df, feature_names, time_horizon)

# Tune Hyperparameters
tuner, best_model, best_hps = tune_hyperparameters(filtered_train_df, filtered_test_df, tuning_params, symbol, max_trials=5, executions_per_trial=1)

# Train the best model
best_model, history = train_best_model(tuner, filtered_train_df, filtered_test_df)

# Get the MinMaxScaler from the ColumnTransformer
scaler = preprocessor.named_transformers_['num']

# Get the best sequence length
best_sequence_length = best_hps.get('sequence_length')
print(f"Best sequence length: {best_sequence_length}")

# Prepare test data with the best sequence length
X_test, y_test = prepare_sequences(filtered_test_df, best_sequence_length)

# Forecast future values
last_known_sequence = X_test[-best_sequence_length:]  # Use the last known sequence from X_test
time_horizon = len(X_test)  # Use the same length as the test set for comparison

# Walk Forward Validation
predictions, y_test_original = walk_forward_validation(best_model, X_test, y_test, scaler)

# Calculate MAPE
mape = np.mean(np.abs((y_test_original - predictions) / y_test_original)) * 100
print(f'MAPE: {mape:.2f}%')

# Predict Future Values
predictions_future, y_test_original_future = predict_future_values(best_model, X_test, y_test, n_future_steps=60, close_scaler=scaler)

# Rescale the data if needed
train_data = data['Close'][:len(train_df)]
test_data = data['Close'][-len(predictions):]  # Get the actual test period data

# Create future dates
last_test_date = test_df.index[-1]
future_dates = pd.date_range(start=last_test_date + pd.Timedelta(days=1), 
                           periods=len(predictions_future), 
                           freq='B')

# Create a clean data dictionary for plotting
plot_data = {
    'dates': {
        'train': train_df.index,
        'test': test_df.index[-len(predictions):],
        'future': future_dates
    },
    'values': {
        'train': train_data,
        'test': test_data,
        'test_predictions': predictions,
        'future_predictions': predictions_future
    }
}

# Now call the plotting function with clean data
plot_predictions(plot_data, symbol)
