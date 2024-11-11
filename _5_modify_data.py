from _1_config import *
from _2_fetch_data import *

def make_tz_naive(df):
    df.index = df.index.tz_localize(None)
    return df

def preprocess_data(data): # Data Preprocessing nan and interpolate, weekend removal
    data = data.infer_objects()  # Infer object types before interpolation
    data = data.interpolate().ffill().bfill()
    data = data[data.index.dayofweek < 5]  # Remove weekends
    print("\nNAN values in data after preprocessing:\n", data.isna().sum())
    return data

def perform_correlation_analysis(data, threshold):
    print("\n--- Correlation Analysis ---")
    
    correlations = data.corr()
    
    # Correlation with target variable (assuming 'Close' is the target)
    target_correlations = correlations['Close'].sort_values(ascending=False)
    target_correlations = target_correlations.drop('Close').dropna()  # Drop NaN values
    
    print("\nTop 10 Positive Correlations with Close:")
    print(target_correlations[target_correlations > 0].head(10).to_string())
    
    print("\nTop 10 Negative Correlations with Close:")
    print(target_correlations[target_correlations < 0].tail(10).to_string())
    
    # Filter features based on correlation threshold
    strong_corr = target_correlations[abs(target_correlations) >= threshold]
    weak_corr = target_correlations[abs(target_correlations) < threshold]
    
    print(f"\nFeatures with strong correlation (|r| >= {threshold}): {len(strong_corr)}")
    print(f"Features with weak correlation (|r| < {threshold}): {len(weak_corr)}")
    
    return strong_corr, weak_corr

def filter_significant_features(train_df, test_df, strong_corr):
    # Filter features based on correlation analysis
    filtered_train_df = train_df[strong_corr.index.tolist() + ['Close']]
    filtered_test_df = test_df[strong_corr.index.tolist() + ['Close']]
        
    return filtered_train_df, filtered_test_df

def train_test_split(data, test_size):
    # using sklearn train_test_split in _1_config.py
    # split the data into train and test sets, and transform into categorical and numeric columns
    train_data, test_data = sklearn_train_test_split(data, test_size=test_size, shuffle=False)

    # Identify categorical and numeric columns
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns
    numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns

    # Create a ColumnTransformer for onehot encoding and minmax scaler
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

    return train_processed_df, test_processed_df, feature_names, preprocessor

def prepare_sequences(data, sequence_length):
    X, y = [], []
    # Ensure 'Close' is in the DataFrame
    if 'Close' not in data.columns:
        raise ValueError("The DataFrame must contain the 'Close' column.")

    print("Preparing sequences with DataFrame shape:", data.shape)
    print("Columns in DataFrame:", data.columns.tolist())

    for i in range(len(data) - sequence_length):
        # Select the sequence of features and drop 'Close'
        X.append(data.iloc[i:(i + sequence_length)].drop('Close', axis=1).values)
        y.append(data.iloc[i + sequence_length]['Close'])

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Check the shape of X to ensure it matches the expected input shape
    print("Shape of X after preparation:", X.shape)
    print("Shape of y after preparation:", y.shape)

    if X.shape[1] != sequence_length or X.shape[2] != 30:  # Adjust 30 to the expected number of features
        raise ValueError(f"Expected input shape (None, {sequence_length}, 30), but got {X.shape}")

    return X, y