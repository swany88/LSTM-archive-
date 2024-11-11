from _1_config import *
from _3_utils import *
from _5_modify_data import *

def calculate_cycle_score(row, data): # cycle score calculated based off of yes no indicators to growth in different areas
    score = 0
    score += 1 if row['Close'] > row['SMA_20'] else -1
    score += 1 if row['GDP'] > data['GDP'].rolling(window=20).mean().loc[row.name] else -1
    score += 1 if row['Industrial_Production'] > data['Industrial_Production'].rolling(window=20).mean().loc[row.name] else -1
    score += 1 if row['Unemployment_Rate'] < data['Unemployment_Rate'].rolling(window=20).mean().loc[row.name] else -1
    score += 1 if row['Consumer_Confidence'] > data['Consumer_Confidence'].rolling(window=20).mean().loc[row.name] else -1
    score += 1 if row['Corporate_Profits'] > data['Corporate_Profits'].rolling(window=20).mean().loc[row.name] else -1
    score += 1 if row['SP500'] > data['SP500'].rolling(window=20).mean().loc[row.name] else -1
    return score

def determine_business_cycle(score): # determine the business cycle phase based on the cycle score
    if score >= 5:
        return 'Expansion'
    elif 2 <= score < 5:
        return 'Peak'
    elif -2 <= score < 2:
        return 'Contraction'
    else:
        return 'Trough'

def iterative_bci_assignment(data): # calls calculate_cycle_score and determine_business_cycle
    data['Cycle_Score'] = data.apply(lambda row: calculate_cycle_score(row, data), axis=1)
    data['Business_Cycle_Phase'] = data['Cycle_Score'].apply(determine_business_cycle)
    
    bci_periods = []
    current_period = data['Business_Cycle_Phase'].iloc[0]
    transition_count = 0
    transition_threshold = 20  # Adjust this value as needed

    for _, row in data.iterrows():
        if row['Business_Cycle_Phase'] == current_period:
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

def avg_BCI_timeframe(data): # get the average duration of each business cycle phase
    phase_durations = defaultdict(list)
    current_phase = data['Business_Cycle_Phase'].iloc[0]
    phase_start = data.index[0]
    
    for date, phase in zip(data.index[1:], data['Business_Cycle_Phase'].iloc[1:]):
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

def add_technical_indicators(data): # calls iterative BCI Assignment
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

    # New volatility indicator: Average True Range (ATR)
    tr1 = result['High'] - result['Low']
    tr2 = abs(result['High'] - result['Close'].shift())
    tr3 = abs(result['Low'] - result['Close'].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    result['ATR'] = tr.rolling(window=14).mean()
    iterative_bci_assignment(result)  # Add this line to assign BCI

    return result

def add_rolling_statistics(data): # add rolling mean, std, skew, kurtosis
    print("\n--- Adding Rolling Statistics ---")
    window_sizes = [5, 10, 20, 50, 100]
    new_features = {}
    close = data['Close']
    
    for window in window_sizes:
        new_features[f'rolling_mean_{window}'] = close.rolling(window=window, min_periods=1).mean()

        # Rolling Standard Deviation
        std = close.rolling(window=window, min_periods=2).std()
        new_features[f'rolling_std_{window}'] = std.bfill()

        # Rolling Skewness
        skew = close.rolling(window=window, min_periods=3).skew()
        new_features[f'rolling_skew_{window}'] = skew.bfill()

        # Rolling Kurtosis
        kurt = close.rolling(window=window, min_periods=4).apply(kurtosis)
        new_features[f'rolling_kurt_{window}'] = kurt.bfill()

        print(f"Added rolling statistics for window size {window}")
    
    result = pd.concat([data, pd.DataFrame(new_features, index=data.index)], axis=1)
    result = result.bfill().ffill()
    return result

def add_seasonal_indicators(data): # add month, quarter, dayofweek, istaxseason, isearningsseason, isyearend, isjanuary, issummerslowdown
    data['Month'] = data.index.month
    data['Quarter'] = data.index.quarter
    data['DayOfWeek'] = data.index.dayofweek
    data['IsTaxSeason'] = (data.index.month == 4).astype(int)
    data['IsEarningsSeason'] = data.index.month.isin([1, 4, 7, 10]).astype(int)
    data['IsYearEnd'] = (data.index.month == 12).astype(int)
    data['IsJanuary'] = (data.index.month == 1).astype(int)
    data['IsSummerSlowdown'] = data.index.month.isin([6, 7, 8]).astype(int)
    return data

def engineer_features(data): # calls preprocess_data, add_seasonal_indicators, add_rolling_statistics, add_technical_indicators
    data = preprocess_data(data)
    data = add_seasonal_indicators(data)
    data = add_rolling_statistics(data)
    data = add_technical_indicators(data)
    return preprocess_data(data)
