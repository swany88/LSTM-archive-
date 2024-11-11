from _1_config import *
from _3_utils import *
from _5_modify_data import *

def set_tuning_params(data, feature_names, time_horizon):
    num_samples, num_features = data.shape
    num_features -= 1  # Subtract 1 to account for 'Close'

    params = {}
    params['lstm_units_min'] = min(32, num_features)
    params['lstm_units_max'] = min(256, num_features * 8)
    params['dense_units_min'] = min(16, num_features)
    params['dense_units_max'] = min(128, num_features * 4)
    params['max_lstm_layers'] = min(5, max(1, int(np.log2(num_samples / time_horizon))))
    params['optimizers'] = ['adam', 'rmsprop', 'sgd']
    params['batch_sizes'] = [2**i for i in range(5, min(10, int(np.log2(num_samples)) + 1))]
    params['sequence_length_min'] = 2 * time_horizon
    params['sequence_length_max'] = 3 * time_horizon
    params['min_features'] = max(5, num_features // 4)
    params['max_features'] = num_features
    params['early_stopping_patience_min'] = 5
    params['early_stopping_patience_max'] = min(50, num_samples // (10 * time_horizon))
    params['use_lr_schedule'] = [True, False]
    params['lr_schedule_patience_min'] = 3
    params['lr_schedule_patience_max'] = 10
    params['lr_schedule_factor_min'] = 0.1
    params['lr_schedule_factor_max'] = 0.5
    params['dropout_min'] = 0.0
    params['dropout_max'] = min(0.5, 1000 / num_samples)
    params['l1_min'] = 1e-6
    params['l1_max'] = 1e-3
    params['l2_min'] = 1e-6
    params['l2_max'] = 1e-3
    params['loss_functions'] = ['mse', 'mae', 'huber']
    params['learning_rate_min'] = 1e-4
    params['learning_rate_max'] = 1e-2

    return params

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
        
        inputs = tf.keras.Input(shape=(sequence_length, self.input_shape[1]))
        
        x = inputs
        for i in range(hp.Int('num_lstm_layers', 1, self.tuning_params['max_lstm_layers'])):
            x = tf.keras.layers.LSTM(units=hp.Int(f'lstm_units_{i}', 
                                  self.tuning_params['lstm_units_min'], 
                                  self.tuning_params['lstm_units_max'], 
                                  step=32),
                     return_sequences=(i < hp.Int('num_lstm_layers', 1, self.tuning_params['max_lstm_layers']) - 1),
                     kernel_regularizer=tf.keras.regularizers.l1_l2(
                         l1=hp.Float(f'l1_{i}', self.tuning_params['l1_min'], self.tuning_params['l1_max'], sampling='LOG'),
                         l2=hp.Float(f'l2_{i}', self.tuning_params['l2_min'], self.tuning_params['l2_max'], sampling='LOG')))(x)
            x = tf.keras.layers.Dropout(hp.Float(f'dropout_{i}', self.tuning_params['dropout_min'], self.tuning_params['dropout_max']))(x)
        
        x = tf.keras.layers.Dense(hp.Int('dense_units', 
                         self.tuning_params['dense_units_min'], 
                         self.tuning_params['dense_units_max'], 
                         step=16), 
                  activation='relu')(x)
        
        outputs = tf.keras.layers.Dense(1)(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        optimizer_choice = hp.Choice('optimizer', self.tuning_params['optimizers'])
        learning_rate = hp.Float('learning_rate', self.tuning_params['learning_rate_min'], self.tuning_params['learning_rate_max'], sampling='LOG')

        if optimizer_choice == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_choice == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

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
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True)
        
        callbacks = [early_stopping, SilentCallback()]
        
        # Learning rate schedule setup
        if hp.Boolean('use_lr_schedule'):
            lr_schedule_patience = hp.Int('lr_schedule_patience', 
                                          self.tuning_params['lr_schedule_patience_min'], 
                                          self.tuning_params['lr_schedule_patience_max'])
            lr_schedule_factor = hp.Float('lr_schedule_factor', 
                                          self.tuning_params['lr_schedule_factor_min'], 
                                          self.tuning_params['lr_schedule_factor_max'])
            lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=lr_schedule_factor, patience=lr_schedule_patience)
            callbacks.append(lr_schedule)
        
        # Update callbacks if they're provided in kwargs
        if 'callbacks' in kwargs:
            existing_callbacks = kwargs['callbacks']
            callbacks.extend([cb for cb in existing_callbacks if not isinstance(cb, (tf.keras.callbacks.EarlyStopping, SilentCallback, tf.keras.callbacks.ReduceLROnPlateau))])
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
    
def tune_hyperparameters(filtered_train_df, filtered_test_df, tuning_params, symbol, max_trials, executions_per_trial):
    # Get the current script name without the .py extension
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Create a directory name based on the stock symbol
    stock_dir = f"tuning_results_{symbol}"
    # Create the full path for the tuning results
    tuning_dir = os.path.join(script_dir, stock_dir)
    # Create the directory if it doesn't exist
    try:
        os.makedirs(tuning_dir, exist_ok=True)
        print(f"Directory created: {tuning_dir}")
    except Exception as e:
        print(f"Error creating directory: {e}")

    tuner_algorithm = "BayesianOptimization"
    objective = "val_loss"
    tuner_objective = "minimize"

    input_shape = (None, filtered_train_df.shape[1] - 1)
    time_horizon = len(filtered_test_df)

    tuner = BayesianOptimization(
        ComplexLSTMHyperModel(input_shape=input_shape, max_trials=max_trials, tuning_params=tuning_params, time_horizon=time_horizon),
        objective=objective,
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        directory=tuning_dir,
        project_name=script_name
    )

    print("\n" + "="*50)
    print(f"Starting {tuner_algorithm} for {symbol}")
    print(f"Script: {script_name}")
    print(f"Number of trials: {max_trials * executions_per_trial}")
    print(f"Tuner: {tuner_algorithm}")
    print(f"Objective: {tuner_objective} {objective}")
    print(f"Executions per trial: {executions_per_trial}")
    print(f"Results will be saved in: {os.path.join(tuning_dir, script_name)}")
    print("="*50 + "\n")

    tuner.search(x=filtered_train_df, epochs=100, verbose=0)

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

    return tuner, best_model, best_hps  # Return tuner, best model, and best hyperparameters

def train_best_model(tuner, filtered_train_df, filtered_test_df):
    # Get the best model and hyperparameters from the tuner
    best_model = tuner.get_best_models(num_models=1)[0]
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    best_sequence_length = best_hps.get('sequence_length')

    # Debugging: Check the columns of filtered_train_df
    print("Columns in filtered_train_df:", filtered_train_df.columns.tolist())
    print("Shape of filtered_train_df:", filtered_train_df.shape)

    # Check for 'Close' column
    if 'Close' not in filtered_train_df.columns:
        raise ValueError("The DataFrame must contain the 'Close' column.")

    # Prepare sequences for training and testing
    X_train, y_train = prepare_sequences(filtered_train_df, best_sequence_length)
    X_test, y_test = prepare_sequences(filtered_test_df, best_sequence_length)

    # Get the best early stopping patience
    best_early_stopping_patience = best_hps.get('early_stopping_patience')
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                     patience=best_early_stopping_patience, 
                                                     restore_best_weights=True)

    # Get the best optimizer and learning rate
    best_optimizer_choice = best_hps.get('optimizer')
    best_learning_rate = best_hps.get('learning_rate')

    if best_optimizer_choice == 'adam':
        best_optimizer = tf.keras.optimizers.Adam(learning_rate=best_learning_rate)
    elif best_optimizer_choice == 'rmsprop':
        best_optimizer = tf.keras.optimizers.RMSprop(learning_rate=best_learning_rate)
    else:
        best_optimizer = tf.keras.optimizers.SGD(learning_rate=best_learning_rate)

    # Recompile the best model with the best optimizer
    best_model.compile(optimizer=best_optimizer, loss=best_model.loss)

    # Train the best model
    print("Training the best model...")
    history = best_model.fit(X_train, y_train,
                             epochs=200,
                             validation_split=0.2,
                             callbacks=[early_stopping, SilentCallback()],
                             verbose=0)

    # Evaluate the model
    test_loss = best_model.evaluate(X_test, y_test, verbose=0)
    print(f'\nTest loss: {test_loss}')

    return best_model, history

def time_series_cv(X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_index, val_index in tscv.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        yield X_train, X_val, y_train, y_val

def walk_forward_validation(model, X_test, y_test, scaler):
    predictions = []
    input_seq = X_test[0]  # First sequence

    for i in range(len(X_test)):
        input_seq_reshaped = input_seq.reshape((1, input_seq.shape[0], input_seq.shape[1]))
        pred = model.predict(input_seq_reshaped, verbose=0)
        predictions.append(pred[0, 0])
        
        if i + 1 < len(X_test):
            input_seq = np.vstack([input_seq[1:], X_test[i + 1, -1]])
            input_seq[-1, -1] = y_test[i]

    # Create a dummy array with the same number of features as the original data
    n_features = len(scaler.get_feature_names_out())
    dummy_array = np.zeros((len(predictions), n_features))
    
    # Find the index of 'Close' in the feature names
    close_index = list(scaler.get_feature_names_out()).index('Close')
    
    # Put our predictions in the Close column
    dummy_array[:, close_index] = predictions
    
    # Inverse transform the entire array
    inverse_transformed = scaler.inverse_transform(dummy_array)
    
    # Extract just the Close predictions
    inverse_predictions = inverse_transformed[:, close_index]

    # Do the same for y_test
    dummy_array_y = np.zeros((len(y_test), n_features))
    dummy_array_y[:, close_index] = y_test
    inverse_transformed_y = scaler.inverse_transform(dummy_array_y)
    y_test_original = inverse_transformed_y[:, close_index]

    return inverse_predictions, y_test_original

def predict_future_values(model, X_test, y_test, n_future_steps, close_scaler):
    predictions = []
    input_seq = X_test[0]  # Start with the last sequence in the test set

    # First, predict the remaining test set
    for i in range(len(X_test)):
        input_seq_reshaped = input_seq.reshape((1, input_seq.shape[0], input_seq.shape[1]))
        pred = model.predict(input_seq_reshaped, verbose=0)
        predictions.append(pred[0, 0])
        
        if i + 1 < len(X_test):
            input_seq = np.vstack([input_seq[1:], X_test[i + 1, -1]])
            input_seq[-1, -1] = y_test[i]

    # Then, predict future values
    for _ in range(n_future_steps):
        input_seq_reshaped = input_seq.reshape((1, input_seq.shape[0], input_seq.shape[1]))
        pred = model.predict(input_seq_reshaped, verbose=0)
        predictions.append(pred[0, 0])
        
        # Update the input sequence
        input_seq = np.vstack([input_seq[1:], np.append(input_seq[-1, :-1], pred[0, 0])])

    # Create a dummy array with the same number of features as the original data
    n_features = len(close_scaler.get_feature_names_out())
    dummy_array = np.zeros((len(predictions), n_features))
    
    # Find the index of 'Close' in the feature names
    close_index = list(close_scaler.get_feature_names_out()).index('Close')
    
    # Put our predictions in the Close column
    dummy_array[:, close_index] = predictions
    
    # Inverse transform the entire array
    inverse_transformed = close_scaler.inverse_transform(dummy_array)
    
    # Extract just the Close predictions
    inverse_predictions = inverse_transformed[:, close_index]

    # Do the same for y_test
    dummy_array_y = np.zeros((len(y_test), n_features))
    dummy_array_y[:, close_index] = y_test
    inverse_transformed_y = close_scaler.inverse_transform(dummy_array_y)
    y_test_original = inverse_transformed_y[:, close_index]

    return inverse_predictions, y_test_original