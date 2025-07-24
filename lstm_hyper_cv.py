import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import keras_tuner as kt
import itertools


# Load and preprocess data-------------------
stock_data = pd.read_csv("aapl.csv")
stock_data = stock_data[['Date', 'Open', 'Close']]
stock_data.set_index('Date', drop=True, inplace=True)

# Scale the data--------------------
scale_data = MinMaxScaler()
scaled_stock_data = scale_data.fit_transform(stock_data)

# Split into train and test sets-----------------
training_size = round(len(stock_data) * 0.80)
train_data = scaled_stock_data[:training_size]
test_data = scaled_stock_data[training_size:]

# Define sequence creation function to create sequence or series of data--------------
def create_sequence(dataset, seq_length):
    sequences, labels = [], []
    for start_idx in range(len(dataset) - seq_length):
        sequences.append(dataset[start_idx:start_idx + seq_length])
        labels.append(dataset[start_idx + seq_length])
    return np.array(sequences), np.array(labels)

# Defining hyperparameter grid-----------------
param_grid = {
    'units': [16, 32, 64],
    'num_layers': [1, 2],
    'dropout_rate': [0.1, 0.2, 0.3],
    'learning_rate': [1e-4, 1e-3, 1e-2],
    'batch_size': [8, 16],
    'epochs': [50, 100],
    'seq_length': [5, 10, 20]
}

# Model building 
def build_model(hp):
    model = Sequential()
    seq_length = hp.Choice('seq_length', param_grid['seq_length'])
    
    # Generate sequences with the chosen sequence length
    train_seq, train_label = create_sequence(train_data, seq_length)
    test_seq, test_label = create_sequence(test_data, seq_length)
    
    model.add(Input(shape=(seq_length, train_data.shape[1])))
    for i in range(hp.Int('num_layers', min_value=min(param_grid['num_layers']), max_value=max(param_grid['num_layers']))):
        model.add(LSTM(
            units=hp.Choice(f'units_{i}', param_grid['units']),
            return_sequences=(i < hp.Int('num_layers', min_value=min(param_grid['num_layers']), max_value=max(param_grid['num_layers'])) - 1)
        ))    
        model.add(Dropout(hp.Choice(f'dropout_{i}', param_grid['dropout_rate'])))
    
    model.add(Dense(2))
    model.compile(
        optimizer=Adam(learning_rate=hp.Choice('learning_rate', param_grid['learning_rate'])),
        loss='mean_squared_error',
        metrics=['mean_absolute_error', 'mean_absolute_percentage_error']
    )
    return model, (train_seq, train_label, test_seq, test_label)

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Initialize time-series cross-validation
n_splits = 3
tscv = TimeSeriesSplit(n_splits=n_splits)

best_model = None
best_val_loss = float('inf')
best_hps = None
best_data = None

# Generate all combinations of batch_size and epochs
batch_epoch_combinations = list(itertools.product(param_grid['batch_size'], param_grid['epochs']))

for batch_size, epochs in batch_epoch_combinations:
    print(f"\nTuning with batch_size={batch_size}, epochs={epochs}")
    
    # Initialize tuner for this combination
    tuner = kt.RandomSearch(
        lambda hp: build_model(hp)[0],
        objective='val_loss',
        max_trials=5,  
        executions_per_trial=2,
        directory='tuner_results',
        project_name=f'lstm_tuning_batch{batch_size}_epochs{epochs}'
    )
    

    hp = kt.HyperParameters()
    _, (train_seq, train_label, test_seq, test_label) = build_model(hp)
    
  

    # cross validation
    cv_val_losses = []

    for fold, (train_index,val_index) in enumerate(tscv.split(train_seq)):
        print(f"Fold {fold + 1}/{n_splits}")
        x_train,x_val = train_seq[train_index], train_seq[val_index]
        y_train,y_val = train_label[train_index], train_label[val_index]

        tuner.search(
        train_seq, train_label,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping],
        verbose=1
        )

        # Get the best model for this batch_size and epochs
        current_best_model = tuner.get_best_models(num_models=1)[0]
        val_loss = current_best_model.evaluate(x_val,y_val, verbose=0)
        cv_val_losses.append(val_loss)

    avg_val_loss = np.mean(cv_val_losses)         
    # Update best model if current is better
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model = current_best_model
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        best_data = (train_seq, train_label, test_seq, test_label)

# Print best hyperparameters
print("\nBest Hyperparameters:")
print(f"Units: {best_hps.get('units_0')}")
print(f"Number of Layers: {best_hps.get('num_layers')}")
print(f"Dropout Rate: {best_hps.get('dropout_0')}")
print(f"Learning Rate: {best_hps.get('learning_rate')}")
print(f"Sequence Length: {best_hps.get('seq_length')}")
print(f"Best Validation Loss: {best_val_loss}")


# Evaluate the best model
train_seq, train_label, test_seq, test_label = best_data
best_model.evaluate(test_seq, test_label)


# Visualize predictions
test_predicted = best_model.predict(test_seq)
test_inverse_predicted = scale_data.inverse_transform(test_predicted)
pred = pd.DataFrame(test_inverse_predicted, columns=['open_predicted', 'close_predicted'])
actual = pd.DataFrame(scale_data.inverse_transform(test_label), columns=['open', 'close'])
date_indices = stock_data.index[-test_label.shape[0]:]
# date_indices = stock_data[training_size:].index[-len(test_label):]
all_data = pd.concat([pred, actual], axis=1)
all_data.index = date_indices
print(all_data)

plt.figure(figsize=(14, 7))
plt.plot(all_data.index, all_data['open'], label='Actual Open', color='blue', marker='o')
plt.plot(all_data.index, all_data['open_predicted'], label='Predicted Open', color='cyan', linestyle='--', marker='x')
plt.plot(all_data.index, all_data['close'], label='Actual Close', color='red', marker='o')
plt.plot(all_data.index, all_data['close_predicted'], label='Predicted Close', color='orange', linestyle='--', marker='x')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Actual vs Predicted Stock Prices')
plt.legend()
plt.grid(True)
# plt.xticks(rotation=45)
# plt.tight_layout()
plt.show()

