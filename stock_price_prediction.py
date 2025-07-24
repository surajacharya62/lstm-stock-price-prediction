import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional


stock_data = pd.read_csv("goog.csv")

# print(stock_data.head())

stock_data = stock_data [['Date','Open','Close']] 
stock_data.set_index('Date',drop=True,inplace=True) 
print(stock_data.head())

#visualizing the data-------------
# fg, ax =plt.subplots(1,2,figsize=(20,7))
# ax[0].plot(stock_data ['Open'],label='Open',color='green')
# ax[0].set_xlabel('Date',size=15)
# ax[0].set_ylabel('Price',size=15)
# ax[0].legend()

# ax[1].plot(stock_data ['Close'],label='Close',color='red')
# ax[1].set_xlabel('Date',size=15)
# ax[1].set_ylabel('Price',size=15)
# ax[1].legend()
# fg.show()

#preprocessing the data ----------------

scale_data = MinMaxScaler()
scaled_stock_data = scale_data.fit_transform(stock_data)
# print(stock_data)

## creating pipeline---------------------
pipeline_steps = [
    ('scaler', MinMaxScaler()),   
    
]

# splitting the data into train and test set-----------------
training_size = round(len(stock_data ) * 0.80)


train_data = scaled_stock_data[:training_size]
test_data  = scaled_stock_data[training_size:]


# print(train_data)

def create_sequence(dataset):
    sequences = []
    # print(dataset[49])
    # print(len(dataset))
    labels = []
    start_idx = 0
    for stop_idx in range(5, len(dataset)):
        # print(stop_idx)
        sequences.append(dataset[start_idx:stop_idx])
        labels.append(dataset[stop_idx])
        start_idx += 1
    return (np.array(sequences), np.array(labels))

train_seq, train_label = create_sequence(train_data)
test_seq, test_label = create_sequence(test_data)


# creating the model -----------
model = Sequential()
model.add(LSTM
          (
            units=15, 
            return_sequences=True,
            input_shape=(train_seq.shape[1],train_seq.shape[2])
            
            ))
model.add(Dropout(0.1))
model.add(LSTM(units=15))
model.add(Dense(2))
model.compile(loss="mean_squared_error", 
              optimizer=Adam(learning_rate=0.001), metrics=["mean_absolute_error"])
model.summary()


# training the model
model.fit(train_seq, train_label, epochs=50,validation_data=(test_seq, test_label), verbose=1)
test_predicted = model.predict(test_seq)
print(test_predicted.shape)
test_inverse_predicted = scale_data.inverse_transform(test_predicted)
# print(test_predicted)
# print("-------------------------------")
# print(test_inverse_predicted)

#Merging actual and predicted data for better visualization

date_indices = stock_data.index[-test_label.shape[0]:]
# print(date_indices)
# print(test_label.shape)
# print(test_seq.shape)

pred = pd.DataFrame(test_inverse_predicted,columns=['open_predicted','close_predicted'])
actual = scale_data.inverse_transform(test_label)
actual = pd.DataFrame(actual,columns=['open','close'])

all_data = pd.concat([pred,actual],axis=1)
all_data.index=date_indices
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
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

