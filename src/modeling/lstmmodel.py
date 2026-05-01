import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import r2_score

# 1. Load the data
print("Loading data...")
df = pd.read_csv("data/raw/aapl_ticker.csv")

# We will use the 'Close' price to predict the trend
data = df.filter(['Close'])
dataset = data.values

# Get the number of rows to train the model on (80% of data)
training_data_len = int(len(dataset) * 0.8)

# 2. Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# 3. Create the training data set
train_data = scaled_data[0:int(training_data_len), :]

# Split the data into x_train and y_train data sets
x_train = []
y_train = []

# Using 60 days of historical data to predict the next day
sequence_length = 60

for i in range(sequence_length, len(train_data)):
    x_train.append(train_data[i-sequence_length:i, 0])
    y_train.append(train_data[i, 0])
    
# Convert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data for LSTM model (samples, time steps, features)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# 4. Build the LSTM model
print("Building the model...")
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(64))
model.add(Dropout(0.2))

model.add(Dense(1))

# 5. Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# 6. Train the model
print("Training the model...")
model.fit(x_train, y_train, batch_size=32, epochs=10)

# 7. Create the testing data set
test_data = scaled_data[training_data_len - sequence_length: , :]

# Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(sequence_length, len(test_data)):
    x_test.append(test_data[i-sequence_length:i, 0])
    
# Convert the data to a numpy array
x_test = np.array(x_test)

# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# 8. Get the models predicted price values 
print("Predicting...")
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# 9. Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
print(f"RMSE: {rmse}")

r2 = r2_score(y_test, predictions)
print(f"R2 Score: {r2}")

# Calculate direction accuracy
actual_direction = np.sign(y_test[1:] - y_test[:-1])
predicted_direction = np.sign(predictions[1:] - y_test[:-1])
direction_accuracy = np.mean(actual_direction == predicted_direction)
print(f"Direction Accuracy: {direction_accuracy * 100:.2f}%")

# 10. Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

plt.figure(figsize=(16,8))
plt.title('LSTM Model for AAPL Stock Price Prediction')
plt.xlabel('Time', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Actual Val', 'Predictions'], loc='lower right')
plt.show()
