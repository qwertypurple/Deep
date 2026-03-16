pip install tensorflow pandas scikit-learn matplotlib

# Time Series Forecasting using LSTM

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# ---------------------------------
# 1. Generate Sample Time Series
# ---------------------------------

time = np.arange(0, 100, 0.1)
data = np.sin(time)

df = pd.DataFrame(data, columns=['value'])


# ---------------------------------
# 2. Data Scaling
# ---------------------------------

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df)


# ---------------------------------
# 3. Create Sequences
# ---------------------------------

def create_sequences(data, time_step=10):

    X, y = [], []

    for i in range(len(data) - time_step):
        X.append(data[i:i+time_step, 0])
        y.append(data[i+time_step, 0])

    return np.array(X), np.array(y)


time_step = 10
X, y = create_sequences(scaled_data, time_step)


# ---------------------------------
# 4. Reshape for LSTM
# ---------------------------------

X = X.reshape(X.shape[0], X.shape[1], 1)


# ---------------------------------
# 5. Train-Test Split
# ---------------------------------

train_size = int(len(X) * 0.8)

X_train = X[:train_size]
X_test = X[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]


# ---------------------------------
# 6. Build LSTM Model
# ---------------------------------

model = Sequential()

model.add(LSTM(50, input_shape=(time_step,1)))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')


# ---------------------------------
# 7. Train Model
# ---------------------------------

model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)


# ---------------------------------
# 8. Predictions
# ---------------------------------

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)


# ---------------------------------
# 9. Evaluation
# ---------------------------------

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

y_train_actual = scaler.inverse_transform(y_train.reshape(-1,1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1,1))

train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_predict))
test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_predict))

print("Train RMSE:", train_rmse)
print("Test RMSE:", test_rmse)


# ---------------------------------
# 10. Plot Results
# ---------------------------------

plt.figure(figsize=(10,5))

plt.plot(df.values, label='Actual Data')

train_plot = np.empty_like(df.values)
train_plot[:] = np.nan
train_plot[time_step:train_size + time_step] = train_predict

test_plot = np.empty_like(df.values)
test_plot[:] = np.nan
test_plot[train_size + time_step:] = test_predict

plt.plot(train_plot, label='Training Prediction')
plt.plot(test_plot, label='Testing Prediction')

plt.title("LSTM Time Series Forecasting")
plt.xlabel("Time")
plt.ylabel("Value")

plt.legend()
plt.show()
