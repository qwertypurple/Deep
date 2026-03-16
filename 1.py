# Program 1: Linear Regression using TensorFlow

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Training data
x = np.array([0, 1, 2, 3, 4, 5], dtype=float)
y = 3 * x + 2

# Create neural network model
model = keras.Sequential([
    layers.Dense(1, input_shape=[1])
])

# Compile model
model.compile(optimizer='adam', loss='mse')

print("Training the model...")

# Train model
history = model.fit(x, y, epochs=500, verbose=0)

# Get learned weights
weights = model.layers[0].get_weights()

print("Learned weight (slope):", weights[0][0][0])
print("Learned bias:", weights[1][0])

# Test prediction
test_value = np.array([10.0])
prediction = model.predict(test_value)[0][0]

print("Prediction for x=10:", prediction)

# Plot training loss
plt.plot(history.history['loss'])
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.show()
