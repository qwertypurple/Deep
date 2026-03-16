import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Input dataset
inputs = np.array([[0,0],
                   [0,1],
                   [1,0],
                   [1,1]])

# Choose ONE logic gate

# AND Gate
# outputs = np.array([[0],[0],[0],[1]])

# OR Gate
# outputs = np.array([[0],[1],[1],[1]])

# XOR Gate
outputs = np.array([[0],[1],[1],[0]])

# Build model
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(inputs, outputs, epochs=3000, verbose=0)

# Predictions
predictions = model.predict(inputs)

print("\nPredictions:")

for i, p in enumerate(predictions):
    print(f"{inputs[i]} => {round(float(p))} (raw: {float(p):.4f})")
