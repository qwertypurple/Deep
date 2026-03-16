import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Input dataset
inputs = np.array([[0,0],
                   [0,1],
                   [1,0],
                   [1,1]])

# Output dataset: [AND, OR, XOR]
outputs = np.array([[0,0,0],
                    [0,1,1],
                    [0,1,1],
                    [1,1,0]])

# Build model
model = Sequential()

model.add(Dense(6, input_dim=2, activation='relu'))
model.add(Dense(3, activation='sigmoid'))

# Compile model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(inputs, outputs, epochs=4000, verbose=0)

# Predictions
predictions = model.predict(inputs)

print("\nPredictions:")

for i, p in enumerate(predictions):
    print(f"{inputs[i]} => AND:{round(p[0])} OR:{round(p[1])} XOR:{round(p[2])} (raw:{p})")
