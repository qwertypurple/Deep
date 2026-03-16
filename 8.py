pip install tensorflow opencv-python matplotlib numpy

# Character Recognition using CNN

import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Step 1: Load MNIST dataset
# -----------------------------

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


# -----------------------------
# Step 2: Normalize images
# -----------------------------

x_train = x_train / 255.0
x_test = x_test / 255.0


# -----------------------------
# Step 3: Reshape for CNN
# -----------------------------

x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)


# -----------------------------
# Step 4: Build CNN model
# -----------------------------

model = models.Sequential([

    layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),

    layers.Dense(64,activation='relu'),

    layers.Dense(10,activation='softmax')
])


# -----------------------------
# Step 5: Compile model
# -----------------------------

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# -----------------------------
# Step 6: Train model
# -----------------------------

model.fit(x_train, y_train, epochs=5)


# -----------------------------
# Step 7: Load input image
# -----------------------------

img = cv2.imread("digit.png", cv2.IMREAD_GRAYSCALE)


# -----------------------------
# Step 8: Resize image
# -----------------------------

img = cv2.resize(img,(28,28))


# -----------------------------
# Step 9: Normalize
# -----------------------------

img = img / 255.0


# -----------------------------
# Step 10: Reshape for prediction
# -----------------------------

img = img.reshape(1,28,28,1)


# -----------------------------
# Step 11: Predict digit
# -----------------------------

prediction = model.predict(img)

predicted_digit = np.argmax(prediction)

print("Predicted Digit:", predicted_digit)


# -----------------------------
# Step 12: Show image
# -----------------------------

plt.imshow(img.reshape(28,28), cmap='gray')
plt.title("Predicted: " + str(predicted_digit))
plt.show()
