pip install tensorflow numpy matplotlib opencv-python

# Autoencoder Neural Network using MNIST Dataset

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model


# -------------------------------
# Load MNIST dataset
# -------------------------------

(x_train, _), (x_test, _) = mnist.load_data()


# -------------------------------
# Normalize data
# -------------------------------

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.


# -------------------------------
# Flatten images (28x28 → 784)
# -------------------------------

x_train = x_train.reshape((len(x_train), 784))
x_test = x_test.reshape((len(x_test), 784))


# -------------------------------
# Input Layer
# -------------------------------

input_img = Input(shape=(784,))


# -------------------------------
# Encoder
# -------------------------------

encoded = Dense(128, activation='relu')(input_img)

encoded = Dense(64, activation='relu')(encoded)

encoded = Dense(32, activation='relu')(encoded)


# -------------------------------
# Decoder
# -------------------------------

decoded = Dense(64, activation='relu')(encoded)

decoded = Dense(128, activation='relu')(decoded)

decoded = Dense(784, activation='sigmoid')(decoded)


# -------------------------------
# Autoencoder Model
# -------------------------------

autoencoder = Model(input_img, decoded)


# -------------------------------
# Compile Model
# -------------------------------

autoencoder.compile(optimizer='adam',
                    loss='binary_crossentropy')


# -------------------------------
# Train Model
# -------------------------------

autoencoder.fit(x_train, x_train,
                epochs=20,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))


# -------------------------------
# Predict reconstructed images
# -------------------------------

decoded_imgs = autoencoder.predict(x_test)


# -------------------------------
# Display results
# -------------------------------

n = 10

plt.figure(figsize=(20,4))

for i in range(n):

    # Original Image
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Reconstructed Image
    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
