# RGB to Grayscale Conversion

from skimage import data
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

# Setting plot size
plt.figure(figsize=(15, 15))

# Load sample image
coffee = data.coffee()

# Show original image
plt.subplot(1, 2, 1)
plt.imshow(coffee)
plt.title("Original RGB Image")

# Convert to grayscale
gray_coffee = rgb2gray(coffee)

# Show grayscale image
plt.subplot(1, 2, 2)
plt.imshow(gray_coffee, cmap="gray")
plt.title("Grayscale Image")

plt.show()
