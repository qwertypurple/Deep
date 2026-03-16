# RGB to HSV Conversion

from skimage import data
from skimage.color import rgb2hsv
import matplotlib.pyplot as plt

# Setting the plot size
plt.figure(figsize=(15, 15))

# Sample image from scikit-image
coffee = data.coffee()

# Display original RGB image
plt.subplot(1, 2, 1)
plt.imshow(coffee)
plt.title("Original RGB Image")

# Convert RGB to HSV
hsv_coffee = rgb2hsv(coffee)

# Display HSV image
plt.subplot(1, 2, 2)
hsv_coffee_colorbar = plt.imshow(hsv_coffee)
plt.title("HSV Image")

# Add colorbar
plt.colorbar(hsv_coffee_colorbar, fraction=0.046, pad=0.04)

plt.show()
