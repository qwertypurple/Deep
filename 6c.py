# Supervised Image Segmentation using Thresholding

from skimage import data
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

# Load image
coffee = data.coffee()

# Convert to grayscale
gray_coffee = rgb2gray(coffee)

# Set figure size
plt.figure(figsize=(15, 15))

# Apply different thresholds
for i in range(10):

    binarized_gray = (gray_coffee > i * 0.1) * 1

    plt.subplot(5, 2, i + 1)

    plt.title("Threshold: >" + str(round(i * 0.1, 1)))

    plt.imshow(binarized_gray, cmap='gray')

plt.tight_layout()
plt.show()
