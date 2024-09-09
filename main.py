
"""
import cv2
import matplotlib.pyplot as plt

# Load the image from the provided file path
image_path = "OriginalImg.jpeg"
image = cv2.imread(image_path)

# Apply Gaussian Blur
# Kernel size (5, 5) and sigmaX = 0 (OpenCV calculates it automatically based on the kernel size)
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

# Save and display the blurred image
blurred_image_path = "blurred_receipt.jpeg"
cv2.imwrite(blurred_image_path, blurred_image)

# Display the original and blurred images using matplotlib
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB))
plt.title('Blurred Image')
plt.axis('off')

plt.show()
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image_path = "OriginalImg.jpeg"
image = cv2.imread(image_path)

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Convert to binary (thresholding)
_, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Apply Gaussian blur
blurred_image = cv2.GaussianBlur(binary_image, (5, 5), 0)

# Find contours
contours, _ = cv2.findContours(blurred_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a copy of the original image to draw bounding boxes
contour_image = image.copy()

# Draw bounding boxes around detected text regions
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    # Filter out small contours that may not be text
    if w > 20 and h > 10:
        cv2.rectangle(contour_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the images
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title('Grayscale Image')
plt.imshow(gray_image, cmap='gray')

plt.subplot(1, 3, 2)
plt.title('Binary Image')
plt.imshow(binary_image, cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Contours Detected')
plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))

plt.show()

# Print the result
print("Text regions have been detected and outlined in green.")

