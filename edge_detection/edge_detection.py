import cv2
import numpy as np

# Load the image
image = cv2.imread('C:/Users/User/Desktop/edge_detection/Recepts.png')

# Check if the image was successfully loaded
if image is None:
    print("Error: Could not open or find the image.")
else:
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and improve edge detection
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 1)

    # Perform Canny edge detection
    edges = cv2.Canny(blurred_image, threshold1=100, threshold2=200)

    # Display the original and edge-detected images
    cv2.imshow('Original Image', image)
    cv2.imshow('Canny Edge Detection', edges)

    # Save the edge-detected image
    cv2.imwrite('canny_edges-recepts.jpg', edges)

    cv2.waitKey(0)
    cv2.destroyAllWindows()