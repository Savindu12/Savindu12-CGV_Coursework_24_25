import cv2
import numpy as np


image = cv2.imread('C:/Users/User/Desktop/edge_detection/Recepts.png')

if image is None:
    print("Error: Could not open or find the image.")
else:

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 1)

    edges = cv2.Canny(blurred_image, threshold1=100, threshold2=200)

    cv2.imshow('Original Image', image)
    cv2.imshow('Canny Edge Detection', edges)

    cv2.imwrite('canny_edges-recepts.jpg', edges)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
