


# # C:/Users/Pabasara/Music/perspective correction/reciept.jpg

import cv2
import numpy as np
import os  # Import os module for directory operations

def stack_images(scale, img_array):
    """
    Stack images in a grid with a specified scale.
    
    Parameters:
    - scale: Float, scaling factor for resizing images.
    - img_array: List of images to stack. Can be a list of lists for multiple rows and columns.
    
    Returns:
    - ver: The stacked image.
    """
    rows = len(img_array)
    cols = len(img_array[0])
    rows_available = isinstance(img_array[0], list)
    width = img_array[0][0].shape[1]
    height = img_array[0][0].shape[0]
    
    if rows_available:
        for x in range(rows):
            for y in range(cols):
                img_array[x][y] = cv2.resize(img_array[x][y], (width, height), None, scale, scale)
                if len(img_array[x][y].shape) == 2: 
                    img_array[x][y] = cv2.cvtColor(img_array[x][y], cv2.COLOR_GRAY2BGR)
        image_blank = np.zeros((height, width, 3), np.uint8)
        hor = [image_blank] * rows
        for x in range(rows):
            hor[x] = np.hstack(img_array[x])
        ver = np.vstack(hor)
    else:
        for x in range(rows):
            img_array[x] = cv2.resize(img_array[x], (width, height), None, scale, scale)
            if len(img_array[x].shape) == 2: 
                img_array[x] = cv2.cvtColor(img_array[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(img_array)
        ver = hor
    return ver

def get_contours(img):
    """
    Find the largest contour with 4 points in the image.
    
    Parameters:
    - img: Grayscale or binary image to find contours in.
    
    Returns:
    - largest_contour: The largest contour with 4 points or None if not found.
    """
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = None
    max_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 5000:  # Filtering out small contours based on area
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) == 4 and area > max_area:
                largest_contour = approx
                max_area = area
    
    return largest_contour

def corrective_perspective(image, points):
    """
    Apply corrective perspective transformation to the image based on the given points.
    
    Parameters:
    - image: The original image.
    - points: Array of 4 points (coordinates) for corrective perspective transformation.
    
    Returns:
    - output: The transformed image with corrected perspective.
    """
    points = points.reshape((4, 2))
    pts1 = np.float32(points)
    width = max(abs(points[0][0] - points[1][0]), abs(points[2][0] - points[3][0]))
    height = max(abs(points[0][1] - points[2][1]), abs(points[1][1] - points[3][1]))
    pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    output = cv2.warpPerspective(image, matrix, (width, height))
    return output

# Create output directory if it does not exist
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the image
image_path = 'C:/Users/Pabasara/Music/perspective correction/reciept.jpg'
image = cv2.imread(image_path)

# Step 1: Preprocess the image
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 1)  # Apply Gaussian Blur to reduce noise
canny_image = cv2.Canny(blurred_image, 50, 150)  # Apply Canny edge detection

# Step 2: Find contours
largest_contour = get_contours(canny_image)

# Step 3: Corrective perspective based on contour detection
if largest_contour is not None:
    corrected_image = corrective_perspective(image, largest_contour)
    cv2.imshow('Corrected Image', corrected_image)
    cv2.imwrite(os.path.join(output_dir, 'corrected_receipt.jpg'), corrected_image)
else:
    print("No appropriate contour found")

# Step 4: Original corrective perspective with predefined points
input_points = np.float32([[83, 18], [342, 53], [14, 389], [295, 436]])  # Original points to be corrected

# Output image size (for A4 paper dimensions)
width = 400
height = int(width * 1.414)  # Aspect ratio for A4

converted_points = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

# Apply corrective perspective transformation
matrix = cv2.getPerspectiveTransform(input_points, converted_points)
img_output = cv2.warpPerspective(image, matrix, (width, height))

cv2.imshow('Original', image)
cv2.imshow('Corrective Perspective', img_output)
cv2.imwrite(os.path.join(output_dir, 'corrective_perspective.jpg'), img_output)

# Step 5: Image stacking for visualization
img_stack = stack_images(0.6, [[image, gray_image, canny_image]])
cv2.imshow('Stacked Images', img_stack)
cv2.imwrite(os.path.join(output_dir, 'stacked_images.jpg'), img_stack)

# Wait until a key is pressed and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()

