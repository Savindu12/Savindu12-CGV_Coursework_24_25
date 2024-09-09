# importing necessary libraries

import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt

# Tesseract Path (Adjust based on your setup)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Function to preprocess the image and improve text extraction
def preprocess_image(img):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to remove noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive Thresholding for better binarization under different lighting conditions
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    # Morphological operations (optional but improves text structure)
    kernel = np.ones((1, 1), np.uint8)
    morph_img = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Edge detection (optional step)
    edges = cv2.Canny(morph_img, 100, 200)

    return edges, thresh, morph_img


# Function to extract text using Tesseract
def extract_text_from_image(image_path):
    # Load the image
    img = cv2.imread(image_path)

    # Preprocess image
    edges, thresh, morph_img = preprocess_image(img)

    # Extract text using pytesseract
    text = pytesseract.image_to_string(morph_img, config='--psm 6')

    return text, edges, thresh, morph_img


# Function to summarize receipt based on extracted text
def summarize_receipt(text):
    lines = text.split('\n')
    summary = {}
    for line in lines:
        if 'Sub Total' in line:
            summary['Sub Total'] = line.split()[-1]
        if 'Cash' in line:
            summary['Cash'] = line.split()[-1]
        if 'Change' in line:
            summary['Change'] = line.split()[-1]
    return summary


# Path to receipt image
image_file = 'D:/4th Year/2nd Semester/CGV - CS402.3/CGV - group assignment/Savindu12-CGV_Coursework_24_25/assets/Recept-I.png'

# Extract text from the receipt image
extracted_text, edges_img, thresh_img, morph_img = extract_text_from_image(image_file)

# Summarize the receipt
receipt_summary = summarize_receipt(extracted_text)

# Display each stage separately
# 1. Original Image
plt.figure()
plt.imshow(cv2.imread(image_file))
plt.title("Original Image")
plt.show()

# 2. Thresholded Image
plt.figure()
plt.imshow(thresh_img, cmap='gray')
plt.title("Thresholded Image")
plt.show()

# Print the extracted text and the summarized receipt
print("Extracted Text:\n", extracted_text)
print("Receipt Summary:\n", receipt_summary)
