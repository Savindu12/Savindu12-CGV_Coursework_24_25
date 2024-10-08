# importing necessary libraries
import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt

# Tesseract Path (Adjust based on your setup)
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'

# Function to display histograms for a given image
def display_histogram(image, title="Histogram"):
    plt.figure()
    plt.title(title)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.hist(image.ravel(), bins=256, range=(0, 256))  # Plot the histogram for the image
    plt.show()


# Function to preprocess the image and improve text extraction
def preprocess_image(img):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Display histogram for grayscale image
    display_histogram(gray, "Grayscale Image Histogram")

    # Apply Gaussian blur to remove noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive Thresholding for better binarization under different lighting conditions
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    # Display histogram for thresholded image
    display_histogram(thresh, "Thresholded Image Histogram")

    # Morphological operations (optional but improves text structure)
    kernel = np.ones((1, 1), np.uint8)
    morph_img = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Display histogram for morphologically processed image
    display_histogram(morph_img, "Morphologically Processed Image Histogram")

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
image_file = '/Users/savindudhamsara/Documents/4th Year/CGV/Group Assignment/Savindu12-CGV_Coursework_24_25/receipt_reader_CGV/assets/Recept-II.png'

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

# 3. Edge Detected Image
plt.figure()
plt.imshow(edges_img, cmap='gray')
plt.title("Edge Detected Image")
plt.show()

# 4. Morphologically Processed Image
plt.figure()
plt.imshow(morph_img, cmap='gray')
plt.title("Morphologically Processed Image")
plt.show()

# Print the extracted text and the summarized receipt
print("Extracted Text:\n", extracted_text)
print("Receipt Summary:\n", receipt_summary)
