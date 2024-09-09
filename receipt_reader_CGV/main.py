# Code for reading a receipt image, extracting text, summarizing the receipt, and displaying the results.

import cv2
import pytesseract
import matplotlib.pyplot as plt
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'

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

# Function to extract text from image using pytesseract
def extract_text_from_image(image_path):
    # Load the image
    img = cv2.imread(image_path)

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply binarization (thresholding)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Extract text using pytesseract
    text = pytesseract.image_to_string(thresh, config='--psm 6')
    return text, thresh

# Sample image file path (You can replace it with one of the uploaded file paths)
image_file = '/Users/savindudhamsara/Documents/4th Year/CGV/Group Assignment/Savindu12-CGV_Coursework_24_25/receipt_reader_CGV/assets/Recept-I.png'

# Extract text from the receipt image
extracted_text, processed_image = extract_text_from_image(image_file)

# Summarize the receipt
receipt_summary = summarize_receipt(extracted_text)

# Display the processed image and the extracted text
plt.imshow(processed_image, cmap='gray')
plt.title('Processed Receipt Image')
plt.show()

print("Extracted Text:\n", extracted_text)
print("Receipt Summary:\n", receipt_summary)
