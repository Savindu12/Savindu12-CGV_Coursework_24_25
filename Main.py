import cv2
import pytesseract
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Set the Tesseract command path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def preprocess_image(image_path):
    # Read the image
    img = cv2.imread(image_path)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Define the kernel for morphological operations
    kernel = np.ones((2, 2), np.uint8)

    # Apply dilation
    dilated = cv2.dilate(gray, kernel, iterations=1)

    # Apply erosion
    eroded = cv2.erode(gray, kernel, iterations=1)

    return img, gray, dilated, eroded


def extract_text_from_receipt(image_path):
    original_img, gray, dilated, eroded = preprocess_image(image_path)
    img_pil = Image.fromarray(eroded)  # Change to eroded or any other processed image based on what you want to OCR
    custom_config = r'--oem 3 --psm 6'  # Adjust PSM based on your specific needs
    text = pytesseract.image_to_string(img_pil, config=custom_config)

    # Displaying the images for comparison
    plt.figure(figsize=(10, 8))
    titles = ['Original Image', 'Grayscale Image', 'Dilated Image', 'Eroded Image']
    images = [original_img, gray, dilated, eroded]

    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(images[i], cmap='gray' if i != 0 else 'gray')
        plt.title(titles[i])
        plt.axis('off')

    plt.show()

    return text


# Path to your image file
image_path = 'D:/Character Recognition/pythonProject1/Images/img_2.png'
extracted_text = extract_text_from_receipt(image_path)
print(extracted_text)
