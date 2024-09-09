# Receipt Reader Application

The **Receipt Reader Application** automates the extraction of key data from receipt images. It uses **image processing** techniques such as grayscale conversion, adaptive thresholding, and morphological operations to enhance the image for better **OCR** (Optical Character Recognition). The application extracts text from the processed images and summarizes key details like **Sub Total**, **Cash**, and **Change**. Additionally, the app visualizes the extracted data using **bar graphs**, **pie charts**, and **line graphs**.

## Features

- **Text Extraction**: Automatically extracts text from receipt images using **Tesseract OCR**.
- **Image Preprocessing**: Enhances images using techniques such as:
  - Grayscale Conversion
  - Adaptive Thresholding
  - Morphological Operations
  - Edge Detection
- **Data Visualization**: Creates:
  - Bar graphs for item quantities and prices.
  - Pie charts for total cost distribution.
  - Line graphs to visualize price trends across multiple receipts.

## Technologies Used

- **Python 3**
- **OpenCV** for image preprocessing
- **Pytesseract** for text recognition
- **Matplotlib** for visualizing the data

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/receipt-reader-application.git
   cd receipt-reader-application
2. **Install the required libraries**:
   ```bash
   pip install opencv-python pytesseract matplotlib numpy
3. **Setup Tesseract**:
   ```bash
   pip install tesseract
   pytesseract.pytesseract.tesseract_cmd = r'/path_to_your_tesseract'
4. **Run the Application**:
   ```bash
   python receipt_reader.py

### Instructions:
- Replace the `git clone` URL with the actual URL of your repository.
- Make sure to update the **Tesseract path** in the usage section to match your system.
- If there are any additional features or file paths, customize the **Project Structure** and **Features** sections as needed.

## Project Structure

```plaintext
receipt-reader-application/
│
├── assets/
│   └── (add your receipt images here)
├── receipt_reader.py
└── readme.md
