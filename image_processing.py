import cv2
import numpy as np

def preprocess_image_advanced(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to the entire image
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    
    # Adaptive thresholding on blurred image
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)

    # Morphological closing to connect broken parts of characters
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, closing_kernel)

    # Apply erosion to separate vertical lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    eroded = cv2.erode(closed, kernel, iterations=1)
    
    return eroded

def enlarge_image(image, scale_factor=3):
    enlarged_image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LANCZOS4)
    return enlarged_image

def enhance_quality(image):
    sharpened = cv2.filter2D(image, -1, np.array([[-1, -1, -1], [-1,  9, -1], [-1, -1, -1]]))
    denoised = cv2.fastNlMeansDenoisingColored(sharpened, None, 10, 10, 7, 21)
    return denoised

# -----------------------------------------------------------------------------------------------------------

# Function to preprocess an image
def preprocess_image_basic(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
    return thresh

# -----------------------------------------------------------------------------------------------------------

def crop_white_background(image):
    """Crop white background from an image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(binary > 0))
    
    if len(coords) > 0:
        y1, x1 = coords.min(axis=0)
        y2, x2 = coords.max(axis=0)
        return image[y1:y2 + 1, x1:x2 + 1]
    return image

# -----------------------------------------------------------------------------------------------------------

# Preprocess the input image
def preprocess_image_to_predict(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to read image at path: {image_path}")
    image = cv2.resize(image, (32, 32))  # Resize to match the model's input size
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image