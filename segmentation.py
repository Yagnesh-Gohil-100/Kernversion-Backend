import cv2
import numpy as np
import os
import fitz
from PIL import Image
from image_processing import enlarge_image, enhance_quality, preprocess_image_advanced
from mapping import create_mapping, assign_number

def extract_alphabets(pdf_path, output_folder, aspect_ratio_threshold=3):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    pdf_document = fitz.open(pdf_path)
    all_coordinates = []  # To store coordinates for all pages
    all_row_mappings = []  # To store row mappings for all pages
    last_row_number = 0  # To ensure row numbers continue across pages

    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        page_image = page.get_pixmap()
        np_page_image = np.frombuffer(page_image.samples, dtype=np.uint8).reshape((page_image.height, page_image.width, page_image.n))

        processed_image = preprocess_image_advanced(np_page_image)

        contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Extract coordinates for the current page
        coordinates = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            coordinates.append((x, y, w, h))

        if not coordinates:
            print(f"No contours found on page {page_num}.")
            all_coordinates.append([])
            all_row_mappings.append([])
            continue

        # Sort coordinates by y-axis (rows)
        coordinates_sorted_by_y = sorted(coordinates, key=lambda item: item[1])

        # Create row mapping for the current page
        row_mapping = create_mapping(coordinates_sorted_by_y, aspect_ratio_threshold, is_row=True)

        # Adjust row numbers to continue from the last row number of the previous page
        adjusted_row_mapping = []
        for num, lower_limit, upper_limit in row_mapping:
            adjusted_row_mapping.append((num + last_row_number, lower_limit, upper_limit))

        # Update last_row_number for the next page
        last_row_number = adjusted_row_mapping[-1][0] if adjusted_row_mapping else last_row_number

        # Store coordinates and row mapping for the current page
        all_coordinates.append(coordinates)
        all_row_mappings.append(adjusted_row_mapping)

        # Print row mapping and coordinates for the current page
        print(f"Page {page_num} Row Mapping:", adjusted_row_mapping)
        print(f"Page {page_num} Coordinates:", coordinates)

        # Process and save alphabet regions for the current page
        for (x, y, w, h) in coordinates:
            if h / w > aspect_ratio_threshold:
                continue

            row_num = assign_number(y, adjusted_row_mapping)

            if row_num == -1:
                continue

            alphabet_region = np_page_image[y:y+h, x:x+w]
            enlarged_region = enlarge_image(alphabet_region)
            enhanced_region = enhance_quality(enlarged_region)

            if w < 6 and h < 6:
                continue

            base_filename = f"{page_num}_row{row_num}_x{x}_y{y}_w{w}_h{h}"
            counter = 1
            filename = f"{base_filename}.png"
            while os.path.exists(os.path.join(output_folder, filename)):
                filename = f"{base_filename}_{counter}.png"
                counter += 1

            alphabet_image = Image.fromarray(enhanced_region)
            alphabet_image.save(os.path.join(output_folder, filename))
    
    return all_coordinates, all_row_mappings


# ----------------------------------------------------------------------------------------------------------

from identifications import check_articulation
from save_and_load import save_segment_swar_and_kann_swar
from image_processing import crop_white_background

def find_separation_line_swar_and_kann_swar(binary_image, image_height):
    """Find the optimal separation line in a binary image."""
    vertical_projection = np.sum(binary_image, axis=1) / 255
    lower_bound = int(image_height * 0.3)
    upper_bound = int(image_height * 0.5)
    
    valid_range = vertical_projection[lower_bound:upper_bound]
    if valid_range.size > 0:
        separation_row_in_range = np.argmin(valid_range)
        return lower_bound + separation_row_in_range
    return image_height // 2  # Default to middle

def split_image_by_row(image, separation_row):
    """Split image into upper and lower parts at separation_row."""
    return image[:separation_row, :], image[separation_row:, :]

def segment_swar_and_kann_swar(img, subgroup_range, col):
    """Process a single outlier image to separate kann swar and swar parts."""
    x, y, w, h, image_path = img
    
    # Load and check articulation
    outlier_image = cv2.imread(image_path)
    is_articulated = check_articulation(outlier_image)
    
    if is_articulated:
        return [image_path], []  # swar_list, kann_swar_list
    
    # Process image to find separation
    gray = cv2.cvtColor(outlier_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 21, 5)
    
    # Morphological operations
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, closing_kernel)
    
    # Find separation line
    separation_row = find_separation_line_swar_and_kann_swar(closed, outlier_image.shape[0])
    
    # Split and crop parts
    upper_part, lower_part = split_image_by_row(outlier_image, separation_row)
    upper_part_cropped = crop_white_background(upper_part)
    lower_part_cropped = crop_white_background(lower_part)
    
    # Save segments
    original_filename = os.path.basename(image_path)
    upper_part_path = save_segment_swar_and_kann_swar(upper_part_cropped, subgroup_range, col, 'upper', original_filename)
    lower_part_path = save_segment_swar_and_kann_swar(lower_part_cropped, subgroup_range, col, 'lower', original_filename)
    
    return [lower_part_path], [upper_part_path]  # swar_list, kann_swar_list


def filter_kann_swar_from_swar(non_outlier_images):
    """Process non-outlier images to identify hidden kann swars."""
    if len(non_outlier_images) == 1:
        return [non_outlier_images[0][4]], []  # swar_list, kann_swar_list
    
    # Multiple images - sort by y-value
    sorted_images = sorted(non_outlier_images, key=lambda x: x[1])
    return [sorted_images[1][4]], [sorted_images[0][4]]  # swar_list, kann_swar_list

# ----------------------------------------------------------------------------------------------------------

from image_processing import preprocess_image_basic

# Function to extract meend and kann swar segments
def segment_meend_and_kann_swar(image_path):
    """
    Function to perform vertical segmentation on an image.
    
    Parameters:
    - image_path: Path to the image file.
    
    Returns:
    - left_part: Left part of the image (kann swar or None).
    - mid_part: Mid part of the image (meend).
    - right_part: Right part of the image (kann swar or None).
    """
    image = cv2.imread(image_path)
    if image is None:
        return None, None, None

    processed_image = preprocess_image_basic(image)
    contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_coords = []
    all_coords = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 10 <= h <= 25 and w > 25:
            valid_coords.append((x, y, w, h))
        else:
            all_coords.append((x, y, w, h))

    left_part, mid_part, right_part = None, None, None
    
    if valid_coords:
        valid_coords = sorted(valid_coords, key=lambda coord: coord[0])
        leftmost_valid = valid_coords[0]
        rightmost_valid = valid_coords[-1]

        left_cut = None
        x1, y1, w1, h1 = leftmost_valid
        for x, y, w, h in all_coords:
            if x <= x1 and (y + h) >= y1 and h > 10 and w > 10:
                left_cut = x1
                break

        right_cut = None
        x2, y2, w2, h2 = rightmost_valid
        for x, y, w, h in all_coords:
            if x >= (x2 + w2) and (y + h) >= y2 and h > 10 and w > 10:
                right_cut = x2 + w2
                break

        if left_cut is not None and right_cut is not None:
            left_part = image[:, :left_cut]
            mid_part = image[:, left_cut:right_cut]
            right_part = image[:, right_cut:]

        elif left_cut is not None:
            left_part = image[:, :left_cut]
            right_part = image[:, left_cut:]
        
        elif right_cut is not None:
            left_part = image[:, :right_cut]
            right_part = image[:, right_cut:]

    # If no segmentation occurred, treat the entire image as mid_part
    if left_part is None and right_part is None:
        mid_part = image

    return left_part, mid_part, right_part

# ----------------------------------------------------------------------------------------------------------

from image_processing import preprocess_image_basic

# Function to separate articulation in an image
def separate_articulation(image):
    processed_image = preprocess_image_basic(image)
    contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 10 < h < 21 and w > 25:
            upper_part = image[:y, :]
            if upper_part.shape[0] > 0:
                return upper_part, True  # Return the upper part and a flag indicating segmentation was successful
            break
    
    return image, False  # Return the original image and a flag indicating no segmentation

# ----------------------------------------------------------------------------------------------------------

from skimage.morphology import binary_erosion, binary_dilation, square
from skimage import img_as_ubyte

# Function to segment a word into multiple images
def break_word_into_segments(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply simple binary thresholding and invert the image
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Define structuring elements
    structuring_element2 = np.ones((2, 2), dtype=bool)
    structuring_element_erosion = square(3)

    # Apply binary dilation to fill gaps
    dilated = binary_dilation(binary, footprint=structuring_element2)

    # Apply binary erosion to separate connected components
    eroded = binary_erosion(dilated, footprint=structuring_element_erosion)
    eroded = img_as_ubyte(eroded)  # Convert to uint8 for display purposes

    # Perform vertical projection to find potential cut lines
    vertical_projection = np.sum(eroded, axis=0)

    # Find cut points by identifying valleys in the projection with heuristic
    threshold = 0.15 * np.max(vertical_projection)
    valleys = [x for x, y in enumerate(vertical_projection) if y < threshold]

    # Apply heuristic: if two consecutive valleys are close, take the right one
    cut_points = []
    min_distance = 13
    i = 0
    while i < len(valleys) - 1:
        if (valleys[i + 1] - valleys[i]) < min_distance:
            cut_points.append(valleys[i + 1])
            i += 2  # Skip the next valley since we took the right one
        else:
            cut_points.append(valleys[i])
            i += 1
    if i == len(valleys) - 1:
        cut_points.append(valleys[i])  # Add the last valley if it's not processed

    # Ensure no duplicate cut points and sort them
    cut_points = sorted(set(cut_points))

    # Separate the image at cut points
    cut_images = []
    start = 0
    for cut_point in cut_points:
        if cut_point - start > 10:  # Ensure segments are large enough
            cut_image = img[:, start:cut_point]
            cut_images.append(cut_image)
            start = cut_point

    # Add the last segment
    cut_images.append(img[:, start:])

    return cut_images

# Function to merge segments based on height-to-width ratio
def merge_broken_segments(segments):
    final_images = []
    i = 0
    while i < len(segments):
        current_image = segments[i]
        current_ratio = current_image.shape[0] / current_image.shape[1]

        ratio_threshold = 1.8

        if current_image.shape[0] > 35:
            ratio_threshold = 2.9
        
        # If the ratio is greater than the threshold and it's the first segment
        if current_ratio > ratio_threshold and i == 0:
            # Merge with the next segment
            if i + 1 < len(segments):
                current_image = np.hstack((current_image, segments[i + 1]))
                final_images.append(current_image)
                i += 2
            else:
                final_images.append(current_image)
                i += 1
        # If two or more consecutive segments have a ratio greater than the threshold
        elif i < len(segments) - 1 and (segments[i + 1].shape[0] / segments[i + 1].shape[1]) > ratio_threshold:
            while i < len(segments) - 1 and (segments[i + 1].shape[0] / segments[i + 1].shape[1]) > ratio_threshold:
                current_image = np.hstack((current_image, segments[i + 1]))
                i += 1
            final_images.append(current_image)
            i += 1
        # If the ratio is greater than the threshold and it's not the first segment
        elif current_ratio > ratio_threshold and i != 0:
            # Merge with the previous segment
            if final_images:
                final_images[-1] = np.hstack((final_images[-1], current_image))
            else:
                final_images.append(current_image)
            i += 1
        else:
            final_images.append(current_image)
            i += 1

    return final_images

# Function to process a single image, segment, and save the results in the provided folder
def segment_word(image_path, output_folder):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        return []
    
    # Segment the image
    segmented_images = break_word_into_segments(img)
    
    # Merge segments based on height-to-width ratio
    final_images = merge_broken_segments(segmented_images)
    
    # Save the segmented images
    image_base_name = os.path.splitext(os.path.basename(image_path))[0]
    segmented_paths = []
    for i, segmented_image in enumerate(final_images):
        seg_image_path = os.path.normpath(os.path.join(output_folder, f'{image_base_name}_seg{i+1}.png'))
        cv2.imwrite(seg_image_path, segmented_image)
        segmented_paths.append(seg_image_path)
    
    return segmented_paths