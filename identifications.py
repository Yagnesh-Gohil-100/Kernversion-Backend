def calculate_valid_sam_positions(taal_name):
    """
    Calculate all valid positions where the sam (X) can occur for a given taal.
    """
    if taal_name not in taal_info:
        raise ValueError(f"Taal '{taal_name}' not found in the database.")
    
    # Get the divisions for the taal
    divisions = taal_info[taal_name]["divisions"]
    
    # Calculate valid sam positions
    valid_positions = [1]  # The first beat is always valid
    cumulative_sum = 0
    
    # Start from the end of the divisions and add cumulatively
    for i in range(len(divisions) - 1, -1, -1):
        cumulative_sum += divisions[i]
        if cumulative_sum < taal_info[taal_name]["beat_count"]:
            valid_positions.append(cumulative_sum + 1)
    
    return sorted(valid_positions)

def calculate_divisions_and_vibhaag(taal_name, sam_beat):
    """
    Calculate the divisions and vibhaag for the given taal and sam beat.
    """
    if taal_name not in taal_info:
        raise ValueError(f"Taal '{taal_name}' not found in the database.")
    
    # Get the divisions and vibhaag for the taal
    divisions = taal_info[taal_name]["divisions"]
    vibhaag = taal_info[taal_name]["vibhaag"]
    
    # If sam is at the first beat, no change is needed
    if sam_beat == 1:
        return divisions, vibhaag
    
    # Calculate the rotation index
    cumulative_sum = 0
    rotation_index = 0
    
    # Start from the end of the divisions and add cumulatively
    for i in range(len(divisions) - 1, -1, -1):
        cumulative_sum += divisions[i]
        if cumulative_sum + 1 == sam_beat:
            rotation_index = i
            break
    
    # Rotate divisions and vibhaag
    new_divisions = divisions[rotation_index:] + divisions[:rotation_index]
    new_vibhaag = vibhaag[rotation_index:] + vibhaag[:rotation_index]
    
    return new_divisions, new_vibhaag

# ------------------------------------------------------------------------------------------------------

import json

# Load taal_info from JSON file
with open("taal_info.json", "r") as file:
    taal_info = json.load(file)

def get_sam_and_taalis_rows(row_image_count, taal_name):
    """
    Calculate important rows based on the number of images in each row and the taal info.

    Args:
        row_image_count (dict): A dictionary where keys are row numbers and values are the number of images in that row.
        taal_name (str): The name of the taal (e.g., "Rupak").
        start_row (int): The starting row number (inclusive).
        end_row (int): The ending row number (inclusive).

    Returns:
        list: A list of row numbers that are considered important (rows with exactly `divisions_size` images).
    """
    # Get the size of the divisions list from taal_info
    divisions_size = len(taal_info[taal_name]["divisions"])

    # Create the important_rows list
    important_rows = [
        row_num for row_num, count in row_image_count.items()
        if count == divisions_size
    ]

    return important_rows

# ------------------------------------------------------------------------------------------------------

import os
from collections import defaultdict
from filename_utils import get_image_details_with_row_and_col
from config import PATHS

def is_articulation(w, h):
    return 4 < h < 9 and w > 9

def classify_rows(subgroup_coords):
    articulation_rows = []
    kann_swar_rows = []
    swar_rows = []
    lyrics_rows = []

    # Group images by rows
    row_groups = defaultdict(list)
    for image, page_num, row_num, x, y, w, h in subgroup_coords:
        row_groups[row_num].append((image, page_num, x, y, w, h))

    # Check for articulation rows
    non_articulation_rows = []
    for row_num, images in row_groups.items():
        if all(is_articulation(w, h) for _, _, x, y, w, h in images):
            articulation_rows.append(row_num)
        else:
            non_articulation_rows.append((row_num, images))

    # Sort non-articulation rows by row number
    non_articulation_rows.sort(key=lambda item: item[0])
    remaining_rows = len(non_articulation_rows)

    # Classify remaining rows based on cases
    if remaining_rows == 3:
        kann_swar_rows.append(non_articulation_rows[0][0])
        swar_rows.append(non_articulation_rows[1][0])
        lyrics_rows.append(non_articulation_rows[2][0])

    elif remaining_rows == 2:
        row1_images = non_articulation_rows[0][1]
        row2_images = non_articulation_rows[1][1]
        if (abs(len(row1_images) - len(row2_images)) <= 2) or (len(row1_images) > len(row2_images)):
            swar_rows.append(non_articulation_rows[0][0])
            lyrics_rows.append(non_articulation_rows[1][0])
        else:
            kann_swar_rows.append(non_articulation_rows[0][0])
            swar_rows.append(non_articulation_rows[1][0])

    elif remaining_rows == 1:
        swar_rows.append(non_articulation_rows[0][0])

    return articulation_rows, kann_swar_rows, swar_rows, lyrics_rows

def classify_rows_in_subgroups(subgroup_ranges):
    composition_folder = PATHS['working_composition']
    images = os.listdir(composition_folder)
    coordinates = []

    # Parse image details and store them
    for image in images:
        details = get_image_details_with_row_and_col(image)
        if details:
            page_num, row_num, col_num, x, y, w, h = details
            coordinates.append((image, page_num, row_num, x, y, w, h))

    articulation_rows_all = []
    kann_swar_rows_all = []
    swar_rows_all = []
    lyrics_rows_all = []

    # Process each subgroup range
    for start_row, end_row in subgroup_ranges:
        subgroup_coords = [
            (image, page_num, row_num, x, y, w, h) for image, page_num, row_num, x, y, w, h in coordinates
            if start_row <= row_num <= end_row
        ]

        # Classify rows within the subgroup
        articulation_rows, kann_swar_rows, swar_rows, lyrics_rows = classify_rows(subgroup_coords)

        # Add rows to the respective lists
        articulation_rows_all.extend(articulation_rows)
        kann_swar_rows_all.extend(kann_swar_rows)
        swar_rows_all.extend(swar_rows)
        lyrics_rows_all.extend(lyrics_rows)

    # Print the results
    print("Articulation Rows: ", articulation_rows_all)
    print("Kann Swar Rows: ", kann_swar_rows_all)
    print("Swar Rows: ", swar_rows_all)
    print("Lyrics Rows: ", lyrics_rows_all)

    return {
        "articulation": articulation_rows_all,
        "kann_swar": kann_swar_rows_all,
        "swar": swar_rows_all,
        "lyrics": lyrics_rows_all
    }
    
# ------------------------------------------------------------------------------------------------------

import cv2
from image_processing import preprocess_image_basic

# Function to check articulation in an image
def check_articulation(image):
    processed_image = preprocess_image_basic(image)
    contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 10 < h < 21 and w > 25:
            upper_part = image[:y, :]
            if upper_part.shape[0] > 0:
                return True  # Return true if there is articulation
            break
    
    return False  # Return false if there is no articulation

# ------------------------------------------------------------------------------------------------------

# Function to identify meend and kann swar
def identify_meend_and_kann_swar(left_part, mid_part, right_part):
    """
    Function to identify and structure meend and kann swar based on the width of the segments.
    
    Parameters:
    - left_part: The left part of the image.
    - mid_part: The mid part of the image.
    - right_part: The right part of the image.
    
    Returns:
    - left_part: The left part (kann swar or None).
    - mid_part: The mid part (meend).
    - right_part: The right part (kann swar or None).
    """
    # Determine which part is meend based on width
    parts = {
        "left": left_part,
        "mid": mid_part,
        "right": right_part
    }

    # Filter out None parts
    valid_parts = {k: v for k, v in parts.items() if v is not None}

    # If no segmentation occurred (only mid_part exists)
    if len(valid_parts) == 1 and "mid" in valid_parts:
        # Treat the entire image as meend
        left_part = None
        right_part = None
        mid_part = valid_parts["mid"]
    
    # If there are only two parts, identify meend based on width
    elif len(valid_parts) == 2:
        # Find the part with the maximum width (meend)
        meend_key = max(valid_parts, key=lambda k: valid_parts[k].shape[1])
        kann_swar_key = [k for k in valid_parts.keys() if k != meend_key][0]

        # Reassign parts to ensure meend is in the middle
        if meend_key == "left":
            mid_part = valid_parts[meend_key]
            right_part = valid_parts[kann_swar_key]
            left_part = None
        elif meend_key == "right":
            mid_part = valid_parts[meend_key]
            left_part = valid_parts[kann_swar_key]
            right_part = None
        else:
            # If meend is already in the middle, no changes needed
            pass

    # If there are three parts, meend is always in the middle
    elif len(valid_parts) == 3:
        mid_part = valid_parts["mid"]
        left_part = valid_parts["left"]
        right_part = valid_parts["right"]

    return left_part, mid_part, right_part