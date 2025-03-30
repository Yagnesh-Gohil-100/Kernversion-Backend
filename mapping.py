def create_mapping(coordinates, aspect_ratio_threshold, is_row=True):
    mapping = []
    number = 1

    if not coordinates:
        return mapping

    for i, (x, y, w, h) in enumerate(coordinates):
        if is_row:
            if h / w > aspect_ratio_threshold:
                continue
            coord = y
            size = h
        else:
            if w / h > aspect_ratio_threshold and w > 14 and h < 8:  
                continue
            coord = x
            size = w

        if i == 0:
            upper_limit = coord + int(size / 2)
            lower_limit = coord
            mapping.append((number, lower_limit, upper_limit))
        elif mapping and coord > mapping[-1][2]:
            number += 1
            lower_limit = coord
            upper_limit = coord + int(size / 2)
            mapping.append((number, lower_limit, upper_limit))
        else:
            upper_limit = max(mapping[-1][2], coord + int(size / 2))
            mapping[-1] = (number, mapping[-1][1], upper_limit)

    return mapping

def assign_number(coord, mapping):
    for num, lower_limit, upper_limit in mapping:
        if lower_limit <= coord <= upper_limit:
            return num
    return -1

# ------------------------------------------------------------------------------------------------------

from collections import defaultdict
import os

def get_row_image_counts(folder_path):
    """
    Calculate the number of images for each row in the folder.

    Args:
        folder_path (str): Path to the folder containing segmented images.

    Returns:
        dict: A dictionary where keys are row numbers and values are the number of images in that row.
    """
    # Dictionary to store the count of images for each row
    row_image_count = defaultdict(int)

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file matches the expected format
        if filename.endswith(".png"):  # Assuming images are in PNG format
            # Extract row number from the filename
            try:
                parts = filename.split("_")
                row_num = int(parts[1].replace("row", ""))  # Extract row number
            except (IndexError, ValueError):
                # Skip files that don't match the expected format
                continue

            # Increment the count for this row
            row_image_count[row_num] += 1

    return row_image_count

# ------------------------------------------------------------------------------------------------------

from filename_utils import get_image_details_with_row, update_filename_with_col_or_suffix
from config import PATHS

def find_general_boundaries(coordinates):
    min_x = min(coordinates, key=lambda item: item[3])[3]
    max_x = max(coordinates, key=lambda item: item[3] + item[5])[3] + max(coordinates, key=lambda item: item[3] + item[5])[5]
    return min_x, max_x

def is_row_centered(row_coords, general_min_x, general_max_x, threshold=0.15):
    min_x = min(row_coords, key=lambda item: item[0])[0]
    max_x = max(row_coords, key=lambda item: item[0] + item[2])[0] + max(row_coords, key=lambda item: item[0] + item[2])[2]
    
    center_region_left = general_min_x + (general_max_x - general_min_x) * threshold
    center_region_right = general_max_x - (general_max_x - general_min_x) * threshold
    
    return center_region_left <= min_x and max_x <= center_region_right

def find_first_valid_row(coordinates, row_mapping, subgroup_lower_bound):
    general_min_x, general_max_x = find_general_boundaries(coordinates)
    first_valid_row = None
    
    for i, (row_num, lower_limit, upper_limit) in enumerate(row_mapping):
        if row_num < subgroup_lower_bound:
            continue
        
        row_coords = [(x, y, w, h) for _, _, _, x, y, w, h in coordinates if lower_limit <= y <= upper_limit]
        
        if len(row_coords) >= 1:  # Ensure there is at least one image in the row
            if not is_row_centered(row_coords, general_min_x, general_max_x):
                if first_valid_row is None:
                    first_valid_row = row_num
                    # Check the previous row only if the first valid row is not the lower bound of the subgroup
                    
                    if row_num > subgroup_lower_bound:
                        if i > 0:
                            prev_row_num, prev_lower_limit, prev_upper_limit = row_mapping[i-1]
                            prev_row_coords = [(x, y, w, h) for _, _, _, x, y, w, h in coordinates if prev_lower_limit <= y <= prev_upper_limit]

                            if len(prev_row_coords) > 2:
                                return prev_row_num
                            elif len(prev_row_coords) <= 2:
                                valid_prev_row = False
                                for (x, y, w, h) in prev_row_coords:
                                    if ((w / h > 1.6) and h > 8) or (w < 5):
                                        valid_prev_row = True
                                        break
                                if not valid_prev_row:
                                    return prev_row_num
                    return first_valid_row
                else:
                    return first_valid_row
    return None  # In case no valid row is found

def assign_column_numbers(all_row_mappings, first_row, sam_and_taalis_rows, aspect_ratio_threshold=1.6):
    composition_folder  = PATHS['working_composition']
    images = os.listdir(composition_folder)
    coordinates = []
    subgroup_ranges = []  # Store subgroup ranges

    # Filter images within the start_row and end_row range
    for image in images:
        details = get_image_details_with_row(image)
        if details:
            page_num, row_num, x, y, w, h = details
            coordinates.append((image, page_num, row_num, x, y, w, h))

    # Flatten row_mapping for the rows within the range
    flattened_row_mapping = []
    for row_mapping in all_row_mappings:
        for row_num, lower_limit, upper_limit in row_mapping:
            flattened_row_mapping.append((row_num, lower_limit, upper_limit))

    # Sort flattened_row_mapping by row_num
    flattened_row_mapping = sorted(flattened_row_mapping, key=lambda item: item[0])

    # Adjust sam_and_taalis_rows to include the start_row
    sam_and_taalis_rows = [first_row - 1] + sam_and_taalis_rows

    for i in range(len(sam_and_taalis_rows) - 1):
        start_subgroup = sam_and_taalis_rows[i] + 1
        end_subgroup = sam_and_taalis_rows[i + 1]
        subgroup_coords = [
            (image, page_num, row_num, x, y, w, h) for image, page_num, row_num, x, y, w, h in coordinates
            if start_subgroup <= row_num <= end_subgroup
        ]
        if not subgroup_coords:
            continue
        
        # Print the current subgroup range
        print(f"Subgroup range: [{start_subgroup}, {end_subgroup}]")
        
        # Find and mark invalid rows in the subgroup
        invalid_rows = []
        first_valid_row_in_subgroup = find_first_valid_row(subgroup_coords, flattened_row_mapping, start_subgroup)

        # Print the first valid row in the subgroup
        print("First valid row in subgroup: ", first_valid_row_in_subgroup)

        # Store the current subgroup range as a tuple
        subgroup_ranges.append((first_valid_row_in_subgroup, end_subgroup - 1))
        
        for image, page_num, row_num, x, y, w, h in subgroup_coords:
            if row_num < first_valid_row_in_subgroup:
                invalid_rows.append(image)
                update_filename_with_col_or_suffix(composition_folder, image, "extra")
        
        # Filter out invalid rows
        valid_subgroup_coords = [
            (image, page_num, row_num, x, y, w, h) for image, page_num, row_num, x, y, w, h in subgroup_coords
            if row_num >= first_valid_row_in_subgroup
        ]
        
        if not valid_subgroup_coords:
            continue
        
        valid_subgroup_coords_sorted_by_x = sorted(valid_subgroup_coords, key=lambda item: item[3])  # Sort by x

        # Extract (x, y, w, h) for create_mapping
        valid_subgroup_coords_mapping = [
            (x, y, w, h) for _, _, _, x, y, w, h in valid_subgroup_coords_sorted_by_x
        ]

        column_mapping = create_mapping(valid_subgroup_coords_mapping, aspect_ratio_threshold, is_row=False)

        for image, page_num, row_num, x, y, w, h in valid_subgroup_coords_sorted_by_x:
            col_num = assign_number(x, column_mapping)
            if col_num != -1:
                update_filename_with_col_or_suffix(composition_folder, image, f"col{col_num}")
        
    return subgroup_ranges