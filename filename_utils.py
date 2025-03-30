import os
import re

def get_image_details_with_row(filename):
    pattern = r'(\d+)_row(\d+)_x(\d+)_y(\d+)_w(\d+)_h(\d+)'
    match = re.match(pattern, filename)
    if match:
        page_num = int(match.group(1))
        row_num = int(match.group(2))
        x = int(match.group(3))
        y = int(match.group(4))
        w = int(match.group(5))
        h = int(match.group(6))
        return (page_num, row_num, x, y, w, h)
    return None

def update_filename_with_col_or_suffix(composition_folder, old_filename, new_suffix):
    details = get_image_details_with_row(old_filename)
    if details:
        page_num, row_num, x, y, w, h = details
        new_filename = f"{page_num}_row{row_num}_{new_suffix}_x{x}_y{y}_w{w}_h{h}.png"
        os.rename(os.path.join(composition_folder, old_filename), os.path.join(composition_folder, new_filename))

# ------------------------------------------------------------------------------------------------------

def get_row_and_col_number(filename):
    """
    Extract row and column details from image filename.
    Filename format: '0_row4_col12_x400_y145_w7_h10' or '0_row3_extra_x282_y116_w40_h18'
    Returns (row_num, col_num).
    """
    parts = filename.split('_')
    row_num = None
    col_num = None

    for part in parts:
        if part.startswith('row'):
            row_num = int(part[3:])
        elif part.startswith('col'):
            col_num = int(part[3:])  # Ensure col_num gets a value only if it exists
    
    return row_num, col_num

# ------------------------------------------------------------------------------------------------------

def get_image_details_with_row_and_col(filename):
    pattern = r'(\d+)_row(\d+)_col(\d+)_x(\d+)_y(\d+)_w(\d+)_h(\d+)'
    match = re.match(pattern, filename)
    if match:
        page_num = int(match.group(1))
        row_num = int(match.group(2))
        col_num = int(match.group(3))
        x = int(match.group(4)) 
        y = int(match.group(5))
        w = int(match.group(6))
        h = int(match.group(7))
        return (page_num, row_num, col_num, x, y, w, h)
    return None

# Function to extract information from the image filename
def get_image_details_with_path(filename, image_folder_path):
    pattern = r'(\d+)_row(\d+)(?:_col(\d+))?_x(\d+)_y(\d+)_w(\d+)_h(\d+)'
    match = re.match(pattern, filename)
    if match:
        page_num = int(match.group(1))
        row_num = int(match.group(2))
        col_num = int(match.group(3)) if match.group(3) else None
        x = int(match.group(4))
        y = int(match.group(5))
        width = int(match.group(6))
        height = int(match.group(7))
        # Use os.path.join to handle path separators correctly
        image_path = os.path.normpath(os.path.join(image_folder_path, filename))
        return page_num, row_num, col_num, x, y, width, height, image_path
    return None