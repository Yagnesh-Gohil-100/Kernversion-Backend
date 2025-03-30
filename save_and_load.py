import json
from typing import Dict, Optional

def save_composition_metadata(
    raag_name: str,
    taal_name: str,
    lay: str,
    sthayee_sam: int,
    antara_sam: Optional[int] = None,
    sanchari_sam: Optional[int] = None,
    aabhog_sam: Optional[int] = None,
    source_name: Optional[str] = None,
    page_number: Optional[int] = None,
    filename: str = "composition_metadata.json"
):
    """Save composition metadata to JSON file"""
    data = {
        "raag_name": raag_name,
        "taal_name": taal_name,
        "lay": lay,
        "sthayee": {"sam_beat": sthayee_sam},
        "antara": {"sam_beat": antara_sam} if antara_sam else None,
        "sanchari": {"sam_beat": sanchari_sam} if sanchari_sam else None,
        "aabhog": {"sam_beat": aabhog_sam} if aabhog_sam else None,
        "source": {
            "name": source_name,
            "page": page_number
        } if source_name else None
    }
    
    # Remove None values
    data = {k: v for k, v in data.items() if v is not None}
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_composition_metadata(filename: str = "composition_metadata.json") -> Dict:
    """Load raag metadata from JSON file"""
    if not os.path.exists(filename):
        return {}
    
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_metadata_field(filename: str = "composition_metadata.json", field: str = None):
    """Get specific field from metadata"""
    data = load_composition_metadata(filename)
    if not field:
        return data
    
    # Handle nested fields
    if '.' in field:
        keys = field.split('.')
        value = data
        for key in keys:
            value = value.get(key, {})
        return value if value != {} else None
    
    return data.get(field)

# -----------------------------------------------------------------------------------------

from typing import Any

# Load the taal data once
with open('taal_info.json', 'r', encoding='utf-8') as f:
    _TAAL_DATA = json.load(f)

def get_taal_data(taal_name: str) -> Dict[str, Any]:
    """Get complete data for a specific taal"""
    return _TAAL_DATA.get(taal_name)

def get_taal_field(taal_name: str, field_name: str) -> Any:
    """Get specific field from a taal's data"""
    taal = _TAAL_DATA.get(taal_name)
    return taal.get(field_name) if taal else None

# -----------------------------------------------------------------------------------------

def save_row_categories(row_categories, filename="row_categories.json"):
    with open(filename, "w") as f:
        json.dump(row_categories, f, indent=4)

def load_row_categories(filename="row_categories.json"):
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None  # Handle case when no file exists yet
    
# -----------------------------------------------------------------------------------------

import os
import re
import cv2
from config import PATHS

def save_segment_swar_and_kann_swar(segment, part_type, original_filename):
    """
    Function to save a segmented part and return its path.
    
    Parameters:
    - segment: The segmented image (enlarged by a factor of 3).
    - subgroup_range: The subgroup range.
    - col: The column number.
    - part_type: Type of segment ('upper' or 'lower').
    - original_filename: The original filename of the image before segmentation.
    
    Returns:
    - Path to the saved segment.
    """
    # Extract original image details from the filename
    pattern = r'(\d+)_row(\d+)_col(\d+)_x(\d+)_y(\d+)_w(\d+)_h(\d+)'
    match = re.match(pattern, original_filename)
    if not match:
        raise ValueError(f"Original filename {original_filename} does not match the expected pattern.")
    
    page_num = match.group(1)
    row_num = match.group(2)
    col_num = match.group(3)
    original_x = int(match.group(4))  # x-coordinate (pre-enlarged)
    original_y = int(match.group(5))  # y-coordinate (pre-enlarged)
    original_w = int(match.group(6))  # width (pre-enlarged)
    original_h = int(match.group(7))  # height (pre-enlarged)
    
    # Calculate new coordinates for the segmented part (scaled down by a factor of 3)
    if part_type == 'upper':
        # Upper part: y remains the same, height is the separation row
        new_x = original_x
        new_y = original_y
        new_w = original_w
        new_h = segment.shape[0] // 3  # Height of the upper part (scaled down)
    elif part_type == 'lower':
        # Lower part: y is original_y + height of the upper part, height is adjusted
        new_x = original_x
        new_y = original_y + (original_h - (segment.shape[0] // 3))  # Adjust y for lower part (scaled down)
        new_w = original_w
        new_h = segment.shape[0] // 3  # Height of the lower part (scaled down)
    else:
        raise ValueError("Invalid part_type. Must be 'upper' or 'lower'.")
    
    # Create the new filename
    new_filename = f"{page_num}_row{row_num}_col{col_num}_x{new_x}_y{new_y}_w{new_w}_h{new_h}_{part_type}.png"
    
    # Save the segmented image
    composition_segmented_folder = os.path.normpath(PATHS['working_composition_segmented'])
    os.makedirs(composition_segmented_folder, exist_ok=True)
    segment_path = os.path.join(composition_segmented_folder, new_filename)
    cv2.imwrite(segment_path, segment)
    
    return segment_path

# ------------------------------------------------------------------------------------------------------------

def save_lists_in_subgroups(subgroup_results: Dict[tuple, Dict[str, Any]], 
                          filename = "subgroups.json") -> None:
    """
    Save subgroup processing results to a JSON file with proper serialization.
    
    Args:
        subgroup_results: Dictionary with tuple keys (start_row, end_row) and 
                        nested dictionary values containing:
                        - kann_swar_list: List of lists of image paths
                        - swar_list: List of lists of image paths  
                        - swar_articulation_checks: List of booleans
                        - lyrics_list: List of lists of image paths
                        - lyrics_articulation_checks: List of booleans
        filename: Output JSON file path
    """
    # Convert tuple keys to strings for JSON compatibility
    serializable_results = {
        f"subgroup_{start}_{end}": data  # Convert (26,31) to "subgroup_26_31"
        for (start, end), data in subgroup_results.items()
    }
    
    # Write with proper encoding and formatting
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, ensure_ascii=False, indent=4)

def load_lists_in_subgroups(filename = "subgroups.json") -> Dict[tuple, Dict[str, Any]]:
    """Load saved subgroup results, converting keys back to tuples"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return {
            tuple(map(int, key.split('_')[1:3])): value  # Converts "subgroup_26_31" to (26, 31)
            for key, value in data.items()
        }
    except FileNotFoundError:
        return {}
    
# ------------------------------------------------------------------------------------------------------------

# Function to save segmented kann swar from composite meend and kann swar
def save_kann_swar_segment_from_meend(segment, subgroup_range, index, part_type):
    """
    Function to save a segmented part and return its path.
    
    Parameters:
    - segment: The segmented image.
    - subgroup_range: The subgroup range.
    - index: The index in the list.
    - part_type: Type of segment ('left', 'mid', 'right').
    
    Returns:
    - Path to the saved segment.
    """

    composition_segmented_folder = os.path.normpath(PATHS['working_composition_segmented'])
    os.makedirs(composition_segmented_folder, exist_ok=True)
    
    segment_filename = f"{subgroup_range[0]}_{subgroup_range[1]}_{index}_{part_type}.png"
    segment_path = os.path.join(composition_segmented_folder, segment_filename)
    cv2.imwrite(segment_path, segment)
    
    return segment_path

# ------------------------------------------------------------------------------------------------------------

def save_word_segmented_images(original_path, segmented_image, target_list, index):
    """
    Saves a segmented image while preserving the original filename and updates the target list
    
    Args:
        original_path: Path to the original image
        segmented_image: The segmented image data (numpy array)
        segmented_folder_path: Directory to save segmented images
        target_list: The list to update (swar_list or lyrics_list)
        index: Position in the list to update
    """

    composition_segmented_folder = os.path.normpath(PATHS['working_composition_segmented'])
    os.makedirs(composition_segmented_folder, exist_ok=True)

    original_name = os.path.basename(original_path)
    seg_image_path = os.path.normpath(os.path.join(composition_segmented_folder, original_name))
    cv2.imwrite(seg_image_path, segmented_image)
    target_list[index] = [seg_image_path]

# ------------------------------------------------------------------------------------------------------------

from datetime import datetime
from tensorflow.keras.models import load_model
from config import PATHS

def load_my_model():
    """Simple model loader with basic error handling"""

    model = PATHS['model']
    try:
        return load_model(model)
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def load_classes():
    with open('classes.json') as f:
        data = json.load(f)
    return data['classes']

def save_predictions(predictions, output_file="predictions.json"):
    """Saves predictions with tuple ranges"""
    data = {
        "metadata": {
            "created": datetime.now().isoformat(),
            "classes_version": "1.0",
            "subgroup_count": len(predictions)
        },
        "predictions": {
            f"subgroup_{start}_{end}": {                            # Convert (26,31) to "subgroup_26_31"
                "kann_swar": results["predicted_kann_swar_list"],
                "swar": results["predicted_swar_list"]
            }
            for (start, end), results in predictions.items()  
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_predictions(input_file="predictions.json"):
    """Loads predictions with original range formatting"""
    if not os.path.exists(input_file):
        return None
        
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Reconstruct original format with ranges
    return {
        tuple(map(int, key.split('_')[1:3])): value  # Converts "subgroup_26_31" to (26, 31)
        for key, value in data['predictions'].items()
    }
