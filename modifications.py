import os
from filename_utils import get_row_and_col_number
from config import PATHS

def update_subgroups(subgroups):
    """
    Processes subgroups by checking if the first subgroup needs to be split into two.
    """
    composition_folder = PATHS['working_composition']
    images = os.listdir(composition_folder)
    images.sort(key=lambda x: get_row_and_col_number(x)[0])  # Sort by row number
    
    first_subgroup_start, first_subgroup_end = subgroups[0]
    first_valid_row = first_subgroup_start

    # Only process rows from the first valid row
    first_group_images = [img for img in images if get_row_and_col_number(img)[0] >= first_valid_row]

    # To track if we need to split the first subgroup
    first_row_images = [img for img in first_group_images if get_row_and_col_number(img)[0] == first_valid_row]
    second_row_images = [img for img in first_group_images if get_row_and_col_number(img)[0] == first_valid_row + 1]

    # Ensure we have valid rows and columns to process
    if first_row_images and second_row_images:
        # Sort images by column number and check the first (lowest column number) image
        first_row_images.sort(key=lambda x: get_row_and_col_number(x)[1])
        second_row_images.sort(key=lambda x: get_row_and_col_number(x)[1])
        
        first_row_col = get_row_and_col_number(first_row_images[0])[1]
        second_row_col = get_row_and_col_number(second_row_images[0])[1]

        if first_row_col is not None and second_row_col is not None and first_row_col > 1 and second_row_col > 1:
            # Now, let's iterate through rows to find where col = 1 begins
            new_first_end = first_valid_row  # Default in case we find no rows with col = 1
            for img in first_group_images:
                row, col = get_row_and_col_number(img)
                if row > first_valid_row and col == 1:
                    new_first_end = row
                    break

            # Update the subgroups
            first_subgroup = (first_valid_row, new_first_end)
            second_subgroup = (new_first_end, first_subgroup_end)
            subgroups[0] = first_subgroup
            subgroups.insert(1, second_subgroup)
    
    return subgroups

# ------------------------------------------------------------------------------------------------

# Function to pad lists to match the beat count
def pad_lists(lists, size):
    if len(lists) < size:
        padding = [[] for _ in range(size - len(lists))]
        return padding + lists
    return lists

# ------------------------------------------------------------------------------------------------

import cv2
from save_and_load import load_lists_in_subgroups, save_word_segmented_images, save_lists_in_subgroups
from segmentation import separate_articulation, segment_word

def apply_articulation_segmentation(row_list, articulation_checks):
    """
    Applies articulation segmentation to a row (swar or lyrics)
    
    Args:
        row_list: The list of images (swar_list or lyrics_list)
        articulation_checks: Corresponding articulation checks list
        segmented_folder_path: Path to save segmented images
    """
    for i in range(len(row_list)):
        if not articulation_checks[i] and row_list[i]:  # Check if articulation is False and the list is not empty
            image_path = row_list[i][0]  # Get the image path
            image = cv2.imread(image_path)   # Load the image
            if image is not None:
                segmented_image, is_segmented = separate_articulation(image)
                if is_segmented:
                    articulation_checks[i] = True

                    # Save the segmented image with the original name
                    save_word_segmented_images(image_path, segmented_image, row_list, i)

def apply_word_segmentation(row_list, articulation_checks):
    """
    Applies word segmentation to a row (swar or lyrics)
    
    Args:
        row_list: The list of images (swar_list or lyrics_list)
        articulation_checks: Corresponding articulation checks list
        segmented_folder_path: Path to save segmented images
    """

    composition_segmented_folder = os.path.normpath(PATHS['working_composition_segmented'])
    os.makedirs(composition_segmented_folder, exist_ok=True)

    for i in range(len(row_list)):
        if articulation_checks[i] and row_list[i]:  # Check if articulation is True and the list is not empty
            image_path = row_list[i][0]  # Get the image path
            segmented_paths = segment_word(image_path, composition_segmented_folder)
            if segmented_paths:
                row_list[i] = segmented_paths  # Update the list with segmented image paths

# Function to update and finalize lists after articulation separation and word segmentation
def finalize_segmentation_and_lists():

    subgroup_results = load_lists_in_subgroups()

    for subgroup_range, results in subgroup_results.items():
        swar_list = results['swar_list']
        lyrics_list = results['lyrics_list']
        swar_articulation_checks = results['swar_articulation_checks']
        lyrics_articulation_checks = results['lyrics_articulation_checks']
        
        # Apply articulation segmentation to swar row
        apply_articulation_segmentation(swar_list, swar_articulation_checks)
        
        # Apply articulation segmentation to lyrics row
        apply_articulation_segmentation(lyrics_list, lyrics_articulation_checks)
        
        # Apply word segmentation to swar row
        apply_word_segmentation(swar_list, swar_articulation_checks)
        
        # Apply word segmentation to lyrics row
        apply_word_segmentation(lyrics_list, lyrics_articulation_checks)
        
        # Update the results
        subgroup_results[subgroup_range]['swar_list'] = swar_list
        subgroup_results[subgroup_range]['lyrics_list'] = lyrics_list
        subgroup_results[subgroup_range]['swar_articulation_checks'] = swar_articulation_checks
        subgroup_results[subgroup_range]['lyrics_articulation_checks'] = lyrics_articulation_checks

    save_lists_in_subgroups(subgroup_results)

# ------------------------------------------------------------------------------------------------

def categorize_flatten_predictions(predictions, row_categories):
    """Categorizes predictions into sthayee, antara, sanchari, aabhog"""
    sthayee = {"swar": [], "kann_swar": []} if "sthayee" in row_categories else None
    antara = {"swar": [], "kann_swar": []}  if "antara" in row_categories else None
    sanchari = {"swar": [], "kann_swar": []} if "sanchari" in row_categories else None
    aabhog = {"swar": [], "kann_swar": []} if "aabhog" in row_categories else None

    for subgroup_range, subgroup_data in predictions.items():
        start_row = subgroup_range[0]
        
        # Determine which category this subgroup belongs to
        if "sthayee" in row_categories and start_row > row_categories["sthayee"]:
            if "antara" in row_categories and start_row > row_categories["antara"]:
                if "sanchari" in row_categories and start_row > row_categories["sanchari"]:
                    if "aabhog" in row_categories and start_row > row_categories["aabhog"]:
                        target = aabhog
                    else:
                        target = sanchari if sanchari else antara
                else:
                    target = antara
            else:
                target = sthayee
            
            if target:  # Only add if category exists
                target["swar"].extend(subgroup_data["swar"])
                target["kann_swar"].extend(subgroup_data["kann_swar"])
    
    return {
        "sthayee": sthayee,
        "antara": antara,
        "sanchari": sanchari,
        "aabhog": aabhog
    }