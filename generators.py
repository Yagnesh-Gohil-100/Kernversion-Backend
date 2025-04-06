import os
import shutil
from filename_utils import get_image_details_with_row

def copy_images_in_row_range(input_folder, output_folder, first_row, last_row):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        details = get_image_details_with_row(filename)
        if details:
            _, row_num, _, _, _, _ = details
            if first_row <= row_num <= last_row:
                shutil.copy(os.path.join(input_folder, filename), os.path.join(output_folder, filename))

# ----------------------------------------------------------------------------------------------------------

import numpy as np
from save_and_load import load_row_categories
from modifications import pad_lists
from segmentation import segment_swar_and_kann_swar, filter_kann_swar_from_swar

# Function to process a subgroup and create the lists of lists
def generate_lists(subgroup_range, row_col_images, is_first_subgroup, beat_count):
    start_row, end_row = subgroup_range

    row_categories = load_row_categories()

    # Extract specific row categories
    articulation_rows = row_categories.get("articulation", [])
    kann_swar_rows = row_categories.get("kann_swar", [])
    swar_rows = row_categories.get("swar", [])
    lyrics_rows = row_categories.get("lyrics", [])
    
    # Find the swar row in this subgroup
    swar_row = None
    for row in swar_rows:
        if start_row <= row <= end_row:
            swar_row = row
            break
    
    if not swar_row:
        return None, None, None, None, None
    
    # Find the kann swar row in this subgroup
    kann_swar_row = None
    for row in kann_swar_rows:
        if start_row <= row <= end_row:
            kann_swar_row = row
            break
    
    # Find the articulation rows in this subgroup
    articulation_rows_in_subgroup = [row for row in articulation_rows if start_row <= row <= end_row]
    
    # Find the lyrics row in this subgroup
    lyrics_row = None
    for row in lyrics_rows:
        if start_row <= row <= end_row:
            lyrics_row = row
            break
    
    # Get the swar images and their column numbers
    swar_images = row_col_images[swar_row]
    swar_cols = sorted(swar_images.keys())
    
    # Get the kann swar images and their column numbers (if kann swar row exists)
    kann_swar_images = row_col_images[kann_swar_row] if kann_swar_row else {}
    kann_swar_cols = sorted(kann_swar_images.keys())
    
    # Get the lyrics images (if lyrics row exists)
    lyrics_images = row_col_images[lyrics_row] if lyrics_row else {}
    lyrics_cols = sorted(lyrics_images.keys()) if lyrics_row else []
    
    # Create the lists of lists
    swar_list = []
    kann_swar_list = []
    swar_articulation_checks = [False] * len(swar_cols)
    lyrics_articulation_checks = [False] * len(lyrics_cols)
    lyrics_list = []
    
    # Case 1: If there is an explicit kann swar row
    if kann_swar_row:
        swar_index = 0
        kann_swar_index = 0
        
        while swar_index < len(swar_cols) or kann_swar_index < len(kann_swar_cols):
            swar_col = swar_cols[swar_index] if swar_index < len(swar_cols) else None
            kann_swar_col = kann_swar_cols[kann_swar_index] if kann_swar_index < len(kann_swar_cols) else None
            
            # If both columns exist and match
            if swar_col is not None and kann_swar_col is not None and swar_col == kann_swar_col:
                swar_list.append([x[4] for x in swar_images[swar_col]])  # Store image paths
                kann_swar_list.append([x[4] for x in kann_swar_images[kann_swar_col]])  # Store image paths
                swar_index += 1
                kann_swar_index += 1
            # If swar column exists but kann swar column doesn't match or is missing
            elif swar_col is not None and (kann_swar_col is None or swar_col < kann_swar_col):
                swar_list.append([x[4] for x in swar_images[swar_col]])  # Store image paths
                kann_swar_list.append([])
                swar_index += 1
            # If kann swar column exists but swar column doesn't match or is missing
            elif kann_swar_col is not None and (swar_col is None or kann_swar_col < swar_col):
                # Assign the kann swar to the next available swar column
                if swar_index < len(swar_cols):
                    swar_list.append([x[4] for x in swar_images[swar_cols[swar_index]]])  # Store image paths
                    kann_swar_list.append([x[4] for x in kann_swar_images[kann_swar_col]])  # Store image paths
                    swar_index += 1
                    kann_swar_index += 1
                else:
                    # If no more swar columns are available, append an empty list
                    swar_list.append([])
                    kann_swar_list.append([x[4] for x in kann_swar_images[kann_swar_col]])  # Store image paths
                    kann_swar_index += 1
    
    # Case 2: If there is no explicit kann swar row, check for hidden kann swars in the swar row
    else:
        for col in swar_cols:
            images_in_col = swar_images[col]
            
            # Separate outliers
            outlier_images = [img for img in images_in_col if img[3] > 25]
            non_outlier_images = [img for img in images_in_col if img[3] <= 25]
            
            # Process outliers
            for img in outlier_images:
                swar, kann_swar = segment_swar_and_kann_swar(img, subgroup_range, col)
                swar_list.append(swar)
                kann_swar_list.append(kann_swar)
            
            # Process non-outliers
            if non_outlier_images:
                swar, kann_swar = filter_kann_swar_from_swar(non_outlier_images)
                swar_list.append(swar)
                kann_swar_list.append(kann_swar)
    
    # Handle articulation rows
    for articulation_row in articulation_rows_in_subgroup:
        # Find the row just before the articulation row
        prev_row = articulation_row - 1
        if prev_row in swar_rows:
            # Swar articulation
            articulation_images = row_col_images[articulation_row]
            articulation_cols = sorted(articulation_images.keys())
            for i, col in enumerate(swar_cols):
                if col in articulation_cols:
                    swar_articulation_checks[i] = True
        elif prev_row in lyrics_rows:
            # Lyrics articulation
            articulation_images = row_col_images[articulation_row]
            articulation_cols = sorted(articulation_images.keys())
            for i, col in enumerate(lyrics_cols):
                if col in articulation_cols:
                    lyrics_articulation_checks[i] = True
    
    # Handle lyrics row (append images one by one without comparing column numbers)
    if lyrics_row:
        # Get all lyrics images in order
        lyrics_cols = sorted(lyrics_images.keys())
        for col in lyrics_cols:
            lyrics_list.append([x[4] for x in lyrics_images[col]])  # Store image paths
    else:
        lyrics_list = [[] for _ in range(len(swar_cols))]
    
    # Pad lists to match the beat count
    if is_first_subgroup:
        swar_list = pad_lists(swar_list, beat_count)
        kann_swar_list = pad_lists(kann_swar_list, beat_count)
        swar_articulation_checks = pad_lists(swar_articulation_checks, beat_count)
        lyrics_articulation_checks = pad_lists(lyrics_articulation_checks, beat_count)
        lyrics_list = pad_lists(lyrics_list, beat_count)
    else:
        if len(swar_list) < beat_count:
            swar_list += [[] for _ in range(beat_count - len(swar_list))]
        if len(kann_swar_list) < beat_count:
            kann_swar_list += [[] for _ in range(beat_count - len(kann_swar_list))]
        if len(swar_articulation_checks) < beat_count:
            swar_articulation_checks += [False for _ in range(beat_count - len(swar_articulation_checks))]
        if len(lyrics_articulation_checks) < beat_count:
            lyrics_articulation_checks += [False for _ in range(beat_count - len(lyrics_articulation_checks))]
        if len(lyrics_list) < beat_count:
            lyrics_list += [[] for _ in range(beat_count - len(lyrics_list))]
    
    return swar_list, kann_swar_list, swar_articulation_checks, lyrics_articulation_checks, lyrics_list


def generate_lists_in_subgroups(subgroup_ranges, row_col_images, beat_count):
    """
    Process all subgroups and return a dictionary of results.

    Args:
        subgroup_ranges (list): A list of tuples where each tuple represents (start_row, end_row).

    Returns:
        dict: A dictionary containing processed results for each subgroup.
    """
    subgroup_results = {}

    for i, subgroup_range in enumerate(subgroup_ranges):
        is_first_subgroup = (i == 0)
        
        # Call process_subgroup for each subgroup
        swar_list, kann_swar_list, swar_articulation_checks, lyrics_articulation_checks, lyrics_list = generate_lists(subgroup_range, row_col_images, is_first_subgroup, beat_count)
        
        if swar_list and kann_swar_list:
            subgroup_results[subgroup_range] = {
                'swar_list': swar_list,
                'kann_swar_list': kann_swar_list,
                'swar_articulation_checks': swar_articulation_checks,
                'lyrics_articulation_checks': lyrics_articulation_checks,
                'lyrics_list': lyrics_list
            }
    
    return subgroup_results

# ----------------------------------------------------------------------------------------------------------

from save_and_load import load_lists_in_subgroups, save_kann_swar_segment_from_meend, save_lists_in_subgroups
from segmentation import segment_meend_and_kann_swar
from identifications import identify_meend_and_kann_swar

# Function to update kann swar list and generate meend lists
def update_kann_swar_and_generate_meend_lists():
    """
    Function to update kann swar and meend lists based on segmentation.
    
    Parameters:
    - subgroup_results: Dictionary containing subgroup results.
    """

    # Load existing lists from JSON
    subgroup_results = load_lists_in_subgroups()
    
    if not subgroup_results:
        print("No data found in JSON file")
        return

    for subgroup_range, results in subgroup_results.items():
        kann_swar_list = results['kann_swar_list']
        swar_list = results['swar_list']
        
        # Initialize meend list with empty values
        meend_list = ['' for _ in range(len(swar_list))]
        
        i = 0
        while i < len(kann_swar_list):
            if kann_swar_list[i]:  # Check if the list is not empty
                image_path = kann_swar_list[i][0]
                # Extract width from the filename
                filename = os.path.basename(image_path)
                width = int(filename.split('_w')[1].split('_')[0])
                
                if width > 20:  # Only process if width > 20
                    # Perform segmentation
                    left_part, mid_part, right_part = segment_meend_and_kann_swar(image_path)
                    
                    # Identify and structure meend and kann swar
                    left_part, mid_part, right_part = identify_meend_and_kann_swar(left_part, mid_part, right_part)
                    
                    if mid_part is not None:  # If meend is found
                        # Mark start of meend
                        meend_list[i] = 'S'
                        
                        # Calculate x + w for the current image
                        x = int(filename.split('_x')[1].split('_')[0])
                        w = width
                        x_end = x + w
                        
                        # Find the end of meend
                        j = i + 1
                        while j < len(swar_list):
                            swar_image_path = swar_list[j][0]
                            swar_filename = os.path.basename(swar_image_path)
                            swar_x = int(swar_filename.split('_x')[1].split('_')[0])
                            
                            if swar_x >= x_end:
                                break  # Stop if swar_x is outside meend area
                            j += 1
                        
                        # Mark end of meend
                        if j > i:
                            meend_list[j - 1] = 'E'
                        
                        # Update kann swar list based on segmentation
                        if left_part is not None:
                            kann_swar_list[i] = [save_kann_swar_segment_from_meend(left_part, subgroup_range, i, 'left')]
                        if right_part is not None:
                            kann_swar_list[j - 1] = [save_kann_swar_segment_from_meend(right_part, subgroup_range, j - 1, 'right')]
                        if left_part is None and right_part is None:
                            kann_swar_list[i] = []  # Remove the original image if no segmentation
                        
                        # Skip processed indices
                        i = j
                    else:
                        i += 1
                else:
                    i += 1
            else:
                i += 1
        
        # Update the subgroup results with the meend list
        subgroup_results[subgroup_range]['meend_list'] = meend_list
    
    # update the subgroup lists
    save_lists_in_subgroups(subgroup_results)

# ----------------------------------------------------------------------------------------------------------

from image_processing import preprocess_image_to_predict
from save_and_load import load_my_model, load_classes

# Pass the image through the model and get predictions
def predict_class(image_path):

    model = load_my_model()

    preprocessed_image = preprocess_image_to_predict(image_path)
    predictions = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(predictions, axis=1)
    max_probability = np.max(predictions, axis=1)
    return predicted_class_index[0], max_probability[0]

def generate_predictions(subgroup_results):
    """
    Generates predictions with proper class name mapping
    Args:
        subgroup_results: Dictionary containing 'swar_list' and 'kann_swar_list'
    Returns:
        Dictionary with predicted class names (not indices)
    """
    predictions = {}
    CLASSES = load_classes()
    
    for subgroup_range, results in subgroup_results.items():
        predictions[subgroup_range] = {
            "predicted_swar_list": [
                [CLASSES[predict_class(path)[0]] for path in paths]  # Convert index to name
                if paths else []
                for paths in results['swar_list']
            ],
            "predicted_kann_swar_list": [
                [CLASSES[predict_class(path)[0]] for path in paths]  # Convert index to name
                if paths else []
                for paths in results['kann_swar_list']
            ]
        }
    
    return predictions

# ----------------------------------------------------------------------------------------------------------

from save_and_load import get_taal_field, get_metadata_field

# Function to generate kern code based on user input
def generate_metadata_header(raag, taal, lay):
    
    time_signature = get_taal_field(taal, field_name='time_signature')
    
    # Set tempo based on lay
    # Tempo mapping
    tempo_map = {
        "vilambit": 60,
        "madhya": 90,
        "drut": 150
    }

    tempo = tempo_map.get(lay.lower(), 60)  # Default to vilambit

    metadata_headers = [
    f"!!!raag: {raag}",
    f"!!!taal: {taal}",
    f"!!!lay: {lay}"
  ]

    source_name = get_metadata_field(field="source.name")
    page_num = get_metadata_field(field="source.page")

    if source_name:
        metadata_headers.append(f"!!!source: {source_name}")
    if page_num:
        metadata_headers.append(f"!!!page: {page_num}")

    # Add reference records with proper spacing
    reference_records = [
        "**kern",
        f"*M{time_signature}",
        f"*MM{tempo}",
        "*clefG2",  # Added treble clef for completeness
        "*c:"
    ]

    metadata_headers.extend(reference_records)

    kern_str = "\n".join(metadata_headers)
    return kern_str.strip(), metadata_headers


# ----------------------------------------------------------------------------------------------------------

from save_and_load import load_kern_map

def handle_extension(kern_output):
    """Handle extension cases for '-' in swar"""
    if not kern_output:
        return
    
    # Pop division markers if they exist
    popped = []
    while kern_output and (kern_output[-1].strip().startswith('==') or kern_output[-1].strip() == '='):
        popped.append(kern_output.pop())
    
    if not kern_output:
        kern_output.extend(popped)
        return
    
    last = kern_output[-1].strip()
    
    # Check if the last element has 'S' suffix
    has_S_suffix = last.endswith('S')
    if has_S_suffix:
        last = last[:-1]  # Remove 'S' temporarily
    
    # Case 3: Ends with ']'
    if last.endswith(']'):
        kern_code = last[:-1]
        kern_output[-1] = kern_code
        modified = modify_kern_code(kern_code)
        if isinstance(modified, tuple):
            kern_output[-1] = modified[0] + ('S' if has_S_suffix else '') + '\n'
            kern_output.append(modified[1] + ('S' if has_S_suffix else '') + '\n')
        else:
            kern_output[-1] = modified + ('S' if has_S_suffix else '') + ']\n'
    
    # Case 4,5,6: Regular kern codes
    else:
        modified = modify_kern_code(last)
        if isinstance(modified, tuple):
            kern_output[-1] = modified[0] + ('S' if has_S_suffix else '') + '\n'
            kern_output.append(modified[1] + ('S' if has_S_suffix else '') + '\n')
        else:
            kern_output[-1] = modified + ('S' if has_S_suffix else '') + '\n'
    
    # Push back any popped division markers
    kern_output.extend(reversed(popped))

def modify_kern_code(kern_code):
    """Modify kern code according to extension rules"""
    # Split into number and code parts
    num_part = ''
    code_part = kern_code
    for i, c in enumerate(kern_code):
        if c.isdigit():
            num_part += c
        else:
            code_part = kern_code[i:]
            break
    
    if not num_part:
        return '2' + code_part
    
    num = int(num_part)
    
    if num == 4:
        return '2' + code_part
    elif num < 4:
        return f'{num}.' + code_part
    else:  # num > 4
        return (f'[{num}{code_part}', f'4{code_part}]')

def generate_kern(flat_kann, flat_swar, divisions, beat_count):
    
    kern_map = load_kern_map()

    kern_output = []
    total_beats = len(flat_kann)
    cumulative_divisions = []
    current_sum = 0
    
    while current_sum <= total_beats:
        for d in divisions:
            current_sum += d
            if current_sum >= total_beats:
                break
            cumulative_divisions.append(current_sum)

    first_empty_flag = True
    
    for i in range(total_beats):
        # Handle major divisions (beat_count)
        if i % beat_count == 0:
            kern_output.append(f"\n=={i // beat_count + 1}\n")
        
        # Handle subdivisions
        if i in cumulative_divisions and i % beat_count != 0:
            kern_output.append("\n=\n")
        
        # Process current beat
        kann = flat_kann[i]
        swar = flat_swar[i]
        
        # Handle empty lists
        if not kann and not swar and first_empty_flag:
            kern_output.append("4ryy\n")
            continue

        # Do not need to add rests at the last
        if not kann and not swar and not first_empty_flag:
            # Join and clean up newlines
            kern_str = ''.join(kern_output)
            kern_str = kern_str.replace('\n\n', '\n')  # Remove duplicate newlines
            return kern_str.strip(), kern_output

        # set flag to false to handle empty lists at the end of composition
        first_empty_flag = False
        
        # Process non-empty lists
        for idx in range(max(len(kann), len(swar))):
            # Process kann if exists
            if idx < len(kann):
                k = kann[idx]
                if k != '':
                    kern_output.append(f"{kern_map[k]}q\n")
            
            # Process swar if exists
            if idx < len(swar):
                s = swar[idx]
                if s == '-':
                    handle_extension(kern_output)
                else:
                    # Handle parenthesized elements in swar
                    is_parenthesized = s.startswith('(') and s.endswith(')')
                    s_clean = s[1:-1] if is_parenthesized else s
                    duration = 4 * len(swar)
                    kern_code = f"{duration}{kern_map[s_clean]}"
                    if is_parenthesized:
                        kern_code += 'S'
                    kern_output.append(f"{kern_code}\n")
    
    # Join and clean up newlines
    kern_str = ''.join(kern_output)
    kern_str = kern_str.replace('\n\n', '\n')  # Remove duplicate newlines
    return kern_str.strip(), kern_output

def generate_transition(kern_output, from_vibhaag, to_vibhaag):
    # count number of divisions passed in last beat cycle, next taali will be at this count index
    count_division = 0

    for i in reversed(range(len(kern_output))):
        if kern_output[i].startswith('\n=='):
            break
        elif kern_output[i] == '\n=\n':
            count_division += 1

    # get index of sam in from vibhaag
    from_sam_index = -1
    for i in range(len(from_vibhaag)):
        if from_vibhaag[i] == 'X':
            from_sam_index = i
            break

    # get index of sam in to vibhaag
    to_sam_index = -1
    for i in range(len(to_vibhaag)):
        if to_vibhaag[i] == 'X':
            to_sam_index = i
            break
    
    i = 0
    x = ((from_sam_index + i) % len(from_vibhaag))

    while ((from_sam_index + i) % len(from_vibhaag)) != count_division:
        i += 1

    # find next taali index in to vibhaag
    next_taali_index = ((to_sam_index + i) % len(from_vibhaag))

    # starting from index as per count_division, number of divisions to repeat
    count_repeat_division = (len(from_vibhaag) - next_taali_index) % len(from_vibhaag)

    count = -1
    i = 0
    while count < count_division:
        item = kern_output[i]
        if item.startswith('\n==') or item == ('\n=\n'):
            count += 1
        i += 1

    
    # transition output starts from index i
    transition_kern = []

    count = 0
    while count < count_repeat_division:
        item = kern_output[i]
        if item.startswith('\n==') or item == ('\n=\n'):
            count += 1
            item = '\n=\n'

        transition_kern.append(item)
        i += 1

    # Join and clean up newlines
    kern_str = ''.join(transition_kern)
    kern_str = kern_str.replace('\n\n', '\n')  # Remove duplicate newlines
    return kern_str.strip(), transition_kern
