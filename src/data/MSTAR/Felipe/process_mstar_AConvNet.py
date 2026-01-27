import os
import re
import numpy as np
import cv2
import json
import shutil
import warnings
from pathlib import Path
import time
import glob
import random

# MSTAR Data Processing - V2 for AConvNet-pytorch compatibility
# This version organizes data according to the structure expected by:
# https://github.com/jangsoopark/AConvNet-pytorch

current_path = os.path.abspath(__file__)

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_path)))))

mstar_dataset_root = os.path.join(project_root, "datasets/MSTAR")

# Input directory (raw CD structure)
RAW_DATA_DIR = os.path.join(mstar_dataset_root, "mstar_raw_data")
# Output directory (organized for training)
OUTPUT_DIR = os.path.join(mstar_dataset_root, "MSTAR_IMG_JSON")

# Fixed size for all output images (standard for MSTAR)
IMG_SIZE = (128, 128)

# --- DATASET ORGANIZATION DEFINITIONS ---

# Class name mapping from headers to standardized names
CLASS_MAP = {
    # Main classes
    '2S1': '2S1',
    '2s1_gun': '2S1',
    'BMP2': 'BMP2', 
    'bmp2_tank': 'BMP2',
    'BRDM2': 'BRDM2',
    'BRDM_2': 'BRDM2',
    'brdm_2': 'BRDM2',
    'brdm2_truck': 'BRDM2',
    'BTR60': 'BTR60',
    'BTR_60': 'BTR60',
    'btr_60': 'BTR60',
    'btr60_transport': 'BTR60',
    'BTR70': 'BTR70',
    'btr_70': 'BTR70',
    'btr70_transport': 'BTR70',    
    'D7': 'D7',
    'd7': 'D7',
    'd7_bulldozer': 'D7',
    'T62': 'T62',
    't_62': 'T62',
    't62_tank': 'T62',
    'T-72': 'T72',
    'T72': 'T72',
    't_72': 'T72',
    't72_tank': 'T72',
    'ZIL131': 'ZIL131',
    'zil_131': 'ZIL131',
    'zil131_truck': 'ZIL131',
    'ZSU234': 'ZSU234',
    'ZSU_23_4': 'ZSU234',
    'zsu_23_4': 'ZSU234',
    'zsu23-4_gun': 'ZSU234',
    'SLICY': 'SLICY',
    'slicy': 'SLICY',
    'slicey': 'SLICY',
}

# File extension mapping per class/serial (MSTAR files have multiple extensions per capture)
# We must use ONLY the specific extension for each class to avoid duplicates
FILE_EXTENSION_MAP = {
    ('2S1', 'b01'): ['000'],
    ('BMP2', '9563'): ['000'],
    ('BMP2', '9566'): ['001'],
    ('BMP2', 'c21'): ['002'],
    ('BRDM2', 'e71'): ['001'],
    ('BTR60', 'k10yt7532'): ['003'],
    ('BTR70', 'c71'): ['004'],
    ('D7', '92v13015'): ['005'],
    ('T62', 'a51'): ['016'],
    ('T62', 'a64'): ['024'],  # T62 A64 variant
    ('T72', '132'): ['015'],
    ('T72', '812'): ['016'],
    ('T72', 's7'): ['017'],
    ('T72', 'a04'): ['017'],
    ('T72', 'a05'): ['018'],
    ('T72', 'a07'): ['019'],
    ('T72', 'a10'): ['020'],
    ('T72', 'a32'): ['021'],  # Corrected from .017 to .021
    ('T72', 'a62'): ['022'],  # Corrected from .018 to .022
    ('T72', 'a63'): ['023'],  # Corrected from .019 to .023
    ('T72', 'a64'): ['024'],  # Uses .024 (not .020)
    ('ZIL131', 'e12'): ['025'],
    ('ZSU234', 'd08'): ['026'],
}

# Serial number definitions for dataset organization
# SOC Training serials (17 degrees) - 10 military vehicles
SOC_TRAIN_SERIALS = {
    '2S1': ['b01'],
    'BMP2': ['9563'],
    'BRDM2': ['e-71', 'e71'],
    'BTR60': ['k10yt7532'],
    'BTR70': ['c71'],
    'D7': ['92v13015'],
    'T62': ['a51'],
    'T72': ['132'],
    'ZIL131': ['e12'],
    'ZSU234': ['d08'],
}

# BMP2 and T72 variants for EOC-2 Version Variants
EOC2_VV_SERIALS = {
    'BMP2': ['9566', 'c21'],
    'T72': ['812', 'a04', 'a05', 'a07', 'a10'],  # S7 removed - it's only in CV
}

# T72 configuration variants for EOC-2 Configuration Variants
EOC2_CV_SERIALS = {
    'T72': ['s7', 'a32', 'a62', 'a63', 'a64'],
}

# EOC-1 specific serials (large depression angle change)
EOC1_SERIALS = {
    '2S1': ['b01'],
    'BRDM2': ['e-71', 'e71'],
    'T72': ['a64'],  # T72 variant A64 for EOC-1 
    'ZSU234': ['d08'],
}

# Outlier Rejection - Known targets (only 3 classes)
OUTLIER_KNOWN = {
    'BMP2': ['9563'],
    'BTR70': ['c71'],
    'T72': ['132'],
}

# Outlier Rejection - Confuser targets
OUTLIER_CONFUSER = {
    '2S1': ['b01'],
    'ZIL131': ['e12'],
}

# Mapping class_name -> class_id
target_name_soc = ('2S1', 'BMP2', 'BRDM2', 'BTR60', 'BTR70', 'D7', 'T62', 'T72', 'ZIL131', 'ZSU234')

target_name_eoc_1 = ('2S1', 'BRDM2', 'T72', 'ZSU234')

target_name_eoc_2 = ('BMP2', 'BRDM2', 'BTR70', 'T72')

target_name_confuser_rejection = ('BMP2', 'BTR70', 'T72', '2S1', 'ZIL131')

target_name_all = set(target_name_soc + target_name_eoc_1 + target_name_eoc_2 + target_name_confuser_rejection)

# serial_number = {
#     # 2S1
#     'b01': 0, 

#     # BMP2
#     '9563': 1,
#     '9566': 1,
#     'c21': 1,
    
#     # BRDM2
#     'E-71': 2,

#     # BTR60
#     'k10yt7532': 3,

#     # BTR70
#     'c71': 4,

#     # D7
#     '92v13015': 5,

#     # T62
#     'A51': 6,

#     # T72    
#     '132': 7,
#     '812': 7,
#     's7': 7,
#     'A04': 7,
#     'A05': 7,
#     'A07': 7,
#     'A10': 7,
#     'A32': 7,
#     'A62': 7,
#     'A63': 7,
#     'A64': 7,

#     # ZIL131
#     'E12': 8,

#     # ZSU234
#     'd08': 9
# }

# --- PROCESSING FUNCTIONS ---

def parse_mstar_file(file_path):
    """
    Parse a single MSTAR file.
    Extracts metadata from Phoenix header and magnitude data.
    """
    metadata = {}
    
    try:
        with open(file_path, 'rb') as f:
            # Read ASCII header (Phoenix Header)
            header_lines = []
            while True:
                line = f.readline()
                if not line:
                    return None, None

                # Header ends with this marker
                if b'EndofPhoenixHeader' in line:
                    break

                # Decode and ignore errors
                line_str = line.decode('ascii', errors='ignore').strip()
                header_lines.append(line_str)

                # Extract key-value pairs using regex
                match = re.match(r'([\w\/]+)\s*=\s*(.*)', line_str)
                if match:
                    key = match.group(1).strip()
                    value = match.group(2).strip()
                    metadata[key] = value

            # Extract dimensions
            rows = int(metadata.get('NumberOfRows', 0))
            cols = int(metadata.get('NumberOfColumns', 0))

            if rows == 0 or cols == 0:
                return None, None

            # Read Binary Data (Magnitude block)
            data_type = np.dtype('>f4')  # Big-endian float32
            magnitude_data = np.fromfile(f, dtype=data_type, count=rows * cols)
            
            if magnitude_data.size < rows * cols:
                return None, None

            magnitude_image = magnitude_data.reshape(rows, cols)
            return metadata, magnitude_image

    except Exception as e:
        return None, None

def process_image(mag_image, target_size=(128, 128)):
    """
    Apply logarithmic scaling (dB) and 8-bit normalization.
    """
    # Logarithmic scaling (Amplitude to Decibels)
    epsilon = 1e-6
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        image_db = 20 * np.log10(mag_image + epsilon)
    
    # 8-bit normalization using percentiles
    p1, p99 = np.percentile(image_db, [1, 99])
    image_db = np.clip(image_db, p1, p99)

    # Scale from [p1, p99] to [0, 255]
    if p99 > p1:
        image_8bit = (image_db - p1) / (p99 - p1)
        image_8bit = (image_8bit * 255).astype(np.uint8)
    else:
        image_8bit = np.zeros_like(image_db, dtype=np.uint8)

    # Resize to fixed size
    image_resized = cv2.resize(image_8bit, target_size, interpolation=cv2.INTER_AREA)
    
    return image_resized

def extract_metadata_from_path(file_path):
    """
    Extract metadata from file path when not available in header.
    """
    path_parts = Path(file_path).parts
    metadata = {'file_path': str(file_path)}
    
    # Extract depression angle from path
    for part in path_parts:
        if '_DEG' in part.upper():
            try:
                angle = int(part.split('_')[0])
                metadata['depression_angle'] = angle
            except:
                pass
    
    # Extract class from path
    parent_dir = os.path.basename(os.path.dirname(file_path))
    
    if parent_dir in CLASS_MAP:
        metadata['target_class'] = CLASS_MAP[parent_dir]
        metadata['serial_num'] = parent_dir.lower()
    else:
        for part in reversed(path_parts):
            if part in CLASS_MAP:
                metadata['target_class'] = CLASS_MAP[part]
                if part.startswith('SN_'):
                    metadata['serial_num'] = part[3:].lower()
                elif part in ['A04', 'A05', 'A07', 'A10', 'A32', 'A62', 'A63', 'A64']:
                    metadata['serial_num'] = part.lower()
                    metadata['target_class'] = 'T72'
                else:
                    metadata['serial_num'] = part.lower()
                break
    
    # Extract serial if SN_xxxx in path
    for part in path_parts:
        if part.startswith('SN_'):
            metadata['serial_num'] = part[3:].lower()
            break
    
    return metadata

def normalize_serial(serial):
    """
    Normalize serial number for comparison (lowercase, no hyphens).
    """
    return serial.lower().strip().replace('-', '')


class ErrorCounter:
    count = 0

def get_partition_soc(class_name, serial_num, depression):
    """
    Determine partition for SOC (Standard Operating Condition).
    
    Training: 17 degrees, 10 military classes with nominal serials
    Test: 15 degrees, same 10 classes
    """
    # Normalize serial
    serial_norm = normalize_serial(serial_num)
    
    # Check if class is in SOC (10 military vehicles, no SLICY)
    if class_name not in SOC_TRAIN_SERIALS:
        return None
    
    # Get nominal serials for this class
    nominal_serials = [normalize_serial(s) for s in SOC_TRAIN_SERIALS[class_name]]
    
    # Check for special cases (D7 starts with 92, etc.)
    is_nominal = serial_norm in nominal_serials
    if class_name == 'D7' and serial_norm.startswith('92'):
        is_nominal = True
    
    if not is_nominal:
        return None
    
    class_id = target_name_soc.index(class_name)
    
    # SOC Training: 17 degrees
    if depression == 17:
        return (f"SOC/train/{class_name}", class_id)
    
    # SOC Test: 15 degrees
    elif depression == 15:
        return (f"SOC/test/{class_name}", class_id)
    
    return None

def get_partition_eoc1(class_name, serial_num, depression):
    """
    Determine partition for EOC-1 (Large depression angle change).
    
    Training: 17 degrees (2S1, BRDM2, T72-A64, ZSU234)
    Test: 30 degrees (same 4 classes)
    """
    serial_norm = normalize_serial(serial_num)
    
    if class_name not in EOC1_SERIALS:
        return None
    
    # Get serials for this class
    target_serials = [normalize_serial(s) for s in EOC1_SERIALS[class_name]]
    is_match = serial_norm in target_serials
    
    if not is_match:
        return None
    
    # EOC-1 Training: 17 degrees
    if depression == 17:
        return (f"EOC-1/train/{class_name}", target_name_eoc_1.index(class_name))
    
    # EOC-1 Test: 30 degrees
    elif depression == 30:
        return (f"EOC-1/test/{class_name}", target_name_eoc_1.index(class_name))
    
    return None

def get_partition_eoc2_cv(class_name, serial_num, depression):
    """
    Determine partition for EOC-2 Configuration Variants.
    
    Training: 17 degrees (BMP2-9563, BRDM2-E71, BTR70-C71, T72-132)
    Test: 15 and 17 degrees (T72 variants: S7, A32, A62, A63, A64)
    """
    serial_norm = normalize_serial(serial_num)
    
    # Training uses SOC nominal serials at 17 degrees
    if depression == 17:
        if class_name in ['BMP2', 'BRDM2', 'BTR70', 'T72']:
            nominal_serials = [normalize_serial(s) for s in SOC_TRAIN_SERIALS[class_name]]
            if serial_norm in nominal_serials:
                return (f"EOC-2-CV/train/{class_name}", target_name_eoc_2.index(class_name))
    
    # Test uses T72 configuration variants at 15 and 17 degrees
    if class_name == 'T72' and depression in [15, 17]:
        cv_serials = [normalize_serial(s) for s in EOC2_CV_SERIALS['T72']]
        if serial_norm in cv_serials:
            return (f"EOC-2-CV/test/{class_name}", target_name_eoc_2.index(class_name))
    
    return None

def get_partition_eoc2_vv(class_name, serial_num, depression):
    """
    Determine partition for EOC-2 Version Variants.
    
    Training: 17 degrees (BMP2-9563, BRDM2-E71, BTR70-C71, T72-132)
    Test: 15 and 17 degrees (BMP2 variants: 9566, C21; T72 variants: 812, A04, A05, A07, A10)
    """
    serial_norm = normalize_serial(serial_num)
    
    # Training uses SOC nominal serials at 17 degrees
    if depression == 17:
        if class_name in ['BMP2', 'BRDM2', 'BTR70', 'T72']:
            nominal_serials = [normalize_serial(s) for s in SOC_TRAIN_SERIALS[class_name]]
            if serial_norm in nominal_serials:
                return (f"EOC-2-VV/train/{class_name}", target_name_eoc_2.index(class_name))
    
    # Test uses version variants at 15 and 17 degrees
    if depression in [15, 17]:
        if class_name in EOC2_VV_SERIALS:
            vv_serials = [normalize_serial(s) for s in EOC2_VV_SERIALS[class_name]]
            if serial_norm in vv_serials:
                return (f"EOC-2-VV/test/{class_name}", target_name_eoc_2.index(class_name))
    
    return None

def get_partition_outlier(class_name, serial_num, depression):
    """
    Determine partition for Outlier Rejection.
    
    Training: 17 degrees (Known: BMP2-9563, BTR70-C71, T72-132; Confuser: 2S1, ZIL131)
    Test: 15 degrees (Known + Confuser)
    """
    serial_norm = normalize_serial(serial_num)
    
    # Known targets
    if class_name in OUTLIER_KNOWN:
        known_serials = [normalize_serial(s) for s in OUTLIER_KNOWN[class_name]]
        if serial_norm in known_serials:
            if depression == 17:
                return (f"OUTLIER/train/known/{class_name}", target_name_confuser_rejection.index(class_name))
            elif depression == 15:
                return (f"OUTLIER/test/known/{class_name}", target_name_confuser_rejection.index(class_name))
    
    # Confuser targets
    if class_name in OUTLIER_CONFUSER:
        confuser_serials = [normalize_serial(s) for s in OUTLIER_CONFUSER[class_name]]
        if serial_norm in confuser_serials:
            if depression == 17:
                return (f"OUTLIER/train/confuser/{class_name}", target_name_confuser_rejection.index(class_name))
            elif depression == 15:
                return (f"OUTLIER/test/confuser/{class_name}", target_name_confuser_rejection.index(class_name))
    
    return None

def should_process_file(file_path, class_name, serial_num):
    """
    Check if this specific file should be processed based on its extension.
    Each class/serial combination should use only specific file extensions.
    """
    serial_norm = normalize_serial(serial_num)
    ext = os.path.splitext(file_path)[1]
    
    if not ext or ext == '':
        return False
    
    # Remove the dot from extension
    ext_num = ext[1:] if ext.startswith('.') else ext
    
    # Check if this class/serial combination has a specific extension mapping
    key = (class_name, serial_norm)
    if key in FILE_EXTENSION_MAP:
        allowed_extensions = FILE_EXTENSION_MAP[key]
        return ext_num in allowed_extensions
    
    # If no specific mapping, allow .000 files (default/most common)
    return ext_num == '000'

def get_all_partitions(metadata, path_metadata, file_path):
    """
    Get all applicable partitions for a single image.
    An image can belong to multiple datasets (SOC, EOC-1, EOC-2, Outlier).
    
    Returns: List of (partition_path, meta_data) tuples
    """
    partitions = []
    
    try:
        # Extract metadata
        target_class = metadata.get('TargetType') or path_metadata.get('target_class')
        serial_num = metadata.get('TargetSerNum') or path_metadata.get('serial_num', '')
        
        if not target_class or not serial_num:
            return []
        
        # Map to clean class name
        class_name = None
        for orig_name, clean_name in CLASS_MAP.items():
            if target_class.lower() == orig_name.lower():
                class_name = clean_name
                break
        
        if not class_name:
            return []
        
        # Check if we should process this specific file based on extension
        if not should_process_file(file_path, class_name, serial_num):
            return []
        
        # Get depression angle
        depression_str = metadata.get('DesiredDepression') or metadata.get('MeasuredDepression') or str(path_metadata.get('depression_angle', 0))
        depression = int(float(depression_str) + 0.5)
        
        # Create name suffix
        name_suffix = f"{normalize_serial(serial_num)}_{depression}deg"
        
        # Try each partition type
        partition_funcs = [
            get_partition_soc,
            get_partition_eoc1,
            get_partition_eoc2_cv,
            get_partition_eoc2_vv,
            get_partition_outlier
        ]
        
        for func in partition_funcs:
            partition = func(class_name, serial_num, depression) # partition = (path, class_id)
            if partition:
                metadata_json = {
                    "class_name": class_name,
                    "serial_number": serial_num,
                    "depression_angle": depression,
                    "class_id": partition[1]
                }
                partitions.append((partition[0], name_suffix, metadata_json))
        
        return partitions
        
    except Exception as e:
        print(f"Error in get_all_partitions: {e}")
        print(file_path)
        return []

def create_directory_structure():
    """
    Create the necessary directory structure for all datasets.
    """
    print("Creating directory structure...")
    
    # All unique classes
    all_classes = list(set(CLASS_MAP.values()))
    
    # SOC directories
    for split in ['train', 'test']:
        for class_name in SOC_TRAIN_SERIALS.keys():
            os.makedirs(os.path.join(OUTPUT_DIR, 'SOC', split, class_name), exist_ok=True)
    
    # EOC-1 directories
    for split in ['train', 'test']:
        for class_name in EOC1_SERIALS.keys():
            os.makedirs(os.path.join(OUTPUT_DIR, 'EOC-1', split, class_name), exist_ok=True)
    
    # EOC-2-CV directories
    for split in ['train', 'test']:
        for class_name in ['BMP2', 'BRDM2', 'BTR70', 'T72']:
            os.makedirs(os.path.join(OUTPUT_DIR, 'EOC-2-CV', split, class_name), exist_ok=True)
    
    # EOC-2-VV directories
    for split in ['train', 'test']:
        for class_name in ['BMP2', 'BRDM2', 'BTR70', 'T72']:
            os.makedirs(os.path.join(OUTPUT_DIR, 'EOC-2-VV', split, class_name), exist_ok=True)
    
    # Outlier directories
    for split in ['train', 'test']:
        for category in ['known', 'confuser']:
            for class_name in (OUTLIER_KNOWN.keys() if category == 'known' else OUTLIER_CONFUSER.keys()):
                os.makedirs(os.path.join(OUTPUT_DIR, 'OUTLIER', split, category, class_name), exist_ok=True)
    
    print("Directory structure created.")

def process_all_files(max_files=None):
    """
    Process all MSTAR files and organize into multiple dataset structures.
    max_files: maximum number of files to process (None = all)
    """
    print(f"=== MSTAR DATA PROCESSOR V2 - AConvNet Compatible ===")
    print(f"Input directory: {RAW_DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Clean and create structure
    if os.path.exists(OUTPUT_DIR):
        print(f"Cleaning existing output directory...")
        shutil.rmtree(OUTPUT_DIR)
    
    create_directory_structure()
    
    # Counters
    file_count = 0
    processed_count = 0
    total_assignments = 0
    skipped_count = 0
    error_count = 0
    
    # Statistics per dataset
    dataset_counts = {
        'SOC': {'train': {}, 'test': {}},
        'EOC-1': {'train': {}, 'test': {}},
        'EOC-2-CV': {'train': {}, 'test': {}},
        'EOC-2-VV': {'train': {}, 'test': {}},
        'OUTLIER': {'train': {'known': {}, 'confuser': {}}, 'test': {'known': {}, 'confuser': {}}}
    }
    
    start_time = time.time()
    
    print("Starting processing...")
    
    # Walk through all files
    for root, dirs, files in os.walk(RAW_DATA_DIR):
        # Skip SCENE3, SCENE2, etc. - only process SCENE1 and root-level data
        # The AConvNet repository uses only specific scenes
        if 'SCENE2' in root or 'SCENE3' in root or 'SCENE4' in root:
            continue
            
        for file in files:
            # Skip non-MSTAR files
            if file.upper().endswith(('.TXT', '.HTM', '.HTML', '.JPG', '.JPEG', '.PNG', '.GIF')):
                continue
            
            # Check if it's a potential MSTAR file
            ext = os.path.splitext(file)[1]
            is_mstar_file = False
            
            if ext == '':  # No extension
                is_mstar_file = True
            elif ext.startswith('.') and ext[1:].isdigit():  # Numeric extension
                is_mstar_file = True
            
            if not is_mstar_file:
                continue
                
            file_path = os.path.join(root, file)
            file_count += 1
            
            # Limit check
            if max_files and processed_count >= max_files:
                print(f"Limit reached: {max_files} files processed.")
                break
            
            try:
                # Parse file
                metadata, mag_image = parse_mstar_file(file_path)
                
                if metadata is None:
                    skipped_count += 1
                    continue
                
                # Extract path metadata
                path_metadata = extract_metadata_from_path(file_path)
                
                # Get all applicable partitions
                partitions = get_all_partitions(metadata, path_metadata, file_path)
                
                if partitions:
                    # Process image once
                    processed_img = process_image(mag_image, target_size=IMG_SIZE)
                    
                    # Save to all applicable partitions
                    for partition_path, name_suffix, metadata_json in partitions:
                        # Create unique img filename
                        base_name = os.path.splitext(os.path.basename(file))[0]
                        output_filename_png = f"{base_name}_{name_suffix}.png"
                        output_filepath_png = os.path.join(OUTPUT_DIR, partition_path, output_filename_png)
                        
                        # Save image
                        cv2.imwrite(output_filepath_png, processed_img)

                        # Create unique json filename
                        output_filename_json = f"{base_name}_{name_suffix}.json"
                        output_filepath_json = os.path.join(OUTPUT_DIR, partition_path, output_filename_json)

                        # Save JSON metadata
                        with open(output_filepath_json, mode='w', encoding='utf-8') as f:
                            json.dump(metadata_json, f, ensure_ascii=False, indent=2)

                        total_assignments += 1
                        
                        # Update statistics
                        parts = partition_path.split('/')
                        dataset = parts[0]
                        split = parts[1]
                        class_name = parts[-1]
                        
                        if dataset == 'OUTLIER':
                            category = parts[2]
                            if class_name not in dataset_counts[dataset][split][category]:
                                dataset_counts[dataset][split][category][class_name] = 0
                            dataset_counts[dataset][split][category][class_name] += 1
                        else:
                            if class_name not in dataset_counts[dataset][split]:
                                dataset_counts[dataset][split][class_name] = 0
                            dataset_counts[dataset][split][class_name] += 1
                    
                    processed_count += 1
                else:
                    skipped_count += 1
                
            except Exception as e:
                error_count += 1
                print(f"Error processing {file_path}: {e}")
            
            # Progress update every 100 files
            if file_count % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Progress: {file_count} files analyzed, {processed_count} processed ({total_assignments} total assignments), {skipped_count} skipped, {error_count} errors (Time: {elapsed:.1f}s)")
        
        # Break outer loop if limit reached
        if max_files and processed_count >= max_files:
            break
    
    # Display final statistics
    elapsed = time.time() - start_time
    print(f"\n=== PROCESSING COMPLETED ===")
    print(f"Total time: {elapsed:.1f} seconds")
    print(f"Files analyzed: {file_count}")
    print(f"Files successfully processed: {processed_count}")
    print(f"Total assignments (images saved): {total_assignments}")
    print(f"Files skipped: {skipped_count}")
    print(f"Errors: {error_count}")
    
    # Detailed statistics per dataset
    print(f"\n=== SOC STATISTICS ===")
    print(f"Training (17 deg):")
    for class_name, count in sorted(dataset_counts['SOC']['train'].items()):
        print(f"  {class_name}: {count} images")
    print(f"Test (15 deg):")
    for class_name, count in sorted(dataset_counts['SOC']['test'].items()):
        print(f"  {class_name}: {count} images")
    
    print(f"\n=== EOC-1 STATISTICS ===")
    print(f"Training (17 deg):")
    for class_name, count in sorted(dataset_counts['EOC-1']['train'].items()):
        print(f"  {class_name}: {count} images")
    print(f"Test (30 deg):")
    for class_name, count in sorted(dataset_counts['EOC-1']['test'].items()):
        print(f"  {class_name}: {count} images")
    
    print(f"\n=== EOC-2 Configuration Variants STATISTICS ===")
    print(f"Training (17 deg):")
    for class_name, count in sorted(dataset_counts['EOC-2-CV']['train'].items()):
        print(f"  {class_name}: {count} images")
    print(f"Test (15 & 17 deg):")
    for class_name, count in sorted(dataset_counts['EOC-2-CV']['test'].items()):
        print(f"  {class_name}: {count} images")
    
    print(f"\n=== EOC-2 Version Variants STATISTICS ===")
    print(f"Training (17 deg):")
    for class_name, count in sorted(dataset_counts['EOC-2-VV']['train'].items()):
        print(f"  {class_name}: {count} images")
    print(f"Test (15 & 17 deg):")
    for class_name, count in sorted(dataset_counts['EOC-2-VV']['test'].items()):
        print(f"  {class_name}: {count} images")
    
    print(f"\n=== OUTLIER REJECTION STATISTICS ===")
    print(f"Training (17 deg) - Known:")
    for class_name, count in sorted(dataset_counts['OUTLIER']['train']['known'].items()):
        print(f"  {class_name}: {count} images")
    print(f"Training (17 deg) - Confuser:")
    for class_name, count in sorted(dataset_counts['OUTLIER']['train']['confuser'].items()):
        print(f"  {class_name}: {count} images")
    print(f"Test (15 deg) - Known:")
    for class_name, count in sorted(dataset_counts['OUTLIER']['test']['known'].items()):
        print(f"  {class_name}: {count} images")
    print(f"Test (15 deg) - Confuser:")
    for class_name, count in sorted(dataset_counts['OUTLIER']['test']['confuser'].items()):
        print(f"  {class_name}: {count} images")
    
    return processed_count > 0

def test_single_file():
    """
    Test function to process a single file and verify it works.
    """
    print("=== TEST MODE: Processing a single file ===")
    
    # Find first available .000 file
    test_file = None
    for root, dirs, files in os.walk(RAW_DATA_DIR):
        for file in files:
            if file.endswith('.000'):
                test_file = os.path.join(root, file)
                break
        if test_file:
            break
    
    if not test_file:
        print("No .000 file found for testing.")
        return False
    
    print(f"Test file: {test_file}")
    
    # Create directory structure
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    create_directory_structure()
    
    # Process the file
    metadata, mag_image = parse_mstar_file(test_file)
    
    if metadata is None:
        print("Error: Could not parse test file.")
        return False
    
    print("File parsed successfully")
    print("Main metadata:")
    for key in ['TargetType', 'TargetSerNum', 'DesiredDepression', 'NumberOfRows', 'NumberOfColumns']:
        if key in metadata:
            print(f"  {key}: {metadata[key]}")
    
    # Extract path metadata
    path_metadata = extract_metadata_from_path(test_file)
    print("\nPath metadata:")
    for key, value in path_metadata.items():
        print(f"  {key}: {value}")
    
    # Get all partitions
    partitions = get_all_partitions(metadata, path_metadata, test_file)
    
    if partitions:
        print(f"\nAssigned to {len(partitions)} partition(s):")
        for partition_path, name_suffix in partitions:
            print(f"  - {partition_path}")
        
        # Process image
        processed_img = process_image(mag_image, target_size=IMG_SIZE)
        print(f"Image processed: shape {processed_img.shape}, type {processed_img.dtype}")
        
        # Save to all partitions
        for partition_path, name_suffix in partitions:
            output_filename = f"{os.path.basename(test_file)}_{name_suffix}.png"
            output_filepath = os.path.join(OUTPUT_DIR, partition_path, output_filename)
            
            cv2.imwrite(output_filepath, processed_img)
            print(f"Image saved to: {output_filepath}")
        
        return True
    else:
        print("Could not determine any partition for this file.")
        return False
    
def create_mixed_dataset(train_proportion = 0.7):
    """
    Go through all data created and add to MIXED dataset
    """

    MIXED_DIR = os.path.join(OUTPUT_DIR, 'MIXED')

    if os.path.exists(MIXED_DIR):
        print(f"Cleaning existing MIXED data directory...")
        shutil.rmtree(MIXED_DIR)

    # Create MIXED/test and MIXED/train/ folders
    for split in ['train', 'test']:
        for class_name in SOC_TRAIN_SERIALS.keys():
            os.makedirs(os.path.join(MIXED_DIR, split, class_name), exist_ok=True)

    for root, dirs, files in os.walk(OUTPUT_DIR):

        if 'OUTLIER' in root or 'MIXED' in root:
            continue

        # if we are in a folder containing files
        if files:

            class_name = os.path.basename(root)
            split = os.path.basename(os.path.dirname(root))
            dataset = os.path.basename(os.path.dirname(os.path.dirname(root)))
            print(f'Processing {dataset}/{split}/{class_name} ...')

            # read files in current folder
            image_list = glob.glob(os.path.join(root, "*.png"))
            label_list = glob.glob(os.path.join(root, "*.json"))

            image_list.sort()
            label_list.sort()

            # check lengths
            assert len(image_list) == len(label_list), "Mismatch between images and labels"

            # Create (image, label) pairs
            pairs = list(zip(image_list, label_list))

            # Calculate train_size
            train_size = int(len(pairs) * train_proportion)

            # Randomly select
            random.seed(42)
            random.shuffle(pairs)  
            train_pairs = pairs[:train_size]
            test_pairs = pairs[train_size:]

            train_count = 0
            test_count = 0

            # Copy train files
            for img_path, lbl_path in train_pairs:
                img_filename = os.path.basename(img_path)
                lbl_filename = os.path.basename(lbl_path)
                
                dest_img = os.path.join(MIXED_DIR, 'train', class_name, img_filename)
                dest_lbl = os.path.join(MIXED_DIR, 'train', class_name, lbl_filename)
                
                shutil.copy(img_path, dest_img)
                shutil.copy(lbl_path, dest_lbl)
                train_count += 1
            
            print(f'Loaded {train_count} images in train')
            
            # Copy test files
            for img_path, lbl_path in test_pairs:
                img_filename = os.path.basename(img_path)
                lbl_filename = os.path.basename(lbl_path)
                
                dest_img = os.path.join(MIXED_DIR, 'test', class_name, img_filename)
                dest_lbl = os.path.join(MIXED_DIR, 'test', class_name, lbl_filename)
                
                shutil.copy(img_path, dest_img)
                shutil.copy(lbl_path, dest_lbl)

                test_count += 1
            
            print(f'Loaded {test_count} images in test')

    
    # Print statistics
    for split in ['train', 'test']:
        split_path = os.path.join(MIXED_DIR, split)
        print(f"\n{split.upper()} set:")
        for class_name in os.listdir(split_path):
            class_path = os.path.join(split_path, class_name)
            if os.path.isdir(class_path):
                num_images = len(glob.glob(os.path.join(class_path, "*.png")))
                num_labels = len(glob.glob(os.path.join(class_path, "*.json")))
                print(f"  {class_name}: {num_images} images and {num_labels} labels")


def main():
    """
    Main script.
    """
    print("=== MSTAR DATA PROCESSOR ===")
    # print("\nThis version organizes data for the AConvNet-pytorch repository:")
    print("  - SOC: Standard Operating Condition (10 classes)")
    print("  - EOC-1: Extended Operating Condition (large depression angle change)")
    print("  - EOC-2-CV: Configuration Variants")
    print("  - EOC-2-VV: Version Variants")
    print("  - OUTLIER: Outlier Rejection (known vs confuser)")
    print("\nSelect an option:")
    print("1. Run test with a single file")
    print("2. Process the first 50 files")
    print("3. Process the first 500 files")
    print("4. Process ALL files")
    
    # choice = input("Enter your option (1-4): ").strip()
    
    # if choice == '1':
    #     success = test_single_file()
    #     if success:
    #         print("\nTest successful")
    #     else:
    #         print("\nTest failed.")
    
    # elif choice == '2':
    #     print("\nProcessing the first 50 files...")
    #     success = process_all_files(max_files=50)
    #     if success:
    #         print("\nProcessing successful")
        
    # elif choice == '3':
    #     print("\nProcessing the first 500 files...")
    #     success = process_all_files(max_files=500)
    #     if success:
    #         print("\nProcessing successful")
    
    # elif choice == '4':
    #     confirm = input("Are you sure you want to process ALL files? (y/N): ").strip().lower()
    #     if confirm == 'y':
    #         print("\nProcessing ALL files...")
    #         success = process_all_files()
    #         if success:
    #             print("\nComplete processing successful")
    #     else:
    #         print("Operation cancelled.")
    
    # else:
    #     print("Invalid option.")

    create_mixed_dataset()

if __name__ == "__main__":
    main()
