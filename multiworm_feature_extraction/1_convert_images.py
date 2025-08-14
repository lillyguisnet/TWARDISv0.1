"""
This script converts TIFF images to JPEG images to input into the SAM model.
"""

import os
from PIL import Image
import numpy as np
import tifffile

src_dir = 'YOUR_DATA_DIR_PATH'
dst_dir = 'YOUR_DST_DIR_PATH'

def convert_16bit_to_8bit(image):
    """Convert a 16-bit image to 8-bit using tifffile and proper scaling"""
    img_array = tifffile.imread(image)
    
    global_min = img_array.min()
    global_max = img_array.max()
    
    img_adjusted = img_array - global_min
    scaling_factor = 255.0 / (global_max - global_min)
    img_8bit = (img_adjusted * scaling_factor).astype(np.uint8)
    
    return Image.fromarray(img_8bit)


os.makedirs(dst_dir, exist_ok=True)


for root, dirs, files in os.walk(src_dir):
    rel_path = os.path.relpath(root, src_dir)
    
    dst_path = os.path.join(dst_dir, rel_path)
    os.makedirs(dst_path, exist_ok=True)
    
    for file in files:
        if file.lower().endswith('.tif'):
            src_file = os.path.join(root, file)
            dst_file = os.path.join(dst_path, os.path.splitext(file)[0] + '.jpg')
            
            # Open and convert the image
            try:
                # Use tifffile for reading and our conversion function
                img = convert_16bit_to_8bit(src_file)
                
                # Save as JPEG
                img.save(dst_file, 'JPEG', quality=95)
                print(f"Converted: {src_file} -> {dst_file}")
            except Exception as e:
                print(f"Error converting {src_file}: {str(e)}")
