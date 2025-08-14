"""
This script uses the SAM model to segment the entire image.
The resulting cutouts are classified as either 'worm_any' or 'not worm' by a fine-tuned classifier. "Worm_any" also includes partial worms.
Metrics are extracted from the final worms and saved to a CSV file.
"""

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import skimage
from skimage.measure import label
from scipy.ndimage import convolve
import glob
import os
import pickle
from skimage.measure import label
from scipy.ndimage import convolve
import shutil
sys.path.append("PATH_TO_CLONED_SAM2_REPO/segment-anything-2")
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

#Setup the SAM model
checkpoint = "./segment-anything-2/checkpoints/sam2_hiera_large.pt" #Checkpoint for the SAM model
model_cfg = "sam2_hiera_l.yaml" #Configuration file for the SAM model
sam2 = build_sam2(model_cfg, checkpoint, device ='cuda', apply_postprocessing=False)
mask_generator = SAM2AutomaticMaskGenerator(sam2)
mask_generator_2 = SAM2AutomaticMaskGenerator(
    model=sam2,
    pred_iou_thresh=0.85,
    stability_score_thresh=0.85,
    stability_score_offset=0.85
)

#Setup the worm classifier
classifdevice = torch.device("cuda:0")
classif_weights = torchvision.models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1
worm_noworm_classif_model = torchvision.models.vit_h_14(weights=classif_weights)
num_ftrs = worm_noworm_classif_model.heads.head.in_features
worm_noworm_classif_model.heads.head = nn.Linear(num_ftrs, 2)
worm_noworm_classif_model = worm_noworm_classif_model.to(classifdevice)
worm_noworm_classif_model.load_state_dict(torch.load('PATH_TO_WORM_NOWORM_CLASSIFIER_WITH_PERFECT_WEIGHTS.pth', map_location=classifdevice)) #https://huggingface.co/lillyguisnet/celegans-classifier-vit-h-14-finetuned

worm_noworm_classif_model.eval()
class_names = ["notworm", "worm_any"]
data_transforms = {
    'val': transforms.Compose([
        transforms.Resize(518),
        transforms.CenterCrop(518),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            contours, _ = cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0,0,1,0.4), thickness=1) 

    ax.imshow(img)

def is_on_edge(x, y, w, h, img_width, img_height):
    # Check left edge
    if x <= 0:
        return True
    # Check top edge
    if y <= 0:
        return True
    # Check right edge
    if (x + w) >= img_width - 1:
        return True
    # Check bottom edge
    if (y + h) >= img_height - 1:
        return True
    return False

def get_valid_imaging_area(image, margin=5, max_iterations=100):
    """
    Find the actual microscope field of view in the image.
    """
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Warning: No valid imaging area found")
        return np.ones_like(gray, dtype=bool), False
    
    # Find the largest contour that's not the entire image
    valid_contours = [cnt for cnt in contours 
                     if 0.1 < cv2.contourArea(cnt) / (gray.shape[0] * gray.shape[1]) < 0.99]
    
    if not valid_contours:
        print("Warning: No valid contours found within acceptable size range")
        return np.ones_like(gray, dtype=bool), False
    
    largest_contour = max(valid_contours, key=cv2.contourArea)
    
    # Create mask of valid area
    valid_area_mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(valid_area_mask, [largest_contour], -1, 255, -1)
    
    # Erode the mask by margin pixels with iteration limit
    if margin > 0:
        kernel = np.ones((3, 3), np.uint8)  # Using smaller kernel for more controlled erosion
        eroded_mask = valid_area_mask.copy()
        for _ in range(min(margin, max_iterations)):
            temp_mask = cv2.erode(eroded_mask, kernel)
            if np.sum(temp_mask) < 1000:
                break
            eroded_mask = temp_mask
        valid_area_mask = eroded_mask
    
    return valid_area_mask > 0, True

def get_nonedge_masks(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    img_height, img_width = image.shape[:2]

    # Generate masks
    masks2 = mask_generator_2.generate(image)

    valid_area, success = get_valid_imaging_area(image)
    
    nonedge_masks = []

    if success:
        # Use valid area method
        for mask in masks2:
            segmentation = mask['segmentation']
            if np.all(segmentation * valid_area == segmentation):
                nonedge_masks.append(segmentation)
    else:
        # Fall back to simple edge detection
        print(f"Falling back to simple edge detection for {img_path}")
        for mask in masks2:
            segmentation = mask['segmentation']
            coords = np.where(segmentation)
            y1, x1 = np.min(coords[0]), np.min(coords[1])
            y2, x2 = np.max(coords[0]), np.max(coords[1])
            h, w = (y2 - y1 + 1), (x2 - x1 + 1)
            
            if not is_on_edge(x1, y1, w, h, img_width, img_height):
                nonedge_masks.append(segmentation)

    return image, img_height, img_width, nonedge_masks


def save_mask_cutouts(image, nonedge_masks, output_dir='PATH_TO_TEMP_CUTOUTS_DIR'):
    """
    Save cutouts of the masks from the image to the specified directory.
    Refreshes the output directory each time.
    """
    # Refresh temp directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving {len(nonedge_masks)} non-edge cutouts to")
    
    for i, mask in enumerate(nonedge_masks):
        # Get bounding box coordinates of the mask
        coords = np.where(mask)
        y1, x1 = np.min(coords[0]), np.min(coords[1])
        y2, x2 = np.max(coords[0]), np.max(coords[1])
        
        # Create a 3D mask by repeating the 2D mask for each color channel
        mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        
        # Apply mask to original image
        cutout = image * mask_3d
        
        # Crop to bounding box
        cutout = cutout[y1:y2+1, x1:x2+1]
        
        # Save the cutout as jpg
        cutout_path = os.path.join(output_dir, f'{i}.jpg')
        cv2.imwrite(cutout_path, cv2.cvtColor(cutout, cv2.COLOR_RGB2BGR))

def classify_cutouts(nonedge_masks, cutouts_dir='PATH_TO_TEMP_CUTOUTS_DIR'):
    """
    Classify each cutout image as either 'worm' or 'not worm' using the pre-trained classifier.
    """
    classifications = []
    for i in range(len(nonedge_masks)):
        cutout_path = os.path.join(cutouts_dir, f'{i}.jpg')
        imgg = Image.open(cutout_path)
        imgg = data_transforms['val'](imgg)
        imgg = imgg.unsqueeze(0)
        imgg = imgg.to(classifdevice)
        
        outputs = worm_noworm_classif_model(imgg)
        _, preds = torch.max(outputs, 1)
        classifications.append(class_names[preds])
    
    return classifications


def merge_and_clean_worm_masks(classifications, nonedge_masks, overlap_threshold=0.95, min_area=25):
    """
    Merge overlapping worm masks and clean the results by removing small regions and keeping only the largest connected component.
    Also checks for and removes masks with holes.
    """
    worm_masks = []
    for i, classification in enumerate(classifications):
        if classification == "worm_any":
            worm_masks.append(nonedge_masks[i])

    if worm_masks:
        # Initialize list to track which masks have been merged
        merged_masks = []
        final_masks = []
        
        # Compare each mask with every other mask
        for i in range(len(worm_masks)):
            if i in merged_masks:
                continue
                
            current_mask = worm_masks[i]
            current_area = np.sum(current_mask)
            merged = False
            
            for j in range(i + 1, len(worm_masks)):
                if j in merged_masks:
                    continue
                    
                other_mask = worm_masks[j]
                # Calculate overlap
                overlap = np.sum(current_mask & other_mask)
                overlap_ratio = overlap / min(current_area, np.sum(other_mask))
                
                # If overlap is more than threshold, merge the masks
                if overlap_ratio > overlap_threshold:
                    current_mask = current_mask | other_mask
                    current_area = np.sum(current_mask)
                    merged_masks.append(j)
                    merged = True
            
            final_masks.append(current_mask)
        
        # Clean final masks - remove regions smaller than min_area pixels and handle discontinuous segments
        worm_masks = []
        for i, mask in enumerate(final_masks):
            if np.sum(mask) >= min_area:
                # Check for holes using contour hierarchy
                contours, hierarchy = cv2.findContours((mask * 255).astype(np.uint8), 
                                                     cv2.RETR_TREE, 
                                                     cv2.CHAIN_APPROX_SIMPLE)
                
                has_holes = False
                if hierarchy is not None:
                    hierarchy = hierarchy[0]  # Get the first dimension
                    for h in hierarchy:
                        if h[3] >= 0:  # If has parent, it's a hole
                            has_holes = True
                            print(f"Skipping mask {i} due to holes in the mask")
                            break
                
                if not has_holes:
                    # Find connected components in the mask
                    num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
                    
                    if num_labels > 2:  # More than one segment (label 0 is background)
                        # Get sizes of each segment
                        unique_labels, label_counts = np.unique(labels[labels != 0], return_counts=True)
                        # Keep only the largest segment
                        largest_label = unique_labels[np.argmax(label_counts)]
                        mask = (labels == largest_label).astype(np.uint8)
                    
                    worm_masks.append(mask)
        
        num_distinct_worms = len(worm_masks)
        print(f"Number of distinct worm regions: {num_distinct_worms}")
    else:
        num_distinct_worms = 0

    return worm_masks, num_distinct_worms


def filter_worms(allworms_metrics, threshold):
    filtered_metrics = []
    for worm in allworms_metrics:
        if worm['area'] > threshold * np.mean([worm['area'] for worm in allworms_metrics]):
            filtered_metrics.append(worm)
    return filtered_metrics

def extract_worm_metrics(worm_masks, img_path, img_height, img_width, threshold=0.75):
    """
    Extract metrics for each worm mask including area, perimeter, medial axis measurements, etc.
    """
    # Get image ID (filename without extension)
    img_id = os.path.splitext(os.path.basename(img_path))[0]
    
    allworms_metrics = []
    for i, npmask in enumerate(worm_masks):
        print(f"Processing worm {i}")

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((npmask * 255).astype(np.uint8), connectivity=8)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        largest_component_mask = (labels == largest_label).astype(np.uint8)

        area = np.sum(largest_component_mask)

        contours, hierarchy = cv2.findContours((largest_component_mask*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 

        perimeter = cv2.arcLength(contours[0], True)

        # Get medial axis and distance transform
        medial_axis, distance = skimage.morphology.medial_axis(largest_component_mask > 0, return_distance=True)   
        structuring_element = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
        neighbours = convolve(medial_axis.astype(np.uint8), structuring_element, mode='constant', cval=0)
        end_points = np.where(neighbours == 11, 1, 0)
        branch_points = np.where(neighbours > 12, 1, 0)
        labeled_branches = label(branch_points, connectivity=2)
        branch_indices = np.argwhere(labeled_branches > 0)
        end_indices = np.argwhere(end_points > 0)
        indices = np.concatenate((branch_indices, end_indices), axis=0)
        
        # Find longest path through medial axis
        paths = []
        for start in range(len(indices)):
            for end in range(len(indices)):
                startid = tuple(indices[start])
                endid = tuple(indices[end])
                route, weight = skimage.graph.route_through_array(np.invert(medial_axis), startid, endid)
                length = len(route)
                paths.append([startid, endid, length, route, weight])
        
        longest_length = max(paths, key=lambda x: x[2])
        pruned_mediala = np.zeros((img_height, img_width), dtype=np.uint8)
        for coord in range(len(longest_length[3])):
            pruned_mediala[longest_length[3][coord]] = 1
            
        # Get measurements along medial axis
        medial_axis_distances_sorted = [distance[pt[0], pt[1]] for pt in longest_length[3]]
        medialaxis_length_list = 0 + np.arange(0, len(medial_axis_distances_sorted))
        pruned_medialaxis_length = np.sum(pruned_mediala)
        mean_wormwidth = np.mean(medial_axis_distances_sorted)
        mid_length = medial_axis_distances_sorted[int(len(medial_axis_distances_sorted)/2)]

        worm_metrics = {
            "img_id": img_id, 
            "worm_id": i,
            "area": area, 
            "perimeter": perimeter, 
            "medial_axis_distances_sorted": medial_axis_distances_sorted, 
            "medialaxis_length_list": np.ndarray.tolist(medialaxis_length_list), 
            "pruned_medialaxis_length": pruned_medialaxis_length, 
            "mean_wormwidth": mean_wormwidth, 
            "mid_length_width": mid_length,
            "mask": largest_component_mask
        }
        allworms_metrics.append(worm_metrics)
    
    return filter_worms(allworms_metrics, threshold = threshold)


def save_worms(allworms_metrics, original_image=None, cutouts_dir='PATH_TO_FINAL_CUTOUTS_DIR', 
               metrics_dir='PATH_TO_FINAL_METRICS_DIR'):
    """
    Filter worms by area and save the filtered cutouts and metrics.
    """
    if not allworms_metrics:
        print("No worm metrics provided")
        return []
          
    img_id = allworms_metrics[0]["img_id"]
    
    # Save cutouts of filtered worms for visualization
    for i, worm in enumerate(allworms_metrics):
        cutout_path = os.path.join(cutouts_dir, f'{img_id}_worm_{i}.png')
        cutout_name = f'{img_id}_worm_{i}'  # The name that will appear on the image
        
        if original_image is not None:
            overlay = original_image.copy()
            overlay[worm["mask"] > 0] = [0, 255, 0]  # Green color
            
            alpha = 0.4  # Back to original 40% transparency
            blended = cv2.addWeighted(original_image, 1 - alpha, overlay, alpha, 0)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(blended, cutout_name, (10, 30), font, 1, (255, 255, 255), 2)
            
            cv2.imwrite(cutout_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
        else:
            # Fall back to saving just the mask if original image not provided
            cv2.imwrite(cutout_path, (worm["mask"] * 255).astype(np.uint8))
    
    # Save metrics as pickle using img_id as filename
    metrics_path = os.path.join(metrics_dir, f'{img_id}.pkl')
    with open(metrics_path, 'wb') as f:
        pickle.dump(allworms_metrics, f)
    print(f"Saved filtered metrics to {metrics_path}")
    
    return allworms_metrics



def process_folder(input_folder, temp_cutouts_dir='PATH_TO_TEMP_CUTOUTS_DIR', 
                  final_cutouts_dir='PATH_TO_FINAL_CUTOUTS_DIR',
                  metrics_dir='PATH_TO_FINAL_METRICS_DIR',
                  noworms_file='PATH_TO_NOWORMS_FILE.csv'):
    """
    Process all images in a folder through the complete worm analysis pipeline.
    """
    
    # Create/check noworms CSV file
    if not os.path.exists(noworms_file):
        with open(noworms_file, 'w') as f:
            f.write('image_path\n')

    # Create/check final cutouts directory
    if not os.path.exists(final_cutouts_dir):
        os.makedirs(final_cutouts_dir)
    
    # Create/check metrics directory
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)
    
    # Get list of image files
    image_files = glob.glob(os.path.join(input_folder, '*.jpg'))
    
    # Process each image
    for img_path in image_files:
        print(f"\nProcessing {img_path}")
        try:
            # Extract masks
            image, img_height, img_width, nonedge_masks = get_nonedge_masks(img_path)
            if len(nonedge_masks) == 0:
                print(f"No valid masks found in {img_path}")
                print(f"Number of worms in image: 0")
                with open(noworms_file, 'a') as f:
                    f.write(f'{img_path}\n')
                continue
                
            save_mask_cutouts(image, nonedge_masks, temp_cutouts_dir)
            
            classifications = classify_cutouts(nonedge_masks, temp_cutouts_dir)
            
            worm_masks, num_distinct_worms = merge_and_clean_worm_masks(
                classifications, nonedge_masks, temp_cutouts_dir)
            if num_distinct_worms == 0:
                print(f"No worms detected in {img_path}")
                print(f"Number of worms in image: 0")
                with open(noworms_file, 'a') as f:
                    f.write(f'{img_path}\n')
                continue
                
            worm_metrics = extract_worm_metrics(worm_masks, img_path, img_height, img_width)
            
            print(f"Final number of worms in image: {len(worm_metrics)}")
            
            save_worms(worm_metrics, original_image=image, cutouts_dir=final_cutouts_dir, metrics_dir=metrics_dir)
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            print(f"Number of worms in image: 0")
            continue
    
    print(f"\nAnalysis complete. Processed {len(image_files)} images.")
    print(f"Results saved to {metrics_dir}")



input_folder = 'PATH_TO_INPUT_FOLDER'

process_folder(input_folder)