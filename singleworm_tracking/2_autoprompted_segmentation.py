"""
This script is used for high-definition segmentation of a worm crawling video.
It firts performs full frame segmentation of the video using a generic prompt frame pool.
Then, it performs a second pass of segmentation on the cropped prompt frames for high-definition segmentation.
There are several visualization helper functions and it also allows for the addition of prompts to the prompt pool and the adjustment of prompts in the prompt pool.
"""
import os
import sys
import json
import shutil
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import time
import cv2
from scipy.ndimage import binary_dilation
import tqdm
sys.path.append("PATH_TO_CLONED_REPO/segment-anything-2")
from sam2.build_sam import build_sam2_video_predictor
import random
import multiprocessing
import pickle
import copy
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

sam2_checkpoint = "./segment-anything-2/checkpoints/sam2_hiera_large.pt" #Checkpoint for the SAM model
model_cfg = "sam2_hiera_l.yaml" #Configuration file for the SAM model
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

#region [functions]

def _process_frame_for_video(args):
    """Helper function to process frames for video creation."""
    frame_idx, frame_name, video_dir, masks_for_frame, colors, alpha, new_size = args
    
    image_path = os.path.join(video_dir, frame_name)
    
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if new_size != (image.shape[1], image.shape[0]):
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

        if masks_for_frame:
            overlay = np.zeros_like(image)
            
            for mask_id, mask in masks_for_frame.items():
                if mask_id is None: 
                    continue
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy()
                
                if mask.dtype != bool:
                    mask = mask > 0.5
                
                if mask.ndim > 2:
                    mask = mask.squeeze()
                
                mask_resized = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                
                color = colors.get(mask_id)
                if color is None:
                    continue

                colored_mask = np.zeros_like(image)
                colored_mask[mask_resized == 1] = color
                
                overlay = cv2.addWeighted(overlay, 1, colored_mask, 1, 0)
            
            overlaid_image = cv2.addWeighted(image, 1, overlay, alpha, 0)
        else:
            overlaid_image = image

        # Return in BGR format for cv2.VideoWriter
        return cv2.cvtColor(overlaid_image, cv2.COLOR_RGB2BGR)

    except Exception as e:
        print(f"Error processing frame {frame_idx}: {str(e)}")
        original_frame = cv2.imread(image_path)
        if original_frame is not None:
            if new_size != (original_frame.shape[1], original_frame.shape[0]):
                original_frame = cv2.resize(original_frame, new_size, interpolation=cv2.INTER_AREA)
            return original_frame
        
        print(f"Could not read original frame {frame_idx}, returning blank frame.")
        return np.zeros((new_size[1], new_size[0], 3), dtype=np.uint8)

def show_mask(mask, ax, obj_id=None, random_color=False):
    color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0) if random_color else \
            np.array([*plt.get_cmap("tab10")(0 if obj_id is None else obj_id)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=26):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def add_prompt_frames_to_video(video_dir, prompt_dir):
    existing_frames = [f for f in os.listdir(video_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
    last_frame_num = max(int(os.path.splitext(f)[0]) for f in existing_frames)

    prompt_frames = sorted([f for f in os.listdir(prompt_dir) if f.lower().endswith(('.jpg', '.jpeg'))],
                           key=lambda x: int(os.path.splitext(x)[0]))

    frame_mapping = {}
    for i, prompt_frame in enumerate(prompt_frames, start=1):
        new_frame_num = last_frame_num + i
        new_frame_name = f"{new_frame_num:06d}.jpg"
        shutil.copy(os.path.join(prompt_dir, prompt_frame), os.path.join(video_dir, new_frame_name))
        frame_mapping[new_frame_num] = int(os.path.splitext(prompt_frame)[0])  # Store original frame number

    final_frame_count = last_frame_num + len(prompt_frames) + 1
    print(f"Added {len(prompt_frames)} prompt frames to the video directory. There are now {final_frame_count} frames in the video directory.")
    return frame_mapping

def remove_prompt_frames_from_video(video_dir, frame_mapping):
    for frame_num in frame_mapping.keys():
        frame_name = f"{frame_num:06d}.jpg"
        frame_path = os.path.join(video_dir, frame_name)
        if os.path.exists(frame_path):
            os.remove(frame_path)
    
    print(f"Removed {len(frame_mapping)} prompt frames from the video directory.")

def filter_prompt_frames_from_segments(video_segments, frame_mapping):
    """Remove prompt frame predictions from the video_segments dictionary."""
    filtered_segments = {
        frame: segments for frame, segments in video_segments.items()
        if frame not in frame_mapping
    }
    print(f"Removed {len(frame_mapping)} prompt frames from video_segments dictionary.")
    return filtered_segments

def add_prompts(inference_state, frame_idx, obj_id, points, labels):
    _, out_obj_ids, out_mask_logits = predictor.add_new_points(
        inference_state=inference_state,
        frame_idx=frame_idx,
        obj_id=obj_id,
        points=points,
        labels=labels
    )
    
    #Prompt frame visualization if needed
    plt.figure(figsize=(12, 8))
    plt.title(f"Frame {frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, f"{frame_idx:06d}.jpg")))
    show_points(points, labels, plt.gca())    
    for i, out_obj_id in enumerate(out_obj_ids):
        show_points(points, labels, plt.gca())
        show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)    
    plt.savefig(f"prompt_frame.png")
    plt.close()
    
def check_overlap(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    overlap_pixels = np.sum(intersection)
    iou = overlap_pixels / np.sum(union) if np.sum(union) > 0 else 0
    return overlap_pixels > 0, iou, overlap_pixels

def check_distance(mask1, mask2, max_distance=10):
    if mask1.sum() == 0 or mask2.sum() == 0:
        return True
    
    dilated_mask1 = binary_dilation(mask1, iterations=max_distance)
    dilated_mask2 = binary_dilation(mask2, iterations=max_distance)
    
    return np.any(np.logical_and(dilated_mask1, dilated_mask2))

def analyze_masks(video_segments):
    """Error flagging: empty frames, frames with small and large predictions, overlapping masks, distant masks, low consecutive overlap."""
    results = {'empty': {}, 'high': {}, 'overlapping': {}, 'distant': {}, 'low_consecutive_overlap': {}}
    max_counts = {'empty': 0, 'high': 0, 'overlapping': 0, 'distant': 0, 'low_consecutive_overlap': 0}
    max_frames = {'empty': None, 'high': None, 'overlapping': None, 'distant': None, 'low_consecutive_overlap': None}

    for frame, mask_dict in video_segments.items():
        mask_ids = [mask_id for mask_id in mask_dict.keys() if mask_id is not None]
        
        empty_masks = set()
        for mask_id in mask_ids:
            mask = mask_dict[mask_id]
            if mask is not None:
                mask_sum = mask.sum()
                if mask_sum == 0:
                    results['empty'].setdefault(frame, []).append(mask_id)
                    empty_masks.add(mask_id)
                elif mask_sum >= 40000:
                    results['high'].setdefault(frame, []).append(mask_id)

        # Process overlaps
        for i in range(len(mask_ids)):
            mask_id = mask_ids[i]
            if mask_id in empty_masks:  # Skip empty masks for overlap checking
                continue
                
            mask = mask_dict[mask_id]
            if mask is not None:
                # Check for overlaps with other masks in the same frame, if any
                for j in range(i + 1, len(mask_ids)):
                    other_mask_id = mask_ids[j]
                    if other_mask_id in empty_masks:  # Skip empty masks
                        continue
                        
                    other_mask = mask_dict[other_mask_id]
                    if other_mask is not None:
                        is_overlapping, iou, overlap_pixels = check_overlap(mask, other_mask)
                        if is_overlapping:
                            results['overlapping'].setdefault(frame, []).append((mask_id, other_mask_id, iou, overlap_pixels))

        for category in ['empty', 'high', 'overlapping', 'distant']:
            if frame in results[category]:
                count = len(results[category][frame])
                if count > max_counts[category]:
                    max_counts[category] = count
                    max_frames[category] = frame

    # Check consecutive frame overlap after processing all frames
    sorted_frames = sorted(video_segments.keys())
    for i in range(len(sorted_frames) - 1):
        current_frame = sorted_frames[i]
        next_frame = sorted_frames[i + 1]
        
        if current_frame >= 600 or next_frame >= 600:
            continue
            
        if next_frame - current_frame != 1:
            continue
            
        current_masks = video_segments[current_frame]
        next_masks = video_segments[next_frame]
        
        for obj_id in current_masks.keys():
            if (obj_id in next_masks and 
                current_masks[obj_id] is not None and next_masks[obj_id] is not None and
                hasattr(current_masks[obj_id], 'sum') and hasattr(next_masks[obj_id], 'sum') and
                current_masks[obj_id].sum() > 0 and next_masks[obj_id].sum() > 0):
                
                is_overlapping, iou, overlap_pixels = check_overlap(current_masks[obj_id], next_masks[obj_id])
                
                if iou < 0.20:  # 20% threshold
                    results['low_consecutive_overlap'].setdefault(next_frame, []).append((obj_id, current_frame, iou))

    for category in ['low_consecutive_overlap']:
        for frame in results[category]:
            count = len(results[category][frame])
            if count > max_counts[category]:
                max_counts[category] = count
                max_frames[category] = frame

    return results, max_counts, max_frames

def collect_results(result_dict, condition, max_count, max_frame):
    detailed_output = []
    summary_output = []
    
    if result_dict:
        detailed_output.append(f"!!! Frames with masks {condition}:")
        for frame, data in result_dict.items():
            if condition == "overlapping":
                overlap_info = [f"{a}-{b} ({iou:.2%}, {pixels} pixels)" for a, b, iou, pixels in data]
                detailed_output.append(f"  Frame {frame}: Overlapping Mask ID pairs {', '.join(overlap_info)}")
            elif condition == "distant":
                detailed_output.append(f"  Frame {frame}: Distant Mask ID pairs {data}")
            elif condition == "low_consecutive_overlap":
                overlap_info = [f"Object {obj_id} (from frame {prev_frame}, overlap: {iou:.2%})" for obj_id, prev_frame, iou in data]
                detailed_output.append(f"  Frame {frame}: Low consecutive overlap {', '.join(overlap_info)}")
            else:
                detailed_output.append(f"  Frame {frame}: Mask IDs {data}")
        if max_count > 0:
            summary_output.append(f"Latest frame with highest number of {condition} masks: {max_frame} (Count: {max_count})")
    else:
        summary_output.append(f"Yay! No masks {condition} found!")
    
    return detailed_output, summary_output

def fill_single_missing_frames(video_segments):
    """
    Fill frames with empty masks or low consecutive overlap by interpolating between adjacent frames.
    If it's the first or last frame, use the next/previous available frame with non-empty masks.
    Only handles cases with exactly 1 consecutive frame with empty masks or low consecutive overlap.
    Returns a copy of video_segments with filled masks.
    """      
    video_segments_copy = copy.deepcopy(video_segments)
    
    frame_indices = sorted(video_segments_copy.keys())
    if len(frame_indices) < 2:
        return video_segments_copy
    
    empty_mask_frames = []
    for frame_idx in frame_indices:
        frame_masks = video_segments_copy[frame_idx]
        for obj_id, mask in frame_masks.items():
            if mask is not None and hasattr(mask, 'sum') and mask.sum() == 0:
                empty_mask_frames.append((frame_idx, obj_id))
                break
    
    low_overlap_frames = []
    for i in range(len(frame_indices) - 1):
        current_frame = frame_indices[i]
        next_frame = frame_indices[i + 1]
        
        if current_frame >= 600 or next_frame >= 600:
            continue
            
        if next_frame - current_frame != 1:
            continue
            
        current_masks = video_segments_copy[current_frame]
        next_masks = video_segments_copy[next_frame]
        
        for obj_id in current_masks.keys():
            if (obj_id in next_masks and 
                current_masks[obj_id] is not None and next_masks[obj_id] is not None and
                hasattr(current_masks[obj_id], 'sum') and hasattr(next_masks[obj_id], 'sum') and
                current_masks[obj_id].sum() > 0 and next_masks[obj_id].sum() > 0):
                
                is_overlapping, iou, overlap_pixels = check_overlap(current_masks[obj_id], next_masks[obj_id])
                
                if iou < 0.20:  # 20% threshold
                    low_overlap_frames.append((next_frame, obj_id))
    
    if not empty_mask_frames and not low_overlap_frames:
        print("No frames with empty masks or low consecutive overlap detected.")
        return video_segments_copy
    
    print(f"Found {len(empty_mask_frames)} frames with empty masks: {[f[0] for f in empty_mask_frames]}")
    print(f"Found {len(low_overlap_frames)} frames with low consecutive overlap: {[f[0] for f in low_overlap_frames]}")
    
    filled_count = 0
    
    if empty_mask_frames:
        frames_with_empty_masks = {}
        for frame_idx, obj_id in empty_mask_frames:
            if frame_idx not in frames_with_empty_masks:
                frames_with_empty_masks[frame_idx] = []
            frames_with_empty_masks[frame_idx].append(obj_id)
        
        for empty_frame in sorted(frames_with_empty_masks.keys()):
            if (empty_frame - 1 in frames_with_empty_masks or empty_frame + 1 in frames_with_empty_masks):
                continue
            
            empty_obj_ids = frames_with_empty_masks[empty_frame]
            filled_count += process_problematic_frame(video_segments_copy, empty_frame, empty_obj_ids, frame_indices, "empty")
    
    if low_overlap_frames:
        frames_with_low_overlap = {}
        for frame_idx, obj_id in low_overlap_frames:
            if frame_idx not in frames_with_low_overlap:
                frames_with_low_overlap[frame_idx] = []
            frames_with_low_overlap[frame_idx].append(obj_id)
        
        for low_overlap_frame in sorted(frames_with_low_overlap.keys()):
            low_overlap_obj_ids = frames_with_low_overlap[low_overlap_frame]
            filled_count += process_problematic_frame(video_segments_copy, low_overlap_frame, low_overlap_obj_ids, frame_indices, "low_overlap")

    print(f"Successfully filled {filled_count} problematic masks (empty or low consecutive overlap).")
    
    verification_count = 0
    for frame_idx in frame_indices:
        for obj_id, mask in video_segments_copy[frame_idx].items():
            if mask is not None and hasattr(mask, 'sum') and mask.sum() > 0:
                verification_count += 1
    
    print(f"Verification: {verification_count} non-empty masks remain in the filled video_segments.")
    return video_segments_copy

def process_problematic_frame(video_segments_copy, problem_frame, problematic_obj_ids, frame_indices, problem_type):
    """Helper function to process a single problematic frame"""
    local_filled_count = 0
    
    for obj_id in problematic_obj_ids:
        prev_frame = None
        next_frame = None
        
        for frame_idx in sorted([f for f in frame_indices if f < problem_frame and f < 600], reverse=True):
            if (obj_id in video_segments_copy[frame_idx] and 
                video_segments_copy[frame_idx][obj_id] is not None and
                hasattr(video_segments_copy[frame_idx][obj_id], 'sum') and
                video_segments_copy[frame_idx][obj_id].sum() > 0):
                prev_frame = frame_idx
                break
        
        for frame_idx in sorted([f for f in frame_indices if f > problem_frame and f < 600]):
            if (obj_id in video_segments_copy[frame_idx] and 
                video_segments_copy[frame_idx][obj_id] is not None and
                hasattr(video_segments_copy[frame_idx][obj_id], 'sum') and
                video_segments_copy[frame_idx][obj_id].sum() > 0):
                next_frame = frame_idx
                break
        
        if prev_frame is None and next_frame is not None:
            video_segments_copy[problem_frame][obj_id] = video_segments_copy[next_frame][obj_id].copy()
            local_filled_count += 1
            print(f"Filled frame {problem_frame} object {obj_id} ({problem_type}, first frame) using frame {next_frame}")
            
        elif prev_frame is not None and next_frame is None:
            video_segments_copy[problem_frame][obj_id] = video_segments_copy[prev_frame][obj_id].copy()
            local_filled_count += 1
            print(f"Filled frame {problem_frame} object {obj_id} ({problem_type}, last frame) using frame {prev_frame}")
            
        elif prev_frame is not None and next_frame is not None:
            prev_mask = video_segments_copy[prev_frame][obj_id]
            next_mask = video_segments_copy[next_frame][obj_id]
            
            if hasattr(prev_mask, 'dtype') and hasattr(next_mask, 'dtype'):
                avg_mask = (prev_mask.astype(np.float32) + next_mask.astype(np.float32)) / 2.0
                if prev_mask.dtype == bool:
                    avg_mask = avg_mask > 0.5
                else:
                    avg_mask = avg_mask.astype(prev_mask.dtype)
                video_segments_copy[problem_frame][obj_id] = avg_mask
            else:
                video_segments_copy[problem_frame][obj_id] = prev_mask.copy()
            
            local_filled_count += 1
            print(f"Filled frame {problem_frame} object {obj_id} ({problem_type}) by interpolating between frames {prev_frame} and {next_frame}")
    
    return local_filled_count

def analyze_and_print_results(video_segments):
    analysis_results, max_counts, max_frames = analyze_masks(video_segments)

    all_detailed_outputs = []
    all_summary_outputs = []
    problematic_frame_counts = {
        'empty': 0,
        'high': 0,
        'overlapping': 0,
        'distant': 0,
        'low_consecutive_overlap': 0
    }

    total_frames = len(video_segments)

    for category in ['empty', 'high', 'overlapping', 'distant', 'low_consecutive_overlap']:
        detailed, summary = collect_results(analysis_results[category], category, max_counts[category], max_frames[category])
        all_detailed_outputs.extend(detailed)
        all_summary_outputs.extend(summary)
        problematic_frame_counts[category] = len(analysis_results[category])

    for line in all_detailed_outputs:
        print(line)

    for line in all_summary_outputs:
        print(line)

    print("\nNumber of problematic frames:")
    for category, count in problematic_frame_counts.items():
        percentage = (count / total_frames) * 100
        print(f"Frames with {category} masks: {count} out of {total_frames} ({percentage:.2f}%)")

    unique_problematic_frames = set()
    for category, frames in analysis_results.items():
        unique_problematic_frames.update(frames.keys())
    
    unique_problematic_count = len(unique_problematic_frames)
    unique_problematic_percentage = (unique_problematic_count / total_frames) * 100
    print(f"\nTotal number of unique problematic frames: {unique_problematic_count} out of {total_frames} ({unique_problematic_percentage:.2f}%)")
    
    if 'low_consecutive_overlap' in analysis_results and analysis_results['low_consecutive_overlap']:
        print(f"\n*** Detected {len(analysis_results['low_consecutive_overlap'])} frames with low consecutive overlap ***")

    if video_segments:
        objects_per_frame = [len([obj for obj in segments.keys() if obj is not None]) for segments in video_segments.values()]
        if objects_per_frame:
            avg_objects_per_frame = sum(objects_per_frame) / len(objects_per_frame)
            print(f"\nAverage number of objects per frame: {avg_objects_per_frame:.2f}")
            print(f"Minimum objects in a frame: {min(objects_per_frame)}")
            print(f"Maximum objects in a frame: {max(objects_per_frame)}")
        else:
            print("\nNo objects found to analyze.")
    else:
        print("\nNo segments to analyze for statistics.")

def create_mask_overlay_video(video_dir, frame_names, video_segments, output_video_path, fps=10, alpha=0.99, num_workers=None, scale_factor=1.0):
    COLORS = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (128, 0, 0),    # Maroon
        (128, 0, 128),  # Purple
        (0, 0, 128),    # Navy
        (128, 128, 0),  # Olive
        (0, 128, 0),    # Dark Green
        (0, 128, 128),  # Teal
        (255, 128, 0),  # Orange
        (255, 0, 128),  # Deep Pink
        (128, 255, 0),  # Lime
        (255, 255, 0),  # Yellow	
        (0, 255, 128)   # Spring Green
    ]

    frame = cv2.imread(os.path.join(video_dir, frame_names[0]))
    if frame is None:
        raise ValueError(f"Could not read first frame from {os.path.join(video_dir, frame_names[0])}")
    height, width, _ = frame.shape
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    new_size = (new_width, new_height)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, new_size)

    all_mask_ids = set()
    for masks in video_segments.values():
        all_mask_ids.update(masks.keys())
    colors = {}
    for i, mask_id in enumerate(all_mask_ids):
        if mask_id is not None:
            colors[mask_id] = COLORS[i % len(COLORS)]
            
    # Setup for parallel processing
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    tasks = []
    for frame_idx, frame_name in enumerate(frame_names):
        masks_for_frame = video_segments.get(frame_idx, {})
        tasks.append((frame_idx, frame_name, video_dir, masks_for_frame, colors, alpha, new_size))

    with multiprocessing.Pool(processes=num_workers) as pool:
        with tqdm.tqdm(total=len(tasks), desc="Creating overlay video") as pbar:
            for result_frame in pool.imap(_process_frame_for_video, tasks):
                if result_frame is not None:
                    out.write(result_frame)
                pbar.update(1)

    out.release()

    print(f"Video saved to {output_video_path}")

def overlay_predictions_on_frame(video_dir, frame_idx, video_segments, alpha=0.99):
    COLORS = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (128, 0, 0),    # Maroon
        (128, 0, 128),  # Purple
        (0, 0, 128),    # Navy
        (128, 128, 0),  # Olive
        (0, 128, 0),    # Dark Green
        (0, 128, 128),  # Teal
        (255, 128, 0),  # Orange
        (255, 0, 128),  # Deep Pink
        (128, 255, 0),  # Lime
        (255, 255, 0),  # Yellow	
        (0, 255, 128)   # Spring Green
    ]

    image_path = os.path.join(video_dir, f"{frame_idx:06d}.jpg")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    overlay = np.zeros_like(image)

    if frame_idx in video_segments:
        masks = video_segments[frame_idx]

        colors = {mask_id: COLORS[i % len(COLORS)] for i, mask_id in enumerate(masks.keys())}

        for mask_id, mask in masks.items():
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()

            if mask.dtype != bool:
                mask = mask > 0.5

            if mask.ndim > 2:
                mask = mask.squeeze()

            mask_resized = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

            color = colors[mask_id]

            colored_mask = np.zeros_like(image)
            colored_mask[mask_resized == 1] = color

            overlay = cv2.addWeighted(overlay, 1, colored_mask, 1, 0)

        overlaid_image = cv2.addWeighted(image, 1, overlay, alpha, 0)
    else:
        print(f"No predictions found for frame {frame_idx}")
        overlaid_image = image

    cv2.imwrite("frame_prediction_overlay.png", cv2.cvtColor(overlaid_image, cv2.COLOR_RGB2BGR))
    print(f"Overlaid image saved to frame_prediction_overlay.png")

def check_prompt_data(frame_idx, prompt_data, video_dir, inference_state, frame_mapping):
    new_frame_number = next((new for new, original in frame_mapping.items() if original == frame_idx), None)
    if new_frame_number is None:
        raise ValueError(f"No mapping found for original frame number {frame_idx}")
    prompts_for_frame = {}
    for obj_id, obj_data in prompt_data[str(frame_idx)].items():
        print(f"Processing frame {new_frame_number}, object {obj_id}")
        points = obj_data["points"]
        labels = obj_data["labels"]
        prompts_for_frame[int(obj_id)] = points, labels
        _, out_obj_ids, out_mask_logits = predictor.add_new_points(
            inference_state=inference_state,
            frame_idx=new_frame_number,
            obj_id=int(obj_id),
            points=points,
            labels=labels
        )
        
        plt.figure(figsize=(12, 8))
        plt.title(f"New frame {new_frame_number}, prompt frame {frame_idx}")
        plt.imshow(Image.open(os.path.join(video_dir, f"{new_frame_number:06d}.jpg")))
        show_points(points, labels, plt.gca())
        for i, out_obj_id in enumerate(out_obj_ids):
            show_points(points, labels, plt.gca())
            show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)   
        plt.savefig(f"prompt_frame_data_check.png")
        print(f"Prompt frame data check saved to prompt_frame_data_check.png")
        plt.close()      
        time.sleep(0.02)  # Optimal delay between iterations for concurrency

    return prompts_for_frame

def analyze_prompt_frames_immediate(video_dir, frame_mapping, prompt_data, inference_state, predictor):
    prompt_frame_results = {}
    pbar = tqdm.tqdm(frame_mapping.items(), desc="Analyzing prompt frames", unit="frame")

    for new_frame_num, original_frame_num in pbar:
        if str(original_frame_num) in prompt_data:
            pbar.set_postfix({"Original Frame": original_frame_num})
            
            # Get the mask predictions for this frame
            _, out_obj_ids, out_mask_logits = predictor.add_new_points(
                inference_state=inference_state,
                frame_idx=new_frame_num,
                obj_id=None,
                points=np.empty((0, 2)),
                labels=np.empty(0)
            )

            masks = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

            # Analyze masks for this prompt frame
            empty_masks = []
            large_masks = []
            overlapping_masks = []
            
            def calculate_overlap(mask1, mask2):
                intersection = np.logical_and(mask1, mask2)
                union = np.logical_or(mask1, mask2)
                overlap_pixels = np.sum(intersection)
                iou = overlap_pixels / np.sum(union)
                return iou, overlap_pixels
            
            for obj_id, mask in masks.items():
                if obj_id is not None: 
                    mask_sum = mask.sum()
                    
                    if mask_sum == 0:
                        empty_masks.append(obj_id)
                    elif mask_sum >= 40000:
                        large_masks.append(obj_id)
                    
                    for other_obj_id, other_mask in masks.items():
                        if other_obj_id is not None and obj_id != other_obj_id:
                            overlap, overlap_pixels = calculate_overlap(mask, other_mask)
                            if overlap > 0.01:  # 1% overlap threshold
                                overlapping_masks.append((obj_id, other_obj_id, overlap, overlap_pixels))
            
            prompt_frame_results[new_frame_num] = {
                'original_frame': original_frame_num,
                'all_objects': list(masks.keys()),  # Store all object IDs
                'empty_masks': empty_masks,
                'large_masks': large_masks,
                'overlapping_masks': overlapping_masks
            }
            
            # Visualize the prompt frame results
            plt.figure(figsize=(12, 8))
            plt.title(f"Prompt Frame {new_frame_num} (Original: {original_frame_num})")
            image = Image.open(os.path.join(video_dir, f"{new_frame_num:06d}.jpg"))
            plt.imshow(image)
            
            for obj_id, mask in masks.items():
                if obj_id is not None:
                    show_mask(mask, plt.gca(), obj_id=obj_id, random_color=True)
            
            plt.savefig(f"prompt_frame_analysis_{new_frame_num}.png")
            plt.close()

    return prompt_frame_results

def print_prompt_frame_analysis(prompt_frame_results):
    print("\nPrompt Frame Analysis Summary:")
    
    problematic_frames = []
    frames_without_issues = []
    all_objects = set()

    def safe_sort(iterable):
        return sorted((item for item in iterable if item is not None), key=lambda x: (x is None, x))

    for frame_num, results in prompt_frame_results.items():
        issues = []
        frame_objects = set(obj for obj in results['all_objects'] if obj is not None)

        if results['empty_masks']:
            issues.append(f"Empty masks: {safe_sort(results['empty_masks'])}")
        if results['large_masks']:
            issues.append(f"Large masks (800+ pixels): {safe_sort(results['large_masks'])}")
        if results['overlapping_masks']:
            overlap_info = [f"{a}-{b} ({overlap:.2%}, {pixels} pixels)" for a, b, overlap, pixels in results['overlapping_masks'] if a is not None and b is not None]
            issues.append(f"Overlapping masks: {', '.join(overlap_info)}")
        
        all_objects.update(frame_objects)
        
        if issues:
            problematic_frames.append((frame_num, results['original_frame'], issues, frame_objects))
        else:
            frames_without_issues.append((frame_num, frame_objects))

    if frames_without_issues:
        print("\nFrames without issues:")
        for frame_num, frame_objects in frames_without_issues:
            print(f"  Frame {frame_num}: Objects present: {safe_sort(frame_objects)}")

    if problematic_frames:
        print("Problematic frames:")
        for frame_num, original_frame, issues, frame_objects in problematic_frames:
            print(f"  Frame {frame_num} (Original: {original_frame}):")
            print(f"    Objects present: {safe_sort(frame_objects)}")
            for issue in issues:
                print(f"    - {issue}")
    else:
        print("No problematic frames detected.")


    print(f"\nTotal frames analyzed: {len(prompt_frame_results)}")
    print(f"Frames with issues: {len(problematic_frames)}")
    print(f"Frames without issues: {len(frames_without_issues)}")

    print(f"\nTotal unique object IDs detected across all frames: {safe_sort(all_objects)}")
    print(f"Number of unique objects: {len(all_objects)}")

def save_hd_video_segments(hd_video_segments, video_dir, output_dir):
    video_folder_name = os.path.basename(os.path.normpath(video_dir))
    
    output_filename = f"{video_folder_name}_hd_segments.pkl"
    output_path = os.path.join(output_dir, output_filename)
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'wb') as file:
        pickle.dump(hd_video_segments, file)
    
    print(f"Saved HD video segments to: {output_path}")

def get_random_unprocessed_video(crop_videos_dir, segmented_videos_dir):
    all_videos = [d for d in os.listdir(crop_videos_dir) if os.path.isdir(os.path.join(crop_videos_dir, d))]
    unprocessed_videos = [
        video for video in all_videos
        if not os.path.exists(os.path.join(segmented_videos_dir, video + "_hd_segments.pkl"))
    ]
    
    if not unprocessed_videos:
        raise ValueError("All videos have been processed.")
    
    return os.path.join(crop_videos_dir, random.choice(unprocessed_videos))

def get_hdsegmentation(ffvideo_segments, crop_size=94):
    hd_video_segments = {}
    for frame_num in sorted(ffvideo_segments.keys()):
        print(frame_num)
        or_mask = next(iter(ffvideo_segments[frame_num].values()))
        # Find the crop box of the segment
        rows, cols = np.where(or_mask[0])
        center_y, center_x = rows.mean(), cols.mean()
        top = max(0, int(center_y - crop_size // 2))
        bottom = min(or_mask.shape[1], top + crop_size)
        left = max(0, int(center_x - crop_size // 2))
        right = min(or_mask.shape[2], left + crop_size)
        or_frame = cv2.imread(os.path.join(video_dir, f"{frame_num:06}.jpg"))
        cropped_arr = or_frame[top:bottom, left:right]
        # Save the cropped frame to prediction folder
        cv2.imwrite("PATH_TO_CROPPED_PROMPT_FRAMES_DIR/<FRAME_NUMBER>.jpg", cropped_arr)

        # Make prediction on the cropped frame
        cropped_dir = "PATH_TO_CROPPED_PROMPT_FRAMES_DIR"
        frame_names = [
            p for p in os.listdir(cropped_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        if frame_num == 0:
            inference_state = predictor.init_state(video_path=cropped_dir)
        else:
            predictor.reset_state(inference_state)
            inference_state = predictor.init_state(video_path=cropped_dir)

        #Add click on the first frame
        ann_frame_idx = 0
        ann_obj_id = 1
        points = np.array([[64, 45]], dtype=np.float32) 
        labels = np.array([1], np.int32)
        _, out_obj_ids, out_mask_logits = predictor.add_new_points(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )
        time.sleep(0.02)
        #Add click on the second frame
        ann_frame_idx = 1
        ann_obj_id = 1
        points = np.array([[70, 45], [65, 44]], dtype=np.float32)
        labels = np.array([1, 0], np.int32)
        _, out_obj_ids, out_mask_logits = predictor.add_new_points(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )
        #add more clicks if needed

        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        cropped_hd_segment = video_segments[2][1]

        # Resize the cropped segment to the original size
        or_shape = or_frame.shape[:2]
        full_hd_segment = np.zeros(or_shape, dtype=bool)
        full_hd_segment[top:bottom, left:right] = cropped_hd_segment

        reshaped_hdseg = np.expand_dims(full_hd_segment, axis=0)
        hd_video_segments[frame_num] = {1: reshaped_hdseg}

    return hd_video_segments

def add_new_prompt(frame_number, video_dir, prompt_dir, prompt_data_file, prompts):
    """
    Add a new prompt image and its associated data based on a frame number from the video directory to the prompt pool.
    
    :param frame_number: Number of the frame to be used as a prompt
    :param video_dir: Directory containing the video frames
    :param prompt_dir: Directory where prompt images are stored
    :param prompt_data_file: Path to the JSON file containing prompt data
    :param prompts: Dictionary containing prompt data in the format {obj_id: (points, labels)}
    """
    os.makedirs(prompt_dir, exist_ok=True)
    
    frame_name = f"{frame_number:06d}.jpg"
    source_frame_path = os.path.join(video_dir, frame_name)
    
    if not os.path.exists(source_frame_path):
        raise FileNotFoundError(f"Frame {frame_name} not found in {video_dir}")

    # Load existing prompt data
    if os.path.exists(prompt_data_file) and os.path.getsize(prompt_data_file) > 0:
        with open(prompt_data_file, 'r') as f:
            existing_prompts = json.load(f)
    else:
        existing_prompts = {}
    
    # Determine the new prompt number
    existing_numbers = [int(num) for num in existing_prompts.keys()]
    new_number = max(existing_numbers) + 1 if existing_numbers else 1
    
    # Copy the frame to the prompt directory with the new number
    new_image_name = f"{new_number}.jpg"
    new_image_path = os.path.join(prompt_dir, new_image_name)
    shutil.copy(source_frame_path, new_image_path)
    
    # Transform the prompts data into the required format for JSON
    new_prompt_data = {}
    for obj_id, (points, labels) in prompts.items():
        new_prompt_data[str(obj_id)] = {
            "points": points.tolist(),
            "labels": labels.tolist()
        }
    
    existing_prompts[str(new_number)] = new_prompt_data
    
    with open(prompt_data_file, 'w') as f:
        json.dump(existing_prompts, f, indent=2)
    
    print(f"Added new prompt image {new_image_name} for frame {frame_number} and updated prompt data.")

def modify_prompt(frame_number, frame_mapping, prompt_data_file, new_prompts):
    """
    Modify existing prompt data or add new data to an existing prompt in the prompt data file,
    using the frame number from the video directory and the existing frame mapping.
    
    :param frame_number: Number of the frame in the video directory to be modified
    :param frame_mapping: Dictionary mapping new frame numbers to original prompt frame numbers
    :param prompt_data_file: Path to the JSON file containing prompt data
    :param new_prompts: Dictionary containing new or updated prompt data in the format {obj_id: (points, labels)}
    """
    # Load existing prompt data
    if os.path.exists(prompt_data_file) and os.path.getsize(prompt_data_file) > 0:
        with open(prompt_data_file, 'r') as f:
            existing_prompts = json.load(f)
    else:
        raise FileNotFoundError(f"Prompt data file not found or is empty: {prompt_data_file}")
    
    # Find the original prompt frame number using the frame_mapping
    original_frame_number = None
    for new_frame, original_frame in frame_mapping.items():
        if new_frame == frame_number:
            original_frame_number = original_frame
            break
    
    if original_frame_number is None:
        raise ValueError(f"No mapping found for frame number {frame_number}")
    
    # Convert the original frame number to string for JSON key
    prompt_number = str(original_frame_number)
    
    if prompt_number not in existing_prompts:
        existing_prompts[prompt_number] = {}
    
    for obj_id, (points, labels) in new_prompts.items():
        existing_prompts[prompt_number][str(obj_id)] = {
            "points": points.tolist(),
            "labels": labels.tolist()
        }
    
    with open(prompt_data_file, 'w') as f:
        json.dump(existing_prompts, f, indent=2)
    
    print(f"Updated prompt data for video frame {frame_number} (original prompt frame {original_frame_number}).")


#endregion [functions]


##Get random video to process
videos_dir = 'PATH_TO_VIDEO_DIR'
segmented_videos_dir = 'OUTPUT_DIR_PATH_FOR_SEGMENTED_VIDEOS'
video_dir = get_random_unprocessed_video(videos_dir, segmented_videos_dir)
print(f"Processing video: {video_dir}")


#region [full frame segmentation]
prompt_dir = "PATH_TO_PROMPT_FRAME_DIR"
prompt_data_path = "PATH_TO_PROMPT_FRAME_DATA_JSON"

# Add prompt frames to the video directory
frame_mapping = add_prompt_frames_to_video(video_dir, prompt_dir)
frame_names = sorted([p for p in os.listdir(video_dir) if p.lower().endswith(('.jpg', '.jpeg'))],
                     key=lambda p: int(os.path.splitext(p)[0]))

inference_state = predictor.init_state(video_path=video_dir)

# Load prompt data from JSON file
with open(prompt_data_path, 'r') as f:
    prompt_data = json.load(f)

for frame_num in prompt_data:
    for obj_id in prompt_data[frame_num]:
        prompt_data[frame_num][obj_id]['points'] = np.array(prompt_data[frame_num][obj_id]['points'], dtype=np.float32)
        prompt_data[frame_num][obj_id]['labels'] = np.array(prompt_data[frame_num][obj_id]['labels'], dtype=np.int32)

# Add prompts for each frame
for new_frame_num, original_frame_num in frame_mapping.items():
    if str(original_frame_num) in prompt_data:
        for obj_id, obj_data in prompt_data[str(original_frame_num)].items():
            print(f"Processing frame {new_frame_num} (original {original_frame_num}), object {obj_id}")
            add_prompts(inference_state, new_frame_num, int(obj_id), obj_data["points"], obj_data["labels"])
            time.sleep(0.02)  # Optimal delay between iterations for concurrency
    print(f"Completed frame {new_frame_num}")

prompt_frame_results = analyze_prompt_frames_immediate(video_dir, frame_mapping, prompt_data, inference_state, predictor)
print_prompt_frame_analysis(prompt_frame_results)

video_segments = {}
last_frame_idx = max(frame_mapping.keys())
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=last_frame_idx, reverse=True):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# Check results
analyze_and_print_results(video_segments)
video_segments_filled = fill_single_missing_frames(video_segments)
# Adjust or add prompts to the prompt pool, if needed

#Make video with masks for visualisation, if needed
create_mask_overlay_video(
    video_dir,
    frame_names,
    video_segments_filled,
    output_video_path="PATH_TO_OUTPUT_VIDEO.mp4",
    fps=10,
    alpha=1.0,
    num_workers=multiprocessing.cpu_count(),
    scale_factor=0.5
)

remove_prompt_frames_from_video(video_dir, frame_mapping)
filtered_video_segments = filter_prompt_frames_from_segments(video_segments_filled, frame_mapping)

#endregion [full frame segmentation]


#region [hd segmentation]

hd_video_segments = get_hdsegmentation(filtered_video_segments, crop_size=94)
analyze_and_print_results(hd_video_segments)
hd_video_segments_filled = fill_single_missing_frames(hd_video_segments)
#Adjust or add prompts to the prompt pool, if needed

#Make video with masks for visualisation, if needed
frame_names = sorted([p for p in os.listdir(video_dir) if p.lower().endswith(('.jpg', '.jpeg'))], key=lambda p: int(os.path.splitext(p)[0]))
create_mask_overlay_video(
    video_dir,
    frame_names,
    hd_video_segments_filled,
    output_video_path="PATH_TO_OUTPUT_VIDEO.mp4",
    fps=10,
    alpha=1.0,
    num_workers=multiprocessing.cpu_count(),
    scale_factor=0.5
)

save_hd_video_segments(hd_video_segments, video_dir, 'PATH_TO_HDSEGMENTATION_OUTPUT_DIR')

#endregion [hd segmentation]