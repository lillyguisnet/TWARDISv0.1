"""
This script performs segmentation on a video with autoprompting of the nrd, nrv and loop compartments.
There are several visualization helper functions.
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
import h5py
import random

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

sam2_checkpoint = "./segment-anything-2/checkpoints/sam2_hiera_large.pt" #Path to SAM2 checkpoint
model_cfg = "sam2_hiera_l.yaml" #Path to SAM2 model config
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

#region [functions]

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
        frame_mapping[new_frame_num] = int(os.path.splitext(prompt_frame)[0])

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

def add_prompts(inference_state, frame_idx, obj_id, points, labels):
    """
    Add prompts points to prompt frames.
    """
    _, out_obj_ids, out_mask_logits = predictor.add_new_points(
        inference_state=inference_state,
        frame_idx=frame_idx,
        obj_id=obj_id,
        points=points,
        labels=labels
    )
    # Visualize the prompt frame, if needed
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
    dilated_mask1 = binary_dilation(mask1, iterations=max_distance)
    dilated_mask2 = binary_dilation(mask2, iterations=max_distance)
    return np.any(np.logical_and(dilated_mask1, dilated_mask2))

def analyze_masks(video_segments):
    results = {'empty': {}, 'high': {}, 'overlapping': {}, 'distant': {}}
    max_counts = {'empty': 0, 'high': 0, 'overlapping': 0, 'distant': 0}
    max_frames = {'empty': None, 'high': None, 'overlapping': None, 'distant': None}

    for frame, mask_dict in video_segments.items():
        mask_ids = [mask_id for mask_id in mask_dict.keys() if mask_id is not None]
        for i in range(len(mask_ids)):
            mask_id = mask_ids[i]
            mask = mask_dict[mask_id]
            if mask is not None:
                mask_sum = mask.sum()
                if mask_sum == 0:
                    results['empty'].setdefault(frame, []).append(mask_id)
                elif mask_sum >= 800:
                    results['high'].setdefault(frame, []).append(mask_id)
            
                for j in range(i + 1, len(mask_ids)):
                    other_mask_id = mask_ids[j]
                    other_mask = mask_dict[other_mask_id]
                    if other_mask is not None:
                        is_overlapping, iou, overlap_pixels = check_overlap(mask, other_mask)
                        if is_overlapping:
                            results['overlapping'].setdefault(frame, []).append((mask_id, other_mask_id, iou, overlap_pixels))
        
        # Check distance between object 3 and 4
        if 3 in mask_dict and 4 in mask_dict and mask_dict[3] is not None and mask_dict[4] is not None:
            if not check_distance(mask_dict[3], mask_dict[4]):
                results['distant'].setdefault(frame, []).append((3, 4))

        for category in ['empty', 'high', 'overlapping', 'distant']:
            if frame in results[category]:
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
            else:
                detailed_output.append(f"  Frame {frame}: Mask IDs {data}")
        if max_count > 0:
            summary_output.append(f"Latest frame with highest number of {condition} masks: {max_frame} (Count: {max_count})")
    else:
        summary_output.append(f"Yay! No masks {condition} found!")
    
    return detailed_output, summary_output

def analyze_and_print_results(video_segments):
    """
    Analyze masks quality.
    """
    analysis_results, max_counts, max_frames = analyze_masks(video_segments)

    all_detailed_outputs = []
    all_summary_outputs = []
    problematic_frame_counts = {
        'empty': 0,
        'high': 0,
        'overlapping': 0,
        'distant': 0
    }
    total_frames = len(video_segments)

    for category in ['empty', 'high', 'overlapping', 'distant']:
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

def create_mask_overlay_video(video_dir, frame_names, video_segments, output_video_path, fps=10, alpha=0.99):
    """
    Create a video with mask predictions overlay.
    """
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

    def overlay_masks_on_image(image_path, masks, colors, alpha=0.99):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        overlay = np.zeros_like(image)
        
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
        return overlaid_image

    frame = cv2.imread(os.path.join(video_dir, frame_names[0]))
    if frame is None:
        raise ValueError(f"Could not read first frame from {os.path.join(video_dir, frame_names[0])}")
    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    all_mask_ids = set()
    for masks in video_segments.values():
        all_mask_ids.update(masks.keys())
    colors = {}
    for i, mask_id in enumerate(all_mask_ids):
        colors[mask_id] = COLORS[i % len(COLORS)]

    for frame_idx in range(len(frame_names)):
        image_path = os.path.join(video_dir, frame_names[frame_idx])
        try:
            if frame_idx in video_segments:
                masks = video_segments[frame_idx]
                overlaid_frame = overlay_masks_on_image(image_path, masks, colors, alpha)
            else:
                overlaid_frame = cv2.imread(image_path)
                if overlaid_frame is None:
                    raise ValueError(f"Could not read image from {image_path}")
                overlaid_frame = cv2.cvtColor(overlaid_frame, cv2.COLOR_BGR2RGB)
            
            out.write(cv2.cvtColor(overlaid_frame, cv2.COLOR_RGB2BGR))
        except Exception as e:
            print(f"Error processing frame {frame_idx}: {str(e)}")
            original_frame = cv2.imread(image_path)
            if original_frame is not None:
                out.write(original_frame)
            else:
                print(f"Could not read original frame {frame_idx}")

    out.release()
    print(f"Video saved to {output_video_path}")

def overlay_predictions_on_frame(video_dir, frame_idx, video_segments, alpha=0.99):
    """
    Overlay mask predictions on a single frame.
    """
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

def analyze_prompt_frames_immediate(video_dir, frame_mapping, prompt_data, inference_state, predictor):
    """
    Check for mask errors in prompt frames.
    """
    prompt_frame_results = {}
    pbar = tqdm.tqdm(frame_mapping.items(), desc="Analyzing prompt frames", unit="frame")

    for new_frame_num, original_frame_num in pbar:
        if str(original_frame_num) in prompt_data:
            pbar.set_postfix({"Original Frame": original_frame_num})
            
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
                    elif mask_sum >= 800:
                        large_masks.append(obj_id)
                    
                    for other_obj_id, other_mask in masks.items():
                        if other_obj_id is not None and obj_id != other_obj_id:
                            overlap, overlap_pixels = calculate_overlap(mask, other_mask)
                            if overlap > 0.01:  # 1% overlap threshold
                                overlapping_masks.append((obj_id, other_obj_id, overlap, overlap_pixels))
            
            prompt_frame_results[new_frame_num] = {
                'original_frame': original_frame_num,
                'all_objects': list(masks.keys()),
                'empty_masks': empty_masks,
                'large_masks': large_masks,
                'overlapping_masks': overlapping_masks
            }
            
            # Visualize the prompt frame results, if needed
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

    # Additional statistics
    objects_per_frame = [len([obj for obj in results['all_objects'] if obj is not None]) for results in prompt_frame_results.values()]
    avg_objects_per_frame = sum(objects_per_frame) / len(objects_per_frame) if objects_per_frame else 0
    print(f"\nAverage number of objects per frame: {avg_objects_per_frame:.2f}")
    print(f"Minimum objects in a frame: {min(objects_per_frame) if objects_per_frame else 0}")
    print(f"Maximum objects in a frame: {max(objects_per_frame) if objects_per_frame else 0}")

def save_video_segments_to_h5(video_segments, video_dir, output_dir, frame_mapping):
    last_folder = os.path.basename(os.path.normpath(video_dir))    
    filename = f"{last_folder}_riasegmentation.h5"    
    os.makedirs(output_dir, exist_ok=True)    
    output_path = os.path.join(output_dir, filename)
    exclude_frames = set(frame_mapping.keys())
    filtered_video_segments = {
        frame: segments for frame, segments in video_segments.items()
        if frame not in exclude_frames
    }

    with h5py.File(output_path, 'w') as f:
        f.attrs['num_frames'] = len(filtered_video_segments)        
        sample_frame = next(iter(filtered_video_segments.values()))
        object_ids = list(sample_frame.keys())
        f.attrs['object_ids'] = [str(obj_id) if obj_id is not None else 'None' for obj_id in object_ids]
        
        # Create datasets for each object
        for obj_id in object_ids:
            obj_id_str = str(obj_id) if obj_id is not None else 'None'
            # Create a dataset for each object, with the first dimension being the number of frames
            f.create_dataset(f'masks/{obj_id_str}', 
                             shape=(len(filtered_video_segments), 1, 110, 110),
                             dtype=bool,
                             compression="gzip")
        
        for i, (frame_idx, objects) in enumerate(sorted(filtered_video_segments.items(), reverse=True)):
            for obj_id, mask in objects.items():
                obj_id_str = str(obj_id) if obj_id is not None else 'None'
                f[f'masks/{obj_id_str}'][i] = mask

    print(f"Saved filtered video segments to: {output_path}")
    print(f"Number of frames saved: {len(filtered_video_segments)}")
    print(f"Number of frames excluded: {len(exclude_frames)}")
    return filtered_video_segments

def get_random_unprocessed_video(crop_videos_dir, segmented_videos_dir):
    all_videos = [d for d in os.listdir(crop_videos_dir) if os.path.isdir(os.path.join(crop_videos_dir, d))]
    unprocessed_videos = [
        video for video in all_videos
        if not os.path.exists(os.path.join(segmented_videos_dir, video + "_riasegmentation.h5"))
    ]
    
    if not unprocessed_videos:
        raise ValueError("All videos have been processed.")
    
    return os.path.join(crop_videos_dir, random.choice(unprocessed_videos))

def add_new_prompt(frame_number, video_dir, prompt_dir, prompt_data_file, prompts):
    """
    Add a new prompt image and its associated data based on a frame number from the video directory.
    """
    os.makedirs(prompt_dir, exist_ok=True)    
    frame_name = f"{frame_number:06d}.jpg"
    source_frame_path = os.path.join(video_dir, frame_name)
    
    if not os.path.exists(source_frame_path):
        raise FileNotFoundError(f"Frame {frame_name} not found in {video_dir}")

    if os.path.exists(prompt_data_file):
        with open(prompt_data_file, 'r') as f:
            existing_prompts = json.load(f)
    else:
        existing_prompts = {}
    
    existing_numbers = [int(num) for num in existing_prompts.keys()]
    new_number = max(existing_numbers) + 1 if existing_numbers else 1
    
    new_image_name = f"{new_number}.jpg"
    new_image_path = os.path.join(prompt_dir, new_image_name)
    shutil.copy(source_frame_path, new_image_path)
    
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
    """
    if os.path.exists(prompt_data_file):
        with open(prompt_data_file, 'r') as f:
            existing_prompts = json.load(f)
    else:
        raise FileNotFoundError(f"Prompt data file not found: {prompt_data_file}")
    
    original_frame_number = None
    for new_frame, original_frame in frame_mapping.items():
        if new_frame == frame_number:
            original_frame_number = original_frame
            break
    
    if original_frame_number is None:
        raise ValueError(f"No mapping found for frame number {frame_number}")
    
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

crop_videos_dir = 'PATH_TO_CROPPED_VIDEO_DIR'
segmented_videos_dir = 'PATH_TO_SEGMENTED_VIDEO_DIR'

video_dir = get_random_unprocessed_video(crop_videos_dir, segmented_videos_dir)
print(f"Processing video: {video_dir}")

prompt_dir = 'PATH_TO_PROMPT_FRAMES_DIR'
prompt_data_file = 'PATH_TO_PROMPT_DATA_FILE'

# Add prompt frames to the video directory
frame_mapping = add_prompt_frames_to_video(video_dir, prompt_dir)

# Load prompt data from JSON file
with open(prompt_data_file, 'r') as f:
    prompt_data = json.load(f)

for frame_num in prompt_data:
    for obj_id in prompt_data[frame_num]:
        prompt_data[frame_num][obj_id]['points'] = np.array(prompt_data[frame_num][obj_id]['points'], dtype=np.float32)
        prompt_data[frame_num][obj_id]['labels'] = np.array(prompt_data[frame_num][obj_id]['labels'], dtype=np.int32)

frame_names = sorted([p for p in os.listdir(video_dir) if p.lower().endswith(('.jpg', '.jpeg'))],
                     key=lambda p: int(os.path.splitext(p)[0]))
inference_state = predictor.init_state(video_path=video_dir)

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

remove_prompt_frames_from_video(video_dir, frame_mapping)
analyze_and_print_results(video_segments)

#Make video with masks, if needed
create_mask_overlay_video(
    video_dir,
    frame_names,
    video_segments,
    output_video_path="PATH_TO_OUTPUT_VIDEO.mp4",
    fps=10,
    alpha=0.99
)

output_dir = 'PATH_TO_OUTPUT_DIR'
filtered_video_segments = save_video_segments_to_h5(video_segments, video_dir, output_dir, frame_mapping)