"""
This script generates a cropped video around the RIA region.
"""
import os
import sys
sys.path.append("PATH_TO_CLONED_REPO/segment-anything-2")
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
from tqdm import tqdm
import cv2
import random

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from sam2.build_sam import build_sam2_video_predictor
sam2_checkpoint = "./segment-anything-2/checkpoints/sam2_hiera_large.pt" #Path to SAM2 checkpoint
model_cfg = "sam2_hiera_l.yaml" #Path to SAM2 model config
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=20):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def get_random_unprocessed_video(parent_dir, crop_dir):
    all_videos = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
    unprocessed_videos = [
        video for video in all_videos
        if not os.path.exists(os.path.join(crop_dir, video + "_crop"))
    ]
    
    if not unprocessed_videos:
        raise ValueError("All videos have been processed.")
    
    return os.path.join(parent_dir, random.choice(unprocessed_videos))

def calculate_fixed_crop_window(video_segments, original_size, crop_size):
    orig_height, orig_width = original_size
    centers = []
    empty_masks = 0
    total_masks = 0

    for frame_num in sorted(video_segments.keys()):
        mask = next(iter(video_segments[frame_num].values()))
        total_masks += 1
        y_coords, x_coords = np.where(mask[0])
        
        if len(x_coords) > 0 and len(y_coords) > 0:
            center_x = (x_coords.min() + x_coords.max()) // 2
            center_y = (y_coords.min() + y_coords.max()) // 2
            centers.append((center_x, center_y))
        else:
            empty_masks += 1
            centers.append((orig_width // 2, orig_height // 2))

    if empty_masks > 0:
        avg_center_x = sum(center[0] for center in centers) // len(centers)
        avg_center_y = sum(center[1] for center in centers) // len(centers)
        centers = [(avg_center_x, avg_center_y)] * len(centers)

    crop_windows = []
    for center_x, center_y in centers:
        left = max(0, center_x - crop_size // 2)
        top = max(0, center_y - crop_size // 2)
        right = min(orig_width, left + crop_size)
        bottom = min(orig_height, top + crop_size)
        
        # Adjust if crop window is out of bounds
        if right == orig_width:
            left = right - crop_size
        if bottom == orig_height:
            top = bottom - crop_size
        
        crop_windows.append((left, top, right, bottom))

    return crop_windows, (crop_size, crop_size), empty_masks, total_masks

def process_frames_fixed_crop(input_folder, output_folder, video_segments, original_size, crop_size):
    frame_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.jpg')])
    
    if not frame_files:
        raise ValueError("No jpg files found in the input folder")    
    os.makedirs(output_folder, exist_ok=True)
    
    crop_windows, (crop_height, crop_width), empty_masks, total_masks = calculate_fixed_crop_window(video_segments, original_size, crop_size) 
    print(f"Empty masks: {empty_masks}/{total_masks}")
    print(f"Crop size: {crop_height}x{crop_width}")
    
    for idx, frame_file in enumerate(tqdm(frame_files, desc="Processing frames")):
        frame = cv2.imread(os.path.join(input_folder, frame_file))
        left, top, right, bottom = crop_windows[idx]
        
        cropped_frame = frame[top:bottom, left:right]
        if cropped_frame.shape[:2] != (crop_height, crop_width):
            cropped_frame = cv2.resize(cropped_frame, (crop_width, crop_height))
        
        output_path = os.path.join(output_folder, frame_file)
        cv2.imwrite(output_path, cropped_frame)
    
    print(f"Cropped frames saved to: {output_folder}")
    return len(frame_files), (crop_height, crop_width)



parent_video_dir = 'PATH_TO_VIDEO_FILES_DIR'
crop_dir = 'PATH_TO_SAVE_CROPPED_VIDEOS_DIR'

random_video_dir = get_random_unprocessed_video(parent_video_dir, crop_dir)
print(f"Processing video: {random_video_dir}")

frame_names = [
    p for p in os.listdir(random_video_dir)
    if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
inference_state = predictor.init_state(video_path=random_video_dir)

points=np.array([[250, 315]], dtype=np.float32) #Generic RIA region prompt
labels=np.array([1], np.int32)
prompts = {}
ann_frame_idx = len(frame_names) - 1
ann_obj_id = 2
prompts[ann_obj_id] = points, labels
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

#Visualize prompt if needed
plt.figure(figsize=(12, 8))
plt.imshow(Image.open(os.path.join(random_video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
plt.savefig("tstclick.png")
plt.close()

video_segments = {}
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=ann_frame_idx, reverse=True):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

#Error flagging
empty_masks = {}
low_detection_masks = {}
high_detection_masks = {}
for frame, mask_dict in video_segments.items():
    for mask_id, mask in mask_dict.items():
        mask_sum = mask.sum()        
        if mask_sum == 0:
            if frame not in empty_masks:
                empty_masks[frame] = []
            empty_masks[frame].append(mask_id)
        elif mask_sum <= 200:
            if frame not in low_detection_masks:
                low_detection_masks[frame] = []
            low_detection_masks[frame].append(mask_id)
        elif mask_sum >= 5000:
            if frame not in high_detection_masks:
                high_detection_masks[frame] = []
            high_detection_masks[frame].append(mask_id)
def print_results(result_dict, condition):
    if result_dict:
        print(f"!!! Frames with masks {condition}:")
        for frame, mask_ids in result_dict.items():
            print(f"  Frame {frame}: Mask IDs {mask_ids}")
    else:
        print(f"Yay! No masks {condition} found, yay!")
print_results(empty_masks, "that are empty")
print_results(low_detection_masks, "having 200 or fewer true elements")
print_results(high_detection_masks, "having 5000 or more true elements")


output_folder = os.path.join(os.path.dirname(crop_dir), os.path.basename(random_video_dir) + "_crop")
first_frame = cv2.imread(os.path.join(random_video_dir, frame_names[0]))
original_size = first_frame.shape[:2]

process_frames_fixed_crop(random_video_dir, output_folder, video_segments, original_size, 110)