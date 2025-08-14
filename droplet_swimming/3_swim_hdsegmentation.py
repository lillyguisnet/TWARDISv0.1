"""
This script performs high-definition frame-by-frame segmentation of the worm in the video, following step 2.
"""

import os
import sys
sys.path.append("PATH_TO_CLONED_SAM2_REPO/segment-anything-2")
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from pathlib import Path
from tqdm import tqdm
import pickle
from sam2.build_sam import build_sam2_video_predictor
import shutil
import h5py
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

sam2_checkpoint = "./segment-anything-2/checkpoints/sam2_hiera_large.pt" #Checkpoint for the SAM model
model_cfg = "sam2_hiera_l.yaml" #Configuration file for the SAM model

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

def calculate_fixed_crop_window(video_segments, original_size, crop_size):
    """
    Create a fixed-sized crop around the mask
    """
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

    #If there are empty masks, use the average center of the non-empty masks and use 800x800 crop size for a double pass
    if empty_masks > 0:
        crop_size = 800
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

def process_frames_fixed_crop(input_folder, output_folder, video_segments, original_size):

    frame_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.jpg')])
    
    if not frame_files:
        raise ValueError("No jpg files found in the input folder")
    
    os.makedirs(output_folder, exist_ok=True)
    
    crop_windows, (crop_height, crop_width), empty_masks, total_masks = calculate_fixed_crop_window(video_segments, original_size, 110)
    
    print(f"Empty masks: {empty_masks}/{total_masks}")
    print(f"Crop size: {crop_height}x{crop_width}")
    
    for idx, frame_file in enumerate(tqdm(frame_files, desc="Processing frames")):
        frame = cv2.imread(os.path.join(input_folder, frame_file))
        
        left, top, right, bottom = crop_windows[idx]
        
        cropped_frame = frame[top:bottom, left:right]
        
        if cropped_frame.shape[:2] != (crop_height, crop_width):
            cropped_frame = cv2.resize(cropped_frame, (crop_width, crop_height))
        
        cv2.imwrite(os.path.join(output_folder, frame_file), cropped_frame)
    
    return len(frame_files), (crop_height, crop_width)


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


def swim_hdsegmentation(video_dir, fframe_segments_file):
    with open(fframe_segments_file, 'rb') as file:
        ffvideo_segments = pickle.load(file)
       
    #If the segmentation file is an intermediate crop, use the temp_cropdir images as base video
    if "segmentation_800.pkl" in fframe_segments_file:
        temp_cropdir = 'PATH_TO_TEMP_CROPDIR'
        temp_cropdir2 = 'PATH_TO_TEMP_CROPDIR2'
        os.makedirs(temp_cropdir2, exist_ok=True)
        # Read one frame to get original dimensions
        frame_files = sorted([f for f in os.listdir(temp_cropdir) if f.endswith('.jpg')])
        first_frame = cv2.imread(os.path.join(temp_cropdir, frame_files[0]))
        original_size = first_frame.shape[:2]
        num_frames, crop_size = process_frames_fixed_crop(temp_cropdir, temp_cropdir2, ffvideo_segments, original_size)
        print(f"Processed {num_frames} intermediate crop frames.")
        print(f"Fixed crop size again: {crop_size[1]}x{crop_size[0]}")
        shutil.rmtree(temp_cropdir)
    else:  #If it's the fframe segmentation, then use original video images    
        temp_cropdir = 'PATH_TO_TEMP_CROPDIR'
        os.makedirs(temp_cropdir, exist_ok=True)
        # Read one frame to get original dimensions
        frame_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.jpg')])
        first_frame = cv2.imread(os.path.join(video_dir, frame_files[0]))
        original_size = first_frame.shape[:2]

        num_frames, crop_size = process_frames_fixed_crop(video_dir, temp_cropdir, ffvideo_segments, original_size)
        print(f"Processed {num_frames} frames.")
        print(f"Fixed crop size: {crop_size[1]}x{crop_size[0]}")

    #Copy prompt frame to the video directory at last position based on crop size
    if "segmentation_800.pkl" in fframe_segments_file:
        prompt_frame = "PATH_TO_PROMPT_FRAME" #Second pass for HD crop size	
        shutil.copy(prompt_frame, os.path.join(temp_cropdir2, "000300.jpg"))
        temp_cropdir=temp_cropdir2         
    elif crop_size[0] == 800: 
        prompt_frame = "PATH_TO_PROMPT_FRAME" #800x800 Intermediate crop size pass if needed
        shutil.copy(prompt_frame, os.path.join(temp_cropdir, "000300.jpg"))
    else:
        prompt_frame = "PATH_TO_PROMPT_FRAME" #First time HD crop size
        shutil.copy(prompt_frame, os.path.join(temp_cropdir, "000300.jpg"))    
    
    frame_names = [
        p for p in os.listdir(temp_cropdir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    inference_state = predictor.init_state(video_path=temp_cropdir)

    #Add click on prompt frame
    ann_frame_idx = 300
    ann_obj_id = 1
    if crop_size[0] == 110:
        points = np.array([[58, 54]], dtype=np.float32) #110x110 crop size
    else:
        points = np.array([[343, 533]], dtype=np.float32) #800x800 crop size
    labels = np.array([1], np.int32)
    _, out_obj_ids, out_mask_logits = predictor.add_new_points(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )

    #Visualize prompt frame if desired
    plt.figure(figsize=(12, 8))
    plt.title(f"frame {ann_frame_idx}")
    plt.imshow(Image.open(os.path.join(temp_cropdir, frame_names[ann_frame_idx])))
    show_points(points, labels, plt.gca())
    show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
    plt.savefig("tstclick.png")
    plt.close()

    video_segments = {} #video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=ann_frame_idx, reverse=True):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    del video_segments[ann_frame_idx]

    #Error flagging: empty frames, frames with small and large predictions
    empty_frames = []
    low_detection_frames = []
    high_detection_frames = []
    for frame, obj_dict in video_segments.items():
        if all(not mask.any() for mask in obj_dict.values()):
            empty_frames.append(frame)
        elif sum(mask.sum() for mask in obj_dict.values()) <= 200:
            low_detection_frames.append(frame)    
        elif sum(mask.sum() for mask in obj_dict.values()) >= 1000:
            high_detection_frames.append(frame)   
    if empty_frames:
        print(f"!!! Empty frames: {empty_frames}")
    else:
        print("Yay! No empty frames found, yay!")
    if low_detection_frames:
        print(f"!!! Frames with 200 or fewer true elements: {low_detection_frames}")
    else:
        print("Yay! No frames with 200 or fewer true elements found, yay!")
    if high_detection_frames:
        print(f"!!! Frames with 1000 or more true elements: {high_detection_frames}")
    else:
        print("Yay! No frames with 1000 or more true elements found, yay!")


    #Save the results as h5 file
    if crop_size[0] == 110:
        shutil.rmtree(temp_cropdir)
        save_name = "PATH_TO_OUTPUT_DIR" + os.path.basename(video_dir) + "_hdsegmentation.h5"
        with h5py.File(save_name, 'w') as hf:
            for key, value in video_segments.items():
                group = hf.create_group(str(key))
                for sub_key, array in value.items():
                    group.create_dataset(str(sub_key), data=array)
        print(f"Saved! {save_name}")
    else:
        os.remove(os.path.join(temp_cropdir, "000300.jpg"))
        save_name = "PATH_TO_INTERMEDIATE_CROP_DIR" + os.path.basename(video_dir) + "_segmentation_800.pkl"
        with open(save_name, 'wb') as file:
            pickle.dump(video_segments, file)
        print(f"!!! We're in a pickle!!! {save_name}")
                

    return video_segments, crop_size[0], save_name


def get_swim_hdsegmentation(or_vid):
    #Get the fframe video directory from the original video path
    or_vid_path = Path(or_vid)
    sub_folder_name = or_vid_path.parent.name
    video_name = or_vid_path.stem
    new_folder_name = f"{sub_folder_name}-{video_name}"
    output_dir = Path("PATH_TO_OUTPUT_DIR")
    ffvideo_dir = str(output_dir / new_folder_name)

    #Infer fframe_segments_file for the video
    fframe_segments_dir = Path("PATH_TO_FFRAME_SEGMENTATIONS_DIR")
    fframe_segments_file = str(fframe_segments_dir / f"{new_folder_name}_fframe_segmentation.pkl")

    video_segments, crop_size, save_name = swim_hdsegmentation(ffvideo_dir, fframe_segments_file)

    if crop_size != 110:
        video_segments, crop_size, save_name = swim_hdsegmentation(ffvideo_dir, save_name)

    return video_segments, crop_size, save_name


or_vid = 'PATH_TO_VIDEO'

video_segments, crop_size, save_name = get_swim_hdsegmentation(or_vid)