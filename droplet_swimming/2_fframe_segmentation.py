"""
This script performs high-level frame-by-frame segmentation of a video using SAM
A generic prompt frame is used to segment each video. This avoids the need to prompt each video individually.
The segmentation for each frame is saved as a pickle file for use in step 3.
"""

import os
import sys
sys.path.append("PATH_TO_CLONED_SAM2_REPO/segment-anything-2")
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
import shutil
from pathlib import Path
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
from sam2.build_sam import build_sam2_video_predictor

sam2_checkpoint = "./segment-anything-2/checkpoints/sam2_hiera_large.pt" #Checkpoint for the SAM model
model_cfg = "sam2_hiera_l.yaml" #Configuration file for the SAM model

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



def segment_fframe_video_withaddedprompt(video_dir):
    #Copy prompt frame to the video directory at last position
    prompt_frame = "PATH_TO_GENERIC_PROMPT_FRAME"
    shutil.copy(prompt_frame, os.path.join(video_dir, "000300.jpg"))

    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    inference_state = predictor.init_state(video_path=video_dir)

    ###Add click to prompt frame
    ann_frame_idx = 300  #frame index
    ann_obj_id = 1  #object id
    points = np.array([[1150, 1175]], dtype=np.float32) #worm on prompt frame
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
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
    show_points(points, labels, plt.gca())
    show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
    plt.savefig("tstclick.png")
    plt.close()

    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=ann_frame_idx, reverse=True):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }
        
    #Remove the prompt frame
    os.remove(os.path.join(video_dir, "000300.jpg"))

    #Error flagging: empty frames, frames with small and large predictions
    empty_frames = []
    low_detection_frames = []
    high_detection_frames = []
    for frame, obj_dict in video_segments.items():
        if all(not mask.any() for mask in obj_dict.values()):
            empty_frames.append(frame)
        elif sum(mask.sum() for mask in obj_dict.values()) <= 200:
            low_detection_frames.append(frame)    
        elif sum(mask.sum() for mask in obj_dict.values()) >= 5000:
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
        print(f"!!! Frames with 5000 or more true elements: {high_detection_frames}")
    else:
        print("Yay! No frames with 5000 or more true elements found, yay!")

    #Save the results
    save_name = "PATH_TO_OUTPUT_DIR" + os.path.basename(video_dir)
    with open(save_name + '_fframe_segmentation.pkl', 'wb') as file:
        pickle.dump(video_segments, file)

    return video_segments


video_dir = 'PATH_TO_VIDEO'

video_segments = segment_fframe_video_withaddedprompt(video_dir)