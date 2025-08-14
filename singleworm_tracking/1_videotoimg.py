"""
This script converts videos to images for the SAM model with the option to parallelize the process.
"""
import cv2
from pathlib import Path
import re
from tqdm import tqdm
from multiprocessing import Pool
import os
import random

def set_cpu_affinity():
    """Set CPU affinity to use only selected cores"""
    try:
        allowed_cores = {0, 2, 4, 6}
        os.sched_setaffinity(0, allowed_cores)
    except (OSError, AttributeError):
        pass

def get_supported_video_extensions():
    return ['.avi', '.mp4', '.mov', '.mkv', '.wmv', '.flv']

def check_video_readability(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, "OpenCV cannot open the video file. The format might be unsupported or the file might be corrupted."
    
    ret, frame = cap.read()
    if not ret:
        return False, "OpenCV can open the file, but cannot read frames from it. The video might be empty or corrupted."
    
    cap.release()
    return True, ""

def get_supported_image_extensions():
    return ['.png', '.jpg', '.jpeg']

def process_single_frame(args):
    set_cpu_affinity()
    
    frame_number, video_path, video_output_dir, output_format, new_width, new_height, orig_width, orig_height = args
    
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None
    
    if (new_width, new_height) != (orig_width, orig_height):
        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    frame_path = video_output_dir / f"{frame_number:06d}.{output_format}"
    
    if output_format == 'jpg':
        cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:  # png
        cv2.imwrite(str(frame_path), frame)
    
    return str(frame_path)

def process_video_for_sam2(video_path, output_dir, fps=None, max_dimension=None, output_format='jpg', force_reprocess=False, convert_existing=False, num_processes=None):
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    
    if num_processes is None:
        num_processes = 1
    
    if output_format not in ['png', 'jpg']:
        raise ValueError("output_format must be either 'png' or 'jpg'")
    
    if video_path.suffix.lower() not in get_supported_video_extensions():
        print(f"Warning: The file extension {video_path.suffix} might not be supported. Attempting to process anyway.")
    
    is_readable, error_message = check_video_readability(video_path)
    if not is_readable:
        raise ValueError(f"Cannot process video: {error_message}")
    
    sub_folder_name = video_path.parent.name
    video_name = video_path.stem
    new_folder_name = f"{sub_folder_name}-{video_name}"
    video_output_dir = output_dir / new_folder_name
    video_output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    cap.release()
    
    if max_dimension and (orig_width > max_dimension or orig_height > max_dimension):
        scale = max_dimension / max(orig_width, orig_height)
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)
    else:
        new_width, new_height = orig_width, orig_height
    
    if fps is None:
        fps = video_fps
    
    frame_interval = max(int(video_fps / fps), 1)
    
    existing_frames = []
    for ext in get_supported_image_extensions():
        existing_frames.extend(video_output_dir.glob(f"*{ext}"))
    existing_frames.sort(key=lambda x: int(re.search(r'\d+', x.stem).group()))
    
    existing_frame_numbers = [int(re.search(r'\d+', f.stem).group()) for f in existing_frames]
    expected_frame_numbers = set(range(0, total_frames, frame_interval))
    
    inconsistencies = []
    
    if existing_frames and not force_reprocess:
        missing_frame_numbers = sorted(expected_frame_numbers - set(existing_frame_numbers))
        if missing_frame_numbers:
            inconsistencies.append(f"Found {len(missing_frame_numbers)} missing frames. They will be processed.")
        extra_frame_numbers = sorted(set(existing_frame_numbers) - expected_frame_numbers)
        if extra_frame_numbers:
            inconsistencies.append(f"Found {len(extra_frame_numbers)} unexpected frames. They will be ignored.")
    else:
        missing_frame_numbers = sorted(expected_frame_numbers)
        if force_reprocess:
            inconsistencies.append("Reprocessing all frames as requested.")
        else:
            print("Processing new video.")
    
    frame_paths = []
    new_frames = 0
    converted_frames = 0
    existing_frames_count = 0
    
    if force_reprocess:
        print(f"Reprocessing all frames using {num_processes} processes")
        frames_to_process = sorted(expected_frame_numbers)
        
        process_args = [(fn, video_path, video_output_dir, output_format, new_width, new_height, orig_width, orig_height) 
                       for fn in frames_to_process]
        
        with Pool(processes=num_processes) as pool:
            results = list(tqdm(pool.imap(process_single_frame, process_args), 
                               total=len(process_args), desc="Processing frames"))
        
        frame_paths = [path for path in results if path is not None]
        new_frames = len(frame_paths)
        
    else:
        existing_frame_dict = {int(re.search(r'\d+', f.stem).group()): f for f in existing_frames}
        frames_to_process = []
        
        for fn in sorted(expected_frame_numbers):
            if fn in existing_frame_dict:
                existing_frame = existing_frame_dict[fn]
                if existing_frame.suffix[1:] != output_format:
                    if convert_existing:
                        img = cv2.imread(str(existing_frame))
                        new_frame_path = existing_frame.with_suffix(f".{output_format}")
                        cv2.imwrite(str(new_frame_path), img)
                        existing_frame.unlink()
                        frame_paths.append(str(new_frame_path))
                        converted_frames += 1
                    else:
                        frame_paths.append(str(existing_frame))
                        existing_frames_count += 1
                else:
                    frame_paths.append(str(existing_frame))
                    existing_frames_count += 1
            else:
                frames_to_process.append(fn)
        
        if frames_to_process:
            print(f"Processing {len(frames_to_process)} new frames using {num_processes} processes")
            process_args = [(fn, video_path, video_output_dir, output_format, new_width, new_height, orig_width, orig_height) 
                           for fn in frames_to_process]
            
            with Pool(processes=num_processes) as pool:
                results = list(tqdm(pool.imap(process_single_frame, process_args), 
                                   total=len(process_args), desc="Processing new frames"))
            
            new_frame_paths = [path for path in results if path is not None]
            frame_paths.extend(new_frame_paths)
            new_frames = len(new_frame_paths)
            
            failed_frames = len(frames_to_process) - len(new_frame_paths)
            if failed_frames > 0:
                inconsistencies.append(f"Failed to process {failed_frames} frames")
    
    frame_paths.sort(key=lambda x: int(re.search(r'\d+', Path(x).stem).group()))
    
    return frame_paths, new_height, new_width, inconsistencies, {
        'new_frames': new_frames,
        'converted_frames': converted_frames,
        'existing_frames': existing_frames_count
    }

def process_video(video_path, output_dir, num_processes=None):
    try:
        frame_paths, video_height, video_width, inconsistencies, frame_stats = process_video_for_sam2(
            video_path, output_dir, output_format='jpg', num_processes=num_processes)

        print(f"Frame processing summary:")
        print(f"- New frames: {frame_stats['new_frames']}")
        print(f"- Converted frames: {frame_stats['converted_frames']}")
        print(f"- Existing frames (unchanged): {frame_stats['existing_frames']}")
        print(f"Total frames: {len(frame_paths)}")
        print(f"Video dimensions: {video_width}x{video_height}")
        print(f"Frames saved in: {Path(output_dir) / Path(video_path).stem}")
        if inconsistencies:
            print("Inconsistencies found:")
            for inc in inconsistencies:
                print(f"- {inc}")
        else:
            print("No inconsistencies found.")
    
    except ValueError as e:
        print(f"Error processing video: {e}")

def get_random_unprocessed_video(original_videos_dir, segmented_videos_dir):
    original_path = Path(original_videos_dir)
    
    all_videos = [f for f in os.listdir(original_videos_dir) 
                  if os.path.isfile(os.path.join(original_videos_dir, f)) 
                  and Path(f).suffix.lower() in get_supported_video_extensions()]
    
    unprocessed_videos = []
    for video in all_videos:
        video_path = original_path / video
        sub_folder_name = video_path.parent.name
        video_name = video_path.stem
        expected_output_dir = f"{sub_folder_name}-{video_name}"
        
        if not os.path.exists(os.path.join(segmented_videos_dir, expected_output_dir)):
            unprocessed_videos.append(video)
    
    if not unprocessed_videos:
        raise ValueError("All videos have been processed.")
    
    return os.path.join(original_videos_dir, random.choice(unprocessed_videos))


original_videos_dir = "PATH_TO_ORIGINAL_VIDEOS_DIR"
output_dir = "PATH_TO_OUTPUT_DIR"

video_path = get_random_unprocessed_video(original_videos_dir, output_dir)
print(f"Processing video: {video_path}")

process_video(video_path, output_dir, num_processes=24)