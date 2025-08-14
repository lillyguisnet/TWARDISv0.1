"""
This script is used for path analysis of the worm trajectory.
There are several visualization helper functions.
"""

import numpy as np
from scipy.ndimage import center_of_mass
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial.distance import euclidean
from scipy.signal import savgol_filter
from matplotlib.patches import Patch
from scipy.signal import medfilt
import os
import cv2
import random
import traceback
import time

# region [functions]

def get_all_unprocessed_videos(shape_analysis_dir, path_analysis_dir):
    """
    Get all unprocessed videos that need path analysis.
    Returns: List of tuples: (video_path, video_filename) for all unprocessed videos
    """
    all_videos = [f for f in os.listdir(shape_analysis_dir) if f.endswith('.pkl')]
    unprocessed_videos = [
        video for video in all_videos
        if not os.path.exists(os.path.join(path_analysis_dir, video.replace('.pkl', '_pathanalysis.pkl')))
    ]
    
    if not unprocessed_videos:
        print("All videos have been processed.")
        return []
    
    print(f"Found {len(unprocessed_videos)} unprocessed videos out of {len(all_videos)} total videos")
    
    return [(os.path.join(shape_analysis_dir, video), video) for video in unprocessed_videos]

def process_single_video(video_path, video_filename, plots_directory, path_analysis_dir, frames_dir, create_videos=True):
    try:
        print(f"\n{'='*50}")
        print(f"PROCESSING VIDEO: {video_filename}")
        print(f"{'='*50}")
        
        with open(video_path, 'rb') as file:
            hdshape = pickle.load(file)
        
        time.sleep(0.2)
        print(f"Loaded video data with {len(hdshape['frames'])} frames")
        
        centroids = {}
        frames = hdshape['frames']
        masks = hdshape['masks']
        
        print("Calculating centroids...")
        for i, frame_num in enumerate(frames):
            if i % 100 == 0:
                print(f"  Processing frame {i+1}/{len(frames)}")
            mask = masks[i]
            centroids[frame_num] = get_centroid(mask)
        smooth_centroids = smooth_path(centroids)   
        
        print(f"Calculated centroids for {len(centroids)} frames")
        
        original_total_distance, original_max_distance = calculate_path_metrics(centroids)
        smooth_total_distance, smooth_max_distance = calculate_path_metrics(smooth_centroids)        
        print(f"Path metrics - Original distance: {original_total_distance:.2f}, Smooth distance: {smooth_total_distance:.2f}")
        
        plot_paths = generate_plot_paths(video_path, plots_directory)
        time.sleep(0.2)
        
        filename = os.path.basename(video_path)        
        plot_paths_with_time_gradient(centroids, smooth_centroids, plot_paths['path_gradient'], filename,
                                    original_total_distance, original_max_distance, 
                                    smooth_total_distance, smooth_max_distance)
        time.sleep(0.2)
        
        # Main movement analysis
        print("Starting movement analysis...")
        results = analyze_worm_movement(centroids, hdshape)        
        # Head/tail corrections
        if results.get('localized_head_tail_corrections', False):
            print(f"\n*** LOCALIZED HEAD/TAIL CORRECTIONS APPLIED ***")
            print(f"Corrected head/tail orientation for {results.get('localized_corrections', 0)}")
        
        if results.get('head_tail_swapped', False):
            print("\n*** GLOBAL HEAD/TAIL SWAP WAS APPLIED ***")
        else:
            print("\nGlobal head/tail orientation analysis: No swap needed")
        
        # Store original classification and metrics for comparison
        original_classification = results['movement_classification'].copy()
        original_metrics = {
            'original_forward_frames': results['forward_frames'],
            'original_backward_frames': results['backward_frames'],
            'original_stationary_frames': results['stationary_frames'],
            'original_forward_bouts': results['forward_bouts'],
            'original_backward_bouts': results['backward_bouts'],
            'original_avg_forward_speed': results['avg_forward_speed'],
            'original_avg_backward_speed': results['avg_backward_speed'],
            'original_avg_forward_bout_length_frames': results['avg_forward_bout_length_frames'],
            'original_avg_backward_bout_length_frames': results['avg_backward_bout_length_frames'],
            'original_avg_forward_bout_length_pixels': results['avg_forward_bout_length_pixels'],
            'original_avg_backward_bout_length_pixels': results['avg_backward_bout_length_pixels']
        }
        results.update(original_metrics)
        
        print("\n=== APPLYING MOVEMENT CORRECTIONS ===")
        corrected_classification = early_frame_head_tail_correction(
            results['movement_classification'], 
            results['smooth_centroids'], 
            early_frame_count=10,
            confidence_threshold=0.7
        )
        
        max_iterations = 5
        total_corrections = 0
        for iteration in range(max_iterations):
            print(f"Comprehensive correction iteration {iteration + 1}...")
            prev_classification = corrected_classification.copy()
            
            corrected_classification = comprehensive_movement_correction(
                corrected_classification, 
                results['smooth_centroids'],
                window_size=20,
                forward_angle_threshold=15,
                backward_angle_threshold=165
            )
            
            changes = sum(1 for f in corrected_classification.keys() 
                         if corrected_classification[f] != prev_classification[f])
            
            if changes == 0:
                print(f"Converged after {iteration + 1} iterations")
                break
            else:
                print(f"Made {changes} additional changes in this iteration")
                total_corrections += changes
        
        print("Applying segment-level correction...")
        corrected_classification = segment_level_correction(corrected_classification, results['smooth_centroids'])
        
        print("Applying local consistency correction...")
        corrected_classification = local_consistency_correction(corrected_classification, results['smooth_centroids'])
        
        print("Applying final stationary correction...")
        corrected_classification = final_stationary_correction(corrected_classification, results['smooth_centroids'], speed_threshold=0.6)
        
        total_changes = sum(1 for f in results['movement_classification'].keys() 
                           if corrected_classification[f] != results['movement_classification'][f])        
        print(f"Total changes made from original classification: {total_changes}")
        
        results['movement_classification'] = corrected_classification
        results['original_classification'] = original_classification
        results['corrected'] = True
        
        print("Recalculating metrics with corrected classifications...")
        results = recalculate_metrics_after_correction(results, corrected_classification)
        
        print("Creating analysis plots...")
        plot_correction_comparison(original_classification, corrected_classification, 
                                 results['smooth_centroids'], plot_paths['correction_comparison'], video_filename)
        
        plot_head_tail_analysis(results['corrected_aligned_data'], results['smooth_centroids'], 
                               results['head_tail_swapped'], plot_paths['head_tail_analysis'], video_filename)
        
        plot_worm_path_with_metrics(results, plot_paths['metrics_speed'], video_filename)
        
        # Create movement video (optional)
        video_output_path = os.path.join(plots_directory, f"{video_filename[:-4]}_movement_analysis.mp4")
        if create_videos:
            print("Creating movement video...")
            frames_directory = get_frames_directory_from_shape_path(video_path, frames_dir)
            create_movement_video(results, frames_directory, hdshape, video_output_path, fps=10, scale_factor=0.5)
        else:
            print("Skipping video creation (disabled)")
        
        print("Saving analysis results...")
        saved_pickle_path = save_path_analysis_results(results, video_path, path_analysis_dir)
        time.sleep(0.2)
        
        print(f"✓ Successfully processed {video_filename}")
        print(f"  - Results saved to: {saved_pickle_path}")
        print(f"  - Plots saved to: {plots_directory}")
        if create_videos:
            print(f"  - Video saved to: {video_output_path}")
        else:
            print(f"  - Video creation skipped")
        
        del hdshape
        del centroids
        del results
        
        return True, None
        
    except Exception as e:
        error_message = f"Error processing {video_filename}: {str(e)}"
        print(f"✗ {error_message}")
        print("Full traceback:")
        traceback.print_exc()
        return False, error_message

def get_random_unprocessed_video(shape_analysis_dir, path_analysis_dir):
    all_videos = [f for f in os.listdir(shape_analysis_dir) if f.endswith('.pkl')]
    unprocessed_videos = [
        video for video in all_videos
        if not os.path.exists(os.path.join(path_analysis_dir, video.replace('.pkl', '_pathanalysis.pkl')))
    ]
    
    if not unprocessed_videos:
        raise ValueError("All videos have been processed.")
    
    selected_video = random.choice(unprocessed_videos)
    video_path = os.path.join(shape_analysis_dir, selected_video)
    print(f"Processing video: {video_path}")
    
    with open(video_path, 'rb') as file:
        video_data = pickle.load(file)
    
    return video_path, video_data

def get_centroid(mask):
    if mask.ndim == 3:
        mask = mask.squeeze()
    cy, cx = center_of_mass(mask)
    return (int(cx), int(cy))

def smooth_path(centroids, window_length=11, poly_order=3):
    frames = sorted(centroids.keys())
    x_coords = [centroids[f][0] for f in frames]
    y_coords = [centroids[f][1] for f in frames]
    
    x_smooth = savgol_filter(x_coords, window_length, poly_order)
    y_smooth = savgol_filter(y_coords, window_length, poly_order)
    
    return {f: (x, y) for f, x, y in zip(frames, x_smooth, y_smooth)}

def calculate_path_metrics(centroids):
    sorted_frames = sorted(centroids.keys())
    coordinates = np.array([centroids[frame] for frame in sorted_frames])
    
    distances = np.sqrt(np.sum(np.diff(coordinates, axis=0)**2, axis=1))
    total_distance = np.sum(distances)
    
    start_point = coordinates[0]
    distances_from_start = np.sqrt(np.sum((coordinates - start_point)**2, axis=1))
    max_distance_from_start = np.max(distances_from_start)
    
    return total_distance, max_distance_from_start

def plot_paths_with_time_gradient(original, filtered, save_path=None, filename=None, 
                                  original_total_distance=None, original_max_distance=None, 
                                  smooth_total_distance=None, smooth_max_distance=None):
    """
    Plot the original and smoothed paths with a time gradient for visualization.
    """
    fig, ax = plt.subplots(figsize=(14, 14))
    
    colors = ['purple', 'blue', 'cyan', 'green', 'yellow', 'red']
    n_bins = 6
    cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)
    
    sorted_frames = sorted(original.keys())
    x_orig, y_orig = zip(*[original[frame] for frame in sorted_frames])
    x_filt, y_filt = zip(*[filtered[frame] for frame in sorted_frames])
    
    for i in range(n_bins):
        start = i * len(sorted_frames) // n_bins
        end = (i + 1) * len(sorted_frames) // n_bins
        ax.plot(x_orig[start:end], y_orig[start:end], color=cmap(i/n_bins), alpha=0.5, linewidth=2)
    
    for i in range(n_bins):
        start = i * len(sorted_frames) // n_bins
        end = (i + 1) * len(sorted_frames) // n_bins
        ax.plot(x_filt[start:end], y_filt[start:end], color=cmap(i/n_bins), alpha=0.9, linewidth=2)
    
    title = "Worm Path: Original vs Filtered (Padded Moving Average)\nColor shows time progression"
    if filename:
        title = f"{filename}\n{title}"
    ax.set_title(title)
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.invert_yaxis()
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=600))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Frame number')
    cbar.set_ticks([50 + i*100 for i in range(6)])
    cbar.set_ticklabels([f'{i*100}-{(i+1)*100}' for i in range(6)])
    
    ax.plot([], [], color='black', alpha=0.5, linewidth=2, label='Original')
    ax.plot([], [], color='black', alpha=0.5, linewidth=2, label='Filtered')
    ax.legend()

    ax.plot(x_orig[0], y_orig[0], 'go', markersize=10, label='Start')
    ax.plot(x_orig[-1], y_orig[-1], 'ro', markersize=10, label='End')
    
    # Highlight furthest point
    distances_from_start = [euclidean(original[sorted_frames[0]], original[frame]) for frame in sorted_frames]
    furthest_frame = sorted_frames[np.argmax(distances_from_start)]
    furthest_point = original[furthest_frame]
    ax.plot(furthest_point[0], furthest_point[1], 'yo', markersize=10, label='Furthest Point')
    
    # Add metrics to the plot
    ax.text(0.05, 0.95, f"Original Total Distance: {original_total_distance:.2f} px", transform=ax.transAxes, verticalalignment='top')
    ax.text(0.05, 0.92, f"Original Max Distance: {original_max_distance:.2f} px", transform=ax.transAxes, verticalalignment='top')
    ax.text(0.05, 0.89, f"Smooth Total Distance: {smooth_total_distance:.2f} px", transform=ax.transAxes, verticalalignment='top')
    ax.text(0.05, 0.86, f"Smooth Max Distance: {smooth_max_distance:.2f} px", transform=ax.transAxes, verticalalignment='top')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Path plot saved to: {save_path}")
    else:
        plt.savefig('plotpath2.png', dpi=300, bbox_inches='tight')
    
    plt.show()
    plt.close()

def generate_plot_paths(original_shape_path, plots_dir):
    original_basename = os.path.basename(original_shape_path)
    
    if original_basename.endswith('.pkl'):
        base_name = original_basename[:-4]
    else:
        base_name = original_basename    
    os.makedirs(plots_dir, exist_ok=True)
    
    paths = {
        'path_gradient': os.path.join(plots_dir, f"{base_name}_path_gradient.png"),
        'correction_comparison': os.path.join(plots_dir, f"{base_name}_correction_comparison.png"),
        'metrics_speed': os.path.join(plots_dir, f"{base_name}_metrics_speed.png"),
        'head_tail_analysis': os.path.join(plots_dir, f"{base_name}_head_tail_analysis.png")
    }
    
    return paths

def calculate_velocity(smooth_centroids):
    """Calculate velocity using central differences (in pixels per frame)"""
    frames = sorted(smooth_centroids.keys())
    positions = np.array([smooth_centroids[f] for f in frames])
    
    velocity = np.gradient(positions, axis=0)
    
    v_x=velocity[:, 0]
    v_y=velocity[:, 1]
    
    return {f: (vx, vy) for f, vx, vy in zip(frames, v_x, v_y)}

def align_data(full_frame_centroids, cropped_analysis):
    """Align the full frame centroids with the cropped video analysis."""
    aligned_data = {}
    for frame in full_frame_centroids.keys():
        if frame in cropped_analysis['frames']:
            aligned_data[frame] = {
                'centroid': full_frame_centroids[frame],
                'head_bend': cropped_analysis['smoothed_head_bends'][cropped_analysis['frames'].index(frame)],
                'smooth_points': cropped_analysis['smooth_points'][cropped_analysis['frames'].index(frame)],
                'head_coord': cropped_analysis['head_positions'][cropped_analysis['frames'].index(frame)],
                'tail_coord': cropped_analysis['tail_positions'][cropped_analysis['frames'].index(frame)]
            }
    return aligned_data

def calculate_orientation_vector(smooth_points, head, tail):
    """Calculate the orientation vector of the worm."""
    segment_point = smooth_points[len(smooth_points) // 10]
    return np.array(head) - np.array(segment_point)

def calculate_movement_features(velocities, orientations, window_size=5):
    speed = np.linalg.norm(velocities, axis=1)
    
    velocity_norms = np.linalg.norm(velocities, axis=1, keepdims=True)
    orientation_norms = np.linalg.norm(orientations, axis=1, keepdims=True)
    
    epsilon = 1e-10
    velocities_norm = velocities / np.maximum(velocity_norms, epsilon)
    orientations_norm = orientations / np.maximum(orientation_norms, epsilon)
    
    # Compute the dot product and get the angles in degrees
    dot_products = np.einsum('ij,ij->i', velocities_norm, orientations_norm)
    
    # Handle cases where original vectors had zero magnitude
    # These represent stationary periods or orientation detection issues
    zero_velocity_mask = (velocity_norms.flatten() < epsilon)
    zero_orientation_mask = (orientation_norms.flatten() < epsilon)
    invalid_mask = zero_velocity_mask | zero_orientation_mask
    
    angles = np.degrees(np.arccos(np.clip(dot_products, -1.0, 1.0)))
    
    for i in range(len(angles)):
        if invalid_mask[i]:
            for j in range(i-1, -1, -1):
                if not invalid_mask[j]:
                    angles[i] = angles[j]
                    break
            else:
                angles[i] = 90.0
    
    smooth_speed = medfilt(speed, kernel_size=window_size)
    smooth_angles = medfilt(angles, kernel_size=window_size)
    
    return smooth_speed, smooth_angles

def classify_movement_window(speeds, angles, speed_threshold=0, angle_threshold=100):
    """First classification pass"""
    if np.mean(speeds) < speed_threshold:
        return 'stationary'
    elif np.mean(np.abs(angles)) < angle_threshold:
        return 'forward'
    else:
        return 'backward'

def analyze_worm_movement(full_frame_centroids, cropped_analysis, fps=10, window_size=5):
    smooth_centroids = smooth_path(full_frame_centroids)
    velocities = calculate_velocity(smooth_centroids, fps)
    aligned_data = align_data(smooth_centroids, cropped_analysis)
    
    # Detect and correct localized head/tail issues based on path analysis
    problematic_segments, segment_analysis = detect_localized_head_tail_issues(aligned_data, smooth_centroids)    
    if problematic_segments:
        print(f"Applying localized head/tail corrections to {len(problematic_segments)} segments")
        aligned_data, localized_corrections = correct_localized_head_tail_issues(aligned_data, problematic_segments)
        localized_head_tail_corrections = True
    else:
        localized_head_tail_corrections = False
        localized_corrections = 0
        aligned_data, head_tail_swapped = detect_and_correct_head_tail_swap(aligned_data, smooth_centroids)
    
    frames = sorted(aligned_data.keys())
    velocity_vectors = np.array([velocities[f] for f in frames])
    orientation_vectors = np.array([calculate_orientation_vector(aligned_data[f]['smooth_points'], aligned_data[f]['head_coord'], aligned_data[f]['tail_coord']) for f in frames])
    speeds, angles = calculate_movement_features(velocity_vectors, orientation_vectors, window_size)
    
    movement_classification = {}
    forward_velocities = []
    backward_velocities = []
    forward_accelerations = []
    backward_accelerations = []
    per_frame_speeds = {}
    per_frame_accelerations = {}
    
    # Bout analysis
    forward_bouts = 0
    backward_bouts = 0
    stationary_bouts = 0
    current_bout_type = None
    bout_lengths_frames = {'forward': [], 'backward': [], 'stationary': []}
    bout_lengths_pixels = {'forward': [], 'backward': [], 'stationary': []}
    current_bout_length_frames = 0
    current_bout_length_pixels = 0.0
    current_bout_start_frame_idx = 0
    
    start_point = np.array(aligned_data[frames[0]]['centroid'])
    max_distance = 0
    furthest_frame = frames[0]
    
    for i, frame in enumerate(frames):
        start = max(0, i - window_size // 2)
        end = min(len(frames), i + window_size // 2 + 1)
        movement_type = classify_movement_window(speeds[start:end], angles[start:end])
        movement_classification[frame] = movement_type
        
        per_frame_speeds[frame] = np.linalg.norm(velocity_vectors[i])
        
        if current_bout_type != movement_type:
            if current_bout_type is not None:
                bout_lengths_frames[current_bout_type].append(current_bout_length_frames)
                bout_lengths_pixels[current_bout_type].append(current_bout_length_pixels)
            
            if movement_type == 'forward':
                forward_bouts += 1
            elif movement_type == 'backward':
                backward_bouts += 1
            elif movement_type == 'stationary':
                stationary_bouts += 1
                
            current_bout_type = movement_type
            current_bout_length_frames = 1
            current_bout_length_pixels = 0.0
            current_bout_start_frame_idx = i
        else:
            current_bout_length_frames += 1
            
        if i > 0:
            current_pos = np.array(aligned_data[frame]['centroid'])
            prev_pos = np.array(aligned_data[frames[i-1]]['centroid'])
            frame_distance = np.linalg.norm(current_pos - prev_pos)
            current_bout_length_pixels += frame_distance
        
        if movement_type == 'forward':
            forward_velocities.append(velocity_vectors[i])
        elif movement_type == 'backward':
            backward_velocities.append(velocity_vectors[i])
        
        # Furthest point
        current_point = np.array(aligned_data[frame]['centroid'])
        distance = np.linalg.norm(current_point - start_point)
        if distance > max_distance:
            max_distance = distance
            furthest_frame = frame
    
    if current_bout_type is not None:
        bout_lengths_frames[current_bout_type].append(current_bout_length_frames)
        bout_lengths_pixels[current_bout_type].append(current_bout_length_pixels)
    
    acceleration = np.gradient(velocity_vectors, axis=0)
    
    # Store per-frame accelerations and collect by movement type
    for i, frame in enumerate(frames):
        per_frame_accelerations[frame] = acceleration[i]  # Store as [ax, ay] vector
        
        if movement_classification[frame] == 'forward':
            forward_accelerations.append(acceleration[i])
        elif movement_classification[frame] == 'backward':
            backward_accelerations.append(acceleration[i])
    
    forward_frames = sum(1 for v in movement_classification.values() if v == 'forward')
    backward_frames = sum(1 for v in movement_classification.values() if v == 'backward')
    stationary_frames = sum(1 for v in movement_classification.values() if v == 'stationary')
    
    # Calculate additional metrics
    total_frames = len(aligned_data)
    positions = np.array([aligned_data[f]['centroid'] for f in sorted(aligned_data.keys())])
    total_distance = np.sum(np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1)))
    avg_speed = total_distance / total_frames
    
    # Calculate sinuosity (total distance / straight-line distance)
    start_point = positions[0]
    end_point = positions[-1]
    straight_line_distance = np.linalg.norm(end_point - start_point)
    sinuosity = total_distance / straight_line_distance if straight_line_distance > 0 else 0
    
    # Calculate average velocity and acceleration for all movements
    avg_velocity = np.mean(velocity_vectors, axis=0)
    avg_acceleration = np.mean(acceleration, axis=0)
    
    avg_forward_velocity = np.mean(forward_velocities, axis=0) if forward_velocities else np.array([0, 0])
    avg_backward_velocity = np.mean(backward_velocities, axis=0) if backward_velocities else np.array([0, 0])
    avg_forward_acceleration = np.mean(forward_accelerations, axis=0) if forward_accelerations else np.array([0, 0])
    avg_backward_acceleration = np.mean(backward_accelerations, axis=0) if backward_accelerations else np.array([0, 0])
    
    avg_forward_speed = np.mean(np.linalg.norm(forward_velocities, axis=1)) if forward_velocities else 0
    avg_backward_speed = np.mean(np.linalg.norm(backward_velocities, axis=1)) if backward_velocities else 0
    
    avg_forward_bout_length_frames = np.mean(bout_lengths_frames['forward']) if bout_lengths_frames['forward'] else 0
    avg_backward_bout_length_frames = np.mean(bout_lengths_frames['backward']) if bout_lengths_frames['backward'] else 0
    avg_stationary_bout_length_frames = np.mean(bout_lengths_frames['stationary']) if bout_lengths_frames['stationary'] else 0
    
    avg_forward_bout_length_pixels = np.mean(bout_lengths_pixels['forward']) if bout_lengths_pixels['forward'] else 0
    avg_backward_bout_length_pixels = np.mean(bout_lengths_pixels['backward']) if bout_lengths_pixels['backward'] else 0
    avg_stationary_bout_length_pixels = np.mean(bout_lengths_pixels['stationary']) if bout_lengths_pixels['stationary'] else 0
    
    return {
        'forward_frames': forward_frames,
        'backward_frames': backward_frames,
        'stationary_frames': stationary_frames,
        'total_frames': total_frames,
        'total_distance': total_distance,
        'avg_speed': avg_speed,
        'sinuosity': sinuosity,
        'avg_velocity': avg_velocity,
        'avg_acceleration': avg_acceleration,
        'avg_forward_velocity': avg_forward_velocity,
        'avg_backward_velocity': avg_backward_velocity,
        'avg_forward_acceleration': avg_forward_acceleration,
        'avg_backward_acceleration': avg_backward_acceleration,
        'avg_forward_speed': avg_forward_speed,
        'avg_backward_speed': avg_backward_speed,
        'movement_classification': movement_classification,
        'smooth_centroids': smooth_centroids,
        'velocities': velocities,
        'per_frame_speeds': per_frame_speeds,
        'per_frame_accelerations': per_frame_accelerations,
        'forward_bouts': forward_bouts,
        'backward_bouts': backward_bouts,
        'stationary_bouts': stationary_bouts,
        'bout_lengths_frames': bout_lengths_frames,
        'bout_lengths_pixels': bout_lengths_pixels,
        'avg_forward_bout_length_frames': avg_forward_bout_length_frames,
        'avg_backward_bout_length_frames': avg_backward_bout_length_frames,
        'avg_stationary_bout_length_frames': avg_stationary_bout_length_frames,
        'avg_forward_bout_length_pixels': avg_forward_bout_length_pixels,
        'avg_backward_bout_length_pixels': avg_backward_bout_length_pixels,
        'avg_stationary_bout_length_pixels': avg_stationary_bout_length_pixels,
        'furthest_point_distance': max_distance,
        'furthest_point_frame': furthest_frame,
        'head_tail_swapped': head_tail_swapped,
        'corrected_aligned_data': aligned_data,
        'localized_head_tail_corrections': localized_head_tail_corrections,
        'localized_corrections': localized_corrections
    }

def comprehensive_movement_correction(movement_classification, smooth_centroids, window_size=10, 
                                    forward_angle_threshold=45, backward_angle_threshold=135):
    """
    Fix movement classification based on path analysis.
    """
    frames = sorted(movement_classification.keys())
    corrected_classification = movement_classification.copy()
    corrections_made = {'backward_to_forward': 0, 'forward_to_backward': 0}
    
    print(f"Analyzing {len(frames)} frames with window_size={window_size}")
    
    for i, frame in enumerate(frames):
        current_type = corrected_classification[frame]
        
        if current_type == 'stationary':
            continue
            
        if i == 0:
            continue
        current_pos = np.array(smooth_centroids[frame])
        prev_pos = np.array(smooth_centroids[frames[i-1]])
        current_direction = current_pos - prev_pos
        
        if np.linalg.norm(current_direction) < 0.1:  # Skip if movement is too small
            continue
            
        current_direction_norm = current_direction / np.linalg.norm(current_direction)
            
        recent_forward_directions = []
        lookback_start = max(0, i - window_size)
        
        is_near_end = i > len(frames) - 10
        
        for j in range(lookback_start, i):
            if corrected_classification[frames[j]] == 'forward':
                if j > 0:
                    pos_j = np.array(smooth_centroids[frames[j]])
                    pos_j_prev = np.array(smooth_centroids[frames[j-1]])
                    direction_j = pos_j - pos_j_prev
                    if np.linalg.norm(direction_j) > 0.1:
                        recent_forward_directions.append(direction_j)
        
        if len(recent_forward_directions) == 0:
            recent_backward_directions = []
            for j in range(lookback_start, i):
                if corrected_classification[frames[j]] == 'backward':
                    if j > 0:
                        pos_j = np.array(smooth_centroids[frames[j]])
                        pos_j_prev = np.array(smooth_centroids[frames[j-1]])
                        direction_j = pos_j - pos_j_prev
                        if np.linalg.norm(direction_j) > 0.1:
                            recent_backward_directions.append(direction_j)
            
            if len(recent_backward_directions) > 0:
                avg_backward_direction = np.mean(recent_backward_directions, axis=0)
                avg_backward_direction = avg_backward_direction / np.linalg.norm(avg_backward_direction)
                
                dot_product = np.dot(current_direction_norm, avg_backward_direction)
                angle_to_backward = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))
                
                if is_near_end and current_type in ['forward', 'backward']:
                    print(f"Frame {frame} (using backward ref): {current_type}, angle_to_backward: {angle_to_backward:.1f}°, refs: {len(recent_backward_directions)}")
                
                if current_type == 'forward' and angle_to_backward < forward_angle_threshold:
                    corrected_classification[frame] = 'backward'
                    corrections_made['forward_to_backward'] += 1
                    print(f"Corrected frame {frame}: forward -> backward (angle to backward ref: {angle_to_backward:.1f}°)")
                    continue
                    
                elif current_type == 'backward' and angle_to_backward > backward_angle_threshold:
                    corrected_classification[frame] = 'forward'
                    corrections_made['backward_to_forward'] += 1
                    print(f"Corrected frame {frame}: backward -> forward (angle to backward ref: {angle_to_backward:.1f}°)")
                    continue
            
            if current_type in ['forward', 'backward'] and (is_near_end or i % 50 == 0):
                print(f"Frame {frame}: No recent reference found (type: {current_type}, near_end: {is_near_end})")
            continue
            
        avg_forward_direction = np.mean(recent_forward_directions, axis=0)
        avg_forward_direction = avg_forward_direction / np.linalg.norm(avg_forward_direction)
        
        # Calculate angle between current movement and recent forward movement
        dot_product = np.dot(current_direction_norm, avg_forward_direction)
        angle = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))
        
        if is_near_end and current_type in ['forward', 'backward']:
            print(f"Frame {frame} (near end): {current_type}, angle: {angle:.1f}°, recent_refs: {len(recent_forward_directions)}")
        
        # Correction logic
        if current_type == 'backward' and angle < forward_angle_threshold:
            # Backward classified but moving in same direction as recent forward
            corrected_classification[frame] = 'forward'
            corrections_made['backward_to_forward'] += 1
            print(f"Corrected frame {frame}: backward -> forward (angle: {angle:.1f}°)")
            
        elif current_type == 'forward' and angle > backward_angle_threshold:
            # Forward classified but moving opposite to recent forward (true reversal)
            corrected_classification[frame] = 'backward'
            corrections_made['forward_to_backward'] += 1
            print(f"Corrected frame {frame}: forward -> backward (angle: {angle:.1f}°)")
    
    total_corrections = sum(corrections_made.values())
    print(f"Made {total_corrections} corrections: {corrections_made['backward_to_forward']} B->F, {corrections_made['forward_to_backward']} F->B")
    return corrected_classification

def analyze_classification_consistency(movement_classification, smooth_centroids):
    frames = sorted(movement_classification.keys())
    
    transitions = []
    for i in range(1, len(frames)):
        prev_type = movement_classification[frames[i-1]]
        curr_type = movement_classification[frames[i]]
        if prev_type != curr_type:
            transitions.append((frames[i-1], frames[i], prev_type, curr_type))
    
    backward_analysis = []
    for i, frame in enumerate(frames):
        if movement_classification[frame] == 'backward' and i > 0:
            current_pos = np.array(smooth_centroids[frame])
            prev_pos = np.array(smooth_centroids[frames[i-1]])
            movement_vector = current_pos - prev_pos
            
            # Look for recent forward movement
            recent_forward_avg = None
            for j in range(max(0, i-10), i):
                if movement_classification[frames[j]] == 'forward' and j > 0:
                    pos_j = np.array(smooth_centroids[frames[j]])
                    pos_j_prev = np.array(smooth_centroids[frames[j-1]])
                    if recent_forward_avg is None:
                        recent_forward_avg = pos_j - pos_j_prev
                    else:
                        recent_forward_avg = (recent_forward_avg + (pos_j - pos_j_prev)) / 2
            
            if recent_forward_avg is not None:
                movement_norm = np.linalg.norm(movement_vector)
                forward_norm = np.linalg.norm(recent_forward_avg)
                
                if movement_norm > 1e-10 and forward_norm > 1e-10:
                    angle = np.degrees(np.arccos(np.clip(
                        np.dot(movement_vector / movement_norm,
                               recent_forward_avg / forward_norm), -1.0, 1.0)))
                    backward_analysis.append((frame, angle))
                elif movement_norm <= 1e-10:
                    # Zero or near-zero movement - worm is stationary
                    # Use angle from previous frame if available, representing maintained state
                    if len(backward_analysis) > 0:
                        prev_angle = backward_analysis[-1][1]
                        backward_analysis.append((frame, prev_angle))
                    else:
                        backward_analysis.append((frame, 90.0))
                else:
                    if len(backward_analysis) > 0:
                        prev_angle = backward_analysis[-1][1]
                        backward_analysis.append((frame, prev_angle))
                    else:
                        backward_analysis.append((frame, 90.0))
    
    return {
        'transitions': transitions,
        'backward_analysis': backward_analysis
    }

def plot_correction_comparison(original_classification, corrected_classification, smooth_centroids, save_path=None, filename=None):
    """
    Create a comparison plot showing movement classifications before and after corrections.
    """
    frames = sorted(smooth_centroids.keys())
    x, y = zip(*[smooth_centroids[f] for f in frames])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    if filename:
        fig.suptitle(f"{filename}\nMovement Classification Comparison", fontsize=16, y=0.98)
    else:
        fig.suptitle("Movement Classification Comparison", fontsize=16, y=0.95)
    
    color_map = {'forward': 'green', 'backward': 'red', 'stationary': 'blue'}
    
    # Original classification plot
    colors1 = [color_map.get(original_classification.get(f, 'stationary'), 'gray') for f in frames]
    ax1.scatter(x, y, c=colors1, s=10, alpha=0.7)
    ax1.plot(x, y, color='gray', alpha=0.3, linewidth=1)
    ax1.set_title("Original Movement Classification")
    ax1.set_xlabel("X position")
    ax1.set_ylabel("Y position")
    ax1.invert_yaxis()
    
    # Corrected classification plot
    colors2 = [color_map.get(corrected_classification.get(f, 'stationary'), 'gray') for f in frames]
    ax2.scatter(x, y, c=colors2, s=10, alpha=0.7)
    ax2.plot(x, y, color='gray', alpha=0.3, linewidth=1)
    ax2.set_title("Corrected Movement Classification")
    ax2.set_xlabel("X position")
    ax2.set_ylabel("Y position")
    ax2.invert_yaxis()
    
    legend_elements = [
        Patch(facecolor='green', edgecolor='green', label='Forward'),
        Patch(facecolor='red', edgecolor='red', label='Backward'),
        Patch(facecolor='blue', edgecolor='blue', label='Stationary')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    ax2.legend(handles=legend_elements, loc='upper right')
    
    # Highlight corrected points
    corrections = []
    for f in frames:
        if original_classification.get(f) != corrected_classification.get(f):
            pos = smooth_centroids[f]
            corrections.append(pos)
            ax2.plot(pos[0], pos[1], 'yo', markersize=8, markeredgecolor='black', markeredgewidth=1)
    
    if corrections:
        ax2.plot([], [], 'yo', markersize=8, markeredgecolor='black', markeredgewidth=1, 
                label=f'Corrected points ({len(corrections)})')
        ax2.legend(handles=legend_elements + [Patch(facecolor='yellow', edgecolor='black', 
                  label=f'Corrected points ({len(corrections)})')], loc='upper right')
    
    # Add statistics
    orig_forward = sum(1 for v in original_classification.values() if v == 'forward')
    orig_backward = sum(1 for v in original_classification.values() if v == 'backward')
    orig_stationary = sum(1 for v in original_classification.values() if v == 'stationary')
    
    corr_forward = sum(1 for v in corrected_classification.values() if v == 'forward')
    corr_backward = sum(1 for v in corrected_classification.values() if v == 'backward')
    corr_stationary = sum(1 for v in corrected_classification.values() if v == 'stationary')
    
    stats_text1 = f"Forward: {orig_forward}\nBackward: {orig_backward}\nStationary: {orig_stationary}"
    stats_text2 = f"Forward: {corr_forward}\nBackward: {corr_backward}\nStationary: {corr_stationary}"
    
    ax1.text(0.05, 0.95, stats_text1, transform=ax1.transAxes, verticalalignment='top', 
             fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    ax2.text(0.05, 0.95, stats_text2, transform=ax2.transAxes, verticalalignment='top', 
             fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Correction comparison plot saved to: {save_path}")
    else:
        plt.savefig('movement_correction_comparison.png', dpi=300, bbox_inches='tight')
    
    plt.show()
    plt.close()

def segment_level_correction(movement_classification, smooth_centroids, min_segment_length=15, 
                           interruption_threshold=3, direction_consistency_threshold=0.8):
    """
    Fix entire segments that are systematically misclassified.
    """
    frames = sorted(movement_classification.keys())
    corrected_classification = movement_classification.copy()
    
    segments = []
    current_segment = {'type': None, 'start': 0, 'frames': []}
    
    for i, frame in enumerate(frames):
        movement_type = movement_classification[frame]
        
        if movement_type != current_segment['type']:
            if len(current_segment['frames']) > 0:
                current_segment['end'] = i - 1
                segments.append(current_segment)
            
            current_segment = {
                'type': movement_type,
                'start': i,
                'frames': [frame],
                'indices': [i]
            }
        else:
            current_segment['frames'].append(frame)
            current_segment['indices'].append(i)
    
    if len(current_segment['frames']) > 0:
        current_segment['end'] = len(frames) - 1
        segments.append(current_segment)
    
    print(f"Found {len(segments)} movement segments")
    for i, segment in enumerate(segments):
        print(f"  Segment {i}: {segment['type']} - {len(segment['frames'])} frames (frames {segment['frames'][0]}-{segment['frames'][-1]})")
    
    corrections_made = 0
    for i, segment in enumerate(segments):
        if (segment['type'] in ['forward', 'backward'] and 
            len(segment['frames']) >= min_segment_length):
            
            print(f"\nAnalyzing long segment {i}: {segment['type']} with {len(segment['frames'])} frames")
            
            interruptions = []
            
            if i > 0:
                prev_segment = segments[i-1]
                print(f"  Previous segment: {prev_segment['type']} - {len(prev_segment['frames'])} frames")
                if (prev_segment['type'] != segment['type'] and 
                    prev_segment['type'] != 'stationary' and
                    len(prev_segment['frames']) <= interruption_threshold):
                    interruptions.append(('before', prev_segment))
                    print(f"    -> Added as interruption (before)")
            
            if i < len(segments) - 1:
                next_segment = segments[i+1]
                print(f"  Next segment: {next_segment['type']} - {len(next_segment['frames'])} frames")
                if (next_segment['type'] != segment['type'] and 
                    next_segment['type'] != 'stationary' and
                    len(next_segment['frames']) <= interruption_threshold):
                    interruptions.append(('after', next_segment))
                    print(f"    -> Added as interruption (after)")            
            print(f"  Found {len(interruptions)} interruptions")
            
            if len(interruptions) > 0:
                # Calculate the actual movement direction of the main segment
                segment_positions = [smooth_centroids[f] for f in segment['frames']]
                segment_start = np.array(segment_positions[0])
                segment_end = np.array(segment_positions[-1])
                overall_direction = segment_end - segment_start
                overall_distance = np.linalg.norm(overall_direction)
                
                print(f"  Overall segment distance: {overall_distance:.2f}")                
                if overall_distance < 1.0:
                    print(f"    -> Skipping: overall movement too small")
                    continue
                
                # Calculate frame-by-frame directions within the segment
                frame_directions = []
                for j in range(1, len(segment['frames'])):
                    pos_curr = np.array(segment_positions[j])
                    pos_prev = np.array(segment_positions[j-1])
                    direction = pos_curr - pos_prev
                    if np.linalg.norm(direction) > 0.1:
                        frame_directions.append(direction)
                
                if len(frame_directions) == 0:
                    print(f"    -> Skipping: no significant frame-by-frame movements")
                    continue
                
                # Check consistency of frame-by-frame directions
                avg_frame_direction = np.mean(frame_directions, axis=0)
                avg_frame_direction = avg_frame_direction / np.linalg.norm(avg_frame_direction)
                
                consistencies = []
                for direction in frame_directions:
                    if np.linalg.norm(direction) > 0.1:
                        direction_norm = direction / np.linalg.norm(direction)
                        consistency = np.dot(direction_norm, avg_frame_direction)
                        consistencies.append(consistency)
                
                avg_consistency = np.mean(consistencies) if consistencies else 0
                print(f"  Average direction consistency: {avg_consistency:.3f}")
                
                # Check if interruptions align with the main movement direction
                should_flip = False
                for pos, interruption in interruptions:
                    print(f"  Checking interruption ({pos}): {interruption['type']} - {len(interruption['frames'])} frames")
                    
                    int_positions = [smooth_centroids[f] for f in interruption['frames']]
                    if len(int_positions) >= 2:
                        int_start = np.array(int_positions[0])
                        int_end = np.array(int_positions[-1])
                        int_direction = int_end - int_start
                        int_distance = np.linalg.norm(int_direction)
                        print(f"    Interruption distance: {int_distance:.2f}")
                        
                        if int_distance > 0.1:
                            int_direction_norm = int_direction / int_distance
                            
                            alignment = np.dot(int_direction_norm, avg_frame_direction)
                            angle = np.degrees(np.arccos(np.clip(alignment, -1.0, 1.0)))
                            
                            print(f"    Alignment angle: {angle:.1f}°")
                            print(f"    Interruption type: {interruption['type']}, Segment type: {segment['type']}")
                            print(f"    Consistency: {avg_consistency:.3f} > {direction_consistency_threshold}? {avg_consistency > direction_consistency_threshold}")
                            
                            context_directions = []
                            for ctx_i in range(max(0, i-2), min(len(segments), i+3)):
                                if ctx_i != i and segments[ctx_i]['type'] in ['forward', 'backward']:
                                    ctx_positions = [smooth_centroids[f] for f in segments[ctx_i]['frames']]
                                    if len(ctx_positions) >= 2:
                                        ctx_start = np.array(ctx_positions[0])
                                        ctx_end = np.array(ctx_positions[-1])
                                        ctx_direction = ctx_end - ctx_start
                                        if np.linalg.norm(ctx_direction) > 1.0:
                                            context_directions.append(ctx_direction / np.linalg.norm(ctx_direction))
                            
                            should_flip = False
                            
                            if len(context_directions) > 0:
                                avg_context_direction = np.mean(context_directions, axis=0)
                                avg_context_direction = avg_context_direction / np.linalg.norm(avg_context_direction)
                                
                                int_context_alignment = np.dot(int_direction_norm, avg_context_direction)
                                main_context_alignment = np.dot(avg_frame_direction, avg_context_direction)
                                
                                int_context_angle = np.degrees(np.arccos(np.clip(abs(int_context_alignment), 0, 1)))
                                main_context_angle = np.degrees(np.arccos(np.clip(abs(main_context_alignment), 0, 1)))
                                
                                print(f"    Context analysis:")
                                print(f"      Interruption-context angle: {int_context_angle:.1f}°")
                                print(f"      Main segment-context angle: {main_context_angle:.1f}°")
                                
                                if (angle > 150 and
                                    avg_consistency > direction_consistency_threshold and
                                    interruption['type'] != segment['type'] and
                                    int_context_angle < main_context_angle - 20):  # interruption aligns much better with context
                                    should_flip = True
                                    print(f"    -> SHOULD FLIP: interruption aligns much better with context")
                                else:
                                    print(f"    -> No flip: insufficient evidence for misclassification")
                            else:
                                print(f"    -> No flip: insufficient context for validation")
                
                if should_flip:
                    new_type = 'forward' if segment['type'] == 'backward' else 'backward'
                    for frame in segment['frames']:
                        corrected_classification[frame] = new_type
                        corrections_made += 1
                    
                    print(f"  *** CORRECTED entire segment of {len(segment['frames'])} frames: {segment['type']} -> {new_type} ***")
                else:
                    print(f"  No correction applied to this segment")
    
    print(f"Segment-level correction made {corrections_made} changes")
    return corrected_classification

def local_consistency_correction(movement_classification, smooth_centroids, max_isolated_length=1):
    """
    Correct isolated short segments that are inconsistent with surrounding longer segments.
    """
    frames = sorted(movement_classification.keys())
    corrected_classification = movement_classification.copy()
    
    segments = []
    current_segment = {'type': None, 'frames': []}
    
    for i, frame in enumerate(frames):
        movement_type = movement_classification[frame]
        
        if movement_type != current_segment['type']:
            if len(current_segment['frames']) > 0:
                segments.append(current_segment)
            current_segment = {'type': movement_type, 'frames': [frame]}
        else:
            current_segment['frames'].append(frame)
    
    if len(current_segment['frames']) > 0:
        segments.append(current_segment)
    
    print(f"Local consistency correction: analyzing {len(segments)} segments")
    corrections_made = 0
    
    for i in range(1, len(segments) - 1):
        current_segment = segments[i]
        prev_segment = segments[i-1]
        next_segment = segments[i+1]
        
        if (len(current_segment['frames']) <= max_isolated_length and
            len(prev_segment['frames']) >= 2 and
            len(next_segment['frames']) >= 2 and
            prev_segment['type'] == next_segment['type'] and
            current_segment['type'] != prev_segment['type'] and
            current_segment['type'] in ['forward', 'backward'] and
            prev_segment['type'] in ['forward', 'backward']):
            
            print(f"Found isolated segment: {len(current_segment['frames'])} {current_segment['type']} frames between {prev_segment['type']} segments")
            
            if len(current_segment['frames']) >= 1:
                isolated_directions = []
                for frame in current_segment['frames']:
                    frame_idx = frames.index(frame)
                    if frame_idx > 0:
                        current_pos = np.array(smooth_centroids[frame])
                        prev_pos = np.array(smooth_centroids[frames[frame_idx-1]])
                        movement = current_pos - prev_pos
                        if np.linalg.norm(movement) > 0.1:
                            isolated_directions.append(movement)
                
                surrounding_directions = []
                
                for frame in prev_segment['frames'][-3:]:
                    frame_idx = frames.index(frame)
                    if frame_idx > 0:
                        current_pos = np.array(smooth_centroids[frame])
                        prev_pos = np.array(smooth_centroids[frames[frame_idx-1]])
                        movement = current_pos - prev_pos
                        if np.linalg.norm(movement) > 0.1:
                            surrounding_directions.append(movement)
                
                for frame in next_segment['frames'][:3]:
                    frame_idx = frames.index(frame)
                    if frame_idx > 0:
                        current_pos = np.array(smooth_centroids[frame])
                        prev_pos = np.array(smooth_centroids[frames[frame_idx-1]])
                        movement = current_pos - prev_pos
                        if np.linalg.norm(movement) > 0.1:
                            surrounding_directions.append(movement)
                
                if len(isolated_directions) > 0 and len(surrounding_directions) > 0:
                    # Calculate average directions
                    avg_isolated = np.mean(isolated_directions, axis=0)
                    avg_surrounding = np.mean(surrounding_directions, axis=0)
                    
                    # Normalize
                    if np.linalg.norm(avg_isolated) > 0 and np.linalg.norm(avg_surrounding) > 0:
                        avg_isolated_norm = avg_isolated / np.linalg.norm(avg_isolated)
                        avg_surrounding_norm = avg_surrounding / np.linalg.norm(avg_surrounding)
                        
                        # Calculate alignment
                        alignment = np.dot(avg_isolated_norm, avg_surrounding_norm)
                        angle = np.degrees(np.arccos(np.clip(abs(alignment), 0, 1)))
                        
                        print(f"  Isolated segment alignment with surroundings: {alignment:.3f}, angle: {angle:.1f}°")
                        
                        # If isolated segment movement aligns well with surrounding movement,
                        # correct it to match the surrounding classification
                        if angle < 45:
                            for frame in current_segment['frames']:
                                corrected_classification[frame] = prev_segment['type']
                                corrections_made += 1
                            print(f"  Corrected isolated {current_segment['type']} -> {prev_segment['type']} ({len(current_segment['frames'])} frames)")
    
    print(f"Local consistency correction made {corrections_made} changes")
    return corrected_classification

def final_stationary_correction(movement_classification, smooth_centroids, speed_threshold=1.0):
    """
    Final post-classification correction to identify truly stationary frames.
    Converts frames to stationary if the movement speed is below the threshold.
    """
    frames = sorted(movement_classification.keys())
    corrected_classification = movement_classification.copy()
    corrections_made = 0
    
    print(f"Final stationary correction: applying speed threshold of {speed_threshold} px/frame")
    
    for i, frame in enumerate(frames):
        movement_distance = None
        
        if i == 0:
            if len(frames) > 1:
                current_pos = np.array(smooth_centroids[frame])
                next_pos = np.array(smooth_centroids[frames[i+1]])
                movement_distance = np.linalg.norm(next_pos - current_pos)
        elif i == len(frames) - 1:
            current_pos = np.array(smooth_centroids[frame])
            prev_pos = np.array(smooth_centroids[frames[i-1]])
            movement_distance = np.linalg.norm(current_pos - prev_pos)
        else:
            current_pos = np.array(smooth_centroids[frame])
            prev_pos = np.array(smooth_centroids[frames[i-1]])
            movement_distance = np.linalg.norm(current_pos - prev_pos)
        
        if movement_distance <= speed_threshold:
            if corrected_classification[frame] != 'stationary':
                corrected_classification[frame] = 'stationary'
                corrections_made += 1
    
    print(f"Final stationary correction: converted {corrections_made} frames to stationary")
    return corrected_classification

def recalculate_metrics_after_correction(results, corrected_classification):
    """
    Recalculate ALL movement metrics using the corrected movement classification.
    This ensures that all bout analysis, speed calculations, and other metrics
    are based on the final corrected data rather than the original classifications.
    """
    print("Recalculating all metrics with corrected movement classification...")
    
    # Get necessary data from original results
    smooth_centroids = results['smooth_centroids']
    velocities = results['velocities']
    frames = sorted(corrected_classification.keys())
    
    # Recalculate frame counts
    forward_frames = sum(1 for v in corrected_classification.values() if v == 'forward')
    backward_frames = sum(1 for v in corrected_classification.values() if v == 'backward')
    stationary_frames = sum(1 for v in corrected_classification.values() if v == 'stationary')
    total_frames = len(corrected_classification)
    
    # Recalculate velocity-based metrics
    velocity_vectors = np.array([velocities[f] for f in frames])
    forward_velocities = []
    backward_velocities = []
    forward_accelerations = []
    backward_accelerations = []
    per_frame_speeds = {}
    per_frame_accelerations = {}
    
    # Calculate accelerations (in pixels per frame^2)
    acceleration = np.gradient(velocity_vectors, axis=0)
    
    # Recalculate bout analysis variables
    forward_bouts = 0
    backward_bouts = 0
    stationary_bouts = 0
    current_bout_type = None
    bout_lengths_frames = {'forward': [], 'backward': [], 'stationary': []}
    bout_lengths_pixels = {'forward': [], 'backward': [], 'stationary': []}
    current_bout_length_frames = 0
    current_bout_length_pixels = 0.0
    
    # Recalculate furthest point analysis
    start_point = np.array(smooth_centroids[frames[0]])
    max_distance = 0
    furthest_frame = frames[0]
    
    # Process each frame with corrected classification
    for i, frame in enumerate(frames):
        movement_type = corrected_classification[frame]
        
        # Per-frame speed and acceleration calculation
        per_frame_speeds[frame] = np.linalg.norm(velocity_vectors[i])
        per_frame_accelerations[frame] = acceleration[i]  # Store as [ax, ay] vector
        
        # Bout analysis - handle all movement types including stationary
        if current_bout_type != movement_type:
            if current_bout_type is not None:
                bout_lengths_frames[current_bout_type].append(current_bout_length_frames)
                bout_lengths_pixels[current_bout_type].append(current_bout_length_pixels)
            
            if movement_type == 'forward':
                forward_bouts += 1
            elif movement_type == 'backward':
                backward_bouts += 1
            elif movement_type == 'stationary':
                stationary_bouts += 1
                
            current_bout_type = movement_type
            current_bout_length_frames = 1
            current_bout_length_pixels = 0.0
        else:
            current_bout_length_frames += 1
            
        if i > 0:
            current_pos = np.array(smooth_centroids[frame])
            prev_pos = np.array(smooth_centroids[frames[i-1]])
            frame_distance = np.linalg.norm(current_pos - prev_pos)
            current_bout_length_pixels += frame_distance
        
        # Collect velocities and accelerations by movement type
        if movement_type == 'forward':
            forward_velocities.append(velocity_vectors[i])
            forward_accelerations.append(acceleration[i])
        elif movement_type == 'backward':
            backward_velocities.append(velocity_vectors[i])
            backward_accelerations.append(acceleration[i])
        
        # Furthest point analysis
        current_point = np.array(smooth_centroids[frame])
        distance = np.linalg.norm(current_point - start_point)
        if distance > max_distance:
            max_distance = distance
            furthest_frame = frame
    
    if current_bout_type is not None:
        bout_lengths_frames[current_bout_type].append(current_bout_length_frames)
        bout_lengths_pixels[current_bout_type].append(current_bout_length_pixels)
    
    # Calculate average velocities and accelerations
    avg_velocity = np.mean(velocity_vectors, axis=0)
    avg_acceleration = np.mean(acceleration, axis=0)
    avg_forward_velocity = np.mean(forward_velocities, axis=0) if forward_velocities else np.array([0, 0])
    avg_backward_velocity = np.mean(backward_velocities, axis=0) if backward_velocities else np.array([0, 0])
    avg_forward_acceleration = np.mean(forward_accelerations, axis=0) if forward_accelerations else np.array([0, 0])
    avg_backward_acceleration = np.mean(backward_accelerations, axis=0) if backward_accelerations else np.array([0, 0])
    
    # Calculate average speeds
    avg_forward_speed = np.mean(np.linalg.norm(forward_velocities, axis=1)) if forward_velocities else 0
    avg_backward_speed = np.mean(np.linalg.norm(backward_velocities, axis=1)) if backward_velocities else 0
    
    # Calculate average bout lengths - both frames and pixels
    avg_forward_bout_length_frames = np.mean(bout_lengths_frames['forward']) if bout_lengths_frames['forward'] else 0
    avg_backward_bout_length_frames = np.mean(bout_lengths_frames['backward']) if bout_lengths_frames['backward'] else 0
    avg_stationary_bout_length_frames = np.mean(bout_lengths_frames['stationary']) if bout_lengths_frames['stationary'] else 0
    
    avg_forward_bout_length_pixels = np.mean(bout_lengths_pixels['forward']) if bout_lengths_pixels['forward'] else 0
    avg_backward_bout_length_pixels = np.mean(bout_lengths_pixels['backward']) if bout_lengths_pixels['backward'] else 0
    avg_stationary_bout_length_pixels = np.mean(bout_lengths_pixels['stationary']) if bout_lengths_pixels['stationary'] else 0
    
    # Calculate stationary-specific metrics
    stationary_percentage = (stationary_frames / total_frames) * 100 if total_frames > 0 else 0
    moving_frames = forward_frames + backward_frames
    moving_percentage = (moving_frames / total_frames) * 100 if total_frames > 0 else 0
    
    # Calculate transition metrics
    transitions = 0
    for i in range(1, len(frames)):
        if corrected_classification[frames[i]] != corrected_classification[frames[i-1]]:
            transitions += 1
    
    # Calculate time ratios - now using frame-based bout lengths
    total_stationary_time = sum(bout_lengths_frames['stationary']) if bout_lengths_frames['stationary'] else 0
    total_forward_time = sum(bout_lengths_frames['forward']) if bout_lengths_frames['forward'] else 0
    total_backward_time = sum(bout_lengths_frames['backward']) if bout_lengths_frames['backward'] else 0
    
    # Calculate distance and movement metrics (these shouldn't change much)
    positions = np.array([smooth_centroids[f] for f in frames])
    total_distance = np.sum(np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1)))
    avg_speed = total_distance / total_frames
    
    # Calculate sinuosity
    start_point = positions[0]
    end_point = positions[-1]
    straight_line_distance = np.linalg.norm(end_point - start_point)
    sinuosity = total_distance / straight_line_distance if straight_line_distance > 0 else 0
    
    # Update results with recalculated metrics
    results.update({
        'movement_classification': corrected_classification,
        'forward_frames': forward_frames,
        'backward_frames': backward_frames,
        'stationary_frames': stationary_frames,
        'total_frames': total_frames,
        'total_distance': total_distance,
        'avg_speed': avg_speed,
        'sinuosity': sinuosity,
        'avg_velocity': avg_velocity,
        'avg_acceleration': avg_acceleration,
        'avg_forward_velocity': avg_forward_velocity,
        'avg_backward_velocity': avg_backward_velocity,
        'avg_forward_acceleration': avg_forward_acceleration,
        'avg_backward_acceleration': avg_backward_acceleration,
        'avg_forward_speed': avg_forward_speed,
        'avg_backward_speed': avg_backward_speed,
        'per_frame_speeds': per_frame_speeds,
        'per_frame_accelerations': per_frame_accelerations,
        'forward_bouts': forward_bouts,
        'backward_bouts': backward_bouts,
        'stationary_bouts': stationary_bouts,
        'bout_lengths_frames': bout_lengths_frames,
        'bout_lengths_pixels': bout_lengths_pixels,
        'avg_forward_bout_length_frames': avg_forward_bout_length_frames,
        'avg_backward_bout_length_frames': avg_backward_bout_length_frames,
        'avg_stationary_bout_length_frames': avg_stationary_bout_length_frames,
        'avg_forward_bout_length_pixels': avg_forward_bout_length_pixels,
        'avg_backward_bout_length_pixels': avg_backward_bout_length_pixels,
        'avg_stationary_bout_length_pixels': avg_stationary_bout_length_pixels,
        'stationary_percentage': stationary_percentage,
        'moving_percentage': moving_percentage,
        'transitions': transitions,
        'total_stationary_time': total_stationary_time,
        'total_forward_time': total_forward_time,
        'total_backward_time': total_backward_time,
        'furthest_point_distance': max_distance,
        'furthest_point_frame': furthest_frame,
        'head_tail_swapped': results['head_tail_swapped'],
        'localized_head_tail_corrections': results['localized_head_tail_corrections'],
        'localized_corrections': results['localized_corrections']
    })
    
    print(f"Metrics recalculated:")
    print(f"  Forward frames: {forward_frames} (was {results.get('original_forward_frames', 'unknown')})")
    print(f"  Backward frames: {backward_frames} (was {results.get('original_backward_frames', 'unknown')})")
    print(f"  Stationary frames: {stationary_frames} (was {results.get('original_stationary_frames', 'unknown')})")
    print(f"  Forward bouts: {forward_bouts}")
    print(f"  Backward bouts: {backward_bouts}")
    print(f"  Stationary bouts: {stationary_bouts}")
    print(f"  Avg forward speed: {avg_forward_speed:.2f} px/frame")
    print(f"  Avg backward speed: {avg_backward_speed:.2f} px/frame")
    print(f"  Avg forward bout length: {avg_forward_bout_length_frames:.1f} frames ({avg_forward_bout_length_pixels:.1f} px)")
    print(f"  Avg backward bout length: {avg_backward_bout_length_frames:.1f} frames ({avg_backward_bout_length_pixels:.1f} px)")
    print(f"  Avg stationary bout length: {avg_stationary_bout_length_frames:.1f} frames ({avg_stationary_bout_length_pixels:.1f} px)")
    print(f"  Stationary percentage: {stationary_percentage:.1f}%")
    print(f"  Moving percentage: {moving_percentage:.1f}%")
    print(f"  Total transitions: {transitions}")
    
    return results

def save_path_analysis_results(results, original_shape_path, output_dir):
    original_basename = os.path.basename(original_shape_path)
    
    if original_basename.endswith('.pkl'):
        new_basename = original_basename[:-4] + '_pathanalysis.pkl'
    else:
        new_basename = original_basename + '_pathanalysis.pkl'
    
    output_path = os.path.join(output_dir, new_basename)    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'wb') as file:
        pickle.dump(results, file)
    
    print(f"Path analysis results saved to: {output_path}")
    return output_path

def plot_worm_path_with_metrics(results, save_path=None, filename=None):
    smooth_centroids = results['smooth_centroids']
    movement_classification = results['movement_classification']
    velocities = results['velocities']

    frames = sorted(smooth_centroids.keys())
    x, y = zip(*[smooth_centroids[f] for f in frames])
    colors = []
    for f in frames:
        if f in movement_classification:
            if movement_classification[f] == 'forward':
                colors.append('green')
            elif movement_classification[f] == 'backward':
                colors.append('red')
            else:
                colors.append('blue')
        else:
            colors.append('gray')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16), gridspec_kw={'height_ratios': [3, 1]})

    if filename:
        fig.suptitle(f"{filename}\nWorm Path Analysis with Movement Classification", fontsize=14, y=0.98)

    ax1.plot(x, y, color='gray', alpha=0.5, linewidth=1)
    ax1.set_title("Worm Path with Movement Classification")
    ax1.set_xlabel("X position")
    ax1.set_ylabel("Y position")
    ax1.invert_yaxis()

    legend_elements = [
        Patch(facecolor='green', edgecolor='green', label='Forward'),
        Patch(facecolor='red', edgecolor='red', label='Backward'),
        Patch(facecolor='blue', edgecolor='blue', label='Stationary')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')

    # Add text annotations for start and end points
    ax1.annotate('Start', (x[0], y[0]), xytext=(10, 10), textcoords='offset points', 
                fontsize=10, fontweight='bold', color='black', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax1.annotate('End', (x[-1], y[-1]), xytext=(10, 10), textcoords='offset points', 
                fontsize=10, fontweight='bold', color='black', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    metrics_text = (
        f"Total Frames: {results['total_frames']}\n"
        f"Forward Frames: {results['forward_frames']}\n"
        f"Backward Frames: {results['backward_frames']}\n"
        f"Stationary Frames: {results['stationary_frames']}\n"
        f"Total Distance: {results['total_distance']:.2f} px\n"
        f"Average Speed: {results['avg_speed']:.2f} px/frame\n"
        f"Sinuosity: {results['sinuosity']:.2f}"
    )
    ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes, verticalalignment='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

    v_x = [velocities[f][0] for f in frames]
    v_y = [velocities[f][1] for f in frames]
    speed = np.sqrt(np.array(v_x)**2 + np.array(v_y)**2)
    ax2.plot(frames, speed, label='Speed')
    ax2.set_title("Worm Speed Over Time")
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Speed (px/frame)")
    ax2.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics plot saved to: {save_path}")
    else:
        plt.savefig('plotspeed2.png', dpi=300, bbox_inches='tight')
    
    plt.close()

def create_movement_video(results, frames_dir, hdshape_data, output_path='PATH_TO_SAVE_VIDEO.mp4', 
                         fps=10, scale_factor=0.5):
    """
    Create a video with movement classification overlay and head position markers.
    """
    
    movement_classification = results['movement_classification']
    
    head_positions = {}
    tail_positions = {}
    for i, frame_num in enumerate(hdshape_data['frames']):
        if i < len(hdshape_data['head_positions']):
            head_positions[frame_num] = hdshape_data['head_positions'][i]
        if i < len(hdshape_data['tail_positions']):
            tail_positions[frame_num] = hdshape_data['tail_positions'][i]
    
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    if not frame_files:
        print("No frame files found!")
        return
    
    first_frame_path = os.path.join(frames_dir, frame_files[0])
    first_frame = cv2.imread(first_frame_path)    
    if first_frame is None:
        print(f"Could not read first frame: {first_frame_path}")
        return
    
    original_height, original_width = first_frame.shape[:2]
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
    
    color_map = {
        'forward': (0, 255, 0),     # Green
        'backward': (0, 0, 255),    # Red
        'stationary': (255, 0, 0),  # Blue
        'unknown': (128, 128, 128)  # Gray
    }
    
    processed_frames = 0
    for frame_file in frame_files:
        frame_num = int(frame_file.split('.')[0])
            
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        
        if frame is None:
            print(f"Could not read frame: {frame_path}")
            continue
        
        # Scale down the frame
        frame_resized = cv2.resize(frame, (new_width, new_height))
        
        movement_type = movement_classification[frame_num]
        text_color = color_map.get(movement_type, color_map['unknown'])
        
        text = f"Frame {frame_num}: {movement_type.upper()}"
        
        overlay = frame_resized.copy()
        cv2.rectangle(overlay, (10, 15), (600, 70), (0, 0, 0), -1)
        frame_resized = cv2.addWeighted(frame_resized, 0.8, overlay, 0.2, 0)
        
        cv2.putText(frame_resized, text, (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 3, cv2.LINE_AA)

        if frame_num in head_positions:
            head_x, head_y = head_positions[frame_num]
            head_x_scaled = int(head_x * scale_factor)
            head_y_scaled = int(head_y * scale_factor)
            
            overlay = frame_resized.copy()
            cv2.circle(overlay, (head_x_scaled, head_y_scaled), 5, (0, 255, 255), -1)
            cv2.circle(overlay, (head_x_scaled, head_y_scaled), 5, (0, 0, 0), 1)
            frame_resized = cv2.addWeighted(frame_resized, 0.7, overlay, 0.3, 0)
        
        out.write(frame_resized)
        processed_frames += 1
        
        if processed_frames % 50 == 0:
            print(f"Processed {processed_frames} frames...")
    
    out.release()    
    print(f"Video created successfully: {output_path}")
    print(f"Total frames processed: {processed_frames}")
    print(f"Video dimensions: {new_width}x{new_height}")
    print(f"Frame rate: {fps} fps")

def create_movement_video_with_segmentation(results, frames_dir, hdshape_data, shape_file_path, segmentation_dir,
                                          output_path='PATH_TO_SAVE_VIDEO_WITH_SEGMENTATION.mp4', 
                                          fps=10, scale_factor=0.5,
                                          segmentation_alpha=0.5):
    """
    Create a video with movement classification overlay, head position markers, and mask segmentation.
    """
    
    movement_classification = results['movement_classification']
    
    shape_basename = os.path.basename(shape_file_path)
    if shape_basename.endswith('_shapeanalysis.pkl'):
        segmentation_filename = shape_basename.replace('_shapeanalysis.pkl', '.pkl')
    else:
        segmentation_filename = shape_basename.replace('.pkl', '_hd_segments.pkl')
    
    segmentation_path = os.path.join(segmentation_dir, segmentation_filename)
    
    try:
        with open(segmentation_path, 'rb') as f:
            segmentation_data = pickle.load(f)
        print(f"Successfully loaded segmentation data from: {segmentation_path}")
    except FileNotFoundError:
        print(f"Warning: Segmentation file not found: {segmentation_path}")
        print("Creating video without segmentation overlay...")
        segmentation_data = None
    except Exception as e:
        print(f"Error loading segmentation data: {e}")
        segmentation_data = None
    
    # Create mapping of frame numbers to head and tail positions
    head_positions = {}
    tail_positions = {}
    for i, frame_num in enumerate(hdshape_data['frames']):
        if i < len(hdshape_data['head_positions']):
            head_positions[frame_num] = hdshape_data['head_positions'][i]
        if i < len(hdshape_data['tail_positions']):
            tail_positions[frame_num] = hdshape_data['tail_positions'][i]
    
    # Create mapping of frame numbers to segmentation masks if available
    segmentation_masks = {}
    if segmentation_data is not None:
        for frame_num in segmentation_data.keys():
            try:
                segmentation_masks[frame_num] = segmentation_data[frame_num][1][0]
            except (IndexError, TypeError):
                print(f"Warning: Could not access segmentation data for frame {frame_num}")
                continue
    
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])    
    if not frame_files:
        print("No frame files found!")
        return
    
    first_frame_path = os.path.join(frames_dir, frame_files[0])
    first_frame = cv2.imread(first_frame_path)    
    if first_frame is None:
        print(f"Could not read first frame: {first_frame_path}")
        return
    
    original_height, original_width = first_frame.shape[:2]
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
    
    color_map = {
        'forward': (0, 255, 0),     # Green
        'backward': (0, 0, 255),    # Red
        'stationary': (255, 0, 0),  # Blue
        'unknown': (128, 128, 128)  # Gray
    }
    
    segmentation_color = (255, 255, 0) # Cyan
    
    processed_frames = 0    
    for frame_file in frame_files:
        frame_num = int(frame_file.split('.')[0])
        if frame_num not in movement_classification:
            continue
            
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        
        if frame is None:
            print(f"Could not read frame: {frame_path}")
            continue
        
        # Add segmentation overlay before scaling
        if frame_num in segmentation_masks:
            mask = segmentation_masks[frame_num]
            
            if mask.ndim == 3:
                mask = mask.squeeze()
            
            if mask.dtype != np.uint8:
                mask = (mask * 255).astype(np.uint8)
            
            colored_overlay = np.zeros_like(frame)
            colored_overlay[mask > 0] = segmentation_color           
            frame = cv2.addWeighted(frame, 1 - segmentation_alpha, colored_overlay, segmentation_alpha, 0)
        
        frame_resized = cv2.resize(frame, (new_width, new_height))
        
        movement_type = movement_classification[frame_num]
        text_color = color_map.get(movement_type, color_map['unknown'])
        
        text = f"Frame {frame_num}: {movement_type.upper()}"        
        overlay = frame_resized.copy()
        cv2.rectangle(overlay, (10, 15), (650, 70), (0, 0, 0), -1)
        frame_resized = cv2.addWeighted(frame_resized, 0.8, overlay, 0.2, 0)
        cv2.putText(frame_resized, text, (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 3, cv2.LINE_AA)
        
        # Add head position marker if available
        if frame_num in tail_positions:
            head_x, head_y = tail_positions[frame_num]
            head_x_scaled = int(head_x * scale_factor)
            head_y_scaled = int(head_y * scale_factor)
            
            overlay = frame_resized.copy()
            cv2.circle(overlay, (head_x_scaled, head_y_scaled), 5, (0, 255, 255), -1)
            cv2.circle(overlay, (head_x_scaled, head_y_scaled), 5, (0, 0, 0), 1)
            frame_resized = cv2.addWeighted(frame_resized, 0.7, overlay, 0.3, 0)
        
        out.write(frame_resized)
        processed_frames += 1
        
        if processed_frames % 50 == 0:
            print(f"Processed {processed_frames} frames...")
    
    out.release()
    
    print(f"Video created successfully: {output_path}")
    print(f"Total frames processed: {processed_frames}")
    print(f"Video dimensions: {new_width}x{new_height}")
    print(f"Frame rate: {fps} fps")
    if segmentation_data is not None:
        print(f"Segmentation overlay applied with alpha: {segmentation_alpha}")
    else:
        print("No segmentation overlay applied")

def detect_and_correct_head_tail_swap(aligned_data, smooth_centroids, confidence_threshold=0.7):
    """
    Detect if head and tail were incorrectly identified by analyzing movement consistency.
    """
    print("Analyzing head/tail consistency...")
    
    frames = sorted(aligned_data.keys())
    if len(frames) < 10:
        print("Not enough frames for head/tail analysis")
        return aligned_data, False
    
    movement_vectors = []
    orientation_vectors = []
    valid_frame_indices = []
    
    for i in range(1, len(frames)):
        frame = frames[i]
        prev_frame = frames[i-1]
        
        current_pos = np.array(smooth_centroids[frame])
        prev_pos = np.array(smooth_centroids[prev_frame])
        movement_vec = current_pos - prev_pos
        
        if np.linalg.norm(movement_vec) < 0.5:
            continue
            
        head = np.array(aligned_data[frame]['head_coord'])
        smooth_points = aligned_data[frame]['smooth_points']
        segment_point = smooth_points[len(smooth_points) // 10]
        orientation_vec = head - np.array(segment_point)
        if np.linalg.norm(orientation_vec) < 0.5:
            continue
            
        movement_vectors.append(movement_vec)
        orientation_vectors.append(orientation_vec)
        valid_frame_indices.append(i)
    
    if len(movement_vectors) < 5:
        print("Not enough valid movement vectors for analysis")
        return aligned_data, False
    
    movement_vectors = np.array(movement_vectors)
    orientation_vectors = np.array(orientation_vectors)
    
    movement_norms = np.linalg.norm(movement_vectors, axis=1, keepdims=True)
    orientation_norms = np.linalg.norm(orientation_vectors, axis=1, keepdims=True)
    
    movement_vectors_norm = movement_vectors / np.maximum(movement_norms, 1e-10)
    orientation_vectors_norm = orientation_vectors / np.maximum(orientation_norms, 1e-10)
    
    # Calculate dot products (alignment between movement and orientation)
    dot_products = np.einsum('ij,ij->i', movement_vectors_norm, orientation_vectors_norm)
    
    # Calculate angles between movement and orientation
    angles = np.degrees(np.arccos(np.clip(np.abs(dot_products), 0, 1)))
    
    forward_aligned = np.sum(dot_products > 0)  # Moving in head direction
    backward_aligned = np.sum(dot_products < 0)  # Moving opposite to head direction
    
    total_movements = len(dot_products)
    forward_ratio = forward_aligned / total_movements
    backward_ratio = backward_aligned / total_movements
    
    print(f"Movement analysis:")
    print(f"  Total valid movements: {total_movements}")
    print(f"  Moving in head direction: {forward_aligned} ({forward_ratio:.2%})")
    print(f"  Moving opposite to head: {backward_aligned} ({backward_ratio:.2%})")
    print(f"  Average alignment angle: {np.mean(angles):.1f}°")
    
    should_swap = False    
    if backward_ratio > confidence_threshold:
        should_swap = True
        print(f"*** HEAD/TAIL SWAP DETECTED ***")
        print(f"  {backward_ratio:.1%} of movements are opposite to head direction")
        print(f"  This suggests head and tail are swapped")
    else:
        print(f"Head/tail orientation appears correct")
        print(f"  {forward_ratio:.1%} of movements align with head direction")
    
    if should_swap:
        print("Applying head/tail swap correction...")
        corrected_aligned_data = {}
        swap_count = 0
        
        for frame in aligned_data:
            corrected_aligned_data[frame] = aligned_data[frame].copy()
            original_head = aligned_data[frame]['head_coord']
            original_tail = aligned_data[frame]['tail_coord']
            corrected_aligned_data[frame]['head_coord'] = original_tail
            corrected_aligned_data[frame]['tail_coord'] = original_head
            swap_count += 1
        
        print(f"Swapped head/tail for {swap_count} frames")
        
        print("Verifying improvement after swap...")        
        test_movement_vectors = []
        test_orientation_vectors = []
        
        for i in range(1, min(len(frames), 20)):
            frame = frames[i]
            prev_frame = frames[i-1]
            
            if frame not in corrected_aligned_data:
                continue
                
            current_pos = np.array(smooth_centroids[frame])
            prev_pos = np.array(smooth_centroids[prev_frame])
            movement_vec = current_pos - prev_pos
            
            if np.linalg.norm(movement_vec) < 0.5:
                continue
                
            head = np.array(corrected_aligned_data[frame]['head_coord'])
            smooth_points = corrected_aligned_data[frame]['smooth_points']
            segment_point = smooth_points[len(smooth_points) // 10]
            orientation_vec = head - np.array(segment_point)
            
            if np.linalg.norm(orientation_vec) < 0.5:
                continue
                
            test_movement_vectors.append(movement_vec)
            test_orientation_vectors.append(orientation_vec)
        
        if len(test_movement_vectors) > 0:
            test_movement_vectors = np.array(test_movement_vectors)
            test_orientation_vectors = np.array(test_orientation_vectors)
            
            test_movement_norms = np.linalg.norm(test_movement_vectors, axis=1, keepdims=True)
            test_orientation_norms = np.linalg.norm(test_orientation_vectors, axis=1, keepdims=True)
            
            test_movement_vectors_norm = test_movement_vectors / np.maximum(test_movement_norms, 1e-10)
            test_orientation_vectors_norm = test_orientation_vectors / np.maximum(test_orientation_norms, 1e-10)
            
            test_dot_products = np.einsum('ij,ij->i', test_movement_vectors_norm, test_orientation_vectors_norm)
            test_forward_aligned = np.sum(test_dot_products > 0)
            test_forward_ratio = test_forward_aligned / len(test_dot_products)
            
            print(f"After swap verification:")
            print(f"  {test_forward_ratio:.1%} of movements now align with head direction")
            
            if test_forward_ratio > forward_ratio + 0.2:  # improvement
                print("Swap verified as beneficial")
                return corrected_aligned_data, True
            else:
                print("Swap did not significantly improve alignment - reverting")
                return aligned_data, False
        else:
            print("Could not verify swap - applying anyway based on original analysis")
            return corrected_aligned_data, True
    
    return aligned_data, False

def plot_head_tail_analysis(aligned_data, smooth_centroids, head_tail_swapped, save_path=None, filename=None):
    """
    Create a visualization showing the head/tail orientation analysis.
    """
    frames = sorted(aligned_data.keys())
    if len(frames) < 10:
        return
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    title_prefix = f"{filename}\n" if filename else ""
    fig.suptitle(f"{title_prefix}Head/Tail Orientation Analysis", fontsize=14, y=0.98)
    
    movement_vectors = []
    orientation_vectors = []
    frame_numbers = []
    dot_products = []
    
    for i in range(1, min(len(frames), 50)):
        frame = frames[i]
        prev_frame = frames[i-1]
        
        current_pos = np.array(smooth_centroids[frame])
        prev_pos = np.array(smooth_centroids[prev_frame])
        movement_vec = current_pos - prev_pos
        
        if np.linalg.norm(movement_vec) < 0.5:
            continue
            
        head = np.array(aligned_data[frame]['head_coord'])
        smooth_points = aligned_data[frame]['smooth_points']
        segment_point = smooth_points[len(smooth_points) // 10]
        orientation_vec = head - np.array(segment_point)
        
        if np.linalg.norm(orientation_vec) < 0.5:
            continue
            
        movement_norm = movement_vec / np.linalg.norm(movement_vec)
        orientation_norm = orientation_vec / np.linalg.norm(orientation_vec)
        
        dot_product = np.dot(movement_norm, orientation_norm)
        
        movement_vectors.append(movement_norm)
        orientation_vectors.append(orientation_norm)
        frame_numbers.append(frame)
        dot_products.append(dot_product)
    
    if len(dot_products) == 0:
        print("No valid movement vectors for head/tail analysis plot")
        return
        
    # Plot 1: Dot products over time
    colors = ['green' if dp > 0 else 'red' for dp in dot_products]
    ax1.scatter(frame_numbers, dot_products, c=colors, alpha=0.7, s=30)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Frame Number')
    ax1.set_ylabel('Movement-Orientation Alignment\n(Dot Product)')
    ax1.set_title('Movement vs Head Direction Alignment')
    ax1.grid(True, alpha=0.3)
    
    forward_count = sum(1 for dp in dot_products if dp > 0)
    backward_count = sum(1 for dp in dot_products if dp < 0)
    total_count = len(dot_products)
    
    forward_pct = (forward_count / total_count) * 100
    backward_pct = (backward_count / total_count) * 100
    
    ax1.text(0.05, 0.95, f"Forward aligned: {forward_count} ({forward_pct:.1f}%)",
             transform=ax1.transAxes, verticalalignment='top', fontsize=10,
             bbox=dict(facecolor='lightgreen', alpha=0.7))
    ax1.text(0.05, 0.85, f"Backward aligned: {backward_count} ({backward_pct:.1f}%)",
             transform=ax1.transAxes, verticalalignment='top', fontsize=10,
             bbox=dict(facecolor='lightcoral', alpha=0.7))
    
    # Plot 2: Movement vectors on path
    x_coords = [smooth_centroids[f][0] for f in frames[:50]]
    y_coords = [smooth_centroids[f][1] for f in frames[:50]]
    ax2.plot(x_coords, y_coords, 'gray', alpha=0.5, linewidth=2)
    
    # Plot head positions and orientation vectors
    for i, frame in enumerate(frames[:20]):
        if frame not in aligned_data:
            continue
            
        centroid = smooth_centroids[frame]
        head = aligned_data[frame]['head_coord']
        tail = aligned_data[frame]['tail_coord']
        
        ax2.plot(centroid[0], centroid[1], 'ko', markersize=4)
        
        ax2.plot(head[0], head[1], 'go', markersize=6, alpha=0.7)
        ax2.plot(tail[0], tail[1], 'ro', markersize=6, alpha=0.7)
        
        # Draw orientation vector (from head toward body)
        smooth_points = aligned_data[frame]['smooth_points']
        if len(smooth_points) > 10:
            segment_point = smooth_points[len(smooth_points) // 10]
            ax2.arrow(head[0], head[1], 
                     segment_point[0] - head[0], segment_point[1] - head[1],
                     head_width=2, head_length=3, fc='blue', ec='blue', alpha=0.6)
    
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    ax2.set_title('Head (green), Tail (red), and Orientation Vectors (blue)')
    ax2.invert_yaxis()
    ax2.set_aspect('equal', adjustable='box')
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', edgecolor='green', label='Head'),
        Patch(facecolor='red', edgecolor='red', label='Tail'),
        Patch(facecolor='blue', edgecolor='blue', label='Orientation Vector'),
        Patch(facecolor='black', edgecolor='black', label='Centroid')
    ]
    ax2.legend(handles=legend_elements, loc='upper right')
    
    swap_text = "HEAD/TAIL SWAP APPLIED" if head_tail_swapped else "No Head/Tail Swap Needed"
    swap_color = 'red' if head_tail_swapped else 'green'
    fig.text(0.5, 0.02, swap_text, ha='center', fontsize=12, fontweight='bold', 
             bbox=dict(facecolor=swap_color, alpha=0.3))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Head/tail analysis plot saved to: {save_path}")
    else:
        plt.savefig('head_tail_analysis.png', dpi=300, bbox_inches='tight')
    
    plt.show()
    plt.close()

def detect_localized_head_tail_issues(aligned_data, smooth_centroids, segment_size=10, confidence_threshold=0.8):
    """
    Detect cases where head/tail detection is wrong only for certain frames.
    """
    print("Analyzing localized head/tail orientation issues...")
    
    frames = sorted(aligned_data.keys())
    if len(frames) < segment_size * 2:
        print("Video too short for segment analysis")
        return [], {}
    
    problematic_segments = []
    segment_analysis = {}
    
    for start_idx in range(0, len(frames) - segment_size + 1, segment_size // 2):
        end_idx = min(start_idx + segment_size, len(frames))
        segment_frames = frames[start_idx:end_idx]
        
        movement_vectors = []
        orientation_vectors = []
        valid_indices = []
        
        for i in range(1, len(segment_frames)):
            frame = segment_frames[i]
            prev_frame = segment_frames[i-1]
            
            current_pos = np.array(smooth_centroids[frame])
            prev_pos = np.array(smooth_centroids[prev_frame])
            movement_vec = current_pos - prev_pos            
            if np.linalg.norm(movement_vec) < 0.5:
                continue
                
            head = np.array(aligned_data[frame]['head_coord'])
            smooth_points = aligned_data[frame]['smooth_points']
            segment_point = smooth_points[len(smooth_points) // 10]
            orientation_vec = head - np.array(segment_point)
            if np.linalg.norm(orientation_vec) < 0.5:
                continue
                
            movement_vectors.append(movement_vec)
            orientation_vectors.append(orientation_vec)
            valid_indices.append(i)
        
        if len(movement_vectors) < 3:
            continue
            
        movement_vectors = np.array(movement_vectors)
        orientation_vectors = np.array(orientation_vectors)
        
        movement_norms = np.linalg.norm(movement_vectors, axis=1, keepdims=True)
        orientation_norms = np.linalg.norm(orientation_vectors, axis=1, keepdims=True)
        
        movement_vectors_norm = movement_vectors / np.maximum(movement_norms, 1e-10)
        orientation_vectors_norm = orientation_vectors / np.maximum(orientation_norms, 1e-10)
        
        dot_products = np.einsum('ij,ij->i', movement_vectors_norm, orientation_vectors_norm)
        
        forward_aligned = np.sum(dot_products > 0)
        backward_aligned = np.sum(dot_products < 0)
        total_movements = len(dot_products)
        
        backward_ratio = backward_aligned / total_movements if total_movements > 0 else 0
        
        segment_info = {
            'start_frame': segment_frames[0],
            'end_frame': segment_frames[-1],
            'start_idx': start_idx,
            'end_idx': end_idx - 1,
            'total_movements': total_movements,
            'forward_aligned': forward_aligned,
            'backward_aligned': backward_aligned,
            'backward_ratio': backward_ratio,
            'is_problematic': backward_ratio > confidence_threshold
        }
        
        segment_analysis[start_idx] = segment_info
        
        print(f"Segment {start_idx}-{end_idx-1} (frames {segment_frames[0]}-{segment_frames[-1]}): "
              f"{backward_aligned}/{total_movements} ({backward_ratio:.1%}) backward-aligned")
        
        if backward_ratio > confidence_threshold:
            problematic_segments.append((start_idx, end_idx - 1))
            print(f"  *** PROBLEMATIC SEGMENT DETECTED ***")
    
    print(f"Found {len(problematic_segments)} problematic segments")
    return problematic_segments, segment_analysis

def correct_localized_head_tail_issues(aligned_data, problematic_segments):
    """
    Correct head/tail positions for specific problematic segments.
    """
    frames = sorted(aligned_data.keys())
    corrected_aligned_data = {}
    corrections_applied = 0
    
    for frame in aligned_data:
        corrected_aligned_data[frame] = aligned_data[frame].copy()
    
    for start_idx, end_idx in problematic_segments:
        print(f"Correcting head/tail swap in segment {start_idx}-{end_idx} (frames {frames[start_idx]}-{frames[end_idx]})")
        
        for i in range(start_idx, end_idx + 1):
            if i < len(frames):
                frame = frames[i]
                original_head = aligned_data[frame]['head_coord']
                original_tail = aligned_data[frame]['tail_coord']
                corrected_aligned_data[frame]['head_coord'] = original_tail
                corrected_aligned_data[frame]['tail_coord'] = original_head
                corrections_applied += 1
    
    print(f"Applied localized head/tail corrections to {corrections_applied} frames")
    return corrected_aligned_data, corrections_applied

def early_frame_head_tail_correction(movement_classification, smooth_centroids, 
                                   early_frame_count=10, confidence_threshold=0.7):
    """
    Special logic for early frames.
    """
    frames = sorted(movement_classification.keys())
    corrected_classification = movement_classification.copy()
    corrections_made = 0
    
    if len(frames) < early_frame_count:
        print("Not enough frames for early frame head/tail correction")
        return corrected_classification
    
    early_frames = frames[:early_frame_count]
    print(f"Analyzing early frame head/tail issues for first {early_frame_count} frames...")
    
    early_backward_count = sum(1 for f in early_frames if movement_classification[f] == 'backward')
    early_forward_count = sum(1 for f in early_frames if movement_classification[f] == 'forward')
    early_stationary_count = sum(1 for f in early_frames if movement_classification[f] == 'stationary')
    
    print(f"Early frames classification: {early_forward_count} forward, {early_backward_count} backward, {early_stationary_count} stationary")
    
    if early_backward_count >= 2:
        print(f"Found {early_backward_count} backward classifications in early frames - investigating...")
        early_start_pos = np.array(smooth_centroids[early_frames[0]])
        early_end_pos = np.array(smooth_centroids[early_frames[-1]])
        overall_early_direction = early_end_pos - early_start_pos
        overall_early_distance = np.linalg.norm(overall_early_direction)
        
        print(f"Overall early movement distance: {overall_early_distance:.2f} pixels")
        
        if overall_early_distance > 2.0:
            overall_early_direction_norm = overall_early_direction / overall_early_distance            
            later_start_idx = early_frame_count
            later_end_idx = min(len(frames), early_frame_count + 20)
            
            if later_end_idx > later_start_idx + 5:
                later_frames = frames[later_start_idx:later_end_idx]
                later_forward_directions = []
                for i in range(len(later_frames) - 1):
                    frame = later_frames[i]
                    next_frame = later_frames[i + 1]
                    
                    if movement_classification[frame] == 'forward':
                        current_pos = np.array(smooth_centroids[frame])
                        next_pos = np.array(smooth_centroids[next_frame])
                        movement_vec = next_pos - current_pos
                        
                        if np.linalg.norm(movement_vec) > 0.5:
                            later_forward_directions.append(movement_vec)
                
                if len(later_forward_directions) >= 3:
                    avg_later_forward = np.mean(later_forward_directions, axis=0)
                    avg_later_forward_norm = avg_later_forward / np.linalg.norm(avg_later_forward)
                    
                    alignment = np.dot(overall_early_direction_norm, avg_later_forward_norm)
                    alignment_angle = np.degrees(np.arccos(np.clip(abs(alignment), 0, 1)))                    
                    print(f"Early movement vs later forward alignment: {alignment:.3f}, angle: {alignment_angle:.1f}°")
                    
                    if alignment > confidence_threshold and alignment_angle < 30:
                        print(f"Early movement aligns well with later forward movement - correcting early backward frames")
                        for frame in early_frames:
                            if movement_classification[frame] == 'backward':
                                frame_idx = frames.index(frame)
                                if frame_idx > 0:
                                    current_pos = np.array(smooth_centroids[frame])
                                    prev_pos = np.array(smooth_centroids[frames[frame_idx - 1]])
                                    frame_movement = current_pos - prev_pos
                                    
                                    if np.linalg.norm(frame_movement) > 0.5:
                                        frame_movement_norm = frame_movement / np.linalg.norm(frame_movement)
                                        frame_alignment = np.dot(frame_movement_norm, avg_later_forward_norm)
                                        frame_angle = np.degrees(np.arccos(np.clip(abs(frame_alignment), 0, 1)))
                                        
                                        if frame_alignment > 0.5 and frame_angle < 45:
                                            corrected_classification[frame] = 'forward'
                                            corrections_made += 1
                                            print(f"Corrected early frame {frame}: backward -> forward (alignment: {frame_alignment:.3f})")
                    else:
                        print(f"Early movement does not align well with later forward movement - no correction applied")
                else:
                    print(f"Not enough later forward movements ({len(later_forward_directions)}) for reference")
            else:
                print("Not enough later frames for reference direction analysis")
        else:
            print("Overall early movement too small for analysis")
    else:
        print("No significant backward classifications in early frames")
    
    if corrections_made > 0:
        print(f"Early frame head/tail correction: made {corrections_made} corrections")
    else:
        print("Early frame head/tail correction: no corrections needed")
    
    return corrected_classification

def get_frames_directory_from_shape_path(shape_path, base_frames_dir):
    shape_basename = os.path.basename(shape_path)
    
    if shape_basename.endswith('.pkl'):
        video_identifier = shape_basename[:-4]
    else:
        video_identifier = shape_basename
    
    print(f"Looking for frames directory for video: {video_identifier}")
    
    frames_directory = os.path.join(base_frames_dir, video_identifier)
    
    if os.path.exists(frames_directory):
        print(f"Found exact match frames directory: {frames_directory}")
        return frames_directory
    else:
        print(f"Exact match not found: {frames_directory}")
        print(f"Searching for similar directories in {base_frames_dir}...")
        
        try:
            if not os.path.exists(base_frames_dir):
                print(f"Base frames directory does not exist: {base_frames_dir}")
                return frames_directory
                
            available_dirs = [d for d in os.listdir(base_frames_dir) if os.path.isdir(os.path.join(base_frames_dir, d))]
            for d in available_dirs:
                if video_identifier in d or d in video_identifier:
                    potential_match = os.path.join(base_frames_dir, d)
                    print(f"Found potential match: {potential_match}")
                    return potential_match
                    
            print(f"No matching directory found for video identifier: {video_identifier}")
            
        except Exception as e:
            print(f"Error searching for directories: {e}")

        return frames_directory

def main(shape_analysis_dir, path_analysis_dir, plots_dir, frames_dir, create_videos=False):
    """
    Main function to process all unprocessed videos.
    create_videos: Boolean indicating whether to create movement videos (default: False for faster processing)
    """
    print("Starting batch processing of all unprocessed videos...")
    print(f"Video creation: {'ENABLED' if create_videos else 'DISABLED'}")
    
    unprocessed_videos = get_all_unprocessed_videos(shape_analysis_dir, path_analysis_dir)
    
    if not unprocessed_videos:
        print("No videos to process. Exiting.")
        return
    
    print(f"\nStarting processing of {len(unprocessed_videos)} videos...")
    
    successful_count = 0
    failed_count = 0
    failed_videos = []    
    start_time = time.time()
    
    for i, (video_path, video_filename) in enumerate(unprocessed_videos):
        print(f"\n[{i+1}/{len(unprocessed_videos)}] Processing: {video_filename}")
        
        success, error_message = process_single_video(video_path, video_filename, plots_dir, path_analysis_dir, frames_dir, create_videos)
        
        if success:
            successful_count += 1
        else:
            failed_count += 1
            failed_videos.append((video_filename, error_message))
        
        elapsed_time = time.time() - start_time
        avg_time_per_video = elapsed_time / (i + 1)
        remaining_videos = len(unprocessed_videos) - (i + 1)
        estimated_remaining_time = avg_time_per_video * remaining_videos
        
        print(f"Progress: {i+1}/{len(unprocessed_videos)} ({(i+1)/len(unprocessed_videos)*100:.1f}%)")
        print(f"Successful: {successful_count}, Failed: {failed_count}")
        print(f"Average time per video: {avg_time_per_video:.1f}s")
        print(f"Estimated remaining time: {estimated_remaining_time/60:.1f} minutes")
    
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Total videos processed: {len(unprocessed_videos)}")
    print(f"Successful: {successful_count}")
    print(f"Failed: {failed_count}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average time per video: {total_time/len(unprocessed_videos):.1f} seconds")
    
    if failed_videos:
        print(f"\nFailed videos ({len(failed_videos)}):")
        for video_filename, error_message in failed_videos:
            print(f"  - {video_filename}: {error_message}")
    
    print(f"\nBatch processing finished!")


# endregion [functions]


# Define directories
SHAPE_ANALYSIS_DIR = "SHAPE_ANALYSIS_DIR_FROM_STEP3"
PATH_ANALYSIS_DIR = "OUTPUT_PATH_ANALYSIS_DIR"
PLOTS_DIR = "OUTPUT_PATH_ANALYSIS_PLOTS_DIR"
FRAMES_DIR = "PATH_TO_FRAMES_DIR"
SEGMENTATION_DIR = "PATH_TO_SEGMENTATION_DIR_FROM_STEP2"
os.makedirs(PATH_ANALYSIS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

main(SHAPE_ANALYSIS_DIR, PATH_ANALYSIS_DIR, PLOTS_DIR, FRAMES_DIR, create_videos=False)