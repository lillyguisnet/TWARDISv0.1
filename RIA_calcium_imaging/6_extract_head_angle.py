"""
This script extracts head angles from head segmentation masks and has visualization helper functions.
"""
import os
import numpy as np
import h5py
from skimage import morphology
from scipy import ndimage
import pandas as pd
import random
import numpy.linalg as la
import re
import cv2

#region [functions]

def load_cleaned_segments_from_h5(filename):
    cleaned_segments = {}
    with h5py.File(filename, 'r') as f:
        num_frames = f.attrs['num_frames']
        object_ids = f.attrs['object_ids']        
        masks_group = f['masks']
        
        for frame_idx in range(num_frames):
            frame_data = {}
            for obj_id in object_ids:
                mask = (masks_group[str(obj_id)][frame_idx] > 0).astype(bool)
                frame_data[obj_id] = mask
            
            cleaned_segments[frame_idx] = frame_data
            print(f"Loading frame {frame_idx}")
    
    print(f"Cleaned segments loaded from {filename}")
    return cleaned_segments

def get_random_unprocessed_video(head_segmentation_dir, final_data_dir):
    all_videos = [f for f in os.listdir(head_segmentation_dir) if f.endswith("_headsegmentation.h5")]
    
    processable_videos = []
    for video in all_videos:
        base_name = video.replace("_headsegmentation.h5", "")
        final_data_base = base_name + "_crop_riasegmentation"        
        if os.path.exists(os.path.join(final_data_dir, final_data_base + "_cleanedalignedsegments.csv")) and \
           not os.path.exists(os.path.join(final_data_dir, final_data_base + "_headangles.csv")):
            processable_videos.append(video)
    
    if not processable_videos:
        raise ValueError("No videos found that need head angle processing.")
    
    return os.path.join(head_segmentation_dir, random.choice(processable_videos))

def get_skeleton(mask):
    return morphology.skeletonize(mask)

def process_all_frames(head_segments):
    """
    Process all frames to generate skeletons from masks.
    """
    skeletons = {}
    skeleton_sizes = []

    for frame_idx, frame_data in head_segments.items():
        frame_skeletons = {}

        for obj_id, mask in frame_data.items():
            skeleton = get_skeleton(mask)
            frame_skeletons[obj_id] = skeleton
            
            size = np.sum(skeleton)
            skeleton_sizes.append(size)
        skeletons[frame_idx] = frame_skeletons

        if frame_idx % 100 == 0:
            print(f"Processed frame {frame_idx}")

    stats = {
        'min_size': np.min(skeleton_sizes),
        'max_size': np.max(skeleton_sizes), 
        'mean_size': np.mean(skeleton_sizes),
        'median_size': np.median(skeleton_sizes),
        'std_size': np.std(skeleton_sizes)
    }

    print("\nSkeleton Statistics:")
    print(f"Minimum size: {stats['min_size']:.1f} pixels")
    print(f"Maximum size: {stats['max_size']:.1f} pixels") 
    print(f"Mean size: {stats['mean_size']:.1f} pixels")
    print(f"Median size: {stats['median_size']:.1f} pixels")
    print(f"Standard deviation: {stats['std_size']:.1f} pixels")

    return skeletons, stats

def truncate_skeleton_fixed(skeleton_dict, keep_pixels=150):
    """
    Keeps only the top specified number of pixels of skeletons.
    """
    truncated_skeletons = {}
    
    for frame_idx, frame_data in skeleton_dict.items():
        frame_truncated = {}
        
        for obj_id, skeleton in frame_data.items():
            skeleton_2d = skeleton[0]
            
            points = np.where(skeleton_2d)
            if len(points[0]) == 0:
                frame_truncated[obj_id] = skeleton
                continue
            
            y_min = np.min(points[0])
            y_max = np.max(points[0])
            original_height = y_max - y_min
            
            cutoff_point = y_min + keep_pixels + 1
            
            truncated = skeleton.copy()
            truncated[0, cutoff_point:, :] = False
            
            new_points = np.where(truncated[0])
            new_height = np.max(new_points[0]) - np.min(new_points[0])
            
            print(f"Frame {frame_idx}, Object {obj_id}:")
            print(f"Original height: {original_height}")
            print(f"New height: {new_height}")
            print(f"Top point: {y_min}")
            print(f"Cutoff point: {cutoff_point}")
            print("------------------------")            
            frame_truncated[obj_id] = truncated            
        truncated_skeletons[frame_idx] = frame_truncated
    
    return truncated_skeletons

def smooth_head_angles(angles, window_size=3, deviation_threshold=15):
    """
    Smooth noise peaks in head angle data by comparing each point with the mean
    of surrounding windows. If a point deviates significantly from both surrounding
    windows' means, it is considered noise and smoothed.
    """    
    angles = np.array(angles)
    smoothed = angles.copy()
    peaks_detected = []
    
    def check_and_smooth_point(i, window_before, window_after):
        if len(window_before) == 0 or len(window_after) == 0:
            return None
            
        mean_before = np.mean(window_before)
        mean_after = np.mean(window_after)
        std_before = np.std(window_before)
        std_after = np.std(window_after)
        
        dev_from_before = abs(angles[i] - mean_before)
        dev_from_after = abs(angles[i] - mean_after)
        
        max_window_std = 10        
        if (dev_from_before > deviation_threshold and 
            dev_from_after > deviation_threshold and
            std_before < max_window_std and 
            std_after < max_window_std):
            
            weight_before = 1 / (std_before + 1e-6)
            weight_after = 1 / (std_after + 1e-6)
            new_value = (mean_before * weight_before + mean_after * weight_after) / (weight_before + weight_after)
            
            return new_value        
        return None
    
    for i in range(len(angles)):
        if i < window_size:
            window_before = angles[0:i]
            window_after = angles[i+1:i+1+window_size]
        elif i >= len(angles) - window_size:
            window_before = angles[i-window_size:i]
            window_after = angles[i+1:]
        else:
            window_before = angles[i-window_size:i]
            window_after = angles[i+1:i+1+window_size]
        
        if i == 0:
            if len(angles) > window_size:
                next_mean = np.mean(angles[1:1+window_size])
                next_std = np.std(angles[1:1+window_size])
                if (abs(angles[i] - next_mean) > deviation_threshold and 
                    next_std < 10):
                    new_value = next_mean
                    peaks_detected.append((i, angles[i], new_value))
                    smoothed[i] = new_value
            continue
            
        if i == len(angles) - 1:
            prev_mean = np.mean(angles[-window_size-1:-1])
            prev_std = np.std(angles[-window_size-1:-1])
            if (abs(angles[i] - prev_mean) > deviation_threshold and 
                prev_std < 10):
                new_value = prev_mean
                peaks_detected.append((i, angles[i], new_value))
                smoothed[i] = new_value
            continue
        
        new_value = check_and_smooth_point(i, window_before, window_after)
        if new_value is not None:
            peaks_detected.append((i, angles[i], new_value))
            smoothed[i] = new_value
    
    return smoothed, peaks_detected

def smooth_head_angles_with_validation(df, id_column='object_id', angle_column='angle_degrees',
                                     window_size=3, deviation_threshold=15):
    """
    Apply head angle smoothing to a DataFrame.
    """
    result_df = df.copy()    
    result_df['original_angle'] = df[angle_column]
    result_df['is_noise_peak'] = False
    result_df['peak_deviation'] = 0.0
    result_df['window_size_used'] = 0
    
    for obj_id in df[id_column].unique():
        obj_mask = df[id_column] == obj_id
        obj_angles = df.loc[obj_mask, angle_column].values
        
        smoothed_angles, peaks = smooth_head_angles(
            obj_angles, 
            window_size=window_size,
            deviation_threshold=deviation_threshold
        )
        
        result_df.loc[obj_mask, angle_column] = smoothed_angles        
        for idx, orig_val, new_val in peaks:
            df_idx = df.loc[obj_mask].iloc[idx].name
            result_df.loc[df_idx, 'is_noise_peak'] = True
            result_df.loc[df_idx, 'peak_deviation'] = abs(orig_val - new_val)
            if idx < window_size:
                result_df.loc[df_idx, 'window_size_used'] = idx
            elif idx >= len(obj_angles) - window_size:
                result_df.loc[df_idx, 'window_size_used'] = len(obj_angles) - idx - 1
            else:
                result_df.loc[df_idx, 'window_size_used'] = window_size
    
    return result_df

def normalize_skeleton_points(points, num_points=100):
    """
    Resample skeleton points to have uniform spacing.
    """
    dists = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
    cum_dists = np.concatenate(([0], np.cumsum(dists)))
    total_length = cum_dists[-1]
    
    even_dists = np.linspace(0, total_length, num_points)
    new_points = np.zeros((num_points, 2))
    for i in range(2):
        new_points[:, i] = np.interp(even_dists, cum_dists, points[:, i])
    
    return new_points, total_length

def gaussian_weighted_curvature(points, window_size=25, sigma=8, restriction_point=0.5):
    """
    Calculate curvature using Gaussian-weighted windows, with stronger smoothing
    and focus on the region between head tip and restriction point.
    """
    valid_points = points[:int(len(points) * restriction_point)]    
    smooth_points = np.zeros_like(valid_points)
    for i in range(2):
        smooth_points[:, i] = ndimage.gaussian_filter1d(valid_points[:, i], sigma=sigma/2)
    
    window_size = window_size if window_size % 2 == 1 else window_size + 1
    pad_width = window_size // 2
    padded_points = np.pad(smooth_points, ((pad_width, pad_width), (0, 0)), mode='edge')
    
    weights = ndimage.gaussian_filter1d(np.ones(window_size), sigma)
    weights /= np.sum(weights)
    
    curvatures = []    
    for i in range(len(smooth_points)):
        window = padded_points[i:i+window_size]
        centroid = np.sum(window * weights[:, np.newaxis], axis=0) / np.sum(weights)
        centered = window - centroid
        cov = np.dot(centered.T, centered * weights[:, np.newaxis]) / np.sum(weights)
        eigvals = la.eigvalsh(cov)
        curvature = eigvals[0] / (eigvals[1] + 1e-10)
        curvatures.append(curvature)
    
    full_curvatures = np.zeros(len(points))
    full_curvatures[:len(curvatures)] = curvatures
    
    return full_curvatures

def calculate_head_angle_with_positions_and_bend(skeleton, prev_angle=None, min_vector_length=5, 
                                             restriction_point=0.4, straight_threshold=3):
    """
    Calculate head angle and bend location along the skeleton.
    Never drops frames - always returns a result dictionary with appropriate error handling.
    For straight worms (angle <= straight_threshold), bend values are set to 0.
    """
    try:
        points = np.column_stack(np.where(skeleton))
        if len(points) == 0:
            return {
                'angle_degrees': prev_angle if prev_angle is not None else 0,
                'error': 'Empty skeleton',
                'head_mag': 0,
                'body_mag': 0,
                'bend_location': 0,
                'bend_magnitude': 0,
                'bend_position': [0, 0],
                'skeleton_points': [[0, 0]],
                'curvature_profile': [0],
                'head_start_pos': [0, 0],
                'head_end_pos': [0, 0],
                'body_start_pos': [0, 0],
                'body_end_pos': [0, 0],
                'head_vector': [0, 0],
                'body_vector': [0, 0]
            }
        
        ordered_points = points[np.argsort(points[:, 0])]
        norm_points, total_length = normalize_skeleton_points(ordered_points, num_points=100)
        
        # Calculate head angle
        head_sections = [0.05, 0.08, 0.1, 0.15]
        body_section = 0.3        
        best_result = None
        max_angle_magnitude = 0
        
        for head_section in head_sections:
            head_end_idx = max(2, int(head_section * len(norm_points)))
            body_start_idx = int((1 - body_section) * len(norm_points))
            
            head_start = norm_points[0]
            head_end = norm_points[head_end_idx]
            body_start = norm_points[body_start_idx]
            body_end = norm_points[-1]
            
            head_vector = head_end - head_start
            body_vector = body_end - body_start
            
            head_mag = np.linalg.norm(head_vector)
            body_mag = np.linalg.norm(body_vector)
            
            if head_mag < min_vector_length or body_mag < min_vector_length:
                continue
            
            dot_product = np.dot(head_vector, body_vector)
            cos_angle = np.clip(dot_product / (head_mag * body_mag), -1.0, 1.0)
            angle_rad = np.arccos(cos_angle)
            angle_deg = np.degrees(angle_rad)
            
            cross_product = np.cross(body_vector, head_vector)
            if cross_product < 0:
                angle_deg = -angle_deg
            
            if prev_angle is not None:
                angle_change = abs(angle_deg - prev_angle)
                if angle_change > 25:
                    continue
            
            if abs(angle_deg) > max_angle_magnitude:
                max_angle_magnitude = abs(angle_deg)
                best_result = {
                    'angle_degrees': float(angle_deg),
                    'head_start_pos': head_start.tolist(),
                    'head_end_pos': head_end.tolist(),
                    'body_start_pos': body_start.tolist(),
                    'body_end_pos': body_end.tolist(),
                    'head_vector': head_vector.tolist(),
                    'body_vector': body_vector.tolist(),
                    'head_mag': float(head_mag),
                    'body_mag': float(body_mag),
                    'head_section': head_section,
                    'skeleton_points': norm_points.tolist(),
                    'error': None
                }
        
        if best_result is None:
            best_result = {
                'angle_degrees': prev_angle if prev_angle is not None else 0,
                'error': 'No valid angle found with current parameters',
                'head_mag': 0,
                'body_mag': 0,
                'head_start_pos': norm_points[0].tolist(),
                'head_end_pos': norm_points[min(5, len(norm_points)-1)].tolist(),
                'body_start_pos': norm_points[max(0, len(norm_points)-10)].tolist(),
                'body_end_pos': norm_points[-1].tolist(),
                'head_vector': [0, 0],
                'body_vector': [0, 0],
                'skeleton_points': norm_points.tolist()
            }
        
        # If the worm is straight (angle within threshold), set all bend-related values to 0
        if abs(best_result['angle_degrees']) <= straight_threshold:
            best_result.update({
                'bend_location': 0,
                'bend_magnitude': 0,
                'bend_position': [0, 0],
                'curvature_profile': np.zeros(len(norm_points)).tolist(),
                'is_straight': True
            })
        else:
            # Calculate curvature only for non-straight worms
            curvatures = gaussian_weighted_curvature(norm_points, window_size=25, sigma=8, 
                                                   restriction_point=restriction_point)
            
            # Find the location of maximum bend (only consider points up to restriction)
            valid_range = int(len(curvatures) * restriction_point)
            max_curvature_idx = np.argmax(np.abs(curvatures[:valid_range]))
            bend_location = max_curvature_idx / len(curvatures)
            bend_magnitude = float(np.abs(curvatures[max_curvature_idx]))
            bend_position = norm_points[max_curvature_idx].tolist()
            
            best_result.update({
                'bend_location': bend_location,
                'bend_magnitude': bend_magnitude,
                'bend_position': bend_position,
                'curvature_profile': curvatures.tolist(),
                'is_straight': False
            })
        
        return best_result
        
    except Exception as e:
        return {
            'angle_degrees': prev_angle if prev_angle is not None else 0,
            'error': f'Unexpected error: {str(e)}',
            'head_mag': 0,
            'body_mag': 0,
            'bend_location': 0,
            'bend_magnitude': 0,
            'bend_position': [0, 0],
            'skeleton_points': [[0, 0]],
            'curvature_profile': [0],
            'head_start_pos': [0, 0],
            'head_end_pos': [0, 0],
            'body_start_pos': [0, 0],
            'body_end_pos': [0, 0],
            'head_vector': [0, 0],
            'body_vector': [0, 0],
            'is_straight': True
        }

def process_skeleton_batch(truncated_skeletons, min_vector_length=5, 
                         restriction_point=0.5, straight_threshold=3,
                         smoothing_window=3, deviation_threshold=15):
    initial_data = []
    frame_results = {}
    
    for frame_idx in sorted(truncated_skeletons.keys()):
        frame_data = truncated_skeletons[frame_idx]
        frame_results[frame_idx] = {}
        
        for obj_id, skeleton_data in frame_data.items():
            skeleton = skeleton_data[0]            
            result = calculate_head_angle_with_positions_and_bend(
                skeleton,
                prev_angle=None,
                min_vector_length=min_vector_length,
                restriction_point=restriction_point,
                straight_threshold=straight_threshold
            )            
            frame_results[frame_idx][obj_id] = result
    
    for frame_idx in sorted(truncated_skeletons.keys()):
        for obj_id in frame_results[frame_idx].keys():
            result = frame_results[frame_idx][obj_id]
            
            if result['error'] is not None:
                prev_valid_frame = None
                prev_valid_result = None
                next_valid_frame = None
                next_valid_result = None
                
                for prev_frame in range(frame_idx - 1, -1, -1):
                    if prev_frame in frame_results and obj_id in frame_results[prev_frame]:
                        prev_result = frame_results[prev_frame][obj_id]
                        if prev_result['error'] is None:
                            prev_valid_frame = prev_frame
                            prev_valid_result = prev_result
                            break
                
                for next_frame in range(frame_idx + 1, max(frame_results.keys()) + 1):
                    if next_frame in frame_results and obj_id in frame_results[next_frame]:
                        next_result = frame_results[next_frame][obj_id]
                        if next_result['error'] is None:
                            next_valid_frame = next_frame
                            next_valid_result = next_result
                            break
                
                if prev_valid_result and next_valid_result:
                    total_frames = next_valid_frame - prev_valid_frame
                    weight_next = (frame_idx - prev_valid_frame) / total_frames
                    weight_prev = 1 - weight_next
                    
                    interpolated_result = interpolate_results(
                        prev_valid_result, next_valid_result, 
                        weight_prev, weight_next,
                        straight_threshold=straight_threshold)
                    interpolated_result['error'] = f"Interpolated between frames {prev_valid_frame} and {next_valid_frame}"
                    frame_results[frame_idx][obj_id] = interpolated_result
                    
                elif prev_valid_result:
                    frames_since_valid = frame_idx - prev_valid_frame
                    decay_factor = 0.9 ** frames_since_valid
                    
                    interpolated_result = decay_result(
                        prev_valid_result, decay_factor,
                        straight_threshold=straight_threshold)
                    interpolated_result['error'] = f"Decayed from previous frame {prev_valid_frame}"
                    frame_results[frame_idx][obj_id] = interpolated_result
                    
                elif next_valid_result:
                    frames_until_valid = next_valid_frame - frame_idx
                    decay_factor = 0.9 ** frames_until_valid
                    
                    interpolated_result = decay_result(
                        next_valid_result, decay_factor,
                        straight_threshold=straight_threshold)
                    interpolated_result['error'] = f"Decayed from next frame {next_valid_frame}"
                    frame_results[frame_idx][obj_id] = interpolated_result
                    
                else:
                    frame_results[frame_idx][obj_id] = {
                        'angle_degrees': 0.0,
                        'head_mag': 0.0,
                        'body_mag': 0.0,
                        'bend_location': 0,
                        'bend_magnitude': 0,
                        'bend_position': [0, 0],
                        'skeleton_points': [[0, 0]],
                        'curvature_profile': [0],
                        'head_start_pos': [0, 0],
                        'head_end_pos': [0, 0],
                        'body_start_pos': [0, 0],
                        'body_end_pos': [0, 0],
                        'head_vector': [0, 0],
                        'body_vector': [0, 0],
                        'is_straight': True,
                        'error': 'No valid frames available for interpolation'
                    }
            
            initial_data.append({
                'frame': frame_idx,
                'object_id': obj_id,
                'angle_degrees': frame_results[frame_idx][obj_id]['angle_degrees'],
                'bend_location': frame_results[frame_idx][obj_id]['bend_location'],
                'bend_magnitude': frame_results[frame_idx][obj_id]['bend_magnitude'],
                'bend_position_y': frame_results[frame_idx][obj_id]['bend_position'][0],
                'bend_position_x': frame_results[frame_idx][obj_id]['bend_position'][1],
                'head_mag': frame_results[frame_idx][obj_id]['head_mag'],
                'body_mag': frame_results[frame_idx][obj_id]['body_mag'],
                'is_straight': frame_results[frame_idx][obj_id].get('is_straight', abs(frame_results[frame_idx][obj_id]['angle_degrees']) <= straight_threshold),
                'error': frame_results[frame_idx][obj_id].get('error', None)
            })
    
    # Convert to DataFrame and apply head angle smoothing
    initial_df = pd.DataFrame(initial_data)
    smoothed_df = smooth_head_angles_with_validation(
        initial_df,
        id_column='object_id',
        angle_column='angle_degrees',
        window_size=smoothing_window,
        deviation_threshold=deviation_threshold
    )
    
    # Final pass: Recalculate bend positions based on smoothed angles
    final_data = []
    for obj_id in smoothed_df['object_id'].unique():
        obj_data = smoothed_df[smoothed_df['object_id'] == obj_id]
        
        prev_angle = None
        for _, row in obj_data.iterrows():
            frame_idx = row['frame']
            skeleton = truncated_skeletons[frame_idx][obj_id][0]            
            new_result = calculate_head_angle_with_positions_and_bend(
                skeleton,
                prev_angle=prev_angle,
                min_vector_length=min_vector_length,
                restriction_point=restriction_point,
                straight_threshold=straight_threshold
            )
            prev_angle = row['angle_degrees']            
            is_straight = abs(row['angle_degrees']) <= straight_threshold
            
            if is_straight:
                bend_location = 0
                bend_magnitude = 0
                bend_position_y = 0
                bend_position_x = 0
            else:
                bend_location = new_result['bend_location']
                bend_magnitude = new_result['bend_magnitude']
                bend_position_y = new_result['bend_position'][0]
                bend_position_x = new_result['bend_position'][1]
            
            final_result = {
                'frame': frame_idx,
                'object_id': obj_id,
                'angle_degrees': row['angle_degrees'],
                'bend_location': bend_location,
                'bend_magnitude': bend_magnitude,
                'bend_position_y': bend_position_y,
                'bend_position_x': bend_position_x,
                'head_mag': new_result['head_mag'],
                'body_mag': new_result['body_mag'],
                'is_noise_peak': row['is_noise_peak'],
                'peak_deviation': row['peak_deviation'],
                'window_size_used': row['window_size_used'],
                'is_straight': is_straight,
                'error': new_result.get('error', None)
            }
            
            final_data.append(final_result)
    
    final_df = pd.DataFrame(final_data)
    
    # Final validation check for correct values
    straight_count = final_df[final_df['is_straight'] == True].shape[0]
    non_straight_with_bends = final_df[(final_df['is_straight'] == False) & (final_df['bend_location'] > 0)].shape[0]
    print(f"Final validation - Straight frames: {straight_count}, Non-straight with bend values: {non_straight_with_bends}")
    final_df['has_warning'] = final_df['error'].notna()
    warning_count = final_df['has_warning'].sum()
    if warning_count > 0:
        print(f"\nWarning Summary:")
        print(f"Total frames with warnings: {warning_count}")
        print("\nSample of warnings:")
        for error in final_df[final_df['has_warning']]['error'].unique()[:5]:
            count = final_df[final_df['error'] == error].shape[0]
            print(f"- {error}: {count} frames")
    
    return final_df

def interpolate_results(prev_result, next_result, weight_prev, weight_next, straight_threshold=3):
    """
    Interpolate between two results based on weights.
    If the interpolated angle is within the straight threshold, all bend values are set to 0.
    """
    def interpolate_value(prev_val, next_val):
        if isinstance(prev_val, (list, np.ndarray)):
            result = []
            for p, n in zip(prev_val, next_val):
                if isinstance(p, (list, np.ndarray)):
                    result.append(interpolate_value(p, n))
                else:
                    result.append(weight_prev * p + weight_next * n)
            return result
        return weight_prev * prev_val + weight_next * next_val
    
    interpolated = {}
    angle_degrees = interpolate_value(prev_result['angle_degrees'], next_result['angle_degrees'])
    
    if abs(angle_degrees) <= straight_threshold:
        return {
            'angle_degrees': angle_degrees,
            'head_mag': interpolate_value(prev_result['head_mag'], next_result['head_mag']),
            'body_mag': interpolate_value(prev_result['body_mag'], next_result['body_mag']),
            'bend_location': 0,
            'bend_magnitude': 0,
            'bend_position': [0, 0],
            'head_start_pos': interpolate_value(prev_result['head_start_pos'], next_result['head_start_pos']),
            'head_end_pos': interpolate_value(prev_result['head_end_pos'], next_result['head_end_pos']),
            'body_start_pos': interpolate_value(prev_result['body_start_pos'], next_result['body_start_pos']),
            'body_end_pos': interpolate_value(prev_result['body_end_pos'], next_result['body_end_pos']),
            'head_vector': interpolate_value(prev_result['head_vector'], next_result['head_vector']),
            'body_vector': interpolate_value(prev_result['body_vector'], next_result['body_vector']),
            'is_straight': True
        }
    
    for key in prev_result.keys():
        if key in ['error', 'is_straight']:
            continue
        interpolated[key] = interpolate_value(prev_result[key], next_result[key])
    
    interpolated['is_straight'] = False
    return interpolated

def decay_result(base_result, decay_factor, straight_threshold=3):
    """
    Apply decay to a result for interpolation.
    """
    def apply_decay(value):
        if isinstance(value, (list, np.ndarray)):
            return [apply_decay(v) for v in value]
        else:
            return value * decay_factor
    
    angle_degrees = base_result['angle_degrees'] * decay_factor    
    if abs(angle_degrees) <= straight_threshold:
        return {
            'angle_degrees': angle_degrees,
            'head_mag': base_result['head_mag'] * decay_factor,
            'body_mag': base_result['body_mag'] * decay_factor,
            'bend_location': 0,
            'bend_magnitude': 0,
            'bend_position': [0, 0],
            'head_start_pos': apply_decay(base_result['head_start_pos']),
            'head_end_pos': apply_decay(base_result['head_end_pos']),
            'body_start_pos': apply_decay(base_result['body_start_pos']),
            'body_end_pos': apply_decay(base_result['body_end_pos']),
            'head_vector': apply_decay(base_result['head_vector']),
            'body_vector': apply_decay(base_result['body_vector']),
            'is_straight': True
        }
    
    decayed = {}
    for key, value in base_result.items():
        if key in ['error', 'is_straight']:
            continue
        decayed[key] = apply_decay(value)
    
    decayed['is_straight'] = False
    return decayed

def save_head_angles_with_side_correction(filename, results_df, final_data_dir):
    base_name = os.path.basename(filename).replace("_headsegmentation.h5", "")
    final_data_base = base_name + "_crop_riasegmentation_cleanedalignedsegments"

    final_data_path = os.path.join(final_data_dir, final_data_base + ".csv")
    final_df = pd.read_csv(final_data_path)

    print(f"Loaded final data from {final_data_path}")
    print(f"Final data shape: {final_df.shape}")
    print(f"Results df shape: {results_df.shape}")

    columns_to_overwrite = [col for col in results_df.columns if col in final_df.columns]
    print(f"Columns that will be overwritten: {columns_to_overwrite}")

    merged_df = pd.merge(final_df, results_df, 
                        left_on=['frame'],
                        right_on=['frame'],
                        how='left',
                        suffixes=('_old', ''))

    for col in columns_to_overwrite:
        if col + '_old' in merged_df.columns:
            merged_df.drop(columns=[col + '_old'], inplace=True)

    merged_df['angle_degrees_corrected'] = merged_df.apply(
        lambda row: -row['angle_degrees'] if row['side_position'] == 'right' else row['angle_degrees'], 
        axis=1
    )

    print(f"Number of right-side angles corrected: {(merged_df['side_position'] == 'right').sum()}")
    print(f"Number of left-side angles: {(merged_df['side_position'] == 'left').sum()}")

    output_path = os.path.join(final_data_dir, final_data_base + "_headangles.csv")

    merged_df.to_csv(output_path, index=False)
    print(f"Saved merged df to: {output_path}")
    print(f"Final merged df shape: {merged_df.shape}")
    print(f"Final columns: {merged_df.columns.tolist()}")

    if os.path.exists(final_data_path):
        os.remove(final_data_path)
        print(f"Deleted existing file: {final_data_path}")
        
    return merged_df

def create_layered_mask_video(image_dir, bottom_masks_dict, top_masks_dict, angles_df,
                           output_path, fps=10, bottom_alpha=0.5, top_alpha=0.7):
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

    def create_mask_overlay(image, frame_masks, mask_colors, alpha):
        overlay = np.zeros_like(image)
        
        for mask_id, mask in frame_masks.items():
            if mask.dtype != bool:
                mask = mask > 0.5
            
            if mask.ndim > 2:
                mask = mask.squeeze()
            
            mask_resized = cv2.resize(mask.astype(np.uint8), 
                                    (image.shape[1], image.shape[0]), 
                                    interpolation=cv2.INTER_NEAREST)            
            colored_mask = np.zeros_like(image)
            colored_mask[mask_resized == 1] = mask_colors[mask_id]            
            overlay = cv2.addWeighted(overlay, 1, colored_mask, 1, 0)
        
        return overlay

    def find_skeleton_tip(mask):
        if mask.dtype != bool:
            mask = mask > 0.5
        
        if mask.ndim > 2:
            mask = mask.squeeze()
            
        y_coords, x_coords = np.where(mask)
        if len(y_coords) == 0:
            return None
            
        top_idx = np.argmin(y_coords)
        return (y_coords[top_idx], x_coords[top_idx])

    def add_angle_text(image, angle, position, font_scale=0.7):
        """Add angle text at the given position with background"""
        if position is None or angle is None:
            return image
            
        y, x = position
        angle_text = f"{angle:.1f} deg"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_thickness = 2        
        (text_width, text_height), baseline = cv2.getTextSize(
            angle_text, font, font_scale, font_thickness)        
        text_x = int(x + 30)
        text_y = int(y)        
        padding = 5
        cv2.rectangle(image, 
                     (text_x - padding, text_y - text_height - padding),
                     (text_x + text_width + padding, text_y + padding),
                     (0, 0, 0), -1)  # Black background
        
        cv2.putText(image, angle_text,
                   (text_x, text_y + text_height),
                   font, font_scale, (255, 255, 255),  # White text
                   font_thickness)
        
        return image

    image_files = [f for f in os.listdir(image_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    frame_numbers = []
    for img_file in image_files:
        match = re.search(r'(\d+)', img_file)
        if match:
            frame_numbers.append((int(match.group(1)), img_file))
    
    frame_numbers.sort(key=lambda x: x[0])
    
    if not frame_numbers:
        raise ValueError(f"No image files found in {image_dir}")

    first_image = cv2.imread(os.path.join(image_dir, frame_numbers[0][1]))
    if first_image is None:
        raise ValueError(f"Could not read first image: {frame_numbers[0][1]}")
    
    height, width, _ = first_image.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    bottom_mask_ids = set()
    top_mask_ids = set()
    for masks in bottom_masks_dict.values():
        bottom_mask_ids.update(masks.keys())
    for masks in top_masks_dict.values():
        top_mask_ids.update(masks.keys())
    
    mid_point = len(COLORS) // 2
    bottom_colors = COLORS[:mid_point]
    top_colors = COLORS[mid_point:] + COLORS[:max(0, len(top_mask_ids) - len(COLORS) // 2)]
    
    bottom_mask_colors = {mask_id: bottom_colors[i % len(bottom_colors)] 
                         for i, mask_id in enumerate(bottom_mask_ids)}
    top_mask_colors = {mask_id: top_colors[i % len(top_colors)] 
                      for i, mask_id in enumerate(top_mask_ids)}

    angles_dict = angles_df.set_index('frame')['angle_degrees_corrected'].to_dict()

    for frame_number, image_file in frame_numbers:
        try:
            image_path = os.path.join(image_dir, image_file)
            frame = cv2.imread(image_path)
            if frame is None:
                raise ValueError(f"Could not read image: {image_file}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if frame_number < 5:
                print(f"Processing frame {frame_number}, file: {image_file}")
                print(f"Bottom masks available: {frame_number in bottom_masks_dict}")
                print(f"Top masks available: {frame_number in top_masks_dict}")
            
            final_frame = frame.copy()
            
            if frame_number in bottom_masks_dict:
                bottom_overlay = create_mask_overlay(frame, 
                                                  bottom_masks_dict[frame_number],
                                                  bottom_mask_colors, 
                                                  bottom_alpha)
                final_frame = cv2.addWeighted(final_frame, 1, bottom_overlay, bottom_alpha, 0)
            
            tip_position = None
            if frame_number in top_masks_dict:
                top_overlay = create_mask_overlay(frame,
                                               top_masks_dict[frame_number],
                                               top_mask_colors,
                                               top_alpha)
                final_frame = cv2.addWeighted(final_frame, 1, top_overlay, top_alpha, 0)
                
                if top_masks_dict[frame_number]:
                    first_mask_id = next(iter(top_masks_dict[frame_number]))
                    tip_position = find_skeleton_tip(top_masks_dict[frame_number][first_mask_id])
            
            if frame_number in angles_dict and tip_position is not None:
                final_frame = add_angle_text(final_frame, angles_dict[frame_number], tip_position)

            out.write(cv2.cvtColor(final_frame, cv2.COLOR_RGB2BGR))

        except Exception as e:
            print(f"Error processing frame {frame_number} ({image_file}): {str(e)}")
            continue

    out.release()
    print(f"Video saved to {output_path}")

#endregion [functions]

head_segmentation_dir = "PATH_TO_HEAD_SEGMENTATION_DIR"
final_data_dir = "PATH_TO_OUTPUT_DATA_DIR"

filename = get_random_unprocessed_video(head_segmentation_dir, final_data_dir)
head_segments = load_cleaned_segments_from_h5(filename)

skeletons, skeleton_stats = process_all_frames(head_segments)
truncated_skeletons = truncate_skeleton_fixed(skeletons, keep_pixels=400)
results_df = process_skeleton_batch(
    truncated_skeletons,
    min_vector_length=5,
    restriction_point=0.5,
    straight_threshold=3,
    smoothing_window=3,
    deviation_threshold=12
)

merged_df = save_head_angles_with_side_correction(filename, results_df, final_data_dir)