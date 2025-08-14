"""
This script is used for shape analysis of the worm skeleton.
There are several visualization helper functions.
"""

import pickle
import numpy as np
from scipy import interpolate
from skimage import morphology, graph, measure
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import binary_dilation
from matplotlib.gridspec import GridSpec
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy.signal import welch, find_peaks
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from collections import defaultdict
import networkx as nx
import os
import gc
import random
import time

# region [functions]

class DiscontinuousSkeletonError(Exception):
    """Debugging error raised when skeleton is discontinuous and cannot find path between endpoints"""
    pass

def smooth_metric(data, window_length=11, poly_order=3):
    return savgol_filter(data, window_length, poly_order)

def crop_around_mask(mask, padding=5):
    if mask.dtype == bool:
        mask_bool = mask
    else:
        mask_bool = mask > 0
    
    coords = np.argwhere(mask_bool)
    
    min_row, min_col = coords.min(axis=0)
    max_row, max_col = coords.max(axis=0)
    
    min_row = max(0, min_row - padding)
    min_col = max(0, min_col - padding)
    max_row = min(mask.shape[0], max_row + padding + 1)
    max_col = min(mask.shape[1], max_col + padding + 1)
    
    cropped_mask = mask[min_row:max_row, min_col:max_col]
    
    bbox = (min_row, min_col, max_row, max_col)
    
    return cropped_mask, bbox

def clean_mask(mask):
    cropped_mask, bbox = crop_around_mask(mask, padding=5)
    
    if cropped_mask.dtype == bool:
        cropped_mask = cropped_mask.astype(np.uint8)
    
    labeled, num_features = ndimage.label(cropped_mask)
    
    if num_features == 0:
        return mask
    
    sizes = ndimage.sum(cropped_mask, labeled, range(1, num_features + 1))
    largest_label = sizes.argmax() + 1
    cleaned_cropped_mask = (labeled == largest_label)
    
    cleaned_mask = np.zeros_like(mask, dtype=bool)
    min_row, min_col, max_row, max_col = bbox
    cleaned_mask[min_row:max_row, min_col:max_col] = cleaned_cropped_mask
    
    return cleaned_mask

def get_skeleton(mask):
    try:
        skeleton = morphology.skeletonize(mask)
        return skeleton
    except Exception as e:
        raise

def find_endpoints_and_junctions(coords):
    if isinstance(coords, np.ndarray) and coords.dtype == bool:
        coords = np.argwhere(coords)
    neighbor_count = defaultdict(int)
    
    def are_neighbors(p1, p2):
        return np.all(np.abs(p1 - p2) <= 1) and not np.all(p1 == p2)

    for i, p1 in enumerate(coords):
        for j, p2 in enumerate(coords):
            if i != j and are_neighbors(p1, p2):
                neighbor_count[tuple(p1)] += 1

    endpoints = []
    junctions = []

    # Classify points based on neighbor count
    for point in coords:
        point_tuple = tuple(point)
        if neighbor_count[point_tuple] == 1:
            endpoints.append(point)
        elif neighbor_count[point_tuple] > 2:
            junctions.append(point)

    return np.array(endpoints), np.array(junctions)

def calculate_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return np.degrees(angle)

def find_furthest_endpoints_along_skeleton(skeleton):
    endpoints, _ = find_endpoints_and_junctions(skeleton)
    
    if len(endpoints) <= 2:
        return None  # Not enough endpoints/Error
    
    max_distance = 0
    furthest_pair = None
    
    for i in range(len(endpoints)):
        for j in range(i+1, len(endpoints)):
            print(f"Checking pair {i} and {j}")
            cost_array = np.where(skeleton, 1, np.inf)
            start = tuple(endpoints[i])
            end = tuple(endpoints[j]) 
            try:
                path_indices, cost = graph.route_through_array(cost_array, start, end)
            except ValueError as e:
                if "no minimum-cost path was found" in str(e):
                    print(f"Discontinuous skeleton detected: {e}")
                    raise DiscontinuousSkeletonError("Skeleton is discontinuous - cannot find path between endpoints")
                else:
                    raise

            path = np.array(path_indices)

            distance = len(path)
            
            if distance > max_distance:
                max_distance = distance
                furthest_pair = (endpoints[i], endpoints[j])
    
    return furthest_pair

def order_segments(segments):
    """
    Order segments coordinates so that endpoints are at the beginning and end
    """
    segments_set = set(map(tuple, segments))
    
    graph_dict = defaultdict(list)
    for x, y in segments_set:
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                neighbor = (x + dx, y + dy)
                if neighbor in segments_set:
                    graph_dict[(x, y)].append(neighbor)
    
    endpoints = [point for point, neighbors in graph_dict.items() if len(neighbors) == 1]
    
    if len(endpoints) != 2:
        raise DiscontinuousSkeletonError(f"Expected exactly two endpoints, but found {len(endpoints)}. This indicates a discontinuous skeleton.")
    
    start, end = endpoints
    ordered = [start]
    current = start
    
    while current != end:
        next_point = [p for p in graph_dict[current] if p not in ordered][0]
        ordered.append(next_point)
        current = next_point
    
    return np.array(ordered)

def calculate_orientation_difference(segment1, segment2, p1, p2):
    idx1 = np.where((segment1 == p1).all(axis=1))[0][0]
    idx2 = np.where((segment2 == p2).all(axis=1))[0][0]

    if idx1 == 0:
        points1 = segment1[:3]
    else:
        points1 = segment1[max(0, idx1-2):idx1+1]
    
    if idx2 == 0:
        points2 = segment2[:3]
    else:
        points2 = segment2[max(0, idx2-2):idx2+1]
       
    vec1 = points1[-1] - points1[0]
    vec2 = points2[-1] - points2[0]
    
    dot_product = np.dot(vec1, vec2)
    norms = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    
    cos_angle = dot_product / norms
    
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    angle_degrees = np.degrees(angle)
    
    return angle_degrees

def calculate_curvature(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    angle = np.arctan2(np.cross(v1, v2), np.dot(v1, v2))
    
    if angle > 0:
        angle = np.pi - angle
    else:
        angle = -np.pi - angle
    
    d1 = np.linalg.norm(v1)
    d2 = np.linalg.norm(v2)
    
    curvature = angle / ((d1 + d2) / 2 + 1e-10)
    return curvature

def close_gap(skeleton, start, end):
    """
    Close gaps in the skeleton by finding the shortest path between two points,
    avoiding both passing through and being neighbors with skeleton elements
    (except start and end points).
    """
    print(f"close_gap: Finding path from {start} to {end}")
    
    skeleton_points = np.argwhere(skeleton)
    
    all_points = np.vstack([skeleton_points, [start], [end]])
    
    padding = 15
    min_row = max(0, all_points[:, 0].min() - padding)
    max_row = min(skeleton.shape[0], all_points[:, 0].max() + padding + 1)
    min_col = max(0, all_points[:, 1].min() - padding)
    max_col = min(skeleton.shape[1], all_points[:, 1].max() + padding + 1)
    
    print(f"close_gap: Original shape {skeleton.shape}, cropped to ({min_row}:{max_row}, {min_col}:{max_col})")
    
    cropped_skeleton = skeleton[min_row:max_row, min_col:max_col]
    
    local_start = (start[0] - min_row, start[1] - min_col)
    local_end = (end[0] - min_row, end[1] - min_col)
    
    print(f"close_gap: Local coordinates - start: {local_start}, end: {local_end}")
    
    cost_map = np.ones_like(cropped_skeleton, dtype=float)
    
    occupied = binary_dilation(cropped_skeleton, iterations=2)
    cost_map[occupied] = np.inf

    cost_map[local_start] = 1
    cost_map[local_end] = 1
    
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if 0 <= local_start[0]+dx < cropped_skeleton.shape[0] and 0 <= local_start[1]+dy < cropped_skeleton.shape[1]:
                cost_map[local_start[0]+dx, local_start[1]+dy] = 1
            if 0 <= local_end[0]+dx < cropped_skeleton.shape[0] and 0 <= local_end[1]+dy < cropped_skeleton.shape[1]:
                cost_map[local_end[0]+dx, local_end[1]+dy] = 1

    height, width = cropped_skeleton.shape
    print(f"close_gap: Creating graph of size {height}x{width} = {height*width} nodes")
    G = nx.grid_2d_graph(height, width)

    for (u, v) in G.edges():
        if cost_map[u] == np.inf or cost_map[v] == np.inf:
            G.remove_edge(u, v)
        else:
            G[u][v]['weight'] = (cost_map[u] + cost_map[v]) / 2

    try:
        local_path = nx.shortest_path(G, local_start, local_end, weight='weight')
        print(f"close_gap: Found path of length {len(local_path)}")
    except nx.NetworkXNoPath:
        print(f"close_gap: No path found")
        return None

    global_path = [(r + min_row, c + min_col) for r, c in local_path]
    
    return global_path

def adjust_self_touching_skeleton(skeleton):
    endpoints, junctions = find_endpoints_and_junctions(skeleton)

    # If no endpoints, it's a perfect loop. Remove point with max curvature.
    if len(endpoints) == 0:
        contours = measure.find_contours(skeleton, 0.5)
        main_contour = max(contours, key=len)

        max_curvature = -np.inf
        max_curvature_point = None

        half_window = 5 // 2

        for i in range(len(main_contour)):
            prev = (i - half_window) % len(main_contour)
            next = (i + half_window) % len(main_contour)

            p1 = main_contour[prev]
            p2 = main_contour[i]
            p3 = main_contour[next]

            curvature = calculate_curvature(p1, p2, p3)

            if curvature > max_curvature:
                max_curvature = curvature
                max_curvature_point = tuple(map(int, p2))
            
            skeleton_points = np.argwhere(skeleton)
            distances = cdist([max_curvature_point], skeleton_points)
            nearest_index = np.argmin(distances)
            point_to_remove = tuple(skeleton_points[nearest_index])

        skeleton[point_to_remove[0], point_to_remove[1]] = False

    endpoints, junctions = find_endpoints_and_junctions(skeleton)
    
    while len(endpoints) > 2:
        try:
            furthest_pair = find_furthest_endpoints_along_skeleton(skeleton)
        except DiscontinuousSkeletonError as e:
            print(f"adjust_self_touching_skeleton: Discontinuous skeleton detected in finding furthest endpoints: {e}")
            raise
        
        if furthest_pair:           
            cost_array = np.where(skeleton, 1, np.inf)
            start = tuple(furthest_pair[0])
            end = tuple(furthest_pair[1]) 
            path_indices, cost = graph.route_through_array(cost_array, start, end)
            path = np.array(path_indices)
            
            new_skeleton = np.zeros_like(skeleton)
            new_skeleton[path[:, 0], path[:, 1]] = True
            skeleton = new_skeleton

        endpoints, junctions = find_endpoints_and_junctions(skeleton)

    while len(junctions) > 0:
        # Remove junctions
        for junction in junctions:
            skeleton[junction[0], junction[1]] = 0
        
        # Find all segments
        labeled_skeleton, num_segments = morphology.label(skeleton, connectivity=2, return_num=True)
        segments = [np.argwhere(labeled_skeleton == i) for i in range(1, num_segments+1)]
        
        while len(segments) > 1:
            segments_to_keep = []
            for seg in segments:
                if len(seg) >= 4:
                    segments_to_keep.append(seg)
                else:
                    for point in seg:
                        skeleton[point[0], point[1]] = False
            
            segments = segments_to_keep
            
            if len(segments) == 1:
                break
            
            ordered_segments = []
            for seg in (segments):
                try:
                    ordered_seg = order_segments(seg)
                    ordered_segments.append(ordered_seg)
                except DiscontinuousSkeletonError as e:
                    print(f"adjust_self_touching_skeleton: Discontinuous skeleton detected in segment ordering: {e}")
                    raise
            segments = ordered_segments
            
            endpoints_segs = []
            for _, segs in enumerate(segments):
                endpoints, junc = find_endpoints_and_junctions(segs)
                endpoints_segs.append(endpoints)
            all_endpoints = np.vstack(endpoints_segs)
            distances = np.linalg.norm(all_endpoints[:, None] - all_endpoints, axis=2)
            np.fill_diagonal(distances, np.inf)

            num_segments = len(segments)
            endpoint_to_segment = {}

            for seg_idx, segment in enumerate(segments):
                for point in segment:
                    endpoint_to_segment[tuple(point)] = seg_idx

            original_min_distance = np.inf
            original_closest_pair = None
            for i in range(len(all_endpoints)):
                for j in range(i + 1, len(all_endpoints)):
                    if endpoint_to_segment[tuple(all_endpoints[i])] != endpoint_to_segment[tuple(all_endpoints[j])]:
                        dist = distances[i, j]
                        if dist < original_min_distance:
                            original_min_distance = dist
                            original_closest_pair = (i, j)

            connect_p1 = None
            connect_p2 = None

            while connect_p1 is None and connect_p2 is None:
                if np.isinf(distances).all():
                    i, j = original_closest_pair
                    connect_p1 = all_endpoints[i]
                    connect_p2 = all_endpoints[j]
                    break

                i, j = np.unravel_index(distances.argmin(), distances.shape)
                p1, p2 = all_endpoints[i], all_endpoints[j]

                seg1_idx = endpoint_to_segment[tuple(p1)]
                seg2_idx = endpoint_to_segment[tuple(p2)]

                if seg1_idx != seg2_idx:
                    angle = calculate_orientation_difference(segments[seg1_idx], segments[seg2_idx], p1, p2)
                    if angle > 120 or angle < 30:
                        connect_p1 = p1
                        connect_p2 = p2
                    else:
                        distances[i, j] = distances[j, i] = np.inf
                else:
                    distances[i, j] = distances[j, i] = np.inf

            path = close_gap(skeleton, connect_p1, connect_p2)
            if path is None:
                i, j = original_closest_pair
                connect_p1 = all_endpoints[i]
                connect_p2 = all_endpoints[j]
                path = close_gap(skeleton, connect_p1, connect_p2)

            if path is None:
                smallest_segment_idx = min(range(len(segments)), key=lambda i: len(segments[i]))
                for point in segments[smallest_segment_idx]:
                    skeleton[tuple(point)] = 0
            else:
                for pixel in path:
                    skeleton[pixel] = 1

            skeleton = get_skeleton(skeleton)

            labeled_skeleton, num_segments = morphology.label(skeleton, connectivity=2, return_num=True)
            segments = [np.argwhere(labeled_skeleton == i) for i in range(1, num_segments+1)]

        endpoints, junctions = find_endpoints_and_junctions(skeleton)

    # Ensure exactly two endpoints
    endpoints, _ = find_endpoints_and_junctions(skeleton)
    if len(endpoints) > 2:
        try:
            furthest_pair = find_furthest_endpoints_along_skeleton(skeleton)
        except DiscontinuousSkeletonError as e:
            print(f"adjust_self_touching_skeleton: Discontinuous skeleton detected in ensuring two endpoints: {e}")
            raise
        
        if furthest_pair:           
            cost_array = np.where(skeleton, 1, np.inf)
            start = tuple(furthest_pair[0])
            end = tuple(furthest_pair[1]) 
            path_indices, cost = graph.route_through_array(cost_array, start, end)

            path = np.array(path_indices)
            
            return path                      
    
    return np.argwhere(skeleton)

def gaussian_weighted_curvature(points, window_size, sigma):
    window_size = window_size if window_size % 2 == 1 else window_size + 1
    pad_width = window_size // 2
    padded_points = np.pad(points, ((pad_width, pad_width), (0, 0)), mode='edge')
    
    weights = ndimage.gaussian_filter1d(np.ones(window_size), sigma)
    weights /= np.sum(weights)
    
    curvatures = []
    for i in range(len(points)):
        window = padded_points[i:i+window_size]
        centroid = np.sum(window * weights[:, np.newaxis], axis=0) / np.sum(weights)
        centered = window - centroid
        cov = np.dot(centered.T, centered * weights[:, np.newaxis]) / np.sum(weights)
        eigvals, eigvecs = np.linalg.eig(cov)
        sort_indices = np.argsort(eigvals)[::-1]
        eigvals = eigvals[sort_indices]
        curvature = eigvals[1] / (eigvals[0] + eigvals[1])
        curvatures.append(curvature)
    
    return np.array(curvatures)

def analyze_periodicity(curvature):
    fft = np.fft.fft(curvature)
    frequencies = np.fft.fftfreq(len(curvature))
    dominant_freq = frequencies[np.argmax(np.abs(fft[1:])) + 1]
    return dominant_freq

def track_endpoints(frames, smooth_points):
    endpoints = []
    for skeleton in smooth_points:
        endpoints.append((skeleton[0], skeleton[-1]))
    return np.array(endpoints)

def group_endpoints(endpoints, window_size):
    """
    Group endpoints with Hungarian algorithm
    Returns grouped endpoints and the mapping of original indices to group indices.
    """
    grouped_endpoints = np.zeros((2, window_size, 2))
    index_mapping = np.zeros((window_size, 2), dtype=int)
    
    # Use the first frame as initial groups
    grouped_endpoints[0, 0] = endpoints[0, 0]
    grouped_endpoints[1, 0] = endpoints[0, 1]
    index_mapping[0] = [0, 1]
    
    for i in range(1, window_size):
        distances = np.linalg.norm(endpoints[i][:, np.newaxis] - grouped_endpoints[:, i-1], axis=2)
        
        row_ind, col_ind = linear_sum_assignment(distances)
        
        grouped_endpoints[col_ind, i] = endpoints[i, row_ind]
        index_mapping[i] = col_ind
    
    return grouped_endpoints, index_mapping

def continuous_head_identification(endpoints, frames, window_size=5, error_threshold=5, close_threshold=10):
    head_positions = []
    tail_positions = []
    confidences = []
    current_head_index = 0
    
    def endpoints_too_close(frame_endpoints, threshold=close_threshold):
        if len(frame_endpoints) < 2:
            return False
        distance = np.linalg.norm(frame_endpoints[0] - frame_endpoints[1])
        return distance < threshold
    
    def filter_valid_frames(window_endpoints, window_start_idx, current_frame_idx):
        valid_endpoints = []
        valid_indices = []
        
        for i, frame_endpoints in enumerate(window_endpoints):
            absolute_frame_idx = window_start_idx + i
            if absolute_frame_idx == current_frame_idx:
                valid_endpoints.append(frame_endpoints)
                valid_indices.append(i)
            elif not endpoints_too_close(frame_endpoints):
                valid_endpoints.append(frame_endpoints)
                valid_indices.append(i)
        
        return np.array(valid_endpoints) if valid_endpoints else None, valid_indices
    
    for i in range(len(frames)):
        start = max(0, i - window_size // 2)
        end = min(len(frames), start + window_size)
        if end == len(frames):
            start = max(0, end - window_size)
        
        current_frame_endpoints = endpoints[i]
        
        if endpoints_too_close(current_frame_endpoints):
            print(f"Frame {i}: Endpoints too close ({np.linalg.norm(current_frame_endpoints[0] - current_frame_endpoints[1]):.1f} px), using fallback method")
            
            if len(head_positions) > 0:
                prev_head_position = head_positions[-1]
                distances = np.linalg.norm(current_frame_endpoints - prev_head_position, axis=1)
                closest_index = np.argmin(distances)
                current_head_position = current_frame_endpoints[closest_index]
                current_tail_position = current_frame_endpoints[1 - closest_index]
                current_head_index = closest_index
                confidence = 0.6
            else:
                current_head_index = 0
                current_head_position = current_frame_endpoints[0]
                current_tail_position = current_frame_endpoints[1]
                confidence = 0.5
        else:
            window_endpoints = endpoints[start:end]
            
            filtered_endpoints, valid_indices = filter_valid_frames(window_endpoints, start, i)
            
            if filtered_endpoints is not None and len(filtered_endpoints) >= 2:
                grouped_endpoints, index_mapping = group_endpoints(filtered_endpoints, len(filtered_endpoints))
                movements = np.diff(grouped_endpoints, axis=1)
                cumulative_movement = np.sum(np.abs(movements), axis=(1, 2))
                head_group_index = np.argmax(cumulative_movement)
                
                current_frame_in_filtered = None
                for idx, valid_idx in enumerate(valid_indices):
                    if start + valid_idx == i:
                        current_frame_in_filtered = idx
                        break
                
                if current_frame_in_filtered is not None:
                    current_head_index = np.where(index_mapping[current_frame_in_filtered] == head_group_index)[0][0]
                    current_head_position = endpoints[i, current_head_index]
                    current_tail_position = endpoints[i, 1 - current_head_index]
                    confidence = 0.9
                else:
                    if len(head_positions) > 0:
                        prev_head_position = head_positions[-1]
                        distances = np.linalg.norm(current_frame_endpoints - prev_head_position, axis=1)
                        closest_index = np.argmin(distances)
                        current_head_position = current_frame_endpoints[closest_index]
                        current_tail_position = current_frame_endpoints[1 - closest_index]
                        current_head_index = closest_index
                        confidence = 0.7
                    else:
                        current_head_index = 0
                        current_head_position = current_frame_endpoints[0]
                        current_tail_position = current_frame_endpoints[1]
                        confidence = 0.5
            else:
                print(f"Frame {i}: Not enough valid frames in window, using fallback method")
                if len(head_positions) > 0:
                    prev_head_position = head_positions[-1]
                    distances = np.linalg.norm(current_frame_endpoints - prev_head_position, axis=1)
                    closest_index = np.argmin(distances)
                    current_head_position = current_frame_endpoints[closest_index]
                    current_tail_position = current_frame_endpoints[1 - closest_index]
                    current_head_index = closest_index
                    confidence = 0.7
                else:
                    current_head_index = 0
                    current_head_position = current_frame_endpoints[0]
                    current_tail_position = current_frame_endpoints[1]
                    confidence = 0.5
        
        # Check for sudden changes (existing error correction logic)
        if len(head_positions) > 0:
            prev_head_position = head_positions[-1]
            distance = np.linalg.norm(current_head_position - prev_head_position)
            if distance > 25:  # If head moved more than portion of worm length
                confidence = min(confidence, 0.5)  # Lower confidence due to change
                
                # Check if this change is consistent with recent history
                if len(head_positions) >= error_threshold:
                    recent_head_positions = head_positions[-error_threshold:]
                    if any(np.linalg.norm(pos - current_head_position) > 25 for pos in recent_head_positions):
                        distances = np.linalg.norm(endpoints[i] - prev_head_position, axis=1)
                        closest_index = np.argmin(distances)
                        current_head_position = endpoints[i, closest_index]
                        current_tail_position = endpoints[i, 1 - closest_index]
                        current_head_index = closest_index
                        confidence = 0.7
            else:
                confidence = max(confidence, 0.8)
        else:
            confidence = max(confidence, 0.9)
            
        previous_head_index = current_head_index
        head_positions.append(current_head_position)
        tail_positions.append(current_tail_position)
        confidences.append(confidence)
    
    return head_positions, tail_positions, confidences

def apply_head_correction(head_positions, confidences, correction_window=5):
    corrected_positions = head_positions.copy()
    for i in range(len(head_positions)):
        if confidences[i] < 0.7:
            start = max(0, i - correction_window)
            end = min(len(head_positions), i + correction_window + 1)
            surrounding_positions = head_positions[start:end]
            corrected_positions[i] = np.median(surrounding_positions, axis=0)
    return corrected_positions

def calculate_head_bend(skeleton, head_index, segment_length=5):
    head_segment = skeleton[0:segment_length] if head_index == 0 else skeleton[-segment_length:]
    body_vector = skeleton[-1] - skeleton[0]
    head_vector = head_segment[-1] - head_segment[0]
    angle = np.arctan2(np.cross(head_vector, body_vector), np.dot(head_vector, body_vector))
    return np.degrees(angle)

def analyze_head_bends(smoothed_head_bends, fps):
    """
    Find peaks and troughs
    """
    peak_threshold = np.std(smoothed_head_bends)/3
    peaks, _ = find_peaks(smoothed_head_bends, prominence=peak_threshold, distance=8)
    troughs, _ = find_peaks(-np.array(smoothed_head_bends), prominence=peak_threshold, distance=8)
    
    num_peaks = len(peaks)
    num_troughs = len(troughs)
    avg_peak_depth = np.mean(np.array(smoothed_head_bends)[peaks])
    avg_trough_depth = np.mean(np.array(smoothed_head_bends)[troughs])
    max_peak_depth = np.max(np.array(smoothed_head_bends)[peaks])
    max_trough_depth = np.min(np.array(smoothed_head_bends)[troughs])
    
    all_extrema = sorted(np.concatenate([peaks, troughs]))
    if len(all_extrema) > 1:
        times = np.arange(len(smoothed_head_bends)) / fps
        extrema_times = times[all_extrema]
        bend_intervals = np.diff(extrema_times)
        avg_bend_frequency = 1 / np.mean(bend_intervals)
    else:
        avg_bend_frequency = 0

    fft = np.fft.fft(smoothed_head_bends)
    freqs = np.fft.fftfreq(len(smoothed_head_bends), 1/fps)
    dominant_freq = freqs[np.argmax(np.abs(fft[1:]) + 1)]
    
    return {
        'num_peaks': num_peaks,
        'num_troughs': num_troughs,
        'avg_peak_depth': avg_peak_depth,
        'avg_trough_depth': avg_trough_depth,
        'max_peak_depth': max_peak_depth,
        'max_trough_depth': max_trough_depth,
        'avg_bend_frequency': avg_bend_frequency,
        'dominant_freq': dominant_freq,
        'peaks': peaks,
        'troughs': troughs,
        'fft': fft,
        'freqs': freqs
    }

def analyze_shape(skeleton, frame_num, head_position, previous_longest_path=None):
    try:
        longest_path = adjust_self_touching_skeleton(skeleton)
        longest_path = order_segments(longest_path)
    except DiscontinuousSkeletonError as e:
        if previous_longest_path is not None:
            longest_path = previous_longest_path.copy()
        else:
            raise
    except Exception as e:
        raise
    
    x, y = longest_path[:, 1], longest_path[:, 0]
    tck, u = interpolate.splprep([x, y], s=0)
    unew = np.linspace(0, 1, num=100)
    smooth_points = np.column_stack(interpolate.splev(unew, tck))

    worm_length = np.sum(np.sqrt(np.sum(np.diff(smooth_points, axis=0)**2, axis=1)))
    
    window_size, sigma = 50, 10
    curvature = gaussian_weighted_curvature(smooth_points, window_size, sigma)

    peaks, _ = find_peaks(curvature)
    if len(peaks) > 1:
        wavelengths = np.diff(peaks)
        avg_wavelength = np.mean(wavelengths) * 2 
    else:
        avg_wavelength = worm_length

    if len(peaks) > 0:
        troughs, _ = find_peaks(-curvature)
        if len(troughs) > 0:
            amplitudes = curvature[peaks] - curvature[troughs[np.searchsorted(troughs, peaks) - 1]]
            max_amplitude = np.max(amplitudes)
            avg_amplitude = np.mean(amplitudes)
        else:
            max_amplitude = np.max(curvature) - np.min(curvature)
            avg_amplitude = max_amplitude / 2
    else:
        max_amplitude = np.max(curvature) - np.min(curvature)
        avg_amplitude = max_amplitude / 2

    wave_number = worm_length / avg_wavelength
    normalized_wavelength = avg_wavelength / worm_length

    spatial_freq = np.abs(np.fft.fft(curvature))
    dominant_spatial_freq = np.abs(np.fft.fftfreq(len(curvature))[np.argmax(spatial_freq[1:]) + 1])

    if head_position is None:
        return {
            'frame': frame_num,
            'smooth_points': smooth_points,
            'curvature': curvature,
            'max_amplitude': max_amplitude,
            'avg_amplitude': avg_amplitude,
            'wavelength': avg_wavelength,
            'worm_length': worm_length,
            'wave_number': wave_number,
            'normalized_wavelength': normalized_wavelength,
            'dominant_spatial_freq': dominant_spatial_freq,
            'head_bend': None,
            'longest_path': longest_path
        }
        
    # Find the index of the point closest to the head position
    head_index = np.argmin(np.sum((smooth_points - head_position)**2, axis=1))
    head_bend = calculate_head_bend(smooth_points, head_index)

    return {
        'frame': frame_num,
        'smooth_points': smooth_points,
        'curvature': curvature,
        'max_amplitude': max_amplitude,
        'avg_amplitude': avg_amplitude,
        'wavelength': avg_wavelength,
        'worm_length': worm_length,
        'wave_number': wave_number,
        'normalized_wavelength': normalized_wavelength,
        'dominant_spatial_freq': dominant_spatial_freq,
        'head_bend': head_bend,
        'longest_path': longest_path
    }

def analyze_video(segmentation_dict, fps=10, window_size=5, overlap=2.5):
    frames = []
    smooth_points = []
    curvatures = []
    max_amplitudes = []
    avg_amplitudes = []
    wavelengths = []
    worm_lengths = []
    wave_numbers = []
    normalized_wavelengths = []
    dominant_spatial_freqs = []
    head_bends = []
    masks = []
    
    # First pass to get endpoints
    previous_longest_path = None
    for frame_num, frame_data in segmentation_dict.items():
        time.sleep(0.1)
        print("Find head: " + str(frame_num))
        mask = frame_data[1][0]
        cleaned_mask = clean_mask(mask)
        skeleton = get_skeleton(cleaned_mask)
        
        frame_results = analyze_shape(skeleton, frame_num, None, previous_longest_path)
        frames.append(frame_results['frame'])
        smooth_points.append(frame_results['smooth_points'])
        curvatures.append(frame_results['curvature'])
        max_amplitudes.append(frame_results['max_amplitude'])
        avg_amplitudes.append(frame_results['avg_amplitude'])
        wavelengths.append(frame_results['wavelength'])
        worm_lengths.append(frame_results['worm_length'])
        wave_numbers.append(frame_results['wave_number'])
        normalized_wavelengths.append(frame_results['normalized_wavelength'])
        dominant_spatial_freqs.append(frame_results['dominant_spatial_freq'])
        head_bends.append(frame_results['head_bend'])
        masks.append(cleaned_mask)
        previous_longest_path = frame_results['longest_path']
        
        gc.collect()

    # Second pass with head information
    endpoints = track_endpoints(frames, smooth_points)
    head_positions, tail_positions, confidences = continuous_head_identification(endpoints, frames)
    corrected_head_positions = head_positions
    frames, smooth_points, curvatures, max_amplitudes, avg_amplitudes, wavelengths, worm_lengths, wave_numbers, normalized_wavelengths, dominant_spatial_freqs, head_bends = ([] for _ in range(11))
    
    previous_longest_path = None
    for frame_num, frame_data in segmentation_dict.items():
        time.sleep(0.1)
        print("Analyze shape: " + str(frame_num))
        mask = frame_data[1][0]
        cleaned_mask = clean_mask(mask)
        skeleton = get_skeleton(cleaned_mask)

        head_index = corrected_head_positions[frame_num]
        
        frame_results = analyze_shape(skeleton, frame_num, head_index, previous_longest_path)
        frames.append(frame_results['frame'])
        smooth_points.append(frame_results['smooth_points'])
        curvatures.append(frame_results['curvature'])
        max_amplitudes.append(frame_results['max_amplitude'])
        avg_amplitudes.append(frame_results['avg_amplitude'])
        wavelengths.append(frame_results['wavelength'])
        worm_lengths.append(frame_results['worm_length'])
        wave_numbers.append(frame_results['wave_number'])
        normalized_wavelengths.append(frame_results['normalized_wavelength'])
        dominant_spatial_freqs.append(frame_results['dominant_spatial_freq'])
        head_bends.append(frame_results['head_bend'])
        previous_longest_path = frame_results['longest_path']
        
        gc.collect()

    curvature_1d = np.array([np.mean(c) for c in curvatures])
    curvature_1d = (curvature_1d - np.mean(curvature_1d)) / np.std(curvature_1d)
    
    nperseg = int(window_size * fps)
    noverlap = int(overlap * fps)
    
    f, psd = welch(curvature_1d, fs=fps, nperseg=nperseg, noverlap=noverlap)
    
    dominant_freqs = []
    time_points = []
    for i in range(0, len(curvature_1d) - nperseg, nperseg - noverlap):
        segment = curvature_1d[i:i+nperseg]
        f_segment, psd_segment = welch(segment, fs=fps, nperseg=nperseg, noverlap=noverlap)
        peaks, _ = find_peaks(psd_segment, height=np.max(psd_segment) * 0.1)
        if len(peaks) > 0:
            dominant_freq_idx = peaks[np.argmax(psd_segment[peaks])]
            dominant_freqs.append(f_segment[dominant_freq_idx])
        else:
            dominant_freqs.append(0)
        time_points.append(i / fps)
    
    frame_numbers = np.arange(len(frames))
    interpolated_freqs = np.interp(frame_numbers / fps, time_points, dominant_freqs)
    
    smoothed_max_amplitudes = smooth_metric(max_amplitudes)
    smoothed_avg_amplitudes = smooth_metric(avg_amplitudes)
    smoothed_wavelengths = smooth_metric(wavelengths)
    smoothed_worm_lengths = smooth_metric(worm_lengths)
    smoothed_wave_numbers = smooth_metric(wave_numbers)
    smoothed_normalized_wavelengths = smooth_metric(normalized_wavelengths)
    smoothed_head_bends = smooth_metric(head_bends)

    head_bend_analysis = analyze_head_bends(smoothed_head_bends, fps)
    
    final_results = {
        'frames': frames,
        'smooth_points': smooth_points,
        'curvatures': curvatures,
        'max_amplitudes': max_amplitudes,
        'avg_amplitudes': avg_amplitudes,
        'wavelengths': wavelengths,
        'worm_lengths': worm_lengths,
        'wave_numbers': wave_numbers,
        'normalized_wavelengths': normalized_wavelengths,
        'dominant_spatial_freqs': dominant_spatial_freqs,
        'head_bends': head_bends,
        'smoothed_max_amplitudes': smoothed_max_amplitudes,
        'smoothed_avg_amplitudes': smoothed_avg_amplitudes,
        'smoothed_wavelengths': smoothed_wavelengths,
        'smoothed_worm_lengths': smoothed_worm_lengths,
        'smoothed_wave_numbers': smoothed_wave_numbers,
        'smoothed_normalized_wavelengths': smoothed_normalized_wavelengths,
        'smoothed_head_bends': smoothed_head_bends,
        'interpolated_freqs': interpolated_freqs,
        'f': f,
        'psd': psd,
        'head_bend_analysis': head_bend_analysis,
        'fps': fps,
        'curvature_time_series': curvature_1d
    }

    # Add head tracking information to the final results
    final_results['head_positions'] = corrected_head_positions
    final_results['tail_positions'] = tail_positions
    final_results['head_position_confidences'] = confidences
    final_results['masks'] = masks
    
    return final_results

def get_all_unprocessed_videos(hdsegmentation_dir, shape_analysis_dir):
    """
    Get all unprocessed videos from the hdsegmentation directory.
    Returns a list of tuples (video_path, video_data) for all unprocessed videos.
    """
    all_videos = [f for f in os.listdir(hdsegmentation_dir) if f.endswith('.pkl')]
    unprocessed_videos = [
        video for video in all_videos
        if not os.path.exists(os.path.join(shape_analysis_dir, video.replace('.pkl', '_shapeanalysis.pkl')))
    ]
    
    if not unprocessed_videos:
        print("All videos have been processed.")
        return []
    
    print(f"Found {len(unprocessed_videos)} unprocessed videos")
    return unprocessed_videos

def get_random_unprocessed_video(hdsegmentation_dir, shape_analysis_dir):
    all_videos = [f for f in os.listdir(hdsegmentation_dir) if f.endswith('.pkl')]
    unprocessed_videos = [
        video for video in all_videos
        if not os.path.exists(os.path.join(shape_analysis_dir, video.replace('.pkl', '_shapeanalysis.pkl')))
    ]
    
    if not unprocessed_videos:
        raise ValueError("All videos have been processed.")
    
    selected_video = random.choice(unprocessed_videos)
    video_path = os.path.join(hdsegmentation_dir, selected_video)
    
    with open(video_path, 'rb') as file:
        video_data = pickle.load(file)
    
    return video_path, video_data

def save_shape_analysis(analysis_results, video_path, output_dir):
    """
    Save shape analysis results with a filename based on the original video file.
    """
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_filename = f"{base_name}_shapeanalysis.pkl"
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"Saving analysis results to: {output_path}")
    with open(output_path, 'wb') as file:
        pickle.dump(analysis_results, file)
    
    print(f"Analysis results saved successfully!")
    return output_path

def visualize_worm_analysis(results, video_path, plot_dir):
    """
    Visualize the worm shape analysis results in a single figure
    """
    frames = results['frames']
    fps = results['fps']
    times = [frame / fps for frame in frames]

    metrics = ['max_amplitudes', 'avg_amplitudes', 'wavelengths', 'worm_lengths', 
               'dominant_spatial_freqs', 'wave_numbers', 'normalized_wavelengths', 'head_bends']
    smoothed_metrics = ['max_amplitudes', 'avg_amplitudes', 'wavelengths', 'worm_lengths', 
                        'wave_numbers', 'normalized_wavelengths', 'head_bends']
    
    data = {metric: results[metric] for metric in metrics}
    smoothed_data = {f'smoothed_{metric}': results[f'smoothed_{metric}'] for metric in smoothed_metrics}

    fig = plt.figure(figsize=(25, 30))
    gs = GridSpec(6, 4, figure=fig)
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    fig.suptitle(f'Shape Analysis: {video_name}', fontsize=20, fontweight='bold', y=0.99)

    # Plot max and avg amplitudes
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(frames, data['max_amplitudes'], label='Max Amplitude')
    ax1.plot(frames, data['avg_amplitudes'], label='Avg Amplitude')
    ax1.plot(frames, smoothed_data['smoothed_max_amplitudes'], label='Smoothed Max Amplitude', linestyle='--')
    ax1.plot(frames, smoothed_data['smoothed_avg_amplitudes'], label='Smoothed Avg Amplitude', linestyle='--')
    ax1.set_title('Amplitudes over time')
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Amplitude')
    ax1.legend()

    # Plot wavelength
    ax2 = fig.add_subplot(gs[2, :2])
    ax2.plot(frames, data['wavelengths'], label='Wavelength')
    ax2.plot(frames, smoothed_data['smoothed_wavelengths'], label='Smoothed Wavelength', linestyle='--')
    ax2.set_title('Wavelength over time')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Wavelength')
    ax2.legend()

    # Plot worm length
    ax3 = fig.add_subplot(gs[3, :2])
    ax3.plot(frames, data['worm_lengths'], label='Worm Length')
    ax3.plot(frames, smoothed_data['smoothed_worm_lengths'], label='Smoothed Worm Length', linestyle='--')
    ax3.set_title('Worm Length over time')
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Length')
    ax3.legend()

    # Plot spatial frequency
    ax4 = fig.add_subplot(gs[0, 2])
    ax4.plot(frames, data['dominant_spatial_freqs'], label='Spatial Frequency')
    ax4.set_title('Spatial Frequency over time')
    ax4.set_xlabel('Frame')
    ax4.set_ylabel('Frequency')
    ax4.legend()

    # Plot worm shape for a sample frame
    sample_frame_index = len(frames) // 2
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.plot(results['smooth_points'][sample_frame_index][:, 0], results['smooth_points'][sample_frame_index][:, 1])
    ax5.set_title(f'Worm Shape (Frame {frames[sample_frame_index]})')
    ax5.set_aspect('equal', 'box')

    # Plot curvature for the sample frame
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.plot(results['curvatures'][sample_frame_index])
    ax6.set_title(f'Curvature (Frame {frames[sample_frame_index]})')
    ax6.set_xlabel('Position along worm')
    ax6.set_ylabel('Curvature')

    # Plot wave number
    ax7 = fig.add_subplot(gs[1, 3])
    ax7.plot(frames, data['wave_numbers'], label='Wave Number')
    ax7.plot(frames, smoothed_data['smoothed_wave_numbers'], label='Smoothed Wave Number', linestyle='--')
    ax7.set_title('Wave Number over time')
    ax7.set_xlabel('Frame')
    ax7.set_ylabel('Wave Number')
    ax7.set_ylim(0, 4)
    ax7.legend()

    # Plot normalized wavelength
    ax8 = fig.add_subplot(gs[2, 3])
    ax8.plot(frames, data['normalized_wavelengths'], label='Normalized Wavelength')
    ax8.plot(frames, smoothed_data['smoothed_normalized_wavelengths'], label='Smoothed Normalized Wavelength', linestyle='--')
    ax8.set_title('Normalized Wavelength over time')
    ax8.set_xlabel('Frame')
    ax8.set_ylabel('Normalized Wavelength')
    ax8.set_ylim(0, 2)
    ax8.legend()
    
    # Plot dominant frequency
    ax9 = fig.add_subplot(gs[4, :2])
    ax9.plot(frames, results['interpolated_freqs'])
    ax9.set_title('Dominant Temporal Frequency Over Time')
    ax9.set_xlabel('Frame Number')
    ax9.set_ylabel('Frequency (cycles per frame)')
    ax9.set_ylim(0, max(results['interpolated_freqs']) * 1.1)
    ax9.grid(True)

    # Plot power spectral density
    ax10 = fig.add_subplot(gs[3, 2:])
    ax10.semilogy(results['f'], results['psd'])
    ax10.set_title('Power Spectral Density')
    ax10.set_xlabel('Frequency (Hz)')
    ax10.set_ylabel('Power/Frequency')
    ax10.grid(True)

    # Plot head bends
    ax11 = fig.add_subplot(gs[1, :2])
    ax11.plot(times, data['head_bends'], alpha=0.5, label='Raw Head Bend', color='lightblue')
    ax11.plot(times, smoothed_data['smoothed_head_bends'], label='Smoothed Head Bend', color='darkblue')
    
    # Add peaks and troughs to the plot
    head_bend_analysis = results['head_bend_analysis']
    ax11.scatter([times[i] for i in head_bend_analysis['peaks']], 
                 [smoothed_data['smoothed_head_bends'][i] for i in head_bend_analysis['peaks']], 
                 color='red', s=50, label='Peaks', zorder=5)
    ax11.scatter([times[i] for i in head_bend_analysis['troughs']], 
                 [smoothed_data['smoothed_head_bends'][i] for i in head_bend_analysis['troughs']], 
                 color='green', s=50, label='Troughs', zorder=5)

    ax11.set_ylabel('Head Bend Angle (degrees)')
    ax11.set_xlabel('Time (seconds)')
    ax11.set_title('Worm Head Bends Over Time')
    ax11.legend()
    ax11.grid(True, which='both', linestyle='--', linewidth=0.5)

    ax12 = fig.add_subplot(gs[4, 2:])
    head_bend_analysis = results['head_bend_analysis']
    ax12.plot(head_bend_analysis['freqs'][1:len(head_bend_analysis['freqs'])//2], 
              np.abs(head_bend_analysis['fft'][1:len(head_bend_analysis['fft'])//2]))
    ax12.set_title('Frequency Spectrum of Head Bends')
    ax12.set_xlabel('Frequency (Hz)')
    ax12.set_ylabel('Magnitude')
    ax12.grid(True)

    # Add text information in the top right corner
    ax_text = fig.add_subplot(gs[0, 3:])
    ax_text.axis('off')
    
    head_bend_analysis = results['head_bend_analysis']
    info_text = (
        f"Head Bend Analysis:\n\n"
        f"Number of peaks: {head_bend_analysis['num_peaks']}\n"
        f"Number of troughs: {head_bend_analysis['num_troughs']}\n"
        f"Average peak depth: {head_bend_analysis['avg_peak_depth']:.2f} degrees\n"
        f"Average trough depth: {head_bend_analysis['avg_trough_depth']:.2f} degrees\n"
        f"Maximum peak depth: {head_bend_analysis['max_peak_depth']:.2f} degrees\n"
        f"Maximum trough depth: {head_bend_analysis['max_trough_depth']:.2f} degrees\n"
        f"Average bending frequency: {head_bend_analysis['avg_bend_frequency']:.2f} Hz\n"
        f"Dominant frequency from FFT: {head_bend_analysis['dominant_freq']:.2f} Hz"
    )
    
    ax_text.text(0, 1, info_text, verticalalignment='top', fontsize=15, fontfamily='monospace')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    plot_filename = f"{video_name}_shapeanalysis.png"
    plot_path = os.path.join(plot_dir, plot_filename)
    
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Shape analysis plot saved to: {plot_path}")

# endregion [functions]


# Define directories
HDSEGMENTATION_DIR = "HDSEGMENTATION_DIR_FROM_STEP2"
SHAPE_ANALYSIS_DIR = "OUTPUT_SHAPE_ANALYSIS_DIR"
PLOT_DIR = "OUTPUT_SHAPE_ANALYSIS_PLOTS_DIR"
os.makedirs(SHAPE_ANALYSIS_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# Process all unprocessed videos
unprocessed_videos = get_all_unprocessed_videos(HDSEGMENTATION_DIR, SHAPE_ANALYSIS_DIR)


if not unprocessed_videos:
    print("No unprocessed videos found. All videos have been processed.")
else:
    print(f"Starting processing of {len(unprocessed_videos)} videos...")
    
    successful_processes = 0
    failed_processes = 0
    
    for i, video_filename in enumerate(unprocessed_videos):
        time.sleep(0.5)
        try:
            video_path = os.path.join(HDSEGMENTATION_DIR, video_filename)
            print(f"\n{'='*60}")
            print(f"Processing video {i+1}/{len(unprocessed_videos)}: {video_filename}")
            print(f"{'='*60}")
            
            with open(video_path, 'rb') as file:
                hd_video_segments = pickle.load(file)
            
            print("Starting shape analysis...")
            fframe_analysis = analyze_video(hd_video_segments)
            
            print("Saving analysis results...")
            save_shape_analysis(fframe_analysis, video_path, SHAPE_ANALYSIS_DIR)
            
            print("Creating visualization...")
            visualize_worm_analysis(fframe_analysis, video_path, PLOT_DIR)
            
            successful_processes += 1
            print(f"✓ Successfully processed {video_filename}")
            
            del hd_video_segments
            del fframe_analysis
            gc.collect()
            
        except Exception as e:
            failed_processes += 1
            print(f"✗ Failed to process {video_filename}: {str(e)}")
            
            try:
                del hd_video_segments
            except:
                pass
            try:
                del fframe_analysis
            except:
                pass
            gc.collect()
            
            continue
    
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Successfully processed: {successful_processes} videos")
    print(f"Failed to process: {failed_processes} videos")
    print(f"Total videos: {len(unprocessed_videos)}")