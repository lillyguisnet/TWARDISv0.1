"""
This script performs shape analysis on the extracted worm masks following step 3.
"""
import numpy as np
import h5py
from scipy import interpolate
from skimage import morphology, graph
from scipy import ndimage
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy.signal import welch, find_peaks
from collections import Counter
from collections import defaultdict
from pathlib import Path
import os
from skimage import measure
from scipy.spatial.distance import cdist
import networkx as nx
from scipy.ndimage import binary_dilation

def smooth_metric(data, window_length=11, poly_order=3):
    return savgol_filter(data, window_length, poly_order)

def clean_mask(mask):
    labeled, num_features = ndimage.label(mask)
    if num_features == 0:
        return mask
    sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
    largest_label = sizes.argmax() + 1
    cleaned_mask = (labeled == largest_label)
    return cleaned_mask

def get_skeleton(mask):
    return morphology.skeletonize(mask)

def find_endpoints_and_junctions(coords):
    """
    Find endpoints and junctions in a skeleton of coordinates
    """
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
        return None  # Not enough endpoints
    
    max_distance = 0
    furthest_pair = None
    
    for i in range(len(endpoints)):
        for j in range(i+1, len(endpoints)):
            
            cost_array = np.where(skeleton, 1, np.inf)
            start = tuple(endpoints[i])
            end = tuple(endpoints[j]) 
            path_indices, cost = graph.route_through_array(cost_array, start, end)

            path = np.array(path_indices)

            distance = len(path)  # The length of the path is the distance along the skeleton
            
            if distance > max_distance:
                max_distance = distance
                furthest_pair = (endpoints[i], endpoints[j])
    
    return furthest_pair

def order_segments(segments):
    """
    Order segments coordinates so that endpoints are at the beginning and end
    """
    segments_set = set(map(tuple, segments))
    
    graph = defaultdict(list)
    for x, y in segments_set:
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                neighbor = (x + dx, y + dy)
                if neighbor in segments_set:
                    graph[(x, y)].append(neighbor)
    
    endpoints = [point for point, neighbors in graph.items() if len(neighbors) == 1]
    
    if len(endpoints) != 2:
        raise ValueError("Expected exactly two endpoints")
    
    start, end = endpoints
    ordered = [start]
    current = start
    
    while current != end:
        next_point = [p for p in graph[current] if p not in ordered][0]
        ordered.append(next_point)
        current = next_point
    
    return np.array(ordered)

def calculate_orientation_difference(segment1, segment2, p1, p2):
    idx1 = np.where((segment1 == p1).all(axis=1))[0][0]
    idx2 = np.where((segment2 == p2).all(axis=1))[0][0]

    if idx1 == 0:
        points1 = segment1[:3]  # First 5 points if at start
    else:
        points1 = segment1[max(0, idx1-2):idx1+1]  # Up to 5 points ending at idx1
    
    if idx2 == 0:
        points2 = segment2[:3]  # First 5 points if at start
    else:
        points2 = segment2[max(0, idx2-2):idx2+1]  # Up to 5 points ending at idx2
       
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
    cost_map = np.ones_like(skeleton, dtype=float)
    
    occupied = binary_dilation(skeleton, iterations=2)
    cost_map[occupied] = np.inf

    start = tuple(start)
    end = tuple(end)
    cost_map[start] = 1
    cost_map[end] = 1
    
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if 0 <= start[0]+dx < skeleton.shape[0] and 0 <= start[1]+dy < skeleton.shape[1]:
                cost_map[start[0]+dx, start[1]+dy] = 1
            if 0 <= end[0]+dx < skeleton.shape[0] and 0 <= end[1]+dy < skeleton.shape[1]:
                cost_map[end[0]+dx, end[1]+dy] = 1

    height, width = skeleton.shape
    G = nx.grid_2d_graph(height, width)

    for (u, v) in G.edges():
        if cost_map[u] == np.inf or cost_map[v] == np.inf:
            G.remove_edge(u, v)
        else:
            G[u][v]['weight'] = (cost_map[u] + cost_map[v]) / 2

    try:
        path = nx.shortest_path(G, start, end, weight='weight')
    except nx.NetworkXNoPath:
        return None  # No path found

    return path

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
        furthest_pair = find_furthest_endpoints_along_skeleton(skeleton)
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
        for junction in junctions:
            skeleton[junction[0], junction[1]] = 0
        
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
            
            # If only one segment remains after removing tiny segments, exit the loop
            if len(segments) == 1:
                break
            
            ordered_segments = []
            for seg in (segments):
                ordered_seg = order_segments(seg)
                ordered_segments.append(ordered_seg)
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

            # Find the original closest pair of endpoints that belong to different segments
            original_min_distance = np.inf
            original_closest_pair = None
            for i in range(len(all_endpoints)):
                for j in range(i + 1, len(all_endpoints)):
                    if endpoint_to_segment[tuple(all_endpoints[i])] != endpoint_to_segment[tuple(all_endpoints[j])]:
                        dist = distances[i, j]
                        if dist < original_min_distance:
                            original_min_distance = dist
                            original_closest_pair = (i, j)

            # Find the closest pair of endpoints that belong to different segments
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
                    # If endpoints are from the same segment, set this distance to infinity and continue
                    distances[i, j] = distances[j, i] = np.inf


            # Connect the closest pair of endpoints from different segments            
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
        furthest_pair = find_furthest_endpoints_along_skeleton(skeleton)
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

def calculate_swimamplitude(smooth_points):
    centerline = np.array([smooth_points[0], smooth_points[-1]])    
    centerline_vector = centerline[1] - centerline[0]    
    centerline_unit = centerline_vector / np.linalg.norm(centerline_vector)
    
    # Calculate perpendicular distances from each point to the centerline
    perpendicular_distances = []
    for point in smooth_points:
        v = point - centerline[0]
        proj = np.dot(v, centerline_unit)
        proj_point = centerline[0] + proj * centerline_unit
        distance = np.linalg.norm(point - proj_point)
        perpendicular_distances.append(distance)
    
    # Maximum amplitude is half of the maximum perpendicular distance
    max_amplitude = max(perpendicular_distances)
    avg_amplitude = np.mean(perpendicular_distances)
    
    return max_amplitude, avg_amplitude

def classify_shape(smooth_points, threshold=8, c_shape_ratio=0.98, epsilon=1e-10):
    """
    Basic shape classification based on centerline and perpendicular distances
    """
    centerline_start = smooth_points[0]
    centerline_end = smooth_points[-1]
    centerline_vector = centerline_end - centerline_start
    
    centerline_unit = centerline_vector / np.linalg.norm(centerline_vector)
    
    perpendicular_distances = []
    sides = []
    for point in smooth_points:
        v = point - centerline_start
        proj = np.dot(v, centerline_unit)
        proj_point = centerline_start + proj * centerline_unit
        distance = np.linalg.norm(point - proj_point)
        perpendicular_distances.append(distance)
        
        cross_product = np.cross(centerline_vector, v)
        if abs(cross_product) < epsilon:
            sides.append(0)
        else:
            sides.append(1 if cross_product > 0 else -1)
    
    unique_sides = np.unique(sides)
    max_distance = max(perpendicular_distances)
    
    positive_count = sum(1 for s in sides if s > 0)
    negative_count = sum(1 for s in sides if s < 0)
    total_points = len(sides)
    
    dominant_side_ratio = max(positive_count, negative_count) / total_points
    
    if dominant_side_ratio >= c_shape_ratio and max_distance > threshold:
        return "C-shape"
    elif max_distance <= threshold:
        return "Straight"
    elif len(unique_sides) > 1:
        return "S-shape"
    else:
        return "Unknown"

def analyze_shape(skeleton, frame_num):
    longest_path = adjust_self_touching_skeleton(skeleton)
    
    longest_path = order_segments(longest_path)
    t = np.arange(len(longest_path))
    x, y = longest_path[:, 1], longest_path[:, 0]
    tck, u = interpolate.splprep([x, y], s=0)
    unew = np.linspace(0, 1, num=100)
    smooth_points = np.column_stack(interpolate.splev(unew, tck))
    
    window_size, sigma = 50, 10
    curvature = gaussian_weighted_curvature(smooth_points, window_size, sigma)

    max_amplitude, avg_amplitude = calculate_swimamplitude(smooth_points)

    worm_length = np.sum(np.sqrt(np.sum(np.diff(smooth_points, axis=0)**2, axis=1)))

    shape = classify_shape(smooth_points)
    if shape == "C-shape":
        wavelength = worm_length * 2  # One full wave is twice the worm length
    elif shape == "Straight":
        wavelength = worm_length * 4  # Assume a very long wavelength
    elif shape == "S-shape":
        peaks, _ = find_peaks(curvature)
        if len(peaks) > 1:
            wavelengths = np.diff(peaks)
            avg_wavelength = np.mean(wavelengths) * 2  # multiply by 2 for full wave
            wavelength = avg_wavelength
        else:
            wavelength = 0
    else:
        wavelength = 0


    wave_number = worm_length / wavelength #Number of waves along the worm body
    normalized_wavelength = wavelength / worm_length #Wavelength as proportion of worm length

    # Frequency analysis (spatial)
    spatial_freq = np.abs(np.fft.fft(curvature))
    dominant_spatial_freq = np.abs(np.fft.fftfreq(len(curvature))[np.argmax(spatial_freq[1:]) + 1])   

    return {
        'frame': frame_num,
        'shape': shape,
        'smooth_points': smooth_points,
        'curvature': curvature,
        'max_amplitude': max_amplitude,
        'avg_amplitude': avg_amplitude,
        'wavelength': wavelength,
        'worm_length': worm_length,
        'wave_number': wave_number,
        'normalized_wavelength': normalized_wavelength,
        'dominant_spatial_freq': dominant_spatial_freq,
    }

def analyze_video(segmentation_dict, fps=10, window_size=5, overlap=2.5):
    frames = []
    shape = []
    smooth_points = []
    curvatures = []
    max_amplitudes = []
    avg_amplitudes = []
    wavelengths = []
    worm_lengths = []
    wave_numbers = []
    normalized_wavelengths = []
    dominant_spatial_freqs = []
    masks = []
    
    # Get shape info, per frame
    for frame_num, frame_data in segmentation_dict.items():
        print("Frame: " + str(frame_num))
        mask = frame_data[1][0]
        cleaned_mask = clean_mask(mask)
        skeleton = get_skeleton(cleaned_mask)

        frame_results = analyze_shape(skeleton, frame_num)
        frames.append(frame_results['frame'])
        shape.append(frame_results['shape'])
        smooth_points.append(frame_results['smooth_points'])
        curvatures.append(frame_results['curvature'])
        max_amplitudes.append(frame_results['max_amplitude'])
        avg_amplitudes.append(frame_results['avg_amplitude'])
        wavelengths.append(frame_results['wavelength'])
        worm_lengths.append(frame_results['worm_length'])
        wave_numbers.append(frame_results['wave_number'])
        normalized_wavelengths.append(frame_results['normalized_wavelength'])
        dominant_spatial_freqs.append(frame_results['dominant_spatial_freq'])
        masks.append(cleaned_mask)

    # Temporal frequency analysis
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
    
    final_results = {
        'frames': frames,
        'shape': shape,
        'smooth_points': smooth_points,
        'curvatures': curvatures,
        'max_amplitudes': max_amplitudes,
        'avg_amplitudes': avg_amplitudes,
        'wavelengths': wavelengths,
        'worm_lengths': worm_lengths,
        'wave_numbers': wave_numbers,
        'normalized_wavelengths': normalized_wavelengths,
        'dominant_spatial_freqs': dominant_spatial_freqs,
        'smoothed_max_amplitudes': smoothed_max_amplitudes,
        'smoothed_avg_amplitudes': smoothed_avg_amplitudes,
        'smoothed_wavelengths': smoothed_wavelengths,
        'smoothed_worm_lengths': smoothed_worm_lengths,
        'smoothed_wave_numbers': smoothed_wave_numbers,
        'smoothed_normalized_wavelengths': smoothed_normalized_wavelengths,
        'interpolated_freqs': interpolated_freqs,
        'f': f,
        'psd': psd,
        'fps': fps,
        'curvature_time_series': curvature_1d
    }

    final_results['masks'] = masks
    
    return final_results

def get_shapeanalysis_file(or_vid):
    #Get hd segmentation file path
    or_vid_path = Path(or_vid)
    sub_folder_name = or_vid_path.parent.name
    video_name = or_vid_path.stem
    seg_file_name = f"{sub_folder_name}-{video_name}_hdsegmentation.h5"
    seg_file_path = Path("PATH_TO_HD_SEGMENTATIONS_DIR") / seg_file_name
    str(seg_file_path)

    shapeanalysis_filepath = "PATH_TO_OUTPUT_DIR"
    filename = f"{shapeanalysis_filepath}/{sub_folder_name}-{video_name}_swimshapeanalysis.h5"

    if os.path.exists(filename):
        raise FileExistsError(f"The file '{filename}' already exists. Analysis aborted to prevent overwriting.")

    #Load the hd segmentation h5 file
    with h5py.File(seg_file_path, 'r') as hf:
        hd_video_segments = {}
        for key in hf.keys():
            hd_video_segments[int(key)] = {}
            for sub_key in hf[key].keys():
                dataset = hf[key][sub_key]
                hd_video_segments[int(key)][int(sub_key)] = np.array(dataset)

    swim_shapeanalysis = analyze_video(hd_video_segments)
    print(Counter(swim_shapeanalysis['shape']))

    def save_dict_to_h5(dict_to_save, filename):
        with h5py.File(filename, 'w') as h5file:
            for key, value in dict_to_save.items():
                if isinstance(value, list):
                    group = h5file.create_group(key)
                    for i, item in enumerate(value):
                        if isinstance(item, np.ndarray):
                            group.create_dataset(f'{i}', data=item)
                        elif isinstance(item, (int, float, str)):
                            group.create_dataset(f'{i}', data=item)
                        else:
                            print(f"Unsupported type {type(item)} for key {key} at index {i}")
                elif isinstance(value, np.ndarray):
                    h5file.create_dataset(key, data=value)
                elif isinstance(value, (int, float)):
                    h5file.create_dataset(key, data=value)
                else:
                    print(f"Unsupported type {type(value)} for key {key}")
        print(f"Saved: '{filename}' ")

    save_dict_to_h5(swim_shapeanalysis, filename)

    return swim_shapeanalysis



or_vid = 'PATH_TO_VIDEO'

swim_shapeanalysis = get_shapeanalysis_file(or_vid)