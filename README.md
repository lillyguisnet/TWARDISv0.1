# TWARDISv0.1

<!-- ![TWARDISv0.1 Banner](docs/images/banner.png) -->

**T**ools for **W**orm **A**utomated **R**ecognition & **D**ynamic **I**maging **S**ystem: A collection of pipelines for automated *C. elegans* video and image analysis using Meta's Segment Anything Model 2 (SAM2).

TWARDISv0.1 includes four independent analysis pipelines:

| Pipeline | Description |
|----------|-------------|
| [Multiworm Feature Extraction](#multiworm-feature-extraction) | Detect and measure multiple worms in images |
| [Droplet Swimming](#droplet-swimming) | Track and analyze swimming behavior in droplets |
| [Single Worm Tracking (Crawling)](#single-worm-tracking-crawling) | Track crawling worms with shape and trajectory analysis |
| [RIA Calcium Imaging in semi-restricted worms](#ria-calcium-imaging) | Segment RIA axonal compartments, extract calcium signals and extrac head angles |

---

## Hardware

The following setup was used for development and testing. The pipelines can also run on CPU (minimum 4GB RAM), but expect slower processing. Long videos of very high-quality might be too large for consumer-sized RAM.

- **OS:** Ubuntu 22.04
- **GPU:** NVIDIA RTX 3090
- **CPU:** 48 cores (used mainly for parallelized frame extraction)
- **Models:** SAM2 (~0.5 GB), worm classifier (~2.5 GB)

---

## Required Packages

Under Python **3.12.3**, scripts were tested with the following versions:
- h5py 3.14.0
- matplotlib 3.10.3
- networkx 3.4.2
- numpy 1.26.4
- opencv-python 4.10.0.84
- pandas 2.3.1
- Pillow 9.4.0
- PyTorch 2.3.1
- scikit-image 0.25.0
- scikit-learn 1.8.0
- scipy 1.14.1
- seaborn 0.13.2
- tifffile 2024.9.20
- torchvision 0.18.1
- tqdm 4.66.1

---

## Installation for Linux

### 1. Clone the repository

```bash
git clone https://github.com/lillyguisnet/TWARDISv0.1.git
cd TWARDISv0.1
```

### 2. Create a virtual environment with uv (optional, but recommended)

```bash
uv venv --python 3.12
source .venv/bin/activate
```

### 3. Install missing dependencies

```bash
uv pip install [...]
```

### 4. Set up SAM2

Clone the SAM2 repository (July 28th 2024 version is our latest tested version) and download the Hierea Large checkpoint:

```bash
git clone https://github.com/facebookresearch/sam2.git segment-anything-2
cd segment-anything-2
# Download the Hiera Large model checkpoint
wget -P checkpoints/ https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
cd ..
```

In each script that uses SAM2, update the path to point to your cloned repo:

```python
sys.path.append("PATH_TO_CLONED_SAM2_REPO/segment-anything-2")
```

### 5. Download the worm classifier (for Multiworm Feature Extraction only)

Download our fine-tuned ViT-H-14 classifier weights from [HuggingFace](https://huggingface.co/lillyguisnet/celegans-classifier-vit-h-14-finetuned) and update the path in `multiworm_feature_extraction/2_extract_wormcutouts.py`.

---

## Multiworm Feature Extraction

Detect and classify multiple worms in images, then extract morphological metrics.

<!-- ![Multiworm Feature Extraction Pipeline](docs/images/multiworm_pipeline.png) -->

<!-- ![Example: detected worms with overlay](docs/images/multiworm_example.gif) -->

```
TIFF images ➜ [1] Convert ➜ JPEG images ➜ [2] Segment, classify, extract features ➜ Metrics + cutout images
```

#### Output data

Per-image pickle file containing a list of dictionaries (one per detected worm):

| Field | Description |
|-------|-------------|
| `img_id` | Image identifier (filename without extension) |
| `worm_id` | Worm index within the image |
| `area` | Worm mask area (pixels) |
| `perimeter` | Worm contour perimeter (Euclidean pixels) |
| `medial_axis_distances_sorted` | Width measurements along the medial axis, sorted along the worm's length (list) |
| `medialaxis_length_list` | Position indices along the medial axis (list) |
| `pruned_medialaxis_length` | Length of the worm (Euclidean pixels) |
| `mean_wormwidth` | Average worm width |
| `mid_length_width` | Width at the midpoint of the medial axis |
| `mask` | Binary mask array of the worm object |

Additionally, a CSV file logs all images where no worms were detected.

### `0_cutout_classifier.py` - Classifier original fine-tuning

A fine-tuned classifier is used to select worm segments after SAM has generated cutouts of the entire image. This classifier can be further fine-tuned if necessary, or replaced with a low-cost language-vision model.

### `1_convert_images.py` — Convert 16-bit TIFF to 8-bit JPEG

Converts 16-bit TIFF microscopy images to 8-bit JPEG with global min/max contrast normalization. Preserves directory structure. If you have other file types, these will need to be converted to .jpeg for input in the second part.

| | |
|---|---|
| **Input** | Directory of `.tif` images (can contain subdirectories) |
| **Output** | Matching directory structure with `.jpg` images |

**Variables to update:**

| Variable | Description |
|----------|-------------|
| `YOUR_DATA_DIR_PATH` | Path to source TIFF image directory |
| `YOUR_DST_DIR_PATH` | Path to destination JPEG directory |

### `2_extract_wormcutouts.py` — SAM2 detection + ViT classification + metric extraction

Uses SAM2 automatic mask generator to detect all objects, classifies each mask as "worm" or "not worm" using our fine-tuned ViT-H-14 classifier, then extracts morphological metrics from worm masks.

| | |
|---|---|
| **Input** | JPEG images from step 1 |
| **Output** | Per-image pickle files with worm metrics (area, perimeter, medial axis length, width), cutout overlay images, CSV log of images with no worms |

**Variables to update:**

| Variable | Description |
|----------|-------------|
| `PATH_TO_INPUT_FOLDER` | Path to folder with JPEG images to process |
| `PATH_TO_TEMP_CUTOUTS_DIR` | Temporary directory for intermediate cutouts |
| `PATH_TO_FINAL_CUTOUTS_DIR` | Directory for final worm overlay images |
| `PATH_TO_FINAL_METRICS_DIR` | Directory for output pickle metric files |
| `PATH_TO_NOWORMS_FILE.csv` | Path to CSV file logging images with no detected worms |

---

## Droplet Swimming

Track and analyze the swimming *C. elegans* in droplets from video recordings.

<!-- ![Droplet Swimming Pipeline](docs/images/swimming_pipeline.png) -->

<!-- ![Example: swimming worm tracking](docs/images/swimming_example.gif) -->

```
Video ➜ [1] Frames ➜ [2] Full-frame segmentation ➜ [3] HD cropped segmentation ➜ [4] Shape analysis
```

#### Output data

<a id="shape-metrics"></a>

H5 file with per-frame shape analysis (also provided by the [Crawling pipeline](#shape-metrics-crawling)):

| Field | Description |
|-------|-------------|
| `frames` | Frame numbers |
| `masks` | Cleaned binary masks per frame |
| `smooth_points` | Interpolated skeleton coordinates (100 points) |
| `curvatures` | Curvature profile along interpolated skeleton |
| `curvature_time_series` | Mean normalized curvature per frame |
| `max_amplitudes` | Maximum curvature amplitude per frame |
| `avg_amplitudes` | Average curvature amplitude per frame |
| `wavelengths` | Estimated body wave wavelength per frame |
| `worm_lengths` | Body length per frame |
| `wave_numbers` | Number of body waves per frame |
| `normalized_wavelengths` | Wavelength / body length ratio per frame |
| `dominant_spatial_freqs` | Dominant spatial frequency per frame |
| `shape` | Shape classification per frame |
| `smoothed_*` | Savitzky-Golay filtered versions of the above metrics |
| `interpolated_freqs` | Temporal bending frequency at each frame |
| `f` | Power spectral density frequency array |
| `psd` | Power spectral density values |
| `fps` | Frames per second |

### `1_videotoimg.py` — Convert video to JPG frames

Extracts frames from a video file at the original FPS. Frames are saved as zero-padded JPGs (`000000.jpg`, `000001.jpg`, ...) in a subdirectory named `{subfolder}-{videoname}/` for SAM2 input requirements.

| | |
|---|---|
| **Input** | Video file (AVI, MP4, MOV) |
| **Output** | Directory of JPG frames |

**Variables to update:**

| Variable | Description |
|----------|-------------|
| `PATH_TO_VIDEO` | Path to the input video file |
| `PATH_TO_OUTPUT_DIR` | Parent directory where the frames folder will be created |

### `2_fframe_segmentation.py` — Frame-by-frame segmentation with generic prompt

Performs full-sized frame segmentation by inserting a generic prompt frame into the video sequence and propagating the segmentation backward through all frames. Flags frames with empty, too-small, or too-large detections.

| | |
|---|---|
| **Input** | JPG frames directory from step 1 |
| **Output** | Pickle file (`{videoname}_fframe_segmentation.pkl`) with per-frame binary masks |

**Variables to update:**

| Variable | Description |
|----------|-------------|
| `PATH_TO_VIDEO` | Path to the video frames directory |
| `PATH_TO_GENERIC_PROMPT_FRAME` | Path to a generic prompt frame with a known worm location |
| `points` | Click coordinates `[[x, y]]` marking the worm on the prompt frame |
| `PATH_TO_OUTPUT_DIR` | Output directory path for the pickle file |

### `3_swim_hdsegmentation.py` — High-definition cropped segmentation

Crops frames around the detected worm using a fixed window (110x110 px), then runs a second-pass SAM2 segmentation on the cropped frames for higher-quality masks. 800x800 intermediary pass when first segmentation fails.

| | |
|---|---|
| **Input** | JPG frames from step 1 + pickle from step 2 |
| **Output** | H5 file (`{videoname}_hdsegmentation.h5`) with per-frame segmentation masks |

**Variables to update:**

| Variable | Description |
|----------|-------------|
| `PATH_TO_PROMPT_FRAME` | Paths to prompt frames for the cropped sizes (110x110, 800x800) |
| `PATH_TO_TEMP_CROPDIR` / `PATH_TO_TEMP_CROPDIR2` | Temporary directories for cropped frames passes |
| `PATH_TO_OUTPUT_DIR` | Path prefix for the output H5 file |
| `PATH_TO_VIDEO` | Path to the original video file |
| `PATH_TO_FFRAME_SEGMENTATIONS_DIR` | Path to full frame segementations from steps 2 |

### `4_shape_analysis.py` — Skeleton and shape metrics

Extracts skeleton, curvature, bend angles, and other shape metrics from the HD segmentation masks.

| | |
|---|---|
| **Input** | H5 segmentation file from step 3 |
| **Output** | H5 file with shape analysis results (skeleton coordinates, curvatures, bend angles, head/tail orientation) |

**Variables to update:**

| Variable | Description |
|----------|-------------|
| `PATH_TO_VIDEO` | Path to the original video file (used to infer the H5 file name) |
| `PATH_TO_HD_SEGMENTATIONS_DIR` | Path to direction with HD segmentations from step 3 |
| `PATH_TO_OUTPUT_DIR` | Path prefix for the output H5 file |

---

## Single Worm Tracking (Crawling)

Comprehensive tracking and analysis of single *C. elegans* crawling behavior, including segmentation, shape analysis, and trajectory analysis.

<!-- ![Single Worm Tracking Pipeline](docs/images/crawling_pipeline.png) -->

<!-- ![Example: crawling worm trajectory](docs/images/crawling_example.gif) -->

```
Videos ➜ [1] Frames ➜ [2] Auto-prompted segmentation ➜ [3] Shape analysis ➜ [4] Path analysis + plots
```

#### Output data

**Shape metrics**

<a id="shape-metrics-crawling"></a>

Same [shape metrics as the Swimming pipeline](#shape-metrics), plus the following additional fields:

| Field | Description |
|-------|-------------|
| `head_positions` | Head coordinates per frame |
| `tail_positions` | Tail coordinates per frame |
| `head_position_confidences` | Confidence score (0–1) for head identification per frame |
| `head_bend_analysis.num_peaks` | Number of peaks in head bend signal |
| `head_bend_analysis.num_troughs` | Number of troughs in head bend signal |
| `head_bend_analysis.avg_peak_depth` | Average peak height |
| `head_bend_analysis.avg_trough_depth` | Average trough depth |
| `head_bend_analysis.max_peak_depth` | Maximum peak height |
| `head_bend_analysis.max_trough_depth` | Maximum trough depth |
| `head_bend_analysis.avg_bend_frequency` | Average frequency of bending oscillations |
| `head_bend_analysis.dominant_freq` | Dominant bending frequency |
| `head_bend_analysis.peaks` | Peak frame indices |
| `head_bend_analysis.troughs` | Trough frame indices |
| `head_bend_analysis.fft` | FFT coefficients of head bend signal |
| `head_bend_analysis.freqs` | Frequency array for FFT |
| `smoothed_head_bends.*` | Above metrics with an 8 frame smoothing |

**Path metrics**

Pickle file with trajectory and behavior data:

| Field | Description |
|-------|-------------|
| `movement_classification` | Per-frame classification: "forward", "backward", or "stationary" |
| `forward_frames` | Total frames classified as forward |
| `backward_frames` | Total frames classified as backward |
| `stationary_frames` | Total frames classified as stationary |
| `total_frames` | Total frames analyzed |
| `total_distance` | Sum of all frame-to-frame distances (pixels) |
| `avg_speed` | Average speed across entire trajectory (pixels/frame) |
| `avg_velocity` | Mean velocity vector [vx, vy] |
| `avg_forward_speed` | Average speed during forward bouts |
| `avg_backward_speed` | Average speed during backward bouts |
| `per_frame_speeds` | Per-frame speed values |
| `velocities` | Per-frame velocity vectors |
| `avg_forward_velocity` | Mean velocity vector during forward movement |
| `avg_backward_velocity` | Mean velocity vector during backward movement |
| `avg_acceleration` | Mean acceleration vector [ax, ay] |
| `avg_forward_acceleration` | Mean acceleration during forward movement |
| `avg_backward_acceleration` | Mean acceleration during backward movement |
| `per_frame_accelerations` | Per-frame acceleration vectors |
| `sinuosity` | Ratio of total distance to straight-line displacement |
| `smooth_centroids` | Smoothed centroid positions per frame (x, y) |
| `furthest_point_distance` | Maximum distance from start point (pixels) |
| `furthest_point_frame` | Frame at which the furthest point was reached |
| `forward_bouts` | Number of forward movement bouts |
| `backward_bouts` | Number of backward movement bouts |
| `stationary_bouts` | Number of stationary bouts |
| `bout_lengths_frames` | Lists of bout durations in frames (per movement type) |
| `bout_lengths_pixels` | Lists of bout distances in pixels (per movement type) |
| `avg_forward_bout_length_frames` | Average forward bout duration (frames) |
| `avg_backward_bout_length_frames` | Average backward bout duration (frames) |
| `avg_stationary_bout_length_frames` | Average stationary bout duration (frames) |
| `avg_forward_bout_length_pixels` | Average forward bout distance (pixels) |
| `avg_backward_bout_length_pixels` | Average backward bout distance (pixels) |
| `avg_stationary_bout_length_pixels` | Average stationary bout distance (pixels) |
| `total_stationary_time` | Total time spent stationary |
| `total_forward_time` | Total time moving forward |
| `total_backward_time` | Total time moving backward |
| `stationary_percentage` | Percentage of time stationary |
| `moving_percentage` | Percentage of time moving |
| `transitions` | Number of movement type changes |


### `1_videotoimg.py` — Parallel video-to-frame extraction

Converts videos to JPG frames using multiprocessing for speed. Automatically picks a random unprocessed video from the input directory.

| | |
|---|---|
| **Input** | Directory of video files (AVI, MP4, MOV) |
| **Output** | Directories of JPG frames (one per video, named `{subfolder}-{videoname}/`) |

**Variables to update:**

| Variable | Description |
|----------|-------------|
| `PATH_TO_ORIGINAL_VIDEOS_DIR` | Path to directory containing input videos |
| `PATH_TO_OUTPUT_DIR` | Parent directory for output frame folders |
| `num_processes` (optional) | Number of parallel processes |

### `2_autoprompted_segmentation.py` — Two-pass auto-prompted segmentation

First performs full-frame segmentation using a generic prompt frame pool, then crops and runs a high-definition second pass. Includes automatic mask quality checking and prompt pool management.

| | |
|---|---|
| **Input** | JPG frames directory from step 1 |
| **Output** | H5 file with cleaned, high-definition segmentation masks |

**Variables to update:**

| Variable | Description |
|----------|-------------|
| `PATH_TO_VIDEO_DIR` | Path to directory containing the video frame folders |
| `OUTPUT_DIR_PATH_FOR_SEGMENTED_VIDEOS` | Output directory for H5 segmentation files |
| `PATH_TO_HDSEGMENTATION_OUTPUT_DIR` | Output directory for H5 segmentation files of final high definition segmentation |
| `PATH_TO_PROMPT_FRAME_DIR` | Path to directory with full-sized prompt frame pool |
| `PATH_TO_PROMPT_FRAME_DATA_JSON` | Path to JSON with prompt data for full frame prompt pool |
| `PATH_TO_OUTPUT_VIDEO.mp4` | Path to save optional quality assurance video |

### `3_shape_analysis.py` — Skeleton and shape metrics

Performs detailed shape analysis on each frame's segmentation mask: skeleton extraction, curvature measurement, bend angle detection... Generates shape metrics visualizations.

| | |
|---|---|
| **Input** | H5 high definition segmentation files from step 2 |
| **Output** | Pickle files with per-frame shape metrics + visualization plots |

**Variables to update:**

| Variable | Description |
|----------|-------------|
| `HDSEGMENTATION_DIR_FROM_STEP2` | Path to directory with H5 files from step 2 |
| `OUTPUT_SHAPE_ANALYSIS_DIR` | Output directory for shape analysis pickle files |
| `OUTPUT_SHAPE_ANALYSIS_PLOTS_DIR` | Output directory for shape visualization plots |

### `4_path_analysis.py` — Trajectory and behavior analysis

Analyzes worm trajectories from shape analysis data: centroid tracking, velocity, acceleration, pause/reversal detection, turn analysis... Generates trajectory visualizations.

| | |
|---|---|
| **Input** | Shape analysis pickle files from step 3 |
| **Output** | Path analysis pickle files + trajectory plots |

**Variables to update:**

| Variable | Description |
|----------|-------------|
| `SHAPE_ANALYSIS_DIR_FROM_STEP3` | Path to shape analysis directory from step 3 |
| `OUTPUT_PATH_ANALYSIS_DIR` | Output directory for path analysis results |
| `OUTPUT_PATH_ANALYSIS_PLOTS_DIR` | Output directory for trajectory plots |
| `PATH_TO_FRAMES_DIR` | Path to original frames directory (for overlay visualizations) |
| `PATH_TO_SEGMENTATION_DIR_FROM_STEP2` | Path to high definition H5 segmentation directory from step 2 |

---

## RIA Calcium Imaging

Segment RIA neuron compartments (nrd, nrv, loop) and extract calcium brightness signals and head orientation from semi-restricted *C. elegans* recordings.

<!-- ![RIA Calcium Imaging Pipeline](docs/images/ria_pipeline.png) -->

<!-- ![Example: RIA segmentation and signal](docs/images/ria_example.gif) -->

```
TIF stacks ➜ [1] JPG frames ➜ [2] Crop RIA region ➜ [3] Segment compartments ➜ [4] Extract brightness
                                       ↓
                                  [5] Segment head ➜ [6] Extract head angle
```

#### Output data

**Brightness and orientation (from step 4)**

| Field | Description |
|--------|-------------|
| `frame` | Frame index |
| `background` | Mean brightness of background pixels |
| `{object_id}` | Mean brightness of compartment in raw image (2 = nrD, 3 = nrV, 4 = loop) |
| `{object_id}_bg_corrected` | Background-subtracted brightness per compartment |
| `{object_id}_pixel_count` | Number of pixels in compartment mask |
| `side_position` | Worm orientation relative to the loop ("left" or "right") |

**Head angle (from step 6)**

| Field | Description |
|--------|-------------|
| `frame` | Frame index |
| `object_id` | Object identifier (only 1 head object) |
| `angle_degrees` | Head angle relative to body orientation (degrees) |
| `angle_degrees_corrected` | Side-corrected angle based on `side_position` |
| `bend_location` | Normalized position of maximum bend along head (0–1) |
| `bend_magnitude` | Curvature magnitude at the maximum bend |
| `bend_position_y` | Y coordinate of maximum bend |
| `bend_position_x` | X coordinate of maximum bend |
| `head_mag` | Magnitude of head direction vector |
| `body_mag` | Magnitude of body direction vector |
| `is_straight` | Whether angle is within straight threshold (<=3 degrees) |
| `is_noise_peak` | Whether the data point was identified as noise |
| `peak_deviation` | Deviation magnitude from neighbors if flagged as noise |
| `window_size_used` | Smoothing window size for bending used at this frame |
| `error` | Error message if analysis failed for this frame |
| `has_warning` | Whether this frame has any warning/error |


### `1_tiftojpg.py` — Convert TIF stacks/videos to JPG frames

Converts TIF/TIFF image stacks or video files to JPG frame sequences. Handles both multi-page TIF stacks and standard video formats.

| | |
|---|---|
| **Input** | Directory of TIF/TIFF stacks |
| **Output** | Directories of JPG frames (one per file) |

**Variables to update:**

| Variable | Description |
|----------|-------------|
| `PATH_TO_VIDEO_FILES_DIR` | Path to directory containing input TIF |
| `PATH_TO_SAVE_JPG_DIR` | Output directory for JPG frame folders |

### `2_crop_RIAregion.py` — Crop around RIA region using SAM2

Uses SAM2 video predictor to segment the RIA region, then creates zoomed 110x110 pixel crops around it for all frames.

| | |
|---|---|
| **Input** | JPG frames from step 1 |
| **Output** | Cropped frames around RIA region |

**Variables to update:**

| Variable | Description |
|----------|-------------|
| `PATH_TO_VIDEO_FILES_DIR` | Path to directory containing video frame folders from step 1 |
| `PATH_TO_SAVE_CROPPED_VIDEOS_DIR` | Output directory for cropped video folders |
| `points` | Click coordinates `[[x, y]]` for generic RIA region prompt |

### `3_autoprompted_RIAsegmentation.py` — Segment RIA compartments

Auto-prompted SAM2 segmentation of RIA compartments with built-in quality checks for empty masks, overlapping masks, and abnormally distant masks.

| | |
|---|---|
| **Input** | Cropped frames from step 2 |
| **Output** | H5 file with segmentation masks per compartment + optional prompt visualization |

**Variables to update:**

| Variable | Description |
|----------|-------------|
| `PATH_TO_CROPPED_VIDEO_DIR` | Path to cropped video directories from step 2 |
| `PATH_TO_SEGMENTED_VIDEO_DIR` | Output directory for H5 segmentation files |
| `PATH_TO_PROMPT_FRAMES_DIR` | Path to directory with RIA compartment prompt frame pool |
| `PATH_TO_PROMPT_DATA_FILE` | Path to JSON with prompt data for prompt pool |
| `PATH_TO_OUTPUT_VIDEO.mp4` | Path to save optional quality assurance video |
| `PATH_TO_OUTPUT_DIR` | Output directory for H5 files |

### `4_extract_RIAbrightness_and_orientation.py` — Extract brightness metrics

Extracts brightness intensity per compartment (nrd, nrv, loop) with background correction, pixel counts, and left/right side orientation relative to the loop.

| | |
|---|---|
| **Input** | H5 segmentation file from step 3 + original JPG frames |
| **Output** | CSV file with per-frame brightness and orientation data |

**Variables to update:**

| Variable | Description |
|----------|-------------|
| `PATH_TO_SEGMENTS_DIR` | Path to H5 segmentation directory from step 3 |
| `OUTPUT_PATH_TO_FINAL_DATA_DIR` | Output directory for CSV files |
| `PATH_TO_VIDEO_DIR` | Path to JPG video frames for brightness extraction |

### `5_head_segmentation.py` — Segment head region

Uses SAM2 video predictor to segment the head region across all frames.

| | |
|---|---|
| **Input** | Video frames from step 1 |
| **Output** | H5 file with head segmentation masks |

**Variables to update:**

| Variable | Description |
|----------|-------------|
| `PATH_TO_VIDEO_DIR` | Path to video frame directories |
| `OUTPUT_PATH_TO_HEAD_SEGMENTATION_DIR` | Output directory for head segmentation H5 files |
| `PATH_TO_OUTPUT_VIDEO.mp4` | Path to save optional quality assurance video |

### `6_extract_head_angle.py` — Extract head angle metrics

Extracts head angle relative to body using skeleton-based orientation from head segmentation masks.

| | |
|---|---|
| **Input** | H5 head segmentation from step 5 + cropped frames |
| **Output** | CSV file with head angle |

**Variables to update:**

| Variable | Description |
|----------|-------------|
| `PATH_TO_HEAD_SEGMENTATION_DIR` | Path to head segmentation H5 directory from step 5 |
| `PATH_TO_OUTPUT_DATA_DIR` | Output directory for CSV files |

---

## Citation

<!-- TODO: Add citation when paper is published -->

If you use TWARDISv0.1 in your research, please cite:

```
Guisnet, A., & Hendricks, M. (2025). Large vision model framework for automated C. elegans analysis: From static morphometry to dynamic neural activity. In bioRxiv (p. 2025.08.18.670800). https://doi.org/10.1101/2025.08.18.670800
```
