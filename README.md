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
- **CPU:** 48 cores (used for parallelized frame extraction)
- **Models:** SAM2 (~0.5 GB), worm classifier (~2.5 GB)

---

## Required Packages

Under Python **3.12.3**, latest stable versions recommended:
- h5py
- matplotlib
- networkx
- numpy
- opencv-python
- pandas
- Pillow
- PyTorch
- scikit-image
- scikit-learn
- scipy
- seaborn
- tifffile
- torchvision
- tqdm
- hydra-core (for SAM)

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
TIFF images ➜ [1] Convert ➜ JPEG images ➜ [2] Segment, classify, extract features ➜ CSV + cutout images
```

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
