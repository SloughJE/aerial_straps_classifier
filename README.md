# Running the Pipelines

## Video Processing and Labeling

### Overview

This code provides functionalities to reduce the quality of videos for quicker labeling and to label frames of videos. The functionalities include:

### 1. **Reduce Video Quality**
Reduces the quality of videos in a given directory to speed up labeling, specifically it reduces the lag from the time a key is pressed until the video skips to the next frame. Already processed videos will be skipped. Videos are named the same as the originals but are placed in this different directory to distinguish them.
- **Reduction Factor**: An integer that specifies the factor by which the dimensions of the videos will be reduced. For example, a reduction factor of 4 would reduce both the width and height of the video to 1/4th of their original size.

### 2. **Video Frame Labeling**
The `label_frames` and `run_labeling` functions are responsible for allowing the user to label frames of videos. Here's how it operates:

- **Labels Mapping**: The labels that can be applied to frames are configured in the `params.yaml` file. This mapping connects a keypress to a specific label. For example, pressing 'm' will label a frame as 'meathook.'
- **Skip Frames**: A configurable number of frames to skip between labeled frames. The same label is applied to skipped frames, which allows labeling of video segments rather than individual frames.
- **Input Video Directory**: The directory containing the videos that need to be labeled. This could be the original videos or the reduced-quality videos.
- **Output Directory**: The directory where the labeled data will be saved. A separate CSV file is output for each video file, named the same as the video. For example, if the video is named "example.mov", the CSV file will be named "example.csv".

The process goes as follows:

1. The video is played frame by frame (skipping the set number of frames).
2. The user is prompted to label the current frame by pressing a key that corresponds to one of the configured labels.
3. The same label is automatically applied to the next 'skip_frames' number of frames.
4. The labeling data is saved to a CSV file, where each row contains the frame number, the filename, and the label.

These functions make it efficient to label video data for machine learning tasks or other analyses, especially when the videos contain continuous segments with the same characteristics.

> **Important Note**: It is recommended to run this part of the code outside of the integrated terminal, such as in the Mac Terminal. Running video playback and labeling within an integrated terminal like the one in VSCode may lead to issues. You may also need to install some additional packages to support video playback in your terminal environment.

### Configuration

You can configure the video processing and labeling process through the `params.yaml` file. The key-value pairs include directories for input and output videos, reduction factor for video size, labels mapping, and number of frames to skip during labeling.

#### Example `params.yaml`

```yaml
video_processing:
  input_video_dir: data/raw/original/
  output_video_dir: data/raw/reduced/
  reduction_factor: 4

labeling:
  labels:
    m: meathook
    n: nutcracker
    l: l hang
    o: other_pose
    r: reverse meathook
    b: back lever
  skip_frames: 60
  input_video_dir: data/raw/reduced/
  output_dir: data/interim/
```

### Usage

To use this code, you can run the `run_pipelines.py` script with the following arguments:

- `--reduce_quality`: Reduce video quality for quicker labeling.
- `--label_data`: Label raw data.

Example commands:

```bash
python run_pipelines.py --reduce_quality
python run_pipelines.py --label_data 
```

### Functions

#### `reduce_video_size(params: dict) -> None`

Reduces the video quality in the given directory. The reduction factor can be specified in the `params` dictionary.

#### `label_frames(params: dict, video_path: str, skip_frames: int) -> list`

Allows labeling frames from the specified video file.

#### `run_labeling(params: dict) -> None`

Executes the labeling process for all video files in the specified input directory and saves the labels to CSV files.


# Directory Structure

```
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── raw            <- The original, immutable data dump.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── results        <- results
│
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment
│
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── data           <- Scripts to download or generate data
│   │   └── load_dataset.py
│   │
│   ├── features       <- Scripts to turn raw data into features for modeling
│   │   └── make_features.py
│   │
│   ├── models         <- Scripts to train models and then use trained models to make
│   │   │                 predictions
│   │   ├── predict_model.py
│   │   └── train_model.py
│   │
│   └── visualization  <- Scripts to create exploratory and results oriented visualizations
│       └── visualize.py
│
└──params.yml          <- parameters 
```

Notes:
Some models had to be manually downloaded. So I download to models/
and the cp to the correct place (based on the download errors)
cp models/pose_landmark_heavy.tflite /usr/local/lib/python3.8/site-packages/mediapipe/modules/pose_landmark