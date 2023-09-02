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

**Summary**:
The `label_frames` function allows the user to manually label frames from a specified video file. The user is presented with a frame every `skip_seconds` (calculated by number of seconds to skip and frame rate of video) and assigns a label to it. This label is then applied to all frames from the previously labeled frame up to and including the current frame. At the end of the video, the user labels the final frame, and this label is applied to all remaining frames. This table respresents the approach.

**Table**:

| Frame Number Displayed | Pressed Key | Label Assigned | Frames Labeled |
|------------------------|-------------|----------------|----------------|
| 0                      | ‘a’         | Apple          | 0              |
| 5                      | ‘b’         | Banana         | 1,2,3,4,5      |
| 10                     | ‘a’         | Apple          | 6,7,8,9,10     |
| 15                     | 'c'         | Cherry         | 11,12,13,14,15 |
| 20                     | ‘c'         | Cherry         | 16,17,18,19,20 |
| 22                     | 'a'         | Apple          | 21,22          |

**Explanation**: 
In this example, there are 22 frames in total, and `skip_frames` is set to 5. The user is first presented with frame 0 and assigns the label 'Apple'. This label is applied to frame 0. Next, the user is presented with frame 5 and assigns the label 'Banana'. This label is then applied to frames 1 through 5. This process continues until the end of the video. Since the last frame displayed by skipping frames is not the final frame of the video, the user is presented with the final frame (frame 22) and assigns the label 'Apple'. This label is then applied to frames 21 and 22.

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

## Make Features

### Overview

This code provides functionality to extract pose landmarks and angles from video frames and save them as intermediate features for further analysis and model training.

### Joint Angle Features

Joint angle features provide insights into the positions and relationships of body parts in videos. These angles can be useful for understanding movement patterns and poses in the videos. Some of the extracted joint angle features include:

- **Elbow Angle**: The angle formed between the shoulder, elbow, and wrist joints.
- **Shoulder Angle**: The angle formed between the elbow, shoulder, and hip joints.
- **Hip Angle**: The angle formed between the shoulder, hip, and knee joints.
- **Knee Angle**: The angle formed between the ankle, knee, and hip joints.
- **Spine Angle**: The angle formed between the left hip, right hip, and head landmarks.
- **Torso Angle**: The angle formed between the left hip, right hip, and neck landmarks.

These joint angles provide insights into the body's orientation and can be valuable features for training machine learning models to classify and analyze video data.

### 1. **Extract Pose Landmarks and Features**

The `extract_landmarks_and_features` function extracts pose landmarks and calculates various angles from the video frames. Here's how it operates:

- **Input Videos**: The videos from which pose landmarks need to be extracted. These videos can be the original ones or the ones with reduced quality.
- **Output Landmarks and Features**: The extracted landmarks and calculated angles are saved as CSV files for each video. The files are named as `{video_name}_landmarks.csv` and `{video_name}_features.csv`.

The process goes as follows:

1. For each video, the function processes each frame to extract pose landmarks using the Mediapipe library.
2. The pose landmarks are used to calculate angles between different body parts, which are important features for analysis.
3. The extracted landmarks and angles are saved in CSV files along with the corresponding video frame number.

### 2. Combine Features from all CSV Files
The `combine_csv_files` function combines CSV files in the given directory, specifically files with names ending in `_features.csv`. The merged data is then merged with labeled data to create the final feature dataset for model training.


### Configuration
The `params.yaml` file contains key-value pairs that configure the process of extracting pose landmarks and features from videos. These parameters control the input and output directories, settings for video processing, and paths to save the extracted features. 

```yaml
features:
  input_video_dir: data/interim/reduced/
  output_video_dir: data/processed/videos/
  interim_features_directory: data/interim/features
  labeled_dir: data/interim/labeled
  write_video: False
  final_features_directory: data/processed/features
```

## Model Training and Evaluation

### Overview

This code provides functionalities for training, evaluating, and saving machine learning models for video analysis tasks. The main components include:

### 1. **Model Training and Saving**
The `train_prod_model` function trains a machine learning model (e.g., XGBoost or RandomForest) on the entire dataset and saves it along with the label encoder.

- **Input Data**: The final feature matrix and labels read from a CSV file.
- **Model Type**: The type of model to be trained (specified in the `params` dictionary).
- **Output**: The trained model and label encoder are saved in appropriate directories.

### 2. **Generating Evaluation Metrics**
Functions such as `generate_roc_curves_and_save`, `generate_pr_curves_and_save`, `generate_visualizations_and_save_metrics`, and `generate_feature_importance_visualization` are used to generate evaluation metrics and visualizations for the trained models.

### 3. **Model Training Pipeline**
The `train_model_pipeline` function manages the process of splitting data, encoding labels, training the classifier, and saving the trained model.

- **Parameters**: Configurable parameters include the file paths, model type, test size, model-specific parameters, and directory paths.

### Configuration

You can configure the model training and evaluation process through the `params.yaml` file. The key-value pairs include file paths, model type, test size, and model-specific parameters.

#### Example `params.yaml`

```yaml
model_dev:
  model_type: xgb
  final_features_filepath: data/processed/features/final_features.csv
  test_size: 0.3
  target_column: label
  predictions_dir: data/results/
  optimize_hyperparams: True

model_prod:
  model_type: xgb
  final_features_filepath: data/processed/features/final_features.csv
  target_column: label
```


## Usage

### 1. Reduce Video Quality

```bash
python run_pipelines.py --reduce_quality
```
### 2. Label Video Data

```bash
python run_pipelines.py --label_data 
```
### 3. Make Features
```
python run_pipelines.py --label_data
```
### 4. Combine Feature CSVs
```bash
python run_pipelines.py --combine_feature_csv
```
### 5. Train Development Model
```bash
python run_pipelines.py --train_dev_model
```
### 6. Train Production Model
```bash
python run_pipelines.py --train_prod_model
```


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