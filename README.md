# Aerial Straps Pose Classifier

Aerial straps, a demanding discipline of aerial arts, involves performers executing a variety of acrobatic maneuvers while suspended from a pair of straps. The subtleties of each pose and transition, combined with the rapid motion, make automated pose detection a challenging endeavor.

This project introduces an intelligent solution that harnesses the power of machine learning to accurately classify various aerial straps poses from photos and videos. With the rapid advancements in computer vision, we've engineered a comprehensive pipeline to process, label, and train models that can recognize and categorize key poses used in aerial straps routines.

# What is Aerial Straps?
From Wikipedia, [aerial straps](https://en.wikipedia.org/wiki/Aerial_straps) "are a type of aerial apparatus on which various feats of strength and flexibility may be performed, often in the context of a circus performance. It is a cotton or nylon web apparatus that looks like two suspended ribbons. Wrapping the strap ends around hands and wrists, the performer performs holds, twists, rolls and manoeuvres, requiring extreme strength and precision similar to men’s rings in gymnastics." 
If you have seen a [Cique du Soleil](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjTyt32h5aBAxVaFlkFHV7sAhcQFnoECBkQAQ&url=https%3A%2F%2Fwww.cirquedusoleil.com%2F&usg=AOvVaw0VWSr1RfYBuHS09WwR0tD_&opi=89978449) show, you have probably seen an aerial straps performance. There are many examples on [YouTube](https://www.youtube.com/results?search_query=aerial+straps) of course. 
Although many of the performances you may see involve the performer moving through poses quickly or doing more advanced or artistic versions of a pose, there are many specific basic poses that can be identified. Some of these poses overlap with other disciplines such as calisthenics or gymnastics, for example the back lever is a common calisthenics position but can be performed on aerial straps:

| ![back lever on aerial straps](/data/raw/photos/straps_monarca_back_lever.jpg) | 
|:--:| 
| *back lever on aerial straps* |

# Motivation for Project
I've been training aerial straps for a few years, and always thought of combining aerial straps and machine learning. At one point, I saw the MediaPipe pose detection model and tried it out on a video of myself. The results were pretty good. Eventually I had the idea to create an aerial straps classification model using MediaPipe to extract features. This projects also attempts to showcase an ML project from end to end, from conception, , data pre-processing, data collection, data labeling, EDA, feature creation, model development, evaluation, hyperparameter tuning and deployment. 

## Project Highlights:

- **Data Processing**: Process and prepare media, including videos and photos, to create a streamlined dataset.
- **Data Labeling**: Distinguish between intricate poses like the 'meathook', 'nutcracker', and 'l-hang', among others.
- **Feature Extraction**: Extract critical pose landmarks and angles to capture the intricacies of each pose.
- **Model Training**: Train robust machine learning models, evaluate their performance, and refine them for real-world applications.

# Project Details

## 1. Media Processing

### 1.1 Video Processing
To streamline the labeling process, the pipeline provides functionalities to reduce the size of the videos. This ensures quicker loading and processing during the labeling stage. Additionally, the script can produce mirrored versions of the videos which can be particularly beneficial in scenarios where data augmentation is required. Already processed videos will be skipped. Videos are named the same as the originals but are placed in this different directory to distinguish them.

- **Reduction Factor**: An integer that specifies the factor by which the dimensions of the videos will be reduced. For example, a reduction factor of 4 would reduce both the width and height of the video to 1/4th of their original size.

## 1.2 Photo Processing
Similar to videos, the pipeline also offers a tool for preparing photos for labeling. This involves creating mirrored versions of the images. Mirroring photos can be useful for expanding the dataset and ensuring model robustness.

## 2. Labeling

### 2.1 Video Labeling
In this process, each frame of a video is assigned a corresponding label.

#### Steps:
1. **Displaying the Video Frame**:
   - The video frame will be displayed for inspection based on the `skip_seconds` value (in this case, every second). 
   - This assists users in deciding the most appropriate label for the current frame.
  
2. **Key Press Mapping**:
   - Based on the `params.yaml`, we have the following labels and their associated keys:
     - `m`: **meathook**
     - `n`: **nutcracker**
     - `l`: **l-hang**
     - `o`: **other pose or transition**
     - `r`: **reverse meathook**
     - `b`: **back lever**
     - `f`: **front lever**
   - The user is prompted to press the respective key to label the frame. Pressing an unassociated key will result in a reminder of the valid key mappings.

3. **Progress Saving**: 
   - The labeled data is saved in CSV format. 
   - Each row contains the frame number, filename (with "video_" as prefix), and the assigned label.

### **Video Frame Labeling Method Summary**:
The `label_videos` method allows the user to manually label frames from specified videos within a directory. The user is presented with a frame every `skip_seconds` (calculated by number of seconds to skip and frame rate of video) and assigns a label to it. This label is then applied to all frames from the previously labeled frame up to and including the current frame. At the end of the video, the user labels the final frame, and this label is applied to all remaining frames. This table respresents the approach.

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


### 2.2 Photo Labeling

#### Steps:
1. **Displaying the Photo**: 
   - Each photo is displayed for inspection.

2. **Key Press Mapping**:
   - Same as the video labeling, you'll use the keys mentioned above to label the photos.

3. **Progress Saving**: 
   - The labeled data is saved in CSV format.
   - Each row contains the photo filename (with "photo_" as prefix) and the assigned label.


> **Important Note**: It is recommended to run the Labeling part of the code outside of the VS Code integrated terminal, such as in the Mac Terminal. Running video playback and labeling within an integrated terminal like the one in VS Code may lead to issues. You may also need to install some additional packages to support video playback in your terminal environment.

## 3. Mirrored Media Labeling

### Overview:
Once videos and photos are labeled, it's essential to ensure that their mirrored versions also have appropriate labels.

### Steps:
1. **Identifying Mirrored Files**: 
   - The code identifies files in the specified directories that have a "mirrored_" prefix. 
   - It then matches these mirrored files to their original counterparts.

2. **Label Application**: 
   - For every mirrored video or photo, labels from the original file are applied.
   - The filenames in the CSV output for these mirrored files will contain the "mirrored_" prefix to distinguish them from their original counterparts.

3. **Saving Mirrored Labels**: 
   - The mirrored labels are saved in CSV format in the specified output directory.

### Note: 
This step is automated and doesn't require manual labeling. It merely applies existing labels to mirrored versions.


## 4. Features

### Overview

This code extracts pivotal pose landmarks and angles from video frames and photos, optimizing for aerial straps performance analysis.

### Pose Landmark Extraction

Utilize [MediaPipe's Pose Landmarker](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker) to derive critical pose landmarks. Not all landmarks are considered; we focus on those pertinent to aerial straps, disregarding ones like the mouth landmark.

#### How MediaPipe Pose Landmarker Works:

MediaPipe's Pose Landmarker detects key body landmarks using a trained machine learning model. The landmarks represent anatomical points, providing a simplified yet robust skeleton of a person in 2D space.

### Joint Angle Features Calculation with Landmarks

Angles between joints or body segments are invaluable for aerial straps posture analysis. The code calculates the angle formed at a vertex between two other points in 2D space. We use the landmark data to derive specific joint angles vital for identifying poses, such as elbow, shoulder, hip, and knee angles.

Some of the extracted joint angle features include:

- **Elbow Angle**: The angle formed between the shoulder, elbow, and wrist joints.
- **Shoulder Angle**: The angle formed between the elbow, shoulder, and hip joints.
- **Hip Angle**: The angle formed between the shoulder, hip, and knee joints.
- **Knee Angle**: The angle formed between the ankle, knee, and hip joints.
- **Spine Angle**: The angle formed between the left hip, right hip, and head landmarks.
- **Torso Angle**: The angle formed between the left hip, right hip, and neck landmarks.

These joint angles provide insights into the body's orientation and can be valuable features for training machine learning models to classify and analyze image data.

### 1. **Extract Pose Landmarks and Features from Videos**

Use the `extract_landmarks_and_features_for_videos` function to obtain the required data from video frames:

- **Input**: Accepts both original or quality-reduced videos.
- **Output**: Produces two distinct CSV files for every video: `{video_name}_landmarks.csv` and `{video_name}_features.csv`.

**Procedure**:
1. Break down the video into individual frames and employ the Mediapipe library to extract pose landmarks.
2. Compute the relevant angles derived from the extracted pose landmarks.
3. Record the landmarks and angles into separate CSV files, indexed with the video frame number.

### 2. **Extract Pose Landmarks and Features from Photos**

For the analysis of static poses from photos, leverage the `extract_landmarks_and_features_for_photos` function:

- **Input**: Takes in high-resolution or down-scaled photos.
- **Output**: Outputs two CSV files for every photo: `{photo_name}_landmarks.csv` and `{photo_name}_features.csv`.

**Steps**:
1. Process each photo using landmark extraction tools, including the Mediapipe framework.
2. Determine joint angles based on the landmarks obtained.
3. Catalog the landmarks and angles in dedicated CSV files, each labeled with the respective photo's name.

Here is an example of the landmarks (blue dots) extracted with connections drawn between the landmarks:

| ![landmarks extracted with Mediapipe pose model](/data/processed/photos/straps_monarca_back_lever_small.jpeg) | 
|:--:| 
| *landmarks extracted with Mediapipe pose model* |

## 5. Combine Features from all CSV Files

The `combine_csv_files` function consolidates interim feature files with labeled files into a singular CSV. This utility ensures that the extracted features from videos/photos and their corresponding labels are combined in a structured manner, ready for subsequent analysis or model training.

### Functionality
The function reads interim features and labeled files from specified directories and merges them based on the filename and frame number. The merged result, which contains both features and labels, is saved in the final features directory. If any row in the merged DataFrame lacks a matching label, a warning is printed to notify the user with the concerning file. The final DataFrame, which comprises features combined with labels, is saved to the `final_features_directory` with the filename `final_features.csv`.

Ensure the labeled data is consistent and matches the filenames and frame numbers in the features data. Any inconsistency may result in missing labels for some entries.

## 6. Model Training and Evaluation

### Overview

This code provides functionalities for training, evaluating, and saving machine learning models for the image analysis task. The main components include:

### 1. **Model Training Pipeline**
The `train_dev_model` function manages the process of splitting data, encoding labels, training the classifier, and saving the trained model.

### 2. **Generating Evaluation Metrics**
Functions such as `generate_roc_curves_and_save`, `generate_pr_curves_and_save`, `generate_visualizations_and_save_metrics`, and `generate_feature_importance_visualization` are used to generate evaluation metrics and visualizations for the trained models.

## 7. Model Training: Production

### Overview

The `train_prod_model` method trains a machine learning model (e.g., XGBoost or RandomForest) on the entire dataset and saves it along with the label encoder.


# Usage Sequence

### 1. Media Processing
#### 1.1 Process Videos
```bash
python run_pipelines.py --process_videos
```
#### 1.2 Process Photos
```bash
python run_pipelines.py --process_photos
```
### 2. Labeling
#### 2.1 Label Videos

```bash
python run_pipelines.py --label_videos
```
#### 2.2 Label Photos
```bash
python run_pipelines.py --label_photos
```

### 3. Mirrored Media Labeling
```bash
python run_pipelines.py --apply_mirror_labels
```
### 4. Features
#### 4.1 Make Features from Videos
```bash
python run_pipelines.py --make_features_videos
```
#### 4.2 Make Features from Photos
```bash
python run_pipelines.py --make_features_photos
```

### 5. Combine Feature CSVs
```bash
python run_pipelines.py --combine_feature_csv
```

### 6. Train Development Model
```bash
python run_pipelines.py --train_dev_model
```

### 7. Train Production Model
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
