import os
from typing import Tuple, Dict, Union

import numpy as np
import pandas as pd

from .extract_landmarks import extract_landmarks



def calculate_2d_angle(a: Tuple[float, float], 
                       b: Tuple[float, float], 
                       c: Tuple[float, float]) -> float:
    """
    Calculate the angle in 2D space between three points.
    
    Args:
    - a (Tuple[float, float]): Coordinate of the first point.
    - b (Tuple[float, float]): Coordinate of the vertex or joint point.
    - c (Tuple[float, float]): Coordinate of the third point.
    
    Returns:
    - angle (float): The angle in degrees between the three points at point 'b'.
    """
    
    a = np.array([a[0], a[1]])
    b = np.array([b[0], b[1]])
    c = np.array([c[0], c[1]])

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = radians * 180.0 / np.pi

    if angle < 0:
        angle += 360

    return angle



def extract_angles(row: pd.Series) -> pd.Series:
    """
    Extract joint angles from the provided landmarks row.

    This function takes a row from a dataframe containing 2D landmark coordinates 
    and calculates various joint angles, such as elbow, shoulder, hip, knee, 
    spine, and torso angles. It uses the `calculate_2d_angle` function to get 
    these angles.

    Parameters:
    - row (pd.Series): A row from a dataframe containing 2D landmark coordinates.

    Returns:
    - pd.Series: A series containing joint angle names as the index and the 
      corresponding calculated angles as values.
    """
    
    angles = {}

    # Define the landmarks for each angle
    angles_definitions = {
        'elbow_angle_left': ('LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST'), 
        'elbow_angle_right': ('RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST'),
        'shoulder_angle_left': ('LEFT_ELBOW', 'LEFT_SHOULDER', 'LEFT_HIP'),
        'shoulder_angle_right': ('RIGHT_ELBOW', 'RIGHT_SHOULDER', 'RIGHT_HIP'),
        'hip_angle_left': ('LEFT_SHOULDER', 'LEFT_HIP', 'LEFT_KNEE'),
        'hip_angle_right': ('RIGHT_SHOULDER', 'RIGHT_HIP', 'RIGHT_KNEE'),
        'knee_angle_left': ('LEFT_ANKLE', 'LEFT_KNEE', 'LEFT_HIP'),
        'knee_angle_right': ('RIGHT_ANKLE', 'RIGHT_KNEE', 'RIGHT_HIP'),
        'spine_angle': ('LEFT_HIP', 'RIGHT_HIP', 'HEAD'),
        'torso_angle': ('LEFT_HIP', 'RIGHT_HIP', 'NECK'),
    }

    # Calculate each angle
    for angle_name, (a, b, c) in angles_definitions.items():

        point_a = [row[a + '_x'], row[a + '_y']] if isinstance(a, str) else a
        point_b = [row[b + '_x'], row[b + '_y']] if isinstance(b, str) else b
        point_c = [row[c + '_x'], row[c + '_y']] if isinstance(c, str) else c
        angle = calculate_2d_angle(point_a, point_b, point_c)
        angles[angle_name] = angle

    return pd.Series(angles)


def extract_landmarks_and_features_for_videos(params: Dict[str, Union[str, bool]]) -> None:
    """
    Extract landmarks and features from videos present in a given directory.

    Given a set of parameters, this function reads videos from an input directory,
    extracts landmarks and features from these videos, and saves the results to 
    specified directories.

    Parameters:
    - params (dict): A dictionary containing the required parameters. Expected keys are:
        * input_video_dir: The directory containing the input videos.
        * output_video_dir: The directory where the annotated videos will be saved (if enabled).
        * interim_features_directory: The directory where the extracted features will be saved.
        * save_annotated_video: A boolean flag indicating if the annotated video should be saved.

    Returns:
    - None
    """

    input_directory = params['input_video_dir']
    output_directory = params['output_video_dir']
    features_directory = params['interim_features_directory']
    write_video = params['save_annotated_video']
    if write_video:
        # Check if the directory exists, if not, create it
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
    # Get a list of all video files in the input directory
    video_files = [f for f in os.listdir(input_directory) if os.path.isfile(os.path.join(input_directory, f)) and f.endswith(('.mp4', '.mov'))]

    # Filter out videos that have already been processed
    videos_to_process = [filename for filename in video_files if not os.path.exists(os.path.join(features_directory, os.path.splitext(filename)[0] + '_landmarks.csv'))]
    total_videos = len(videos_to_process)
    already_processed = len(video_files) - total_videos
    print(f"Total number of videos to process: {total_videos}.\nAlready processed {already_processed}.")

    for idx, video_file in enumerate(videos_to_process):
        input_video = os.path.join(input_directory, video_file)
        output_video = os.path.join(output_directory, video_file)
        print(f"Processing video: {input_video}")

        df_landmarks = extract_landmarks(input_video, output_video, True, write_video)
        
        video_name = os.path.basename(input_video)
        modified_video_name = "video_" + video_name
        csv_file_path = os.path.join(features_directory, f'{modified_video_name}_landmarks.csv')
        print(f"Landmarks extracted and saved to {csv_file_path}")
        df_landmarks['filename'] = modified_video_name
        df_landmarks.to_csv(csv_file_path, index=False)

        df_features = df_landmarks.apply(extract_angles, axis=1)
        df_features['filename'] = modified_video_name
        df_features['frame_number'] = df_landmarks['frame_number'] # Copying frame_number from df_landmarks to df_features
        csv_file_path_features = os.path.join(features_directory, f'{modified_video_name}_features.csv')
        print(f"Features extracted and saved to {csv_file_path_features}")
        df_features.to_csv(csv_file_path_features, index=False)

        print(f"Processed {video_file} ({idx + 1} of {total_videos})")

    print(f"{total_videos} video(s) processed successfully.")


def extract_landmarks_and_features_for_photos(params: Dict[str, Union[str, bool]]) -> None:
    """
    Extract landmarks and features from photos present in a given directory.

    Given a set of parameters, this function reads photos from an input directory,
    extracts landmarks and features from these photos, and saves the results to 
    specified directories.

    Parameters:
    - params (dict): A dictionary containing the required parameters. Expected keys are:
        * input_photo_dir: The directory containing the input photos.
        * output_photo_dir: The directory where the annotated photos will be saved (if enabled).
        * interim_features_directory: The directory where the extracted features will be saved.
        * save_annotated_photo: A boolean flag indicating if the annotated photo should be saved.

    Returns:
    - None
    """

    input_directory = params['input_photo_dir']
    output_directory = params['output_photo_dir']
    features_directory = params['interim_features_directory']
    write_photo = params['save_annotated_photo']

    photo_files = [f for f in os.listdir(input_directory) if os.path.isfile(os.path.join(input_directory, f))\
                        and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if write_photo:
        # Check if the directory exists, if not, create it
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
    
    photos_to_process = [filename for filename in photo_files \
                         if not os.path.exists(os.path.join(features_directory, os.path.splitext(filename)[0] + '_landmarks.csv'))]
    total_photos = len(photos_to_process)
    already_processed = len(photo_files) - total_photos
    print(f"Total number of photos to process: {total_photos}.\nAlready processed {already_processed}.")

    for idx, photo_file in enumerate(photos_to_process):
        input_photo = os.path.join(input_directory, photo_file)
        output_photo = os.path.join(output_directory, photo_file)
        print(f"Processing photo: {input_photo}")

        df_landmarks = extract_landmarks(input_photo, output_photo, False,  write_photo)

        photo_name = os.path.basename(input_photo)
        modified_photo_name = "photo_" + photo_name  
        csv_file_path = os.path.join(features_directory, f'{modified_photo_name}_landmarks.csv')
        print(f"Landmarks extracted and saved to {csv_file_path}")
        df_landmarks['filename'] = modified_photo_name
        df_landmarks.to_csv(csv_file_path, index=False)

        df_features = df_landmarks.apply(extract_angles, axis=1)
        df_features['filename'] = modified_photo_name
        df_features['frame_number'] = df_landmarks['frame_number'] # Copying frame_number from df_landmarks to df_features
        csv_file_path_features = os.path.join(features_directory, f'{modified_photo_name}_features.csv')
        print(f"Features extracted and saved to {csv_file_path_features}")
        df_features.to_csv(csv_file_path_features, index=False)

        print(f"Processed {photo_file} ({idx + 1} of {total_photos})")

    print(f"{total_photos} photo(s) processed successfully.")


def combine_csv_files(params: Dict[str, str]) -> None:
    """
    Combine interim feature files and labeled files into a single CSV.

    Given a set of parameters, this function reads interim features and labeled 
    files from the respective directories, merges them based on filename and frame number,
    and saves the combined result in the final features directory.

    Parameters:
    - params (dict): A dictionary containing the required parameters. Expected keys are:
        * interim_features_directory: The directory containing the interim features files.
        * final_features_directory: The directory where the final combined features will be saved.
        * labeled_dir: The directory containing the labeled files.

    Returns:
    - None
    """
    
    interim_features_directory = params['interim_features_directory']
    final_features_directory = params['final_features_directory']
    labeled_dir = params['labeled_dir']

    if not os.path.exists(interim_features_directory) or not os.path.exists(labeled_dir):
        print(f"Either the interim features directory or the labeled directory does not exist.")
        return

    # Ensure output directory exists
    if not os.path.exists(final_features_directory):
        os.makedirs(final_features_directory)

    csv_files = [f for f in os.listdir(interim_features_directory) if f.lower().endswith('_features.csv')]
    print(f"Combining {len(csv_files)} interim feature files")

    if not csv_files:
        print("No interim feature files found.")
        return

    list_of_dfs = []
    for file in csv_files:
        try:
            list_of_dfs.append(pd.read_csv(os.path.join(interim_features_directory, file)))
        except Exception as e:
            print(f"Error reading {file}: {e}")

    combined_df = pd.concat(list_of_dfs, ignore_index=True)

    labeled_csv_files = [f for f in os.listdir(labeled_dir) if f.lower().endswith('_labeled.csv')]
    labeled_dfs = []
    for file in labeled_csv_files:
        try:
            labeled_dfs.append(pd.read_csv(os.path.join(labeled_dir, file)))
        except Exception as e:
            print(f"Error reading labeled file {file}: {e}")

    if not labeled_dfs:
        print("No labeled files found.")
        return

    labeled_df = pd.concat(labeled_dfs, ignore_index=True)
    final_df = pd.merge(combined_df, labeled_df, on=['filename', 'frame_number'], how='left')

    if final_df['label'].isna().any():
        missing_label_filenames = final_df.loc[final_df['label'].isna(), 'filename'].unique()
        print("Some rows do not have matching labels. Consider checking your labeled data.")
        print("Files with missing labels:")
        for filename in missing_label_filenames:
            print(filename)

    label_counts = final_df['label'].value_counts()
    print("\nNumber of rows per label:")
    for label, count in label_counts.items():
        print(f"{label}: {count}")

    filepath_features = os.path.join(final_features_directory, "final_features.csv")
    final_df.to_csv(filepath_features, index=False)

    print(f"\nFinal features combined with labels and written to {filepath_features}")

