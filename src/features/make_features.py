import os

import pandas as pd
import numpy as np

from .video_features import extract_landmarks_for_video
from .photo_features import extract_landmarks_for_photo


def calculate_2d_angle( a, b, c):
    a = np.array([a[0], a[1]])
    b = np.array([b[0], b[1]])
    c = np.array([c[0], c[1]])

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = radians * 180.0 / np.pi

    if angle < 0:
        angle += 360

    return angle


def extract_angles(row):
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


def extract_landmarks_and_features_for_videos(params: dict):

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

        df_landmarks = extract_landmarks_for_video(input_video, output_video, write_video)
        
        video_name = os.path.splitext(os.path.basename(input_video))[0]
        csv_file_path = os.path.join(features_directory, f'{video_name}_landmarks.csv')
        print(f"Landmarks extracted and saved to {csv_file_path}")
        df_landmarks['filename'] = video_file
        df_landmarks.to_csv(csv_file_path, index=False)

        df_features = df_landmarks.apply(extract_angles, axis=1)
        df_features['filename'] = video_file
        df_features['frame_number'] = df_landmarks['frame_number'] # Copying frame_number from df_landmarks to df_features
        csv_file_path_features = os.path.join(features_directory, f'{video_name}_features.csv')
        print(f"Features extracted and saved to {csv_file_path_features}")
        df_features.to_csv(csv_file_path_features, index=False)

        print(f"Processed {video_file} ({idx + 1} of {total_videos})")

    print(f"{total_videos} video(s) processed successfully.")


def extract_landmarks_and_features_for_photos(params: dict):
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

        df_landmarks = extract_landmarks_for_photo(input_photo, output_photo, write_photo)

        photo_name = os.path.splitext(os.path.basename(input_photo))[0]
        csv_file_path = os.path.join(features_directory, f'{photo_name}_landmarks.csv')
        print(f"Landmarks extracted and saved to {csv_file_path}")
        df_landmarks['filename'] = photo_file
        df_landmarks.to_csv(csv_file_path, index=False)

        df_features = df_landmarks.apply(extract_angles, axis=1)
        df_features['filename'] = photo_file
        df_features['frame_number'] = df_landmarks['frame_number'] # Copying frame_number from df_landmarks to df_features
        csv_file_path_features = os.path.join(features_directory, f'{photo_name}_features.csv')
        print(f"Features extracted and saved to {csv_file_path_features}")
        df_features.to_csv(csv_file_path_features, index=False)

        print(f"Processed {photo_file} ({idx + 1} of {total_photos})")

    print(f"{total_photos} photo(s) processed successfully.")


def combine_csv_files(params: dict) -> None:
    """
    Combine CSV files in the given directory with filenames ending in '_features.csv' 
    and merge with labeled data to create the final features dataframe.

    Args:
        params (dict): Dictionary containing the following key-value pairs:
            - 'interim_features_directory': Directory containing interim feature CSV files.
            - 'final_features_directory': Directory where the final features CSV will be saved.
            - 'labeled_dir': Directory containing labeled CSV files.

    Returns:
        None
    """
    interim_features_directory = params['interim_features_directory']
    final_features_directory = params['final_features_directory']
    labeled_dir = params['labeled_dir']

    # Get list of all CSV files in the interim_features_directory with filenames ending in '_features.csv'
    csv_files = [f for f in os.listdir(interim_features_directory) if f.endswith('_features.csv')]
    print(f"Combining {len(csv_files)} interim feature files")
    
    # Create a list of dataframes by reading each CSV file
    list_of_dfs = [pd.read_csv(os.path.join(interim_features_directory, f)) for f in csv_files]

    # Concatenate all dataframes into a single dataframe
    combined_df = pd.concat(list_of_dfs, ignore_index=True)

    # Load all labeled CSV files
    labeled_csv_files = [f for f in os.listdir(labeled_dir) if f.endswith('_labeled.csv')]
    labeled_dfs = [pd.read_csv(os.path.join(labeled_dir, f)) for f in labeled_csv_files]
    labeled_df = pd.concat(labeled_dfs, ignore_index=True)

    # Merge the labeled data into the final features dataframe
    final_df = pd.merge(combined_df, labeled_df, on=['filename', 'frame_number'], how='left')

    # Print out the count of rows per unique label
    label_counts = final_df['label'].value_counts()
    print("\nNumber of rows per label:")
    for label, count in label_counts.items():
        print(f"{label}: {count}")

    filepath_features = os.path.join(final_features_directory, "final_features.csv")
    final_df.to_csv(filepath_features, index=False)

    print(f"\nFinal features combined with labels and written to {filepath_features}")

