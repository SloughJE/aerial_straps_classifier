import cv2
import mediapipe as mp
import os

import pandas as pd
import numpy as np


def extract_landmarks(input_video, output_video, write_video=True):

    cap = cv2.VideoCapture(input_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Only create a VideoWriter if we intend to write the video
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height)) if write_video else None

    # Define the columns for the DataFrame
    columns = ['video_frame']
    for landmark in mp.solutions.pose.PoseLandmark:
        columns.extend([
            f'{landmark.name}_x',
            f'{landmark.name}_y',
            f'{landmark.name}_z',
            f'{landmark.name}_visibility',
            f'{landmark.name}_presence'
        ])
    df_landmarks = pd.DataFrame(columns=columns)

    frame_count = 0
    with mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the frame to RGB format.
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False

            # Detect the pose landmarks.
            results = pose.process(frame_rgb)

            # Create a blank frame to draw the landmarks on.
            blank_frame = np.zeros_like(frame)

            # Render the landmarks on the blank frame.
            if results.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(blank_frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

                # Extract all landmarks
                features = [frame_count]
                for landmark in results.pose_landmarks.landmark:
                    features.extend([landmark.x, landmark.y, landmark.z, landmark.visibility, landmark.presence])

                # Append the features to the DataFrame
                df_landmarks.loc[frame_count] = features

            # Write the frame to the video file if write_video is True
            if write_video:
                out.write(blank_frame)
            frame_count += 1

    cap.release()
    if write_video:
        out.release()

    # Compute additional features
    df_landmarks['HEAD_x'] = (df_landmarks['LEFT_EAR_x'] + df_landmarks['RIGHT_EAR_x']) / 2
    df_landmarks['HEAD_y'] = (df_landmarks['LEFT_EAR_y'] + df_landmarks['RIGHT_EAR_y']) / 2
    df_landmarks['HEAD_z'] = (df_landmarks['LEFT_EAR_z'] + df_landmarks['RIGHT_EAR_z']) / 2

    df_landmarks['NECK_x'] = (df_landmarks['LEFT_SHOULDER_x'] + df_landmarks['RIGHT_SHOULDER_x']) / 2
    df_landmarks['NECK_y'] = (df_landmarks['LEFT_SHOULDER_y'] + df_landmarks['RIGHT_SHOULDER_y']) / 2
    df_landmarks['NECK_z'] = (df_landmarks['LEFT_SHOULDER_z'] + df_landmarks['RIGHT_SHOULDER_z']) / 2
    
    df_landmarks['video_frame'] = df_landmarks['video_frame'].astype(int)

    return df_landmarks



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


def extract_landmarks_and_features(params: dict):

    input_directory = params['input_video_dir']
    output_directory = params['output_video_dir']
    features_directory = params['interim_features_directory']
    write_video = params['write_video']
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

        df_landmarks = extract_landmarks(input_video, output_video, write_video)
        
        video_name = os.path.splitext(os.path.basename(input_video))[0]
        csv_file_path = os.path.join(features_directory, f'{video_name}_landmarks.csv')
        print(f"Landmarks extracted and saved to {csv_file_path}")
        df_landmarks['filename'] = video_file
        df_landmarks.to_csv(csv_file_path, index=False)

        df_features = df_landmarks.apply(extract_angles, axis=1)
        df_features['filename'] = video_file
        df_features['video_frame'] = df_landmarks['video_frame'] # Copying frame_number from df_landmarks to df_features
        csv_file_path_features = os.path.join(features_directory, f'{video_name}_features.csv')
        print(f"Features extracted and saved to {csv_file_path_features}")
        df_features.to_csv(csv_file_path_features, index=False)

        print(f"Processed {video_file} ({idx + 1} of {total_videos})")

    print(f"{total_videos} video(s) processed successfully.")


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
    final_df = pd.merge(combined_df, labeled_df, on=['filename', 'video_frame'], how='left')

    # Print out the count of rows per unique label
    label_counts = final_df['label'].value_counts()
    print("\nNumber of rows per label:")
    for label, count in label_counts.items():
        print(f"{label}: {count}")

    filepath_features = os.path.join(final_features_directory, "final_features.csv")
    final_df.to_csv(filepath_features, index=False)

    print(f"\nFinal features combined with labels and written to {filepath_features}")

