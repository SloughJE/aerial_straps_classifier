import cv2
import mediapipe as mp
import os

import pandas as pd
import numpy as np


# Process the video file.
def extract_landmarks(input_video, output_video, save_directory):

    cap = cv2.VideoCapture(input_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

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

            out.write(blank_frame)
            frame_count += 1

    cap.release()
    out.release()

    # Adding mid point between ears for HEAD
    df_landmarks['HEAD_x'] = (df_landmarks['LEFT_EAR_x'] + df_landmarks['RIGHT_EAR_x']) / 2
    df_landmarks['HEAD_y'] = (df_landmarks['LEFT_EAR_y'] + df_landmarks['RIGHT_EAR_y']) / 2
    df_landmarks['HEAD_z'] = (df_landmarks['LEFT_EAR_z'] + df_landmarks['RIGHT_EAR_z']) / 2

    # Adding mid point between shoulders for NECK, close enough?
    df_landmarks['NECK_x'] = (df_landmarks['LEFT_SHOULDER_x'] + df_landmarks['RIGHT_SHOULDER_x']) / 2
    df_landmarks['NECK_y'] = (df_landmarks['LEFT_SHOULDER_y'] + df_landmarks['RIGHT_SHOULDER_y']) / 2
    df_landmarks['NECK_z'] = (df_landmarks['LEFT_SHOULDER_z'] + df_landmarks['RIGHT_SHOULDER_z']) / 2

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
    save_directory = params['save_directory']

    # Get a list of all video files in the input directory
    video_files = [f for f in os.listdir(input_directory) if os.path.isfile(os.path.join(input_directory, f)) and f.endswith(('.mp4', '.mov'))]

    for video_file in video_files:
        input_video = os.path.join(input_directory, video_file)
        output_video = os.path.join(output_directory, video_file) # You can also modify this based on your requirement
        print(f"Processing video: {input_video}")

        df_landmarks = extract_landmarks(input_video, output_video, save_directory)
        # Extract the name of the video file (without extension)
        video_name = os.path.splitext(os.path.basename(input_video))[0]
        # Combine the save directory with the video name to create the CSV file name
        csv_file_path = os.path.join(save_directory, f'{video_name}_landmarks.csv')
        # Save the DataFrame to the CSV file
        print(f"landmarks extracted and saved to {csv_file_path}")
        df_landmarks.to_csv(csv_file_path, index=False)

        df_features = df_landmarks.apply(extract_angles, axis=1)
        # Combine the save directory with the video name to create the CSV file name for features
        csv_file_path_features = os.path.join(save_directory, f'{video_name}_features.csv')
        # Save the DataFrame to the CSV file for features
        print(f"Features extracted and saved to {csv_file_path_features}")
        df_features.to_csv(csv_file_path_features, index=False)
