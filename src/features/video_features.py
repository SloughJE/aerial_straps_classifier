import cv2
import mediapipe as mp

import pandas as pd
import numpy as np


def extract_landmarks_for_video(input_video, output_video, write_video=True):

    cap = cv2.VideoCapture(input_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Only create a VideoWriter if we intend to write the video
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height)) if write_video else None

    # Define the columns for the DataFrame
    columns = ['frame_number']
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
    
    df_landmarks['frame_number'] = df_landmarks['frame_number'].astype(int)

    return df_landmarks
