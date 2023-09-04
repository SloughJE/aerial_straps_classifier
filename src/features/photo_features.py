import cv2
import mediapipe as mp

import pandas as pd
import numpy as np


def extract_landmarks_for_photo(input_photo, output_photo=None, write_photo=True):
    # Read the photo
    image = cv2.imread(input_photo)

    # Convert the frame to RGB format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False

    columns = []
    for landmark in mp.solutions.pose.PoseLandmark:
        columns.extend([
            f'{landmark.name}_x',
            f'{landmark.name}_y',
            f'{landmark.name}_z',
            f'{landmark.name}_visibility',
            f'{landmark.name}_presence'
        ])

    df_landmarks = pd.DataFrame(columns=columns)

    with mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1) as pose:
        # Detect the pose landmarks
        results = pose.process(image_rgb)

        # Create a blank frame to draw the landmarks on
        blank_frame = np.zeros_like(image)

        # Render the landmarks on the blank frame
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(blank_frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

            features = []
            for landmark in results.pose_landmarks.landmark:
                features.extend([landmark.x, landmark.y, landmark.z, landmark.visibility, landmark.presence])

            df_landmarks.loc[0] = features

        # Write the photo if required
        if write_photo and output_photo:
            print(f"writing annoted photo to: {output_photo}")
            cv2.imwrite(output_photo, blank_frame)

    # Compute additional features as before
    df_landmarks['HEAD_x'] = (df_landmarks['LEFT_EAR_x'] + df_landmarks['RIGHT_EAR_x']) / 2
    df_landmarks['HEAD_y'] = (df_landmarks['LEFT_EAR_y'] + df_landmarks['RIGHT_EAR_y']) / 2
    df_landmarks['HEAD_z'] = (df_landmarks['LEFT_EAR_z'] + df_landmarks['RIGHT_EAR_z']) / 2
    df_landmarks['NECK_x'] = (df_landmarks['LEFT_SHOULDER_x'] + df_landmarks['RIGHT_SHOULDER_x']) / 2
    df_landmarks['NECK_y'] = (df_landmarks['LEFT_SHOULDER_y'] + df_landmarks['RIGHT_SHOULDER_y']) / 2
    df_landmarks['NECK_z'] = (df_landmarks['LEFT_SHOULDER_z'] + df_landmarks['RIGHT_SHOULDER_z']) / 2

    df_landmarks['frame_number'] = 0

    return df_landmarks
