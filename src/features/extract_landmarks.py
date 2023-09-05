from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd


def extract_landmarks_from_frame(frame_rgb: np.ndarray, 
                                 pose: mp.solutions.pose.Pose) -> Tuple[List[float], np.ndarray]:
    """
    Extracts landmarks from an RGB frame.

    Args:
        frame_rgb (np.ndarray): Input RGB frame.
        pose (mp.solutions.pose.Pose): MediaPipe Pose object.

    Returns:
        Tuple[List[float], np.ndarray]: A tuple containing the list of extracted landmarks
                                        and the annotated frame.
    """
    
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        # Annotate the frame with landmarks
        annotated_frame = frame_rgb.copy()
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

        features = [
            feature 
            for landmark in results.pose_landmarks.landmark
            for feature in [landmark.x, landmark.y, landmark.z, landmark.visibility, landmark.presence]
        ]
        return features, annotated_frame

    return [], frame_rgb


# Helper function to calculate additional features
def calculate_additional_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates additional features for a dataframe containing pose landmarks.

    The function will modify the passed dataframe to include HEAD and NECK coordinates.

    Args:
        df (pd.DataFrame): The input dataframe containing pose landmarks. The dataframe
                           should have columns named according to landmarks, for instance:
                           'LEFT_EAR_x', 'RIGHT_EAR_x', etc.

    Returns:
        None: The dataframe is modified in-place.
    """
    
    df['HEAD_x'] = (df['LEFT_EAR_x'] + df['RIGHT_EAR_x']) / 2
    df['HEAD_y'] = (df['LEFT_EAR_y'] + df['RIGHT_EAR_y']) / 2
    df['HEAD_z'] = (df['LEFT_EAR_z'] + df['RIGHT_EAR_z']) / 2
    
    df['NECK_x'] = (df['LEFT_SHOULDER_x'] + df['RIGHT_SHOULDER_x']) / 2
    df['NECK_y'] = (df['LEFT_SHOULDER_y'] + df['RIGHT_SHOULDER_y']) / 2
    df['NECK_z'] = (df['LEFT_SHOULDER_z'] + df['RIGHT_SHOULDER_z']) / 2

    return df


def extract_landmarks(input_source: str, 
                      output_destination: str, 
                      is_video: bool=True, 
                      write_output: bool=True) -> pd.DataFrame:
    """
    Extracts landmarks from an input source and optionally saves the annotated frames.

    Args:
        input_source (str): Path to the input source (video or image).
        output_destination (str): Path to save the output (annotated video or image).
        is_video (bool): Flag to indicate if the input source is a video. Default is True.
        write_output (bool): Flag to indicate if the annotated frames should be saved. Default is True.

    Returns:
        pd.DataFrame: A dataframe containing the landmarks for each frame.
    """
    
    columns = ['frame_number'] if is_video else []
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

        if is_video:
            cap = cv2.VideoCapture(input_source)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_destination, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height)) if write_output else None
            
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                landmarks, annotated_frame = extract_landmarks_from_frame(frame_rgb, pose)  # unpack the tuple
                
                if landmarks:
                    df_landmarks.loc[frame_count] = [frame_count] + landmarks
                    if write_output:
                        out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))  # Save the annotated frame to the video

                frame_count += 1

            cap.release()
            if write_output:
                out.release()

        else:  # for photo
            frame = cv2.imread(input_source)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks, annotated_frame = extract_landmarks_from_frame(frame_rgb, pose)
            
            if landmarks:
                df_landmarks.loc[0] = landmarks
            if write_output:
                cv2.imwrite(output_destination, cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
            df_landmarks['frame_number'] = 0  # Adding frame_number for photos

    calculate_additional_features(df_landmarks)
    df_landmarks['frame_number'] = df_landmarks['frame_number'].astype(int)

    return df_landmarks

