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


def extract_spatial_features(df_landmarks: pd.DataFrame) -> pd.DataFrame:
    """
    Extract spatial features from the provided landmarks dataframe.
    
    The function calculates relative positions such as hip_to_shoulder, knee_to_hip, etc.
    using a defined margin of error for labeling positions as 'level'.
    
    Parameters:
    - df_landmarks (pd.DataFrame): DataFrame containing landmark data.
    
    Returns:
    - pd.DataFrame: DataFrame containing the extracted spatial features.
    """
    
    margin_error = 0.08
    prefix = "spatial_"
    
    def relative_position(y1: float, y2: float) -> str:
        """
        Calculate the relative position of y1 with respect to y2.
        
        Parameters:
        - y1 (float): The y-coordinate of the first landmark.
        - y2 (float): The y-coordinate of the second landmark.
        
        Returns:
        - str: 'above', 'below', or 'level' indicating y1's position relative to y2.
        """
        below = y1 > y2 + margin_error
        above = y1 < y2 - margin_error
        return np.where(below, 'below', np.where(above, 'above', 'level'))

    # Lists of landmarks to simplify the creation of the columns
    relations = [('KNEE', 'HIP'), ('KNEE', 'SHOULDER'), ('ELBOW', 'HIP'), 
                 ('KNEE', 'ANKLE'), ('ELBOW', 'SHOULDER'), ('WRIST', 'ELBOW'), 
                 ('WRIST', 'SHOULDER'), ('WRIST', 'HIP'), ('WRIST', 'KNEE'), 
                 ('WRIST', 'ANKLE')]
    
    sides = ['LEFT', 'RIGHT']
    
    for side in sides:
        for rel1, rel2 in relations:
            y1 = df_landmarks[f'{side}_{rel1}_y'] if rel1 != 'HEAD' else df_landmarks['HEAD_y']
            y2 = df_landmarks[f'{side}_{rel2}_y'] if rel2 != 'HEAD' else df_landmarks['HEAD_y']
            column_name = f'{prefix}{side.lower()}_{rel1.lower()}_to_{rel2.lower()}'
            df_landmarks[column_name] = relative_position(y1, y2)

    # Calculate relative positions, may add some later
    #df_landmarks['hip_to_shoulder'] = relative_position(df_landmarks['avg_hip_y'], df_landmarks['avg_shoulder_y'])
    #df_landmarks['knee_to_hip'] = relative_position(df_landmarks['avg_knee_y'], df_landmarks['avg_hip_y'])
    #df_landmarks['knee_to_shoulder'] = relative_position(df_landmarks['avg_knee_y'], df_landmarks['avg_shoulder_y'])
    #df_landmarks['head_to_shoulder'] = relative_position(df_landmarks['HEAD_y'], df_landmarks['avg_shoulder_y'])
    #df_landmarks['elbow_to_hip'] = relative_position(df_landmarks['avg_elbow_y'], df_landmarks['avg_hip_y'])
    #df_landmarks['knee_to_ankle'] = relative_position(df_landmarks['avg_knee_y'], df_landmarks['avg_ankle_y'])

#    Add the general hip_to_shoulder and head_to_shoulder with the prefix
    df_landmarks[f'{prefix}hip_to_shoulder'] = relative_position(df_landmarks['avg_hip_y'], df_landmarks['avg_shoulder_y'])
    df_landmarks[f'{prefix}head_to_shoulder'] = relative_position(df_landmarks['HEAD_y'], df_landmarks['avg_shoulder_y'])

    # Build a list of the new columns to return
    columns = [f'{prefix}{side.lower()}_{rel1.lower()}_to_{rel2.lower()}' for side in sides for rel1, rel2 in relations]
    columns += [f'{prefix}hip_to_shoulder', f'{prefix}head_to_shoulder']

    return df_landmarks[columns]


def extract_features_from_landmarks(params: Dict[str, Union[str, bool]]) -> None:
    """
    Extract spatial and angle features from landmark CSV files and save to new CSV files.

    This function processes a list of CSV files from the provided directory containing landmarks, 
    extracts spatial and angle features, and then saves these features to new CSV files 
    in the specified features directory.

    Parameters:
    - params (Dict[str, Union[str, bool]]): A dictionary containing:
        'interim_landmarks_directory': The path to the directory containing landmark CSV files.
        'interim_features_directory': The path to the directory where feature CSV files will be saved.
    """

    landmarks_directory = params['interim_landmarks_directory']
    features_directory = params['interim_features_directory']

    # Get a list of all landmark CSV files in the directory
    csv_files = [f for f in os.listdir(landmarks_directory) if f.endswith('_landmarks.csv')]

    for csv_file in csv_files:
        df_landmarks = pd.read_csv(os.path.join(landmarks_directory, csv_file))

        # Extract spatial features
        df_spatial = extract_spatial_features(df_landmarks)

        # Extract angles features
        df_angles = df_landmarks.apply(extract_angles, axis=1)

        # Create a new DataFrame for features
        df_features = pd.concat([df_landmarks['filename'], df_landmarks['frame_number'], df_spatial, df_angles], axis=1)

        new_csv_file_name = csv_file.replace("_landmarks", "")
        csv_file_path_features = os.path.join(features_directory, f"{os.path.splitext(new_csv_file_name)[0]}_features.csv")
        print(f"Features extracted and saved to {csv_file_path_features}")
        df_features.to_csv(csv_file_path_features, index=False)

    print(f"Created features for {len(csv_files)} files")


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

