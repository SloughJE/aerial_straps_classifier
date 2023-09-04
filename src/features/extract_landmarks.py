import cv2
import mediapipe as mp
import pandas as pd

# Helper function to extract landmarks from a given RGB frame
def extract_landmarks_from_frame(frame_rgb, pose):
    results = pose.process(frame_rgb)
    
    if results.pose_landmarks:
        # Annotate the frame with landmarks
        annotated_frame = frame_rgb.copy()
        mp.solutions.drawing_utils.draw_landmarks(annotated_frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
        
        features = []
        for landmark in results.pose_landmarks.landmark:
            features.extend([landmark.x, landmark.y, landmark.z, landmark.visibility, landmark.presence])
        return features, annotated_frame
    return [], frame_rgb


# Helper function to calculate additional features
def calculate_additional_features(df):
    df['HEAD_x'] = (df['LEFT_EAR_x'] + df['RIGHT_EAR_x']) / 2
    df['HEAD_y'] = (df['LEFT_EAR_y'] + df['RIGHT_EAR_y']) / 2
    df['HEAD_z'] = (df['LEFT_EAR_z'] + df['RIGHT_EAR_z']) / 2
    df['NECK_x'] = (df['LEFT_SHOULDER_x'] + df['RIGHT_SHOULDER_x']) / 2
    df['NECK_y'] = (df['LEFT_SHOULDER_y'] + df['RIGHT_SHOULDER_y']) / 2
    df['NECK_z'] = (df['LEFT_SHOULDER_z'] + df['RIGHT_SHOULDER_z']) / 2

def extract_landmarks(input_source, output_destination, is_video=True, write_output=True):

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
                # Save the annotated frame, not the original one
                cv2.imwrite(output_destination, cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
            df_landmarks['frame_number'] = 0  # Adding frame_number for photos

    calculate_additional_features(df_landmarks)
    
    df_landmarks['frame_number'] = df_landmarks['frame_number'].astype(int)

    return df_landmarks

