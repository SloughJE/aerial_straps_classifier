import logging
import csv
import os
import shutil
import tempfile
import cv2
import pandas as pd

from typing import List, Tuple, Dict

logger = logging.getLogger(__name__)

import cv2
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

def label_frames(params: Dict[str, Dict[str, str]], video_path: str, skip_seconds: float) -> List[Tuple[int, str]]:
    """
    Manually label frames from a specified video.

    This function allows a user to manually label frames from a video by presenting frames
    to the user at intervals defined by `skip_seconds`. The label assigned by the user 
    is applied to all frames from the last labeled frame up to and including the current frame.
    At the end of the video, the user labels the last frame, which applies the label to 
    all remaining frames.

    Parameters:
    - params (Dict[str, Dict[str, str]]): A dictionary containing label mappings.
    - video_path (str): Path to the video file that needs to be labeled.
    - skip_seconds (float): Number of seconds to skip between frames that are presented for labeling.

    Returns:
    - List[Tuple[int, str]]: A list of tuples, where each tuple contains a frame number and its corresponding label.

    How to Use:
    1. The user is presented with frames at intervals defined by `skip_seconds`.
    2. The user labels the frame using the designated keys as defined in `params`.
    3. The label is applied to all frames from the last labeled frame to the current frame.
    4. The user can press the left arrow key to go back to the previous labeled frame or the last frame if on the final frame.
    5. The user can press 'q' at any time to exit without saving progress.

    Example:
    Given a video of 22 frames with `skip_seconds` set to 5 (assuming each frame represents a second),
    the user labels frames: 0, 5, 10, 15, 20, and 22. The labels applied to frames 0, 5, and 10
    will apply to frames 1-4, 6-9, and 11-14, respectively. The label applied to frame 22 will apply to frame 21.

    Note:
    Frame count starts from 0. However, if there's any confusion, refer to the documentation.
    """
        
    label_mapping = params['labels']
    cap = cv2.VideoCapture(video_path)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    skip_frames = int(fps * skip_seconds)

    frame_number = 0
    last_labeled_frame = -1
    labeled_frames = {}

    print(f"Total frames to label: {total_frames}")

    def get_label_for_frame(frame_no, frame) -> Tuple[str, bool]:
        print(f"Labeling frame number: {frame_no}")
        while True:
            cv2.imshow('Video Labeling', frame)
            key = cv2.waitKey(0)
            char_key = chr(key & 0xFF)

            if key == 81:  # Left arrow key
                return None, True
            elif char_key == 'q':
                logger.warning(f"Quitting labeling for {video_path}. Current progress will not be saved.")
                exit(0)
            elif char_key in label_mapping:
                return label_mapping[char_key], False
            else:
                logger.info("Invalid key pressed. Use the designated labeling keys or 'q' to quit.")

    while frame_number <= total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            break

        label, go_back = get_label_for_frame(frame_number, frame)

        # If user pressed "left" to go back
        if go_back:
            # If at the last frame, jump back to the last labeled frame
            if frame_number == total_frames - 1:
                frame_number = last_labeled_frame
            else:
                frame_number = max(0, frame_number - skip_frames)
            continue

        # Label frames from the last labeled frame to the current frame
        for i in range(last_labeled_frame + 1, frame_number + 1):
            labeled_frames[i] = label

        last_labeled_frame = frame_number

        # If at the last frame, break out of the loop
        if frame_number == total_frames - 1:
            break

        # If close to the last frame, set the frame number to the last frame
        if frame_number + skip_frames >= total_frames:
            frame_number = total_frames - 1
        else:
            frame_number += skip_frames

    labels = sorted([(frame, lbl) for frame, lbl in labeled_frames.items()])
    cap.release()
    cv2.destroyAllWindows()
    
    return labels




def label_photos(params: dict, photo_path: str) -> list:
    label_mapping = params['labels']
    labels = []
    
    image = cv2.imread(photo_path)
    cv2.imshow('Photo', image)

    while True:
        key = cv2.waitKey(0)
        char_key = chr(key & 0xFF)
        if char_key == 'q':
            logger.warning(f"Warning: You are quitting the labeling for {photo_path}. The current progress will not be saved.")
            exit(0)
        elif char_key in label_mapping:
            label = label_mapping[char_key]
            break
        else:
            logger.info("Invalid key pressed. Please press one of the following keys for labeling or 'q' to quit:")
            for k, v in label_mapping.items():
                logger.info(f"  Press '{k}' for {v}")

    labels.append((photo_path, label))
    cv2.destroyAllWindows()
    return labels


def run_labeling(params: dict, mode: str) -> None:
    """
    Executes the labeling process for media files (either videos or photos) in the specified input directory.
    Skips files that have already been labeled or start with 'mirrored_'.
    Labels and filenames (or frame numbers for videos) are saved to CSV files in the specified output directory.
    
    Parameters:
    params (dict): Dictionary containing parameters for labeling, including:
        - 'input_video_dir' (str): The directory containing video files to be labeled.
        - 'input_photo_dir' (str): The directory containing photo files to be labeled.
        - 'output_dir' (str): The directory where labeled CSV files will be saved.
        - 'skip_seconds' (int, optional): The number of seconds to skip between frames when labeling videos. Not used for photos.
        - 'force_relabel' (list of str, optional): A list of filenames to force relabel, even if they have already been labeled.
        - 'force_relabel_all' (bool, optional): A flag indicating whether to force relabel all files, ignoring any existing labels.
    mode (str): Either 'video' or 'photo', depending on the media being labeled.

    Raises:
    ValueError: If an invalid mode is provided.

    Returns:
    None

    Usage:
    - Set the appropriate parameters in your 'params' dictionary.
    - Call this function with the 'params' dictionary and the mode ('video' or 'photo') to start the labeling process.
    """
    
    # Determine the input and output directories and the file extensions based on the mode
    input_dir = params.get('input_video_dir') if mode == "video" else params.get('input_photo_dir')
    output_dir = params.get('output_dir')
    skip_seconds = params.get('skip_seconds', None) 

    force_relabel = params.get('force_relabel', [])
    force_relabel_all = params.get('force_relabel_all', False)

    extensions = {".mov", ".mp4"} if mode == "video" else {".jpg", ".jpeg", ".png"}
    
    if not extensions:
        raise ValueError("Invalid mode provided. Expected 'video' or 'photo'.")

    files = [f for f in os.listdir(input_dir) if not f.lower().startswith("mirrored_") and any(f.lower().endswith(ext) for ext in extensions)]

    if force_relabel_all:
            logger.info("Force relabeling is enabled for all files.")
    elif force_relabel:
        logger.info(f"Force relabeling is enabled for the following files: {', '.join(force_relabel)}")
        
        for file_to_relabel in force_relabel:
            if file_to_relabel not in files:
                logger.warning(f"The file '{file_to_relabel}' listed for force relabeling was not found in the directory.")

    total_files = len(files)
    logger.info(f"Total number of {mode}s to label: {total_files}")

    try:
        for idx, filename in enumerate(files):
            file_path = os.path.join(input_dir, filename)
            base_name = os.path.splitext(filename)[0]

            output_file = os.path.join(output_dir, f"{mode}_") + base_name + "_labeled.csv"
            
            # Check if this file has already been labeled
            if not force_relabel_all and os.path.exists(output_file) and (force_relabel is None or filename not in force_relabel):
                logger.info(f"Skipping {filename} ({idx + 1} of {total_files}) - already labeled.")
                continue

            logger.info(f"Labeling {filename} ({idx + 1} of {total_files})")

            # Label the media based on the mode
            if mode == "video":
                labels = label_frames(params, file_path, skip_seconds)
            else:
                labels = label_photos(params, file_path)
            
            # Save the labels to a temporary CSV file
            temp_file = tempfile.NamedTemporaryFile(mode='w+', newline='', delete=False)
            logger.info(f"Saving temporary output to: {temp_file.name}")
            with temp_file as csvfile:
                fieldnames = ['frame_number', 'filename', 'label']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for item, label in labels:
                    # If mode is 'photo', set frame_number to 0, else keep the original frame number
                    frame_number = 0 if mode == 'photo' else item
                    prefixed_filename = f"{mode}_"+filename  # Add the prefix here
                    writer.writerow({'frame_number': frame_number, 'filename': prefixed_filename, 'label': label})


            # Move the temporary CSV file to the final output directory
            shutil.move(temp_file.name, output_file)
            logger.info(f"Labeled {filename} successfully and saved to {output_file}.")

    except KeyboardInterrupt:
        logger.info("\nLabeling interrupted by user. Exiting.")

        exit(1)

    logger.info(f"All {total_files} {mode}s labeled.")


def apply_mirror_labels(params: dict) -> None:
    """
    Applies labels from original labeled media to the corresponding mirrored media.
    The mirrored labels are saved to CSV files in the specified output directory.

    Parameters:
    params (dict): Dictionary containing the following key-value pairs:
        - 'output_dir': Directory where labeled CSV files will be saved.
        - 'input_video_dir': Directory containing input video files.
        - 'input_photo_dir': Directory containing input photo files.
    """

    labeled_dir = params['output_dir']
    input_video_dir = params['input_video_dir']
    input_photo_dir = params['input_photo_dir']

    # Loop through both video and photo directories
    for input_dir, file_ext, prefix in [(input_video_dir, ['.MOV', '.MP4', '.mov', '.mp4'], 'video_'), 
                                       (input_photo_dir, ['.JPG', '.JPEG', '.PNG', '.jpg', '.jpeg', '.png'], 'photo_')]:
        
        mirrored_filenames = [f for f in os.listdir(input_dir) 
                              if f.lower().startswith("mirrored_") and f.lower().endswith(tuple([ext.lower() for ext in file_ext]))]
        
        if mirrored_filenames:
            for mirrored_filename in mirrored_filenames:
                logger.info(f"Adding label files for: {mirrored_filename}")

                # Determine the correct file extension for label based on mode
                label_ext = '_labeled.csv'

                # Get corresponding non-mirrored label file
                base_name = mirrored_filename.replace('mirrored_', '').rsplit('.', 1)[0]
                corresponding_label_file = prefix + base_name + label_ext
                corresponding_labeled_filepath = os.path.join(labeled_dir, corresponding_label_file)
                
                mirrored_label_filepath = os.path.join(labeled_dir, f'{prefix}mirrored_{base_name}{label_ext}')

                # Read in original label file, add mirrored_ to filename and df column
                df = pd.read_csv(corresponding_labeled_filepath)
                # Extract the file extension from mirrored_filename
                file_ext = os.path.splitext(mirrored_filename)[1]
                df['filename'] = f'{prefix}mirrored_{base_name}{file_ext}'
                df.to_csv(mirrored_label_filepath, index=False)
                logger.info(f"Mirrored labels applied and saved to {mirrored_label_filepath}.")
        else:
            logger.info(f"No mirrored files to process in {input_dir}")
