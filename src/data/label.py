import logging
import csv
import os
import shutil
import tempfile
import cv2
import pandas as pd

from typing import List, Tuple, Dict

logger = logging.getLogger(__name__)


def label_frames(params: Dict[str, Dict[str, str]], video_path: str, skip_seconds: float) -> List[Tuple[int, str]]:
    """
    Annotates frames in a video using key press mappings.

    Args:
        params (Dict[str, Dict[str, str]]): Contains the mapping of keyboard keys to labels.
        video_path (str): Path to the video to be labeled.
        skip_seconds (float): Duration in seconds to skip between labeling frames.

    Returns:
        List[Tuple[int, str]]: A list of tuples where each tuple contains the frame number and its corresponding label.
    """
    
    label_mapping = params['labels']
    cap = cv2.VideoCapture(video_path)
    frame_number = 0
    labels = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    skip_frames = int(fps * skip_seconds)

    def get_label_for_frame(frame) -> str:
        while True:
            cv2.imshow('Video Frame', frame)
            key = cv2.waitKey(0)
            char_key = chr(key & 0xFF)
            if char_key == 'q':
                logger.warning(f"Warning: You are quitting the labeling for {video_path}. "
                      "The current progress will not be saved.")
                exit(0)
            elif char_key in label_mapping:
                return label_mapping[char_key]
            else:
                logger.info("Invalid key pressed. "
                      "Please press one of the following keys for labeling or 'q' to quit:")
                for k, v in label_mapping.items():
                    logger.info(f"  Press '{k}' for {v}")

    while frame_number < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            break

        label = get_label_for_frame(frame)
        
        # Label the current frame and all the previous frames that were skipped
        for i in range(max(0, frame_number - skip_frames + 1), frame_number + 1):
            labels.append((i, label))

        frame_number += skip_frames

    # Process the last frames if there are any left
    if frame_number >= total_frames:
        # Display the last frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
        ret, frame = cap.read()
        if ret:
            label = get_label_for_frame(frame)
            
            # Apply the label to all the remaining frames
            for i in range(frame_number - skip_frames, total_frames):
                labels.append((i, label))

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
