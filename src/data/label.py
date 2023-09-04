import csv
import os
import shutil
import tempfile
import cv2
import pandas as pd


def label_frames(params: dict, video_path: str, skip_seconds: float) -> list:
    label_mapping = params['labels']

    cap = cv2.VideoCapture(video_path)
    frame_number = 0
    labels = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps_rounded = round(fps)
    skip_frames = int(fps_rounded * skip_seconds)

    print(total_frames)
    while frame_number < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('Video Frame', frame)

        while True:
            key = cv2.waitKey(0)
            char_key = chr(key & 0xFF)
            if char_key == 'q':
                print(f"Warning: You are quitting the labeling for {video_path}. \
                      The current progress will not be saved.")
                exit(0)
            elif char_key in label_mapping:
                label = label_mapping[char_key]
                break
            else:
                print("Invalid key pressed. \
                      Please press one of the following keys for labeling or 'q' to quit:")
                for k, v in label_mapping.items():
                    print(f"  Press '{k}' for {v}")

        # Label the current frame and all the previous frames that were skipped
        for i in range(max(0, frame_number - skip_frames + 1), frame_number + 1):
            labels.append((i, label))

        frame_number += skip_frames

    # Process the last frames if there are any left
    if frame_number >= total_frames:
        print("labeling final frame")
        # Display the last frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
        ret, frame = cap.read()
        if ret:
            cv2.imshow('Video Frame', frame)

            while True:
                key = cv2.waitKey(0)
                char_key = chr(key & 0xFF)
                if char_key == 'q':
                    print(f"Warning: You are quitting the labeling for {video_path}. \
                          The current progress will not be saved.")
                    exit(0)
                elif char_key in label_mapping:
                    label = label_mapping[char_key]
                    break
                else:
                    print("Invalid key pressed. \
                          Please press one of the following keys for labeling or 'q' to quit:")
                    for k, v in label_mapping.items():
                        print(f"  Press '{k}' for {v}")

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
            print(f"Warning: You are quitting the labeling for {photo_path}. The current progress will not be saved.")
            exit(0)
        elif char_key in label_mapping:
            label = label_mapping[char_key]
            break
        else:
            print("Invalid key pressed. Please press one of the following keys for labeling or 'q' to quit:")
            for k, v in label_mapping.items():
                print(f"  Press '{k}' for {v}")

    labels.append((photo_path, label))
    cv2.destroyAllWindows()
    return labels


def run_labeling(params: dict, mode: str) -> None:
    """Executes the labeling process for media files (either videos or photos) in the specified input directory.
    Skips files that have already been labeled or start with 'mirrored_'.
    Labels and filenames (or frame numbers for videos) are saved to CSV files in the specified output directory.
    
    Parameters:
    params (dict): Dictionary containing parameters for labeling.
    mode (str): Either 'video' or 'photo', depending on the media being labeled.
    """
    
    # Determine the input directory based on the mode
    input_dir = params['input_video_dir'] if mode == "video" else params['input_photo_dir']
    output_dir = params['output_dir']
    skip_seconds = params.get('skip_seconds', None)  # may not be needed for photos
    
    # Fetch appropriate files from the directory based on their extensions and avoid 'mirrored_' prefix
    if mode == "video":
        extensions = [".mov", ".mp4"]
    elif mode == "photo":
        extensions = [".jpg", ".jpeg", ".png"]
    else:
        raise ValueError("Invalid mode provided. Expected 'video' or 'photo'.")
    
    files = [f for f in os.listdir(input_dir) if not f.lower().startswith("mirrored_") and any(f.lower().endswith(ext) for ext in extensions)]
    
    total_files = len(files)
    print(f"Total number of {mode}s to label: {total_files}")

    for idx, filename in enumerate(files):
        file_path = os.path.join(input_dir, filename)
        base_name = os.path.splitext(filename)[0]

        output_file = os.path.join(output_dir, f"{mode}_") + base_name + "_labeled.csv"
        
        # Check if this file has already been labeled
        if os.path.exists(output_file):
            print(f"Skipping {filename} ({idx + 1} of {total_files}) - already labeled.")
            continue

        print(f"Labeling {filename} ({idx + 1} of {total_files})")

        # Label the media based on the mode
        if mode == "video":
            labels = label_frames(params, file_path, skip_seconds)
        else:
            labels = label_photos(params, file_path)
        
        # Save the labels to a temporary CSV file
        temp_file = tempfile.NamedTemporaryFile(mode='w+', newline='', delete=False)
        print(f"Saving temporary output to: {temp_file.name}")
        with temp_file as csvfile:
            fieldnames = ['frame_number', 'filename', 'label']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for item, label in labels:
                # If mode is 'photo', set frame_number to 0, else keep the original frame number
                frame_number = 0 if mode == 'photo' else item
                writer.writerow({'frame_number': frame_number, 'filename': filename, 'label': label})


        # Move the temporary CSV file to the final output directory
        shutil.move(temp_file.name, output_file)
        print(f"Labeled {filename} successfully and saved to {output_file}.")

    print(f"All {total_files} {mode}s labeled.")


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
                print(f"Adding label files for: {mirrored_filename}")

                # Determine the correct file extension for label based on mode
                label_ext = '_labeled.csv'

                # Get corresponding non-mirrored label file
                base_name = mirrored_filename.replace('mirrored_', '').rsplit('.', 1)[0]
                corresponding_label_file = prefix + base_name + label_ext
                corresponding_labeled_filepath = os.path.join(labeled_dir, corresponding_label_file)
                
                mirrored_label_filepath = os.path.join(labeled_dir, f'{prefix}mirrored_{base_name}{label_ext}')

                # Read in original label file, add mirrored_ to filename and df column
                df = pd.read_csv(corresponding_labeled_filepath)
                df['filename'] = f'{prefix}mirrored_{base_name}'
                df.to_csv(mirrored_label_filepath, index=False)
                print(f"Mirrored labels applied and saved to {mirrored_label_filepath}.")
        else:
            print(f"No mirrored files to process in {input_dir}")
