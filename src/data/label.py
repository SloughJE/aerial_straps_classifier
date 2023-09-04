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


def run_labeling(params: dict) -> None:
    """
    Executes the labeling process for all video files in the specified input directory.
    Skips videos that have already been labeled.
    Labels and frame numbers are saved to CSV files in the specified output directory.

    Parameters:
    params (dict): Dictionary containing the following key-value pairs:
        - 'input_video_dir': Directory containing video files to be labeled.
        - 'output_dir': Directory where labeled CSV files will be saved.
        - 'skip_seconds': Number of frames to skip between labeled frames. 
        Same label is applied to skipped frames.
    """

    input_video_dir = params['input_video_dir']
    output_dir = params['output_dir']
    skip_seconds = params['skip_seconds']

    video_files = [f for f in os.listdir(input_video_dir) if f.endswith(".mov") or f.endswith(".mp4")]
    total_videos = len(video_files)

    print(f"Total number of videos to label: {total_videos}")

    for idx, filename in enumerate(video_files):
        video_path = os.path.join(input_video_dir, filename)
        video_name = os.path.splitext(filename)[0]

        output_file = os.path.join(output_dir, video_name) + "_labeled.csv"
        
        # Check if this video has already been labeled
        if os.path.exists(output_file):
            print(f"Skipping {filename} ({idx + 1} of {total_videos}) - already labeled.")
            continue

        print(f"Labeling {filename} ({idx + 1} of {total_videos})")

        labels = label_frames(params, video_path, skip_seconds)
        temp_file = tempfile.NamedTemporaryFile(mode='w+', newline='', delete=False)

        print(f"Saving temporary output to: {temp_file.name}")
        with temp_file as csvfile:
            fieldnames = ['video_frame', 'filename', 'label']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for video_frame, label in labels:
                writer.writerow({'video_frame': video_frame, 'filename': filename, 'label': label})

        # Move temporary file to final destination once the labeling is completed
        shutil.move(temp_file.name, output_file)

        print(f"Labeled {filename} successfully and saved to {output_file}.")

    print(f"All {total_videos} videos labeled.")


def apply_mirror_labels(params: dict) -> None:
    """
    Applies labels from original labeled videos to the corresponding mirrored videos.
    The mirrored labels are saved to CSV files in the specified output directory.

    Parameters:
    params (dict): Dictionary containing the following key-value pairs:
        - 'labeled_dir': Directory containing labeled CSV files.
        - 'input_video_dir': Directory containing input video files.
    """

    labeled_dir = params['output_dir']
    input_video_dir = params['input_video_dir']

    mirrored_video_filenames = [f for f in os.listdir(input_video_dir) if f.startswith("mirrored_")]

    if len(mirrored_video_filenames)>0:
        for mirrored_video_filename in mirrored_video_filenames:
            print(f"adding label files for: {mirrored_video_filename}")

            # get corresponding non-mirrored label file
            corresponding_label_file = mirrored_video_filename.replace('mirrored_', '').replace('.mov', '_labeled.csv')
            corresponding_labeled_filepath = os.path.join(labeled_dir, corresponding_label_file)
            mirrored_label_filepath = os.path.join(labeled_dir, f'mirrored_{corresponding_label_file}')

            # read in og label file, add mirrored_ to filename and df column
            df = pd.read_csv(corresponding_labeled_filepath)
            df['filename'] = "mirrored_"+df.filename
            df.to_csv(mirrored_label_filepath,index=False)
            print(f"Mirrored labels applied and saved to {mirrored_label_filepath}.")

    else:
        print("no mirrored files to process")