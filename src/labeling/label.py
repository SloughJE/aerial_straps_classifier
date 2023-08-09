import csv
import os
import shutil
import tempfile
import cv2


def reduce_video_size(params: dict) -> None:
    """
    Reduces the video quality for all videos in the given directory. 
    Skips videos that have already been processed.

    Parameters:
    params (dict): Dictionary containing the following key-value pairs:
        - 'input_video_dir': Path to the directory containing input videos.
        - 'output_video_dir': Path to the directory where reduced videos will be saved.
        - 'reduction_factor': Integer factor by which video dimensions will be reduced.

    Returns:
    None: Processes videos and prints progress without returning any value.
    """

    print("Reducing video quality to speed labeling")

    input_video_dir = params['input_video_dir']
    output_video_dir = params['output_video_dir']
    reduction_factor = params['reduction_factor']

    video_files = [f for f in os.listdir(input_video_dir) if f.endswith(".mov") or f.endswith(".mp4")]
    
    # Filter out videos that have already been processed
    videos_to_process = [filename for filename in video_files if not os.path.exists(os.path.join(output_video_dir, filename))]
    total_videos = len(videos_to_process)
    already_processed = len(video_files) - total_videos
    print(f"Total number of videos to process: {total_videos}.\nAlready processed {already_processed}.")

    for idx, filename in enumerate(video_files):
        input_video_path = os.path.join(input_video_dir, filename)
        output_video_path = os.path.join(output_video_dir, filename)
        
        # Check if the file already exists in the output directory
        if os.path.exists(output_video_path):
            print(f"Skipping {filename} as it already exists in the output directory.")
            continue
        print(f"Processing video: {filename}")

        input_video = cv2.VideoCapture(input_video_path)

        original_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        new_width = original_width // reduction_factor
        new_height = original_height // reduction_factor

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = cv2.VideoWriter(output_video_path, fourcc, input_video.get(cv2.CAP_PROP_FPS), (new_width, new_height))

        while True:
            ret, frame = input_video.read()
            if not ret:
                break

            resized_frame = cv2.resize(frame, (new_width, new_height))
            output_video.write(resized_frame)

        input_video.release()
        output_video.release()

        print(f"Processed {filename} ({idx + 1} of {total_videos})")
    
    print(f"{total_videos} video(s) processed successfully.")


def label_frames(params: dict, video_path: str, skip_frames: int) -> list:
    """
    Allows the user to label frames from the specified video file. 
    The user can label a frame, and the same label will be applied to 
    the specified number of subsequent frames (skip_frames).

    Parameters:
    params (dict): Dictionary containing the following key-value pairs:
        - 'labels': Dictionary mapping keypresses to corresponding labels.
    video_path (str): Path to the video file to be labeled.
    skip_frames (int): Number of frames to skip between labeled frames. 
    Same label is applied to skipped frames.

    Returns:
    list: A list of tuples, where each tuple contains the frame number and the corresponding label.
    """

    label_mapping = params['labels']

    cap = cv2.VideoCapture(video_path)
    frame_number = 0
    labels = []
    last_label = 'unknown'  # Keep track of the last label
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while frame_number < total_frames:
        # Seek to the desired frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            break

        # Display the frame
        cv2.imshow('Video Frame', frame)

        while True:  # Keep asking for input until a valid key is pressed
            key = cv2.waitKey(0)
            char_key = chr(key & 0xFF)  # Get the character representation of the key

            if char_key == 'q':  # Press 'q' to quit
                print(f"Warning: You are quitting the labeling for {video_path}. \
                      The current progress will not be saved.")
                exit(0)  # Exit the entire program
            elif char_key in label_mapping:  # Check if the key corresponds to a label
                label = label_mapping[char_key]
                break
            else:
                print("Invalid key pressed. \
                      Please press one of the following keys for labeling or 'q' to quit:")
                for k, v in label_mapping.items():
                    print(f"  Press '{k}' for {v}")
                if char_key == 'q':  # Break outer loop if 'q' was pressed
                    break


        # Store the label for this frame and all skipped frames
        for i in range(frame_number, min(frame_number + skip_frames + 1, total_frames)):  # +1 to include the current frame
            labels.append((i, label))

        last_label = label
        frame_number += skip_frames + 1  # Increment by the number of skipped frames + 1 for the current frame

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
        - 'output_video_dir': Directory where labeled CSV files will be saved.
        - 'skip_frames': Number of frames to skip between labeled frames. 
        Same label is applied to skipped frames.
    """

    input_video_dir = params['input_video_dir']
    output_video_dir = params['output_video_dir']
    skip_frames = params['skip_frames']

    video_files = [f for f in os.listdir(input_video_dir) if f.endswith(".mov") or f.endswith(".mp4")]
    total_videos = len(video_files)

    print(f"Total number of videos to label: {total_videos}")

    for idx, filename in enumerate(video_files):
        video_path = os.path.join(input_video_dir, filename)
        video_name = os.path.splitext(filename)[0]

        output_file = os.path.join(output_video_dir, video_name) + ".csv"
        
        # Check if this video has already been labeled
        if os.path.exists(output_file):
            print(f"Skipping {filename} ({idx + 1} of {total_videos}) - already labeled.")
            continue

        print(f"Labeling {filename} ({idx + 1} of {total_videos})")

        labels = label_frames(params, video_path, skip_frames)
        temp_file = tempfile.NamedTemporaryFile(mode='w+', newline='', delete=False)

        print(f"Saving temporary output to: {temp_file.name}")
        with temp_file as csvfile:
            fieldnames = ['frame_number', 'filename', 'label']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for frame_number, label in labels:
                writer.writerow({'frame_number': frame_number, 'filename': video_path, 'label': label})

        # Move temporary file to final destination once the labeling is completed
        shutil.move(temp_file.name, output_file)

        print(f"Labeled {filename} successfully and saved to {output_file}.")

    print(f"All {total_videos} videos labeled.")