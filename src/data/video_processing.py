import os
import cv2

def reduce_video_size(input_video_path, output_video_path, reduction_factor):
    """
    Reduces the video size by a given reduction factor.

    Parameters:
        input_video_path (str): Path to the input video.
        output_video_path (str): Path to save the output resized video.
        reduction_factor (int): Integer factor by which video dimensions will be reduced.

    Returns:
        None: Processes video and saves the resized video.
    """
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

def mirror_video(input_video_path, output_video_dir):
    """
    Mirrors the input video horizontally and saves the output video in the same directory.

    Parameters:
        input_video_path (str): Path to the input video.
        output_video_dir (str): Directory where the output mirrored video will be saved.

    Returns:
        None: Processes video and saves the mirrored video.
    """
    input_video = cv2.VideoCapture(input_video_path)
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    filename = os.path.basename(input_video_path)
    mirrored_filename = "mirrored_" + filename
    output_video_path = os.path.join(output_video_dir, mirrored_filename)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while True:
        ret, frame = input_video.read()
        if not ret:
            break

        mirrored_frame = cv2.flip(frame, 1)  # Flip horizontally
        output_video.write(mirrored_frame)

    input_video.release()
    output_video.release()


def process_videos(params: dict) -> None:
    """
    Processes videos by reducing their size and optionally mirroring them.

    Parameters:
        params (dict): Dictionary containing the following key-value pairs:
            - 'input_video_dir': Path to the directory containing input videos.
            - 'output_video_dir': Path to the directory where processed videos will be saved.
            - 'reduction_factor': Integer factor by which video dimensions will be reduced.
            - 'mirror_videos': Boolean indicating whether to mirror videos or not.

    Returns:
        None: Processes videos and prints progress without returning any value.
    """
    print("Processing videos")

    input_video_dir = params['input_video_dir']
    output_video_dir = params['output_video_dir']
    reduction_factor = params['reduction_factor']
    mirror_videos = params['mirror_videos']

    video_files = [f for f in os.listdir(input_video_dir) if f.endswith(".mov") or f.endswith(".mp4")]

    for idx, filename in enumerate(video_files):
        input_video_path = os.path.join(input_video_dir, filename)
        output_video_path = os.path.join(output_video_dir, filename)

        if os.path.exists(output_video_path):
            print(f"Skipping {filename} as it already exists in the output directory.")
            continue
        print(f"Processing video: {filename}")

        reduce_video_size(input_video_path, output_video_path, reduction_factor)

        if mirror_videos:
            print("mirroring video: {filename}")
            mirror_video(input_video_path, output_video_dir)
            print(f"Mirrored video: {filename}")

        print(f"Processed {filename} ({idx + 1} of {len(video_files)})")

    print(f"{len(video_files)} video(s) processed successfully.")
