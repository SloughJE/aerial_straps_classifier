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
