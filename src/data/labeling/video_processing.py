import os
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