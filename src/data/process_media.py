import os
import shutil

from .video_processing import reduce_video_size, mirror_video
from .photo_processing import mirror_photo


def process_media(params: dict, media_type: str) -> None:
    """
    Processes media (videos or photos) by reducing their size and optionally mirroring them.

    Parameters:
        params (dict): Dictionary containing the media processing configuration.
        media_type (str): Type of media, either 'video' or 'photo'.

    Returns:
        None: Processes media and prints progress without returning any value.
    """
    media_processing_config = params[f'{media_type}_processing']

    input_media_dir = media_processing_config[f'input_{media_type}_dir']
    output_media_dir = media_processing_config[f'output_{media_type}_dir']
    mirror_media = media_processing_config[f'mirror_{media_type}s']
    if media_type == 'video':
        reduction_factor = media_processing_config['reduction_factor']
    else:
        reduction_factor = None
    print(f"loading media from {input_media_dir}")
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_media_dir):
        print("creating output directory...")
        os.makedirs(output_media_dir)

    # Filter files based on the media type
    valid_extensions = (".mov", ".mp4") if media_type == 'video' else (".jpg", ".jpeg", ".png")
    media_files = [f for f in os.listdir(input_media_dir) if f.lower().endswith(valid_extensions)]
    
    for idx, filename in enumerate(media_files):
        input_media_path = os.path.join(input_media_dir, filename)
        output_media_path = os.path.join(output_media_dir, filename)

        if os.path.exists(output_media_path):
            print(f"Skipping {filename} as it already exists in the output directory.")
            continue
        print(f"Processing {media_type}: {filename}")

        if media_type == 'video':
            reduce_video_size(input_media_path, output_media_path, reduction_factor)
        elif media_type == 'photo':
            # Process photos (e.g., resizing) here if needed
            print(f"copying original photos to: {output_media_path}")
            shutil.copyfile(input_media_path, output_media_path)

        if mirror_media:
            print(f"Mirroring {media_type}: {filename}")
            mirror_function = mirror_video if media_type == 'video' else mirror_photo
            mirror_function(output_media_path, output_media_dir)
            print(f"Mirrored {media_type}: {filename}")

        print(f"Processed {filename} ({idx + 1} of {len(media_files)})")

    print(f"All {len(media_files)} {media_type} processed successfully.")
