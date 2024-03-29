import logging
import os
import shutil

from .video_processing import reduce_video_size, mirror_video
from .photo_processing import mirror_photo

logger = logging.getLogger(__name__)


def process_media(params: dict, media_type: str) -> None:
    """
    Processes media (videos or photos) by reducing their size and optionally mirroring them.

    Parameters:
        params (dict): Dictionary containing the media processing configuration.
        media_type (str): Type of media, either 'video' or 'photo'.

    Returns:
        None: Processes media and prints progress without returning any value.
    """
    media_type_key = f"{media_type}_processing"  
    media_processing_config = params[media_type_key]  

    input_media_dir = media_processing_config[f'input_{media_type}_dir']
    output_media_dir = media_processing_config[f'output_{media_type}_dir']
    mirror_media = media_processing_config[f'mirror_{media_type}s']
    if media_type == 'video':
        reduction_factor = media_processing_config['reduction_factor']
    else:
        reduction_factor = None
    logger.info(f"loading media from {input_media_dir}")
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_media_dir):
        logger.info("creating output directory...")
        os.makedirs(output_media_dir)

    # Filter files based on the media type
    valid_extensions = (".mov", ".mp4") if media_type == 'video' else (".jpg", ".jpeg", ".png")
    media_files = [f for f in os.listdir(input_media_dir) if f.lower().endswith(valid_extensions)]
    
    for idx, filename in enumerate(media_files):
        input_media_path = os.path.join(input_media_dir, filename)
        output_media_path = os.path.join(output_media_dir, filename)

        if os.path.exists(output_media_path):
            logger.info(f"Skipping {filename} as it already exists in the output directory.")
            continue
        logger.info(f"Processing {media_type}: {filename}")

        if media_type == 'video':
            reduce_video_size(input_media_path, output_media_path, reduction_factor)
        elif media_type == 'photo':
            # Process photos (e.g., resizing) here if needed
            logger.info(f"copying original photos to: {output_media_path}")
            shutil.copyfile(input_media_path, output_media_path)

        if mirror_media:
            logger.info(f"Mirroring {media_type}: {filename}")
            mirror_function = mirror_video if media_type == 'video' else mirror_photo
            mirror_function(output_media_path, output_media_dir)
            logger.info(f"Mirrored {media_type}: {filename}")

        logger.info(f"Processed {filename} ({idx + 1} of {len(media_files)})")

    logger.info(f"All {len(media_files)} {media_type} processed successfully.")
