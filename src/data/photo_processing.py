import os
import cv2

def mirror_photo(input_photo_path, output_photo_dir):
    """
    Mirrors the input photo horizontally and saves the output photo in the specified directory.

    Parameters:
        input_photo_path (str): Path to the input photo.
        output_photo_dir (str): Directory where the output mirrored photo will be saved.

    Returns:
        None: Processes photo and saves the mirrored photo.
    """
    input_photo = cv2.imread(input_photo_path)

    filename = os.path.basename(input_photo_path)
    mirrored_filename = "mirrored_" + filename
    output_photo_path = os.path.join(output_photo_dir, mirrored_filename)

    mirrored_photo = cv2.flip(input_photo, 1)  # Flip horizontally
    cv2.imwrite(output_photo_path, mirrored_photo)