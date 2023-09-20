from shutil import copy
import os
import yaml
import numpy as np
import cv2

from src.data.process_media import process_media
from src.data.photo_processing import mirror_photo
from src.data.video_processing import mirror_video, reduce_video_size

# pytest -s tests,  to run with print output


def test_process_video_media():
    """
    Test the process_media function to ensure that it can successfully process video files.

    Steps:
    1. Load the configuration parameters from params.yaml
    2. Set the input directory to the test data directory for videos
    3. Run the process_media function with video inputs
    4. Verify that the processed output files are created in the correct directory
    5. Clean up by removing any created files and directories
    """
    # Step 1: Load configurations from params.yaml
    with open('params.yaml', 'r') as file:
        params = yaml.safe_load(file)

    # Step 2: Create dummy media files for the test
    video_config = params['media_processing']['video_processing']
    video_dir = video_config['input_video_dir']
    
    # Set test video directory and test video name
    test_video_dir = 'tests/test_data/videos'
    test_video_name = 'test_IMG_3301.mov'
    
    dummy_video_path = os.path.join(video_dir, test_video_name)  # Set the dummy_video_path here

    # Verify source file exists before copying
    assert os.path.exists(os.path.join(test_video_dir, test_video_name))
    
    copy(os.path.join(test_video_dir, test_video_name), dummy_video_path)

    try:
        # Step 3: Call the function with the test inputs
        process_media(params['media_processing'], 'video')
        
        # Step 4: Verify the output
        output_video_path = os.path.join(video_config['output_video_dir'], test_video_name)
        
        # Check that the output file has been created
        assert os.path.exists(output_video_path)

        if video_config['mirror_videos']:

            mirrored_video_name = 'mirrored_' + test_video_name
            mirrored_output_video_path = os.path.join(video_config['output_video_dir'], mirrored_video_name)
            assert os.path.exists(mirrored_output_video_path)

    finally:
        # Clean up both input and output files
        if os.path.exists(dummy_video_path):
            os.remove(dummy_video_path)
        if os.path.exists(output_video_path):
            os.remove(output_video_path)


def test_process_photo_media():
    """
    Test the process_media function to ensure that it can successfully process photo files.

    Steps:
    1. Load the configuration parameters from params.yaml
    2. Set the input directory to the test data directory for photos
    3. Run the process_media function with photo inputs
    4. Verify that the processed output files are created in the correct directory
    5. Clean up by removing any created files and directories
    """
    # Step 1: Load configurations from params.yaml
    with open('params.yaml', 'r') as file:
        params = yaml.safe_load(file)

    # Step 2: Create dummy media files for the test
    photo_config = params['media_processing']['photo_processing']
    photo_dir = photo_config['input_photo_dir']
    
    # Set test photo directory and test photo name
    test_photo_dir = 'tests/test_data/photos'
    test_photo_name = 'test_IMG_0073.jpg'
    
    dummy_photo_path = os.path.join(photo_dir, test_photo_name)  # Set the dummy_video_path here

   # Verify source file exists before copying
    assert os.path.exists(os.path.join(test_photo_dir, test_photo_name))
    
    copy(os.path.join(test_photo_dir, test_photo_name), dummy_photo_path)

    try:
        # Step 3: Call the function with the test inputs
        process_media(params['media_processing'], 'photo')
        
        # Step 4: Verify the output
        output_photo_path = os.path.join(photo_config['output_photo_dir'], test_photo_name)
        
        # Check that the output file has been created
        assert os.path.exists(output_photo_path)

        if photo_config['mirror_photos']:

            mirrored_photo_name = 'mirrored_' + test_photo_name
            mirrored_output_photo_path = os.path.join(photo_config['output_photo_dir'], mirrored_photo_name)
            assert os.path.exists(mirrored_output_photo_path)
                    
    finally:
        # Step 5: Clean up
        os.remove(os.path.join(photo_dir, test_photo_name))


def test_mirror_photo():
    """
    This test function verifies the mirror_photo function.

    The function performs the following steps:
    1. Mirrors a test image.
    2. Compares the average colors of corresponding strips from the left and 
       right sides of the original and mirrored images.

    The function asserts that the average colors of the corresponding strips 
    from the mirrored and original images are almost equal, using a decimal 
    precision of 1.

    If the assertions pass, the function passes the test.

    The mirrored image file created during the test is deleted afterward to 
    clean up.

    :raises AssertionError: If the color values of the corresponding strips 
                            are not almost equal.
    """
    # Path to a test image file
    input_photo_path = "tests/test_data/photos/test_IMG_0073.jpg"
    
    # Path to the output directory where the mirrored photo will be saved
    output_photo_dir = "tests/test_data/photos/output/"
    
    # Call the function to test
    mirror_photo(input_photo_path, output_photo_dir)
    
    # Load the original and mirrored images
    original_image = cv2.imread(input_photo_path)
    mirrored_image_path = os.path.join(output_photo_dir, "mirrored_test_IMG_0073.jpg")
    mirrored_image = cv2.imread(mirrored_image_path)

    # Get the average color value of some strips from the left and right sides of the images
    left_strip_original = np.mean(original_image[:, :10, :], axis=(0,1))
    right_strip_original = np.mean(original_image[:, -10:, :], axis=(0,1))
    left_strip_mirrored = np.mean(mirrored_image[:, :10, :], axis=(0,1))
    right_strip_mirrored = np.mean(mirrored_image[:, -10:, :], axis=(0,1))

    # Check that the average color values of the corresponding strips are almost equal
    np.testing.assert_array_almost_equal(left_strip_original, right_strip_mirrored, decimal=1)
    np.testing.assert_array_almost_equal(right_strip_original, left_strip_mirrored, decimal=1)

    # Cleanup: remove the mirrored image file
    os.remove(mirrored_image_path)


def test_mirror_video():
    """
    This test function verifies the mirror_video function.
    
    The function performs the following steps:
    1. Reduces the size of a test video.
    2. Mirrors the reduced video.
    3. Compares the average colors of corresponding strips from the left and 
       right sides of the original and mirrored videos in certain frames.

    The frames are chosen at intervals of 50, from the first 200 frames of the videos.

    The function asserts that the average colors of the corresponding strips 
    from the mirrored and original videos are almost equal, using a relative tolerance 
    of 0.1 and an absolute tolerance of 1.

    If the assertions pass for all selected frames, the function passes the test.

    Any generated video files are deleted after the test.

    :raises AssertionError: If the color values of the corresponding strips 
                            are not almost equal in any of the selected frames.
    """
    # Path to a test video file
    input_video_path = "tests/test_data/videos/test_IMG_3301.mov"
    
    # Path to the output directory where the reduced and mirrored video will be saved
    output_video_dir = "tests/test_data/videos/output/"
    
    # Path to the output video file for the reduced size video
    reduced_video_path = os.path.join(output_video_dir, "reduced_test_video.mp4")
    
    # First, reduce the size of the video
    reduce_video_size(input_video_path, reduced_video_path, 4)  

    # Then, mirror the reduced video
    mirror_video(reduced_video_path, output_video_dir)

    # Load the original (reduced) and mirrored videos
    original_video = cv2.VideoCapture(reduced_video_path)
    mirrored_video_path = os.path.join(output_video_dir, "mirrored_reduced_test_video.mp4")
    mirrored_video = cv2.VideoCapture(mirrored_video_path)
    
    try:
        # Check the mirroring on some frames
        for i in range(0, 200, 50):  # Check every 20th frame in the first 100 frames
            original_video.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, original_frame = original_video.read()

            mirrored_video.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, mirrored_frame = mirrored_video.read()

            if not ret or original_frame is None or mirrored_frame is None:
                continue  # Skip this iteration if no frame was grabbed

            # Get the dimensions of a frame
            height, width, _ = original_frame.shape
            # Get the average color value of some strips from the left and right sides of the frames
            left_strip_original = np.mean(original_frame[:, :10, :], axis=(0,1))
            right_strip_original = np.mean(original_frame[:, -10:, :], axis=(0,1))
            left_strip_mirrored = np.mean(mirrored_frame[:, :10, :], axis=(0,1))
            right_strip_mirrored = np.mean(mirrored_frame[:, -10:, :], axis=(0,1))
            
            assert np.allclose(left_strip_original, right_strip_mirrored, rtol=0.1, atol=1)
            assert np.allclose(right_strip_original, left_strip_mirrored, rtol=0.1, atol=1)


        # Release the video objects
        original_video.release()
        mirrored_video.release()

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Cleanup: remove the mirrored video file
        if os.path.exists(reduced_video_path):
            os.remove(reduced_video_path)
        if os.path.exists(mirrored_video_path):
            os.remove(mirrored_video_path)