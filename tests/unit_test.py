import numpy as np
import pytest
import pandas as pd

from src.data.process_media import process_media
from src.data.photo_processing import mirror_photo
from src.data.video_processing import mirror_video, reduce_video_size
from src.data.label import label_photos, label_frames, apply_mirror_labels
from unittest.mock import patch, mock_open, MagicMock, ANY, PropertyMock, Mock
import cv2
import numpy as np

# pytest -s tests,  to run with print output
# pytest tests/unit_tests.py -m unit -s

@pytest.mark.unit
def test_mirror_photo():
    """
    Test the `mirror_photo` function to ensure it correctly mirrors an image.

    This test:
    - Mocks the image reading process to provide a fake image.
    - Calls the `mirror_photo` function.
    - Checks if the output (mirrored) image is as expected.
    """
    # Dummy data
    fake_image = np.array([[255, 0], [0, 255]])
    mirrored_image = np.array([[0, 255], [255, 0]])

    with patch('cv2.imread', return_value=fake_image), patch('cv2.imwrite') as mock_write, patch('os.path.basename', return_value="test.jpg"):
        mirror_photo("fake_path.jpg", "fake_output_dir")

    # Ensure the mirrored data was written correctly
    written_data = mock_write.call_args.args[1]
    
    assert np.array_equal(written_data, mirrored_image)


@pytest.mark.unit
def test_mirror_video():
    """
    Test the `mirror_video` function to ensure it correctly mirrors video frames.

    This test:
    - Mocks the video capture process to provide fake frames.
    - Calls the `mirror_video` function.
    - Checks if the output frames of the video are mirrored as expected.
    """
    # Dummy data for 2 frames
    frames = [
        np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]]),
        np.array([[0, 255, 0], [0, 0, 255], [255, 0, 0]])
    ]

    mirrored_frames = [
        np.array([[0, 0, 255], [0, 255, 0], [255, 0, 0]]),
        np.array([[0, 255, 0], [255, 0, 0], [0, 0, 255]])  
    ]


    mock_video = MagicMock()
    mock_video.get.side_effect = [3, 3, 24]  # width, height, fps for dummy data
    mock_video.read.side_effect = [(True, frames[0]), (True, frames[1]), (False, None)]

    mock_writer = MagicMock()

    with patch('cv2.VideoCapture', return_value=mock_video) as mock_cap, \
         patch('cv2.VideoWriter', return_value=mock_writer) as mock_vw:

        mirror_video("fake_path.mp4", "fake_output_dir")

        # Extract written frames from mock_writer call arguments
        written_frames = [args[0][0] for args in mock_writer.write.call_args_list]

        # Ensure the mirrored data was written correctly for each frame
        for idx, frame in enumerate(written_frames):
            assert np.array_equal(frame, mirrored_frames[idx])


@pytest.mark.unit
def test_reduce_video_size():
    """
    Test the `reduce_video_size` function to ensure it correctly reduces the size of a single video frame.

    This test:
    - Mocks the video capture process to provide a fake video with a single frame.
    - Calls the `reduce_video_size` function.
    - Verifies if the frame was read, resized, and written correctly.
    - Checks if resources (video capture and writer) were released after processing.
    """
    # Dummy data
    fake_frame = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])
    resized_frame = np.array([[255, 0], [0, 255]])

    mock_video = MagicMock()  # Directly create a MagicMock object
    mock_video.get.side_effect = [3, 3, 24]  # width, height, fps for dummy data
    mock_video.read.side_effect = [(True, fake_frame), (False, None)]
    
    mock_writer = MagicMock()  # Create a MagicMock object for the VideoWriter

    with patch('cv2.VideoCapture', return_value=mock_video) as mock_cap, \
        patch('cv2.resize', return_value=resized_frame), \
        patch('cv2.VideoWriter', return_value=mock_writer) as mock_vw:


        reduce_video_size("fake_path.mp4", "fake_output_path.mp4", 2)
        
        # Check if frames were read and written
        mock_video.read.assert_called()
        mock_writer.write.assert_called_with(resized_frame)

        # Check if resize was called correctly
        cv2.resize.assert_called_with(fake_frame, (1, 1))  # Given your mock data and reduction factor

        # Check if VideoWriter was set up correctly
        mock_vw.assert_called_with('fake_output_path.mp4', ANY, 24, (1, 1))

        # Check that resources were released
        mock_video.release.assert_called()
        mock_writer.release.assert_called()


@pytest.mark.unit
def test_reduce_video_size_multiple_frames():
    """
    Test the `reduce_video_size` function to ensure it correctly reduces the size of multiple video frames.

    This test:
    - Mocks the video capture process to provide a fake video with multiple frames.
    - Calls the `reduce_video_size` function.
    - Verifies that the resizing function was called correctly for each frame.
    - Ensures that each frame was written to the output.
    """
    # Dummy data for 3 frames
    frames = [
        np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]]),
        np.array([[0, 255, 0], [0, 0, 255], [255, 0, 0]]),
        np.array([[0, 0, 255], [255, 0, 0], [0, 255, 0]])
    ]

    resized_frames = [np.array([[255, 0], [0, 255]]),
                      np.array([[0, 255], [0, 0]]),
                      np.array([[0, 0], [255, 0]])]

    mock_video = MagicMock()
    mock_video.get.side_effect = [3, 3, 24]
    mock_video.read.side_effect = [(True, frames[0]), (True, frames[1]), (True, frames[2]), (False, None)]

    mock_writer = MagicMock()

    with patch('cv2.VideoCapture', return_value=mock_video), \
         patch('cv2.resize', side_effect=resized_frames), \
         patch('cv2.VideoWriter', return_value=mock_writer) as mock_vw:

        reduce_video_size("fake_path.mp4", "fake_output_path_multiple_frames.mp4", 2)

        assert mock_writer.write.call_count == 3  # Ensure that write was called for each frame


###### Labeling
@pytest.mark.unit
def test_label_photos_valid_key():
    # Define the test parameters
    params = {
        'labels': {
            'a': 'LabelA'
        }
    }
    photo_path = "dummy_photo.jpg"

    # Mock the cv2.imread function to return a dummy image (2x2 array of zeros)
    mock_image = np.zeros((2, 2))
    with patch("cv2.imread", return_value=mock_image):
        # Mock cv2.imshow so that it doesn't actually show anything
        with patch("cv2.imshow"):
            # Mock cv2.waitKey to simulate pressing 'a'
            with patch("cv2.waitKey", return_value=ord('a')):
                labels = label_photos(params, photo_path)
    # Assert that the correct label was returned
    assert labels == [(photo_path, "LabelA")]


@pytest.mark.unit
@pytest.mark.parametrize(
    "skip_seconds,total_frames,fps", [
        (1.0, 100, 25),
        (0.5, 100, 30),
        (1.5, 120, 24),
        (2.0, 50, 15),
        (0.7, 77, 20)
    ]
)
def test_label_frames_valid_key_with_skip(skip_seconds, total_frames, fps):
    """
    Test the behavior of the `label_frames` function when the 'skip_seconds' parameter is used.

    The purpose of this test is to ensure that:
    1. Frames are labeled at the intervals defined by `skip_seconds`.
    2. All frames between the current frame and the next skipped frame are labeled with the same label.
    3. All frames at the end of the video after the last skipped frame are also labeled correctly.

    Test Methodology:
    1. Mock the `cv2.VideoCapture` to simulate reading frames from a video without actually accessing a real video file.
    2. Set predefined properties for the mocked video capture, such as `total_frames` and `fps`.
    3. Define a sequence of frames that the mock video capture will "read" based on the skip intervals and the total frames.
    4. Mock the `cv2.imshow` to suppress the actual frame display during the test.
    5. Mock the `cv2.waitKey` to simulate pressing a specific key ('a' in this case) for labeling.
    6. Capture the labels applied by the `label_frames` function.
    7. Calculate the expected labels based on the test parameters.
    8. Compare the labels from the function with the expected labels to determine if the function behaved as intended.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If the labels applied by the `label_frames` function do not match the expected labels.
    """
    # Define the test parameters
    
    params = {
        'labels': {
            'a': 'LabelA'
        }
    }
    video_path = "dummy_video.mp4"
    skip_frames = int(skip_seconds * fps)

    # Mock video properties and video reading
    mock_cap = Mock()
    type(mock_cap).isOpened = PropertyMock(return_value=True)
    type(mock_cap).get = Mock(side_effect=[total_frames, fps])

    # Simulate successful reads for frames to be labeled and an end of video
    type(mock_cap).read = Mock(side_effect=[(True, np.zeros((2, 2))) for _ in range(0, total_frames, skip_frames)] + [(True, np.zeros((2, 2))), (False, None)])

    with patch("cv2.VideoCapture", return_value=mock_cap):
        # Mock cv2.imshow so it doesn't actually show anything
        with patch("cv2.imshow"):
            # Mock cv2.waitKey to simulate pressing 'a'
            with patch("cv2.waitKey", return_value=ord('a')):
                # Mock print function to suppress frame numbers being printed
                with patch("builtins.print"):
                    labels = label_frames(params, video_path, skip_seconds)

    # Verify frames are labeled correctly with skip_seconds considered
    expected_labels = []
    for i in range(0, total_frames, skip_frames):
        for j in range(i, min(i + skip_frames, total_frames)):
            expected_labels.append((j, 'LabelA'))

    assert labels == expected_labels


@pytest.mark.unit
def create_sample_csv(path, filenames):
    """
    Create a sample labeled csv file at the specified path with given filenames.

    Parameters:
    - path (str): Path where the CSV file will be created.
    - filenames (list of str): List of filenames to include in the CSV.

    Returns:
    None
    """
    df = pd.DataFrame({
        'filename': filenames,
        'label': ['cat' for _ in filenames]
    })
    df.to_csv(path, index=False)


@pytest.fixture
def setup_files(tmpdir):
    """
    Set up temporary directories and sample labeled csv files for testing.

    Parameters:
    - tmpdir (py.path.local): Temporary directory path provided by pytest.

    Returns:
    tuple: Paths to input video directory, input photo directory, and output directory.
    """
    input_video_dir = tmpdir.mkdir("input_videos")
    input_photo_dir = tmpdir.mkdir("input_photos")
    output_dir = tmpdir.mkdir("output")
    create_sample_csv(output_dir.join('photo_sample1_labeled.csv'), ['sample1.JPG'])
    create_sample_csv(output_dir.join('video_sample2_labeled.csv'), ['sample2.MP4'])

    return input_video_dir, input_photo_dir, output_dir


@pytest.mark.unit
def test_apply_mirror_labels(setup_files):
    """
    Test the function apply_mirror_labels to ensure it correctly labels mirrored media files.

    Parameters:
    - setup_files (fixture): Paths to input video directory, input photo directory, and output directory.

    Returns:
    None
    """

    input_video_dir, input_photo_dir, output_dir = setup_files

    # Create mirrored video and photo files for testing
    input_video_dir.join("mirrored_sample2.MP4").write("dummy content")
    input_photo_dir.join("mirrored_sample1.JPG").write("dummy content")

    params = {
        'output_dir': str(output_dir),
        'input_video_dir': str(input_video_dir),
        'input_photo_dir': str(input_photo_dir)
    }
    
    apply_mirror_labels(params)

    # Assertions for video
    mirrored_video_csv_path = output_dir.join('video_mirrored_sample2_labeled.csv')
    original_video_csv_path = output_dir.join('video_sample2_labeled.csv')

    assert mirrored_video_csv_path.check(file=1), "Mirrored video CSV not created"
    df_video = pd.read_csv(mirrored_video_csv_path)
    df_orig_video = pd.read_csv(original_video_csv_path)
    
    assert df_video['filename'].iloc[0] == 'video_mirrored_sample2.MP4', "Incorrect filename in mirrored video CSV"
    assert df_video['label'].iloc[0] == df_orig_video['label'].iloc[0], "Mirrored video label doesn't match original"

    # Assertions for photo
    mirrored_photo_csv_path = output_dir.join('photo_mirrored_sample1_labeled.csv')
    original_photo_csv_path = output_dir.join('photo_sample1_labeled.csv')

    assert mirrored_photo_csv_path.check(file=1), "Mirrored photo CSV not created"
    df_photo = pd.read_csv(mirrored_photo_csv_path)
    df_orig_photo = pd.read_csv(original_photo_csv_path)
    
    assert df_photo['filename'].iloc[0] == 'photo_mirrored_sample1.JPG', "Incorrect filename in mirrored photo CSV"
    assert df_photo['label'].iloc[0] == df_orig_photo['label'].iloc[0], "Mirrored photo label doesn't match original"

    # Further assertions:
    
    # Ensure that no other files are created in the output directory
    all_output_files = [f.basename for f in output_dir.listdir()]
    expected_files = [
        'video_mirrored_sample2_labeled.csv',
        'photo_mirrored_sample1_labeled.csv',
        'photo_sample1_labeled.csv',
        'video_sample2_labeled.csv'
    ]
    assert set(all_output_files) == set(expected_files), "Unexpected files in output directory"

    # Ensure that files that don't match the "mirrored_" prefix are ignored
    input_video_dir.join("non_mirrored_sample3.MP4").write("dummy content")
    apply_mirror_labels(params)
    non_mirrored_csv_path = output_dir.join('video_non_mirrored_sample3_labeled.csv')
    assert not non_mirrored_csv_path.check(), "CSV generated for non-mirrored file"

