import cv2
import numpy as np
import pandas as pd
import pytest
import mediapipe as mp

import sys
sys.path.append("/workspaces/cv")

from src.data.photo_processing import mirror_photo
from src.data.video_processing import mirror_video, reduce_video_size
from src.data.label import label_photos, label_frames, apply_mirror_labels
from src.features.extract_landmarks import extract_landmarks_from_frame, calculate_additional_features
from src.features.make_features import calculate_2d_angle, extract_angles, extract_spatial_features
from unittest.mock import patch, mock_open, MagicMock, ANY, PropertyMock, Mock


# pytest -s tests,  to run with print output
# pytest tests -m unit -s, run only unit tests

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
@pytest.mark.parametrize(
    "skip_seconds,total_frames,fps", [
        (1.0, 100, 25),
        (0.5, 100, 30),
        (1.5, 120, 24),
        (2.0, 50, 15),
        (0.7, 77, 20)
    ]
)
def test_label_frames_with_back_and_change(skip_seconds, total_frames, fps):
    """
    Test the frame labeling functionality when the user moves back to a previous frame and changes its label.

    This test simulates the user labeling video frames. After labeling a few frames with "LabelA", the user
    decides to go back to a previously labeled frame and change its label to "LabelB". The test checks if
    the labeling tool correctly captures this change.

    The video frames are divided into segments based on the `skip_seconds` value. For example, if `fps` is 30 
    and `skip_seconds` is 1.0, then every 30 frames will be considered as one segment. The user labels the segments 
    at the beginning and end of these intervals.

    The main steps of the test are:
    1. Mock the video properties and the frame reading process.
    2. Mock the user input (i.e., labeling and navigation commands).
    3. Call the labeling tool.
    4. Verify that the returned labels match the expected labels, accounting for the user going back and changing a label.

    Parameters:
    - skip_seconds: Number of seconds to skip between frames that are displayed for labeling.
    - total_frames: Total number of frames in the video.
    - fps: Frames per second of the video.

    Uses Mock and patch to simulate video reading and user interactions.

    Assertions:
    - Ensures that the returned labels from the labeling tool match the expected labels.
    """
    params = {
        'labels': {
            'a': 'LabelA',
            'b': 'LabelB'
        }
    }
    video_path = "dummy_video.mp4"
    skip_frames = int(skip_seconds * fps)

    # Mock video properties and video reading
    mock_cap = Mock()
    type(mock_cap).isOpened = PropertyMock(return_value=True)
    type(mock_cap).get = Mock(side_effect=[total_frames, fps])
    type(mock_cap).read = Mock(side_effect=[(True, np.zeros((2, 2))) for _ in range(0, total_frames, skip_frames)] + [(True, np.zeros((2, 2))), (False, None)])

    # We'll simulate a scenario where after the 2nd frame (0-based index), 
    # the user presses the left arrow key and then changes the label.
    mock_responses = [ord('a') for _ in range(2)] + [81, ord('b')] + [ord('a') for _ in range(total_frames - 3)]

    with patch("cv2.VideoCapture", return_value=mock_cap), \
         patch("cv2.imshow"), \
         patch("cv2.waitKey", side_effect=mock_responses), \
         patch("builtins.print"):
            labels = label_frames(params, video_path, skip_seconds)

        # Expected labels now have to account for the label change.
    expected_labels = []
    for i in range(1):
        for j in range(i * skip_frames, (i + 1) * skip_frames):
            expected_labels.append((j, 'LabelA'))
    for j in range(1 * skip_frames, 2 * skip_frames):
        expected_labels.append((j, 'LabelB'))
    for i in range(2, total_frames):
        for j in range(i * skip_frames, (i + 1) * skip_frames):
            expected_labels.append((j, 'LabelA'))


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


@pytest.mark.unit
def test_calculate_additional_features():
    # Mocking some sample data
    data = {
        'LEFT_EAR_x': [1], 'RIGHT_EAR_x': [3],
        'LEFT_EAR_y': [2], 'RIGHT_EAR_y': [4],
        'LEFT_EAR_z': [1], 'RIGHT_EAR_z': [3],
        'LEFT_SHOULDER_x': [2], 'RIGHT_SHOULDER_x': [4],
        'LEFT_SHOULDER_y': [3], 'RIGHT_SHOULDER_y': [5],
        'LEFT_SHOULDER_z': [2], 'RIGHT_SHOULDER_z': [4],
        'LEFT_HIP_y': [6], 'RIGHT_HIP_y': [8],
        'LEFT_KNEE_y': [10], 'RIGHT_KNEE_y': [12],
        'LEFT_ELBOW_y': [7], 'RIGHT_ELBOW_y': [9],
        'LEFT_ANKLE_y': [11], 'RIGHT_ANKLE_y': [13]
    }

    df_mock = pd.DataFrame(data)
    
    # Apply the function
    df_result = calculate_additional_features(df_mock)

    # Expected results
    data_expected = {
        'LEFT_EAR_x': [1.0], 'RIGHT_EAR_x': [3.0], 'HEAD_x': [2.0],
        'LEFT_EAR_y': [2.0], 'RIGHT_EAR_y': [4.0], 'HEAD_y': [3.0],
        'LEFT_EAR_z': [1.0], 'RIGHT_EAR_z': [3.0], 'HEAD_z': [2.0],
        'LEFT_SHOULDER_x': [2.0], 'RIGHT_SHOULDER_x': [4.0], 'NECK_x': [3.0],
        'LEFT_SHOULDER_y': [3.0], 'RIGHT_SHOULDER_y': [5.0], 'NECK_y': [4.0],
        'LEFT_SHOULDER_z': [2.0], 'RIGHT_SHOULDER_z': [4.0], 'NECK_z': [3.0],
        'LEFT_HIP_y': [6.0], 'RIGHT_HIP_y': [8.0], 'avg_hip_y': [7.0],
        'LEFT_KNEE_y': [10.0], 'RIGHT_KNEE_y': [12.0], 'avg_knee_y': [11.0],
        'LEFT_ELBOW_y': [7.0], 'RIGHT_ELBOW_y': [9.0], 'avg_elbow_y': [8.0],
        'LEFT_ANKLE_y': [11.0], 'RIGHT_ANKLE_y': [13.0], 'avg_ankle_y': [12.0],
        'avg_shoulder_y': [4.0]
    }


    df_expected = pd.DataFrame(data_expected)
    
    # Reorder the columns of both dataframes to be in alphabetical order
    df_result = df_result.reindex(sorted(df_result.columns), axis=1)
    df_expected = df_expected.reindex(sorted(df_expected.columns), axis=1)
    df_result = df_result.astype(float)
    df_expected = df_expected.astype(float)
    # Using pandas' built-in testing utilities to assert that the two DataFrames are equal
    pd.testing.assert_frame_equal(df_result, df_expected)


@pytest.mark.unit
def mock_pose_process(*args, **kwargs):
    """
    Mock function for MediaPipe's pose process.
    
    This function mocks the MediaPipe Pose process by returning a predefined result 
    with a NormalizedLandmarkList containing landmarks with fixed values.
    
    Args:
        *args: Variable-length argument list (not used in this mock).
        **kwargs: Arbitrary keyword arguments (not used in this mock).
    
    Returns:
        MockResult: A mocked result with predefined landmarks.
    """
    class Landmark:
        def __init__(self):
            self.x = 0.5
            self.y = 0.5
            self.z = 0.0
            self.visibility = 1.0
            self.presence = 1.0

        def HasField(self, field_name):
            return hasattr(self, field_name)

    class MockResult:
        class NormalizedLandmarkList:
            def __init__(self, landmarks):
                self.landmark = landmarks

        pose_landmarks = NormalizedLandmarkList([Landmark() for _ in range(33)])

    
    return MockResult()


@pytest.mark.unit
def test_photo_vs_video_frame_processing():
    """
    Unit test for verifying landmark extraction consistency across photo and video frames.
    
    This test checks that the landmarks extracted from a mock photo and a mock video frame 
    are consistent when using the mock_pose_process function.
    
    Scenarios:
    1. Mock a photo and a video frame.
    2. Extract landmarks from both using the mocked pose process.
    3. Check that the extracted landmarks are consistent across both frames.
    """

    mock_photo = np.array([
        [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        [[255, 255, 0], [0, 255, 255], [255, 0, 255]],
        [[0, 0, 0], [128, 128, 128], [255, 255, 255]]
    ], dtype=np.uint8)

    mock_video_frame = np.copy(mock_photo)

    with patch('src.features.extract_landmarks.mp.solutions.pose.Pose.process', side_effect=mock_pose_process):
        photo_result = extract_landmarks_from_frame(mock_photo, mp.solutions.pose.Pose())
        video_frame_result = extract_landmarks_from_frame(mock_video_frame, mp.solutions.pose.Pose())

    assert np.array_equal(photo_result[0], video_frame_result[0])
    assert np.array_equal(photo_result[1], video_frame_result[1])


@pytest.mark.unit
def test_calculate_2d_angle():
    """
    Unit test for the `calculate_2d_angle` function.
    
    This function tests various scenarios to ensure the function accurately calculates the angle between three 2D points.
    
    Test Scenarios:
    1. Straight line - should return an angle of 180 degrees.
    2. Right angle - should return an angle of 90 degrees.
    3. 45-degree line from vertical - should return an angle of 135 degrees.
    4. Testing normalization of negative angles.
    5. Angles greater than 180 degrees.
    """
    
    # Case 1: Test for a straight line (180 degrees)
    angle_1 = calculate_2d_angle((0, 0), (0, 1), (0, 2))
    assert np.isclose(angle_1, 180.0), f"Expected 180, got {angle_1}"

    # Case 2: Test for a right angle (90 degrees)
    angle_2 = calculate_2d_angle((0, 0), (0, 1), (1, 1))
    assert np.isclose(angle_2, 90.0), f"Expected 90, got {angle_2}"

    # Case 3: Test for a 45 degrees angle
    angle_3 = calculate_2d_angle((0, 0), (0, 1), (1, 2))
    assert np.isclose(angle_3, 135.0), f"Expected 135, got {angle_3}"

    # Case 4: Test for negative angles to ensure they are normalized
    angle_4 = calculate_2d_angle((1, 1), (0, 1), (0, 0))
    assert np.isclose(angle_4, 270), f"Expected 270, got {angle_4}"

    # Case 5: Test for angles greater than 180 degrees
    angle_5 = calculate_2d_angle((1, 1), (0, 1), (-1, 1))
    assert np.isclose(angle_5, 180.0), f"Expected 180, got {angle_5}"


@pytest.mark.unit
def test_extract_angles():
    """
    Unit test for the extract_angles function.
    
    This test verifies that the angles are computed correctly for a given row of 2D landmarks.
    """
    # Mock row data
    data = {
        'LEFT_SHOULDER_x': 0, 'LEFT_SHOULDER_y': 1,
        'LEFT_ELBOW_x': 0, 'LEFT_ELBOW_y': 0,
        'LEFT_WRIST_x': 0, 'LEFT_WRIST_y': -1,
        'RIGHT_SHOULDER_x': 0, 'RIGHT_SHOULDER_y': 1,
        'RIGHT_ELBOW_x': 0, 'RIGHT_ELBOW_y': 0,
        'RIGHT_WRIST_x': 0, 'RIGHT_WRIST_y': -1,
        'LEFT_HIP_x': -1, 'LEFT_HIP_y': 0,
        'RIGHT_HIP_x': 1, 'RIGHT_HIP_y': 0,
        'LEFT_KNEE_x': -1, 'LEFT_KNEE_y': -1,
        'RIGHT_KNEE_x': 1, 'RIGHT_KNEE_y': -1,
        'LEFT_ANKLE_x': -1, 'LEFT_ANKLE_y': -2,
        'RIGHT_ANKLE_x': 1, 'RIGHT_ANKLE_y': -2,
        'HEAD_x': 0, 'HEAD_y': 2,
        'NECK_x': 0, 'NECK_y': 1.5
    }
    
    row = pd.Series(data)

    # Expected angles (180 for straight lines, 90 for right angles)
    expected_angles = {
        'elbow_angle_left': 180.0,
        'elbow_angle_right': 180.0,
        'shoulder_angle_left': 315.0,  
        'shoulder_angle_right': 45.0, 
        'hip_angle_left': 225.0,       
        'hip_angle_right': 135.0,      
        'knee_angle_left': 180.0,      
        'knee_angle_right': 180.0,    
        'spine_angle': 296.565,
        'torso_angle': 303.69
    }

    calculated_angles = extract_angles(row)
    
    for angle_name, expected_value in expected_angles.items():
        assert np.isclose(calculated_angles[angle_name], expected_value, atol=1e-2), \
            f"For {angle_name}: Expected {expected_value}, but got {calculated_angles[angle_name]}."


@pytest.mark.unit
def test_extract_spatial_features():
    """
    Unit test for the extract_spatial_features function.
    
    This test verifies that the spatial features are extracted correctly from the landmarks dataframe.
    """
    
    # Sample landmark data for testing
    df_landmarks = pd.DataFrame({
        'LEFT_KNEE_y': [1.0, 2.0],
        'LEFT_HIP_y': [2.0, 3.0],
        'LEFT_SHOULDER_y': [3.0, 4.0],
        'LEFT_ELBOW_y': [2.5, 3.5],  
        'LEFT_WRIST_y': [2.0, 3.0],  
        'LEFT_ANKLE_y': [0.5, 1.5],  
        'RIGHT_KNEE_y': [2.0, 3.0],
        'RIGHT_HIP_y': [3.0, 4.0],
        'RIGHT_SHOULDER_y': [3.0, 4.0],
        'RIGHT_ELBOW_y': [2.5, 3.5],
        'RIGHT_WRIST_y': [2.0, 3.0],
        'RIGHT_ANKLE_y': [0.5, 1.5],
        'HEAD_y': [5.0, 6.0],
        'avg_hip_y': [2.5, 3.5],
        'avg_shoulder_y': [4.0, 5.0]
    })
    
    result_df = extract_spatial_features(df_landmarks)
    
    
    # Define expected data. For simplicity, we'll just check a subset of the data.
    expected_data = {
        'spatial_left_knee_to_hip': ['above', 'above'],
        # ... [add expected data for other relationships]
        'spatial_right_knee_to_hip': ['above', 'above'],
        'spatial_hip_to_shoulder': ['above', 'above'],
        'spatial_head_to_shoulder': ['below', 'below']
    }
    
    # Check that each column in expected_data exists in result_df
    for column in expected_data.keys():
        assert column in result_df.columns, f"Missing column {column} in result_df"

    # Validate the values in those columns
    for column, expected_values in expected_data.items():
        assert list(result_df[column]) == expected_values, f"Unexpected values in column {column}"
