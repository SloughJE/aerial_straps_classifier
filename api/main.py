import json
import logging
import shutil
import time
import uuid
from pathlib import Path
from typing import Dict, Tuple, Union

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from joblib import load

from src.features.extract_landmarks import extract_landmarks
from src.features.make_features import extract_features_from_single_landmark_csv
from src.utils.processing_utils import CustomLabelEncoder, convert_spatial_features_to_categorical
from .visualization.charts import create_probability_chart

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Constants
BASE_DIRECTORY = Path(__file__).parent
UPLOAD_DIR = BASE_DIRECTORY / "image_processing"
MODEL_PATH = BASE_DIRECTORY.parent / "models" / "prod" / "xgb" / "xgb_prod_model.joblib"
LABEL_ENCODER_PATH = BASE_DIRECTORY.parent / "models" / "prod" / "xgb" / "label_encoder.json"

# Create the directory if it doesn't exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

TEMPLATES_DIR = BASE_DIRECTORY / "templates"
templates = Jinja2Templates(directory=TEMPLATES_DIR)

xgb_model = load(MODEL_PATH)

with open(LABEL_ENCODER_PATH, 'r') as f:
    mappings = json.load(f)

# Ensure the keys are strings for valid JSON format
mappings['int_to_label'] = {str(k): v for k, v in mappings['int_to_label'].items()}

label_encoder = CustomLabelEncoder()
label_encoder.set_mappings(mappings['label_to_int'], mappings['int_to_label'])

app.mount("/image_processing", StaticFiles(directory=UPLOAD_DIR), name="image_processing")


@app.get("/", response_model=None)
async def serve_page() -> None:
    """Serve the main page."""
    return templates.TemplateResponse("index.html", {"request": {}})


def cleanup_files(directory: Path, age_minutes: int = 1) -> None:
    """
    Delete files older than a specified age.

    Parameters:
    - directory (Path): Path to the directory containing files.
    - age_minutes (int, optional): Age threshold for file deletion. Defaults to 1.
    """
    age_seconds = age_minutes * 60
    current_time = time.time()

    for filepath in directory.iterdir():
        file_age = current_time - filepath.stat().st_mtime
        if file_age > age_seconds:
            filepath.unlink()


def save_uploaded_file(file: UploadFile) -> Path:
    """
    Save the uploaded file and return its path.

    Parameters:
    - file (UploadFile): Uploaded file.

    Returns:
    - Path: Path to the saved file.
    """
    extension = Path(file.filename).suffix

    if not extension:
        raise HTTPException(status_code=400, detail="File does not have an extension")

    unique_filename = f"{uuid.uuid4()}{extension}"
    input_path = UPLOAD_DIR / unique_filename
    with input_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return input_path


def extract_image_features(input_path: Path) -> Tuple[pd.DataFrame, Path, Path]:
    """
    Extract features from an image.

    Parameters:
    - input_path (Path): Path to the image file.

    Returns:
    - Tuple[pd.DataFrame, Path, Path]: Tuple containing DataFrame of extracted features, path to the original image,
      and path to the annotated image.
    """
    file_stem = input_path.stem
    annotated_img_name = f"annotated_{file_stem}.jpg"
    annotated_img_path = UPLOAD_DIR / annotated_img_name

    df_landmarks = extract_landmarks(str(input_path), str(annotated_img_path), is_video=False, write_output=True)

    if df_landmarks.empty or df_landmarks is None:
        raise ValueError("Unable to detect a human or the human in the image is obscured. Please upload a clear image.")

    landmarks_csv_path = UPLOAD_DIR / f"{file_stem}_landmarks.csv"
    df_landmarks.to_csv(landmarks_csv_path, index=False)

    df_features = extract_features_from_single_landmark_csv(str(landmarks_csv_path), str(UPLOAD_DIR))
    return df_features, input_path, annotated_img_path


def get_pose_prediction(df_features: pd.DataFrame, xgb_model, label_encoder: CustomLabelEncoder) -> Tuple[str, pd.Series, pd.Index]:
    """
    Predict pose and return it along with probabilities and labels.

    Parameters:
    - df_features (pd.DataFrame): DataFrame containing extracted features.
    - xgb_model: XGBoost model for pose prediction.
    - label_encoder (CustomLabelEncoder): Label encoder for decoding predictions.

    Returns:
    - Tuple[str, pd.Series, pd.Index]: Tuple containing predicted pose, prediction probabilities, and pose labels.
    """
    features_for_prediction = df_features.drop(columns=['filename', 'frame_number'])
    features_for_prediction = convert_spatial_features_to_categorical(features_for_prediction)

    pose_encoded = xgb_model.predict(features_for_prediction)[0]
    pose_decoded = label_encoder.inverse_transform([pose_encoded])[0]

    probabilities = xgb_model.predict_proba(features_for_prediction).flatten()
    pose_labels = label_encoder.classes_

    return pose_decoded, probabilities, pose_labels


@app.post("/upload/")
async def process_image(background_tasks: BackgroundTasks, file: UploadFile = File(...)) -> Dict[str, Union[str, float]]:
    """
    Process an uploaded image and return results.

    Parameters:
    - background_tasks (BackgroundTasks): Background tasks for FastAPI.
    - file (UploadFile, optional): Uploaded file. Defaults to File(...).

    Returns:
    - Dict[str, Union[str, float]]: Dictionary containing paths to original and annotated images, predicted pose,
      and path to the probability chart.
    """
    try:
        input_path = save_uploaded_file(file)
        file_stem = input_path.stem

        df_features, og_img_path, annotated_img_path = extract_image_features(input_path)
        pose_decoded, probabilities, pose_labels = get_pose_prediction(df_features, xgb_model, label_encoder)

        chart_filename = UPLOAD_DIR / f"{file_stem}_chart.html"
        create_probability_chart(probabilities, pose_labels, str(chart_filename), str(og_img_path))

        background_tasks.add_task(cleanup_files, UPLOAD_DIR, age_minutes=1)

        return {
            "original_filename": str(input_path),
            "annotated_filename": str(annotated_img_path),
            "predicted_pose": pose_decoded,
            "chart_filename": str(chart_filename)
        }

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        background_tasks.add_task(cleanup_files, UPLOAD_DIR, age_minutes=1)
        return {"error": str(e)}
    
@app.get("/test_image/")
async def process_test_image(background_tasks: BackgroundTasks) -> Dict[str, Union[str, float]]:
    try:
        input_path = BASE_DIRECTORY / "test_images" / "DSC09032.JPG"
        file_stem = input_path.stem

        # Ensure the original image is in a web-accessible location
        web_accessible_original = UPLOAD_DIR / f"{file_stem}.JPG"
        shutil.copy(input_path, web_accessible_original)

        df_features, og_img_path, annotated_img_path = extract_image_features(input_path)
        pose_decoded, probabilities, pose_labels = get_pose_prediction(df_features, xgb_model, label_encoder)

        chart_filename = UPLOAD_DIR / f"{file_stem}_chart.html"
        create_probability_chart(probabilities, pose_labels, str(chart_filename), str(og_img_path))

        background_tasks.add_task(cleanup_files, UPLOAD_DIR, age_minutes=1)

        return {
            "original_filename": str(web_accessible_original),
            "annotated_filename": str(annotated_img_path),
            "predicted_pose": pose_decoded,
            "chart_filename": str(chart_filename)
        }

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        background_tasks.add_task(cleanup_files, UPLOAD_DIR, age_minutes=1)
        return {"error": str(e)}
