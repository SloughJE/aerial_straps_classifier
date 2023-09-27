from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from joblib import load
import uuid
from src.features.extract_landmarks import extract_landmarks
from src.features.make_features import extract_features_from_single_landmark_csv
from src.models.train_model import convert_spatial_features_to_categorical
from .visualization.charts import create_probability_chart
import logging
import shutil
from datetime import datetime
from typing import Tuple, Dict, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Constants
base_directory = Path(__file__).parent

# Constants
UPLOAD_DIR = base_directory / "uploaded_images"
MODEL_PATH = base_directory.parent / "models" / "prod" / "xgb" / "xgb_prod_model.joblib"
LABEL_ENCODER_PATH = base_directory.parent / "models" / "prod" / "xgb" / "label_encoder.pkl"

# Create the directory if it doesn't exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Define the templates directory
TEMPLATES_DIR = base_directory / "templates"
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Loading the model and encoder when the app starts
xgb_model = load(MODEL_PATH)
label_encoder = load(LABEL_ENCODER_PATH)

# Mount static files using the absolute path
app.mount("/uploaded_images", StaticFiles(directory=UPLOAD_DIR), name="uploaded_images")

@app.get("/", response_model=None)
async def serve_page():
    """Serves the main page."""
    return templates.TemplateResponse("index.html", {"request": {}})

import time

def cleanup_files(directory: Path, age_minutes: int = 1):
    """Delete files in the given directory that are older than the specified number of minutes."""
    
    # Calculate the age threshold in seconds
    age_seconds = age_minutes * 60
    current_time = time.time()

    for filepath in directory.iterdir():
        # Get the time the file was last modified
        file_age = current_time - filepath.stat().st_mtime
        
        # If the file age exceeds the threshold, delete the file
        if file_age > age_seconds:
            filepath.unlink()

#def save_uploaded_file(file: UploadFile, new_filename: str = "temp.jpg") -> Path:
#    """Saves the uploaded file and returns its path."""
#    input_path = UPLOAD_DIR / new_filename
#    with input_path.open("wb") as buffer:
#        shutil.copyfileobj(file.file, buffer)
#    return input_path

def save_uploaded_file(file: UploadFile) -> Path:
    """Saves the uploaded file and returns its path."""
    
    # Extract the file extension from the uploaded file's filename
    extension = Path(file.filename).suffix
    
    if not extension:
        raise HTTPException(status_code=400, detail="File does not have an extension")
    
    unique_filename = f"{uuid.uuid4()}{extension}"
    input_path = UPLOAD_DIR / unique_filename
    with input_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return input_path

def extract_image_features(input_path: Path) -> Tuple[object, Path, Path]:
    """Extracts features from the uploaded image."""
    
    # Extract the file stem from the original image path
    file_stem = input_path.stem
    
    # Generate the annotated image path by prefixing with "annotated_" and setting extension to ".jpg"
    annotated_img_name = f"annotated_{file_stem}.jpg"
    annotated_img_path = UPLOAD_DIR / annotated_img_name
    
    df_landmarks = extract_landmarks(str(input_path), str(annotated_img_path), is_video=False, write_output=True)
    
    # Check if landmarks extraction is valid
    if df_landmarks.empty or df_landmarks is None:
        raise ValueError("Unable to detect a human or the human in the image is obscured. Please upload a clear image.")

    landmarks_csv_path = UPLOAD_DIR / f"{file_stem}_landmarks.csv"
    df_landmarks.to_csv(landmarks_csv_path, index=False)
    
    df_features = extract_features_from_single_landmark_csv(str(landmarks_csv_path), str(UPLOAD_DIR))
    return df_features, input_path, annotated_img_path


def get_pose_prediction(df_features: object, xgb_model, label_encoder) -> Tuple[str, object, object]:
    """Predicts the pose and returns it along with probabilities and labels."""
    
    features_for_prediction = df_features.drop(columns=['filename', 'frame_number'])
    features_for_prediction = convert_spatial_features_to_categorical(features_for_prediction)
    
    pose_encoded = xgb_model.predict(features_for_prediction)
    pose_decoded = label_encoder.inverse_transform(pose_encoded)[0]
    probabilities = xgb_model.predict_proba(features_for_prediction).flatten()
    pose_labels = label_encoder.classes_
    return pose_decoded, probabilities, pose_labels



@app.post("/upload/")
async def process_image(background_tasks: BackgroundTasks, file: UploadFile = File(...)) -> Dict[str, Union[str, float]]:
    """Handles file upload, feature extraction, pose prediction, and returns results."""
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
        return {
            "error": str(e)  # Sending the error to the client
        }
