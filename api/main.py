from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import os
import shutil
from datetime import datetime

from joblib import load
from src.features.extract_landmarks import extract_landmarks
from src.features.make_features import extract_features_from_single_landmark_csv
from src.models.train_model import convert_spatial_features_to_categorical
from .visualization.charts import create_probability_chart
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Constants
UPLOAD_DIR = "api/uploaded_images"
MODEL_PATH = "models/dev/xgb/xgb_model.joblib"
LABEL_ENCODER_PATH = "models/dev/xgb/label_encoder.pkl"
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
base_directory = Path(__file__).parent
templates = Jinja2Templates(directory="api/templates")

app.mount("/uploaded_images", StaticFiles(directory=base_directory / "uploaded_images"), name="uploaded_images")

@app.get("/")
async def serve_page():
    return templates.TemplateResponse("index.html", {"request": {}})

def save_uploaded_file(file: UploadFile, new_filename: str = "temp.jpg") -> str:
    input_path = os.path.join(UPLOAD_DIR, new_filename)
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return input_path

def extract_image_features(input_path: str):
    og_img_path = input_path
    annotated_img_path = os.path.join(UPLOAD_DIR, "annotated_temp.jpg")
    df_landmarks = extract_landmarks(input_path, annotated_img_path, is_video=False, write_output=True)
    
    landmarks_csv_path = os.path.join(UPLOAD_DIR, "temp_landmarks.csv")
    df_landmarks.to_csv(landmarks_csv_path, index=False)
    
    df_features = extract_features_from_single_landmark_csv(landmarks_csv_path, UPLOAD_DIR)
    return df_features, og_img_path, annotated_img_path


def get_pose_prediction(df_features):
    xgb_model = load(MODEL_PATH)
    label_encoder = load(LABEL_ENCODER_PATH)
    features_for_prediction = df_features.drop(columns=['filename', 'frame_number'])
    features_for_prediction = convert_spatial_features_to_categorical(features_for_prediction)
    pose_encoded = xgb_model.predict(features_for_prediction)
    pose_decoded = label_encoder.inverse_transform(pose_encoded)[0]
    probabilities = xgb_model.predict_proba(features_for_prediction).flatten()
    pose_labels = label_encoder.classes_
    return pose_decoded, probabilities, pose_labels


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        input_path = save_uploaded_file(file)
        df_features, og_img_path, annotated_img_path = extract_image_features(input_path)
        pose_decoded, probabilities, pose_labels = get_pose_prediction(df_features)
        
        chart_filename = os.path.join(UPLOAD_DIR, "temp_chart.html")
        create_probability_chart(probabilities, pose_labels, chart_filename, og_img_path)

        current_timestamp = datetime.now().timestamp()

        return {
            "original_filename": f"{input_path}?t={current_timestamp}",
            "annotated_filename": f"{annotated_img_path}?t={current_timestamp}",
            "predicted_pose": pose_decoded,
            "chart_filename": f"{chart_filename}?t={current_timestamp}"
        }

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))