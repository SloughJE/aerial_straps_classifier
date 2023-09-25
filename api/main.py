from fastapi import FastAPI, File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path


import os        
from joblib import dump, load
import xgboost as xgb
import shutil
from src.features.extract_landmarks import extract_landmarks
from src.features.make_features import extract_features_from_single_landmark_csv
from src.models.train_model import convert_spatial_features_to_categorical
from .visualization.charts import create_probability_chart

app = FastAPI()

# Mount the static files directory to a path
base_directory = Path(__file__).parent
app.mount("/uploaded_images", StaticFiles(directory=base_directory / "uploaded_images"), name="uploaded_images")

UPLOAD_DIR = "api/uploaded_images"
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

templates = Jinja2Templates(directory="api/templates")

@app.get("/")
async def serve_page():
    return templates.TemplateResponse("index.html", {"request": {}})

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Step 1: Read and Display the Uploaded Image
        filename = file.filename
        input_path = os.path.join(UPLOAD_DIR, filename)
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        og_img_path = os.path.join(UPLOAD_DIR, filename)
        # Step 2: Extract Landmarks and Save the Annotated Image
        annotated_img_path = os.path.join(UPLOAD_DIR, "annotated_" + filename)
        df_landmarks = extract_landmarks(input_path, annotated_img_path, is_video=False, write_output=True)

        # Save df_landmarks to a temporary CSV
        filename_without_extension = os.path.splitext(filename)[0]
        landmarks_csv_path = os.path.join(UPLOAD_DIR, f"{filename_without_extension}_landmarks.csv")
        df_landmarks.to_csv(landmarks_csv_path, index=False)

        # Step 3: Extract Features from Landmarks
        features_directory = UPLOAD_DIR

        df_features = extract_features_from_single_landmark_csv(landmarks_csv_path, features_directory)

        #print(df_features.columns)
        # Load the trained XGBoost model and label encoder
        #MODEL_PATH = "models/prod/xgb/xgb_prod_model.pkl"
        #MODEL_PATH = "models/dev/xgb/xgb_model.pkl"

        # Paths for the trained XGBoost model and label encoder
        MODEL_PATH = "models/dev/xgb/xgb_model.joblib"
        LABEL_ENCODER_PATH = "models/dev/xgb/label_encoder.pkl"

        # Load the trained XGBoost model and label encoder using joblib
        xgb_model = load(MODEL_PATH)
        label_encoder = load(LABEL_ENCODER_PATH)

        # Assuming df_features is your DataFrame with the extracted features
        features_for_prediction = df_features.drop(columns=['filename', 'frame_number'])
        features_for_prediction = convert_spatial_features_to_categorical(features_for_prediction)
        pose_encoded = xgb_model.predict(features_for_prediction)
        pose_decoded = label_encoder.inverse_transform(pose_encoded)[0]  # Decode to actual pose name
        probabilities = xgb_model.predict_proba(features_for_prediction).flatten()

        # Get pose labels from the label encoder
        pose_labels = label_encoder.classes_
        # Create and save the bar chart

        chart_filename = os.path.join(UPLOAD_DIR, f"{filename_without_extension}_chart.html")

        create_probability_chart(probabilities, pose_labels, chart_filename, og_img_path)

        # Step 4: Display the Annotated Image
        return {"original_filename": input_path, "annotated_filename": annotated_img_path, "predicted_pose": pose_decoded,
                "chart_filename": chart_filename
}

    except Exception as e:
        return {"error": str(e)}
