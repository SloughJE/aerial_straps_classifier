import os
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def train_prod_model(params):
    """
    Trains the production model on the entire dataset and saves it.

    Args:
    - params: Dictionary containing parameters for training the production model.
    """
    model_type = params['model_type']
    final_features_filepath = params['final_features_filepath']
    target_column = params['target_column']

    df = pd.read_csv(final_features_filepath)

    # Initialize the model with the specified type
    if model_type == 'xgb':
        model = XGBClassifier()
    elif model_type == 'rf':
        model = RandomForestClassifier()

    X_full = df.drop(columns=[target_column, 'video_frame', 'filename'])
    y_full = df[target_column]

    le = LabelEncoder()
    y_full = le.fit_transform(y_full)

    # Train the model on the full dataset
    model.fit(X_full, y_full)

    # Save the trained model and label encoder
    prod_models_dir = 'models/prod'
    model_dir = os.path.join(prod_models_dir, model_type)
    os.makedirs(model_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(model_dir, f"{model_type}_prod_model.pkl")
    joblib.dump(model, model_path)

    # Save label encoder
    le_path = os.path.join(model_dir, f"label_encoder.pkl")
    joblib.dump(le, le_path)

    print("Production model trained and saved, along with label encoder.")
