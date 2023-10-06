import os
import shutil
import logging
from urllib.parse import urlparse
import mlflow

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def copy_prod_model_to_destination(params: dict):
    model_name = params['model_name']
    dest_path = params['dest_path']
    
    client = mlflow.tracking.MlflowClient()
    model_version_details = client.get_latest_versions(model_name, stages=["Production"])

    if model_version_details:
        source_uri = model_version_details[0].source
        
        parsed_uri = urlparse(source_uri)
        model_base_path = os.path.abspath(os.path.join(parsed_uri.netloc, parsed_uri.path))

        # Model Path
        model_file_path = os.path.join(model_base_path, "model.pkl")

        # Label Encoder Path
        base_path_before_models = model_base_path[:model_base_path.rfind("models")]
        label_encoder_file_path = os.path.join(base_path_before_models, "label_encoder", "label_encoder.json")

        # Correcting the destination paths as per requirement
        dest_model_path = os.path.join(dest_path, "prod_model.pkl")
        dest_label_encoder_path = os.path.join(dest_path, "label_encoder.json")
        
        logger.info(f"Grabbing latest registered production model from: {model_file_path}")
        logger.info(f"Label Encoder file path: {label_encoder_file_path}")

        if os.path.exists(model_file_path) and os.path.exists(label_encoder_file_path):
            os.makedirs(dest_path, exist_ok=True)

            if os.path.exists(dest_model_path):
                logger.warning(f"Note: Overwriting existing model at: {dest_model_path}")
            shutil.copy2(model_file_path, dest_model_path)
            logger.info(f"Latest registered production model copied to: {dest_model_path}")

            if os.path.exists(dest_label_encoder_path):
                logger.warning(f"Note: Overwriting existing label encoder at: {dest_label_encoder_path}")
            shutil.copy2(label_encoder_file_path, dest_label_encoder_path)
            logger.info(f"Label encoder copied to: {dest_label_encoder_path}")

        else:
            missing_files = []
            if not os.path.exists(model_file_path):
                missing_files.append(model_file_path)
            if not os.path.exists(label_encoder_file_path):
                missing_files.append(label_encoder_file_path)
            
            raise FileNotFoundError(f"No such file or directory: {', '.join(missing_files)}")
    else:
        raise ValueError("No production model available.")
