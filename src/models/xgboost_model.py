import json
import logging
import os
from typing import Any, Dict

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
from mlflow.data.pandas_dataset import PandasDataset
from mlflow.models.signature import infer_signature
from pandas import DataFrame
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from .evaluation_metrics import generate_feature_importance_visualization
from .utils import predict_and_evaluate

logger = logging.getLogger(__name__)


def get_class_weights(le: LabelEncoder) -> dict:
    # Get encoded label for 'other pose or transition'
    encoded_other = le.transform(['other pose or transition'])[0]  # Assuming 'o' is the label for 'other pose or transition'
    
    # Create weight dict
    # For example, setting the weight for 'other pose or transition' to x and y for others
    weights = {label: 50 if label != encoded_other else 1 for label in range(len(le.classes_))}
    
    return weights


def train_production_model(X_train: DataFrame, X_test: DataFrame, y_train: np.ndarray, y_test: np.ndarray, best_params: Dict[str, Any], params: Dict[str, Any]) -> None:
    """
    Trains the production model using the full dataset and logs the details to MLFlow.

    Args:
    - X_train (DataFrame): The training features.
    - X_test (DataFrame): The testing features.
    - y_train (np.ndarray): The training target.
    - y_test (np.ndarray): The testing target.
    - best_params (Dict[str, Any]): The best parameters obtained from hyperparameter optimization.
    - params (Dict[str, Any]): Dictionary containing the parameters for the model, including 'model_type', 'MLflow_config', 'predictions_dir', etc.

    Returns:
    - None
    """
    logger.info("Training production model")

    # Drop unnecessary columns
    X_train = X_train.drop(columns=['filename', 'frame_number'], errors='ignore')
    X_test = X_test.drop(columns=['filename', 'frame_number'], errors='ignore')

    # Combine the training and testing datasets to form the full dataset
    X_full = pd.concat([X_train, X_test])
    y_full = pd.concat([y_train, y_test])

    # Define the directory structure for the production model
    model_type = params['model_type']
    prod_models_dir = 'models/prod'
    prod_model_dir = os.path.join(prod_models_dir, model_type)
    os.makedirs(prod_model_dir, exist_ok=True)

    # Define the filepath for MLflow logging
    mlflow_prod_model_filepath = os.path.join(prod_model_dir, model_type)

    with mlflow.start_run(run_name=params['MLflow_config']['run_names']['prod_training'], nested=True):
        # Log parameters and dataset information
        mlflow.log_params(best_params)
        mlflow.log_param('num_samples', len(X_full))
        mlflow.log_param('num_features', X_full.shape[1])
        prod_dataset: PandasDataset = mlflow.data.from_pandas(pd.concat([X_full,y_full],axis=1), 
                                                              targets=params['target_column'], name="Prod Dataset")
        mlflow.log_input(prod_dataset, context="Prod dataset")
        # Train the production model on the full dataset
        prod_model = XGBClassifier(**best_params)
        prod_model.fit(X_full, y_full)

        # Infer the model signature and log the production model to MLFlow
        input_example = X_test.iloc[:5] 
        model_predictions = prod_model.predict(input_example)
        signature = infer_signature(input_example, model_predictions)
        mlflow.xgboost.log_model(xgb_model=prod_model, artifact_path=mlflow_prod_model_filepath, model_format='json', signature=signature)
        
        logger.info("Production model trained and logged to MLFlow.")

        # Generate and save the feature importance visualization
        save_path = os.path.join(params['predictions_dir'], params['model_type'], "prod_feature_importance.png")
        generate_feature_importance_visualization(prod_model, list(X_full.columns), save_path)


def full_train_dataset_training(X_train: DataFrame, y_train: np.ndarray, X_test: DataFrame, y_test: np.ndarray, 
                   best_params: Dict[str, Any], params: Dict[str, Any]) -> None:
    """
    Conducts the training of the XGB model using the best parameters obtained from hyperparameter optimization.

    Args:
    - X_train (DataFrame): The training features.
    - y_train (np.ndarray): The training target.
    - X_test (DataFrame): The testing features.
    - y_test (np.ndarray): The testing target.
    - best_params (Dict[str, Any]): The best parameters obtained from hyperparameter optimization.
    - params (Dict[str, Any]): Dictionary containing the parameters for the model, including 'model_type', 
                               'label_encoder', 'MLflow_config', 'predictions_dir', 'target_column', etc.

    Returns:
    - None
    """
    
    # Directory setup
    models_dir = 'models/dev'
    model_type = params['model_type']
    model_dir = os.path.join(models_dir, model_type)
    os.makedirs(model_dir, exist_ok=True)

    # MLflow setup for final training
    with mlflow.start_run(run_name=params['MLflow_config']['run_names']['final_training'], nested=True):
        
        # Log the best parameters to MLflow
        mlflow.log_params(best_params)

        # Initialize and train the model with the best parameters
        model = XGBClassifier(**best_params)
        sample_weights = np.array([get_class_weights(params['label_encoder'])[label] for label in y_train])
        model.fit(X_train, y_train, sample_weight=sample_weights)

        # Evaluate the model and log the metrics
        train_accuracy, test_accuracy = predict_and_evaluate(model, X_train, y_train, X_test, y_test, params)
        logger.info(f"Train accuracy: {train_accuracy:.2f}")
        logger.info(f"Test accuracy: {test_accuracy:.2f}")

        # Save and log the label encoder
        label_encoder = params['label_encoder']
        label_encoder_filepath = os.path.join(model_dir, 'label_encoder.pkl')
        joblib.dump(label_encoder, label_encoder_filepath)
        mlflow.log_artifact(label_encoder_filepath)

        X_test = X_test.drop(columns=['filename', 'frame_number'])
        # Infer the model signature and log the model
        input_example = X_test.iloc[:5] 
        model_predictions = model.predict(input_example)
        signature = infer_signature(input_example, model_predictions)
        mlflow_model_filepath = os.path.join(model_dir, model_type)
        mlflow.xgboost.log_model(xgb_model=model, artifact_path=mlflow_model_filepath, model_format='json', signature=signature)
        
        # Generate and save the feature importance visualization
        save_path = os.path.join(params['predictions_dir'], params['model_type'], "feature_importance.png")
        generate_feature_importance_visualization(model, list(X_train.columns), save_path)

        mlflow.end_run()


def train_xgb(X_train: DataFrame, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, 
              groups: np.ndarray, params: Dict[str, Any]) -> None:
    """
    Trains an XGBClassifier with optional hyperparameter optimization.
    Also trains a prod model optionally. 

    Args:
    - X_train (DataFrame): The training features.
    - y_train (np.ndarray): The training target.
    - groups (np.ndarray): The groups for the training data.
    - params (dict): The parameters for the model.

    Returns:
    - None
    """   


    optimize_hyperparams = params.pop('optimize_hyperparams', False)
    #weights = get_class_weights(params['label_encoder'])
    #logger.info(f"using class weights: {weights}")
    # sample_weights = np.array([weights[label] for label in y_train])

    def objective(trial):

        param = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
            'max_depth': trial.suggest_int('max_depth', 1, 20),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1),
            'enable_categorical' : True,
            'tree_method': 'hist' # best for 'medium' sized datasets (also only hist and approx work with categorical features)
        }
         
        with mlflow.start_run(run_name=f'Trial_{trial.number}', nested=True):  
            score_metric = params.get('score_metric', 'accuracy')  # Defaulting to accuracy if score_metric isn't provided
            mlflow.log_param("score_metric", score_metric)  

            model = XGBClassifier(**param)
            score = cross_val_score(model, X_train, y_train, cv=5, scoring=score_metric).mean()

            mlflow.log_params(param)  
            mlflow.log_metric(f"cross_val_score_{score_metric}", score)  

        return score

    # Define a variable to hold the parameters to be used for training
    training_params = None

    if optimize_hyperparams:

        logger.info("Optimizing hyperparameters")
        study = optuna.create_study(direction='maximize')
    
        models_dir = 'models/dev'
        model_dir = os.path.join(models_dir, 'xgb')
        os.makedirs(model_dir, exist_ok=True)
        
        with mlflow.start_run(run_name=params['MLflow_config']['run_names']['hyperparameter_optimization'], nested=True):
            study.optimize(objective, n_trials=params['num_trials'])
            study_file_path = os.path.join(model_dir, 'optuna_study.pkl')
            joblib.dump(study, study_file_path)
            mlflow.log_artifact(study_file_path)

        best_params = study.best_params
        logger.info(f"Best hyperparameters found: {best_params}")
        
        # add back the fixed params
        fixed_params = {
            'tree_method': 'hist',
            'enable_categorical': True
            }
        
        # Merge fixed parameters with the optimized parameters
        best_params.update(fixed_params)
        with open(os.path.join(model_dir, 'best_hyperparameters.json'), 'w') as f:
            json.dump(best_params, f)

        training_params = best_params

        logger.info("Training with optimized hyperparameters")
        full_train_dataset_training(X_train, y_train, X_test, y_test, training_params, params)

    else:

        logger.info("Training with default parameters.")
        default_params = {
            'tree_method': 'hist',
            'enable_categorical': True
        }
        training_params = default_params
        full_train_dataset_training(X_train, y_train, X_test, y_test, training_params, params)


    if params.get('train_prod_model', False):
        train_production_model(X_train, X_test, y_train, y_test, training_params, params)