import logging
import json
import os
from typing import Any, Dict

import mlflow
import numpy as np
import optuna
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from .utils import predict_and_evaluate, save_model
from .evaluation_metrics import generate_feature_importance_visualization

logger = logging.getLogger(__name__)


def get_class_weights(le: LabelEncoder) -> dict:
    # Get encoded label for 'other pose or transition'
    encoded_other = le.transform(['other pose or transition'])[0]  # Assuming 'o' is the label for 'other pose or transition'
    
    # Create weight dict
    # For example, setting the weight for 'other pose or transition' to x and y for others
    weights = {label: 50 if label != encoded_other else 1 for label in range(len(le.classes_))}
    
    return weights


def train_xgb(X_train: DataFrame, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, groups: np.ndarray, params: Dict[str, Any]) -> XGBClassifier:
    """
    Trains an XGBClassifier with optional hyperparameter optimization.

    Args:
    - X_train (DataFrame): The training features.
    - y_train (np.ndarray): The training target.
    - groups (np.ndarray): The groups for the training data.
    - params (dict): The parameters for the model.

    Returns:
    - model (XGBClassifier): The trained XGB model.
    """
    mlflow.set_experiment('XGB_Optimization_and_Training') 

    #mlflow.xgboost.autolog(log_models=False)

    # Instantiate MLflowCallback and specify the tracking URI
#    mlflc = MLflowCallback(tracking_uri="http://127.0.0.1:5000", metric_name="model_score",mlflow_kwargs={"nested": True})

    optimize_hyperparams = params.pop('optimize_hyperparams', False)
    weights = get_class_weights(params['label_encoder'])
    logger.info(f"using class weights: {weights}")
    sample_weights = np.array([weights[label] for label in y_train])

#    @mlflc.track_in_mlflow()
    def objective(trial):
        # Ending any active run before starting a new one
        #if mlflow.active_run():
        #    mlflow.end_run()

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
        
        model = XGBClassifier(**param)
 
        with mlflow.start_run(run_name=f'Trial_{trial.number}', nested=True):  # Set a descriptive name for each trial
            score_metric = params.get('score_metric', 'accuracy')  # Defaulting to accuracy if score_metric isn't provided
            mlflow.log_param("score_metric", score_metric)  # Log the score metric

            model = XGBClassifier(**param)
            score = cross_val_score(model, X_train, y_train, cv=5, scoring=score_metric).mean()

            mlflow.log_params(param)  # Log the parameters for this trial
            mlflow.log_metric(f"cross_val_score_{score_metric}", score)  # Log the score for this trial (note the negative sign to make it positive)

        return score

    if optimize_hyperparams:

        logger.info("Optimizing hyperparameters")
        study = optuna.create_study(direction='maximize')

        parent_run_name = "Hyperparameter_Optimization"
        with mlflow.start_run(run_name=parent_run_name, nested=True):  # Start a new run for the optimization step
            study.optimize(objective, n_trials=10)

        best_params = study.best_params
        logger.info(f"Best hyperparameters found: {best_params}")
        
        # add back the fixed params
        fixed_params = {
            'tree_method': 'hist',
            'enable_categorical': True
            }
        
        # Merge fixed parameters with the optimized parameters
        best_params.update(fixed_params)
        # Save the best hyperparameters
        models_dir = 'models/dev'
        model_dir = os.path.join(models_dir, 'xgb')
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, 'best_hyperparameters.json'), 'w') as f:
            json.dump(best_params, f)
                
        # Close the "Hyperparameter_Optimization" run before starting the final training run
        mlflow.end_run()

        with mlflow.start_run(run_name='Final_Training_with_Optimized_Params', nested=True):
            # Log the best parameters to MLFlow
            mlflow.log_params(best_params)
            model = XGBClassifier(**best_params)

            model.fit(X_train, y_train, sample_weight=sample_weights)

            train_accuracy, test_accuracy = predict_and_evaluate(model, X_train, y_train, X_test, y_test, params)
            label_encoder = params['label_encoder']
            save_model(model, params, label_encoder)

            feature_names = list(X_train.columns)
            feature_names = [col for col in feature_names if col not in ['filename', 'frame_number']]
            save_path = os.path.join(params['predictions_dir'], params['model_type'], "feature_importance.png")
            generate_feature_importance_visualization(model, feature_names, save_path)

            logger.info(f"Train accuracy: {train_accuracy:.2f}")
            logger.info(f"Test accuracy: {test_accuracy:.2f}")

        return model

    #else:
    #    model = XGBClassifier(tree_method='hist',enable_categorical=True) # later can: XGBClassifier(**params)
    
    # Start a new run for the final training with the best parameters

    