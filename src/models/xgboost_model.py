import logging
import json
import os
from typing import Any, Dict

import numpy as np
import optuna
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def get_class_weights(le: LabelEncoder) -> dict:
    # Get encoded label for 'other pose or transition'
    encoded_other = le.transform(['other pose or transition'])[0]  # Assuming 'o' is the label for 'other pose or transition'
    
    # Create weight dict
    # For example, setting the weight for 'other pose or transition' to x and y for others
    weights = {label: 50 if label != encoded_other else 1 for label in range(len(le.classes_))}
    
    return weights


def train_xgb(X_train: DataFrame, y_train: np.ndarray, groups: np.ndarray, params: Dict[str, Any]) -> XGBClassifier:
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
    optimize_hyperparams = params.pop('optimize_hyperparams', False)
    weights = get_class_weights(params['label_encoder'])
    logger.info(f"using class weights: {weights}")
    sample_weights = np.array([weights[label] for label in y_train])

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
        
        model = XGBClassifier(**param)
        
        score_metric = params.get('score_metric', 'accuracy')  # Defaulting to accuracy if score_metric isn't provided
        return -cross_val_score(model, X_train, y_train, cv=5, scoring=score_metric).mean()

    if optimize_hyperparams:
        logger.info("Optimizing hyperparameters")
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=3)
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
        
        model = XGBClassifier(**best_params)

    else:
        model = XGBClassifier(tree_method='hist',enable_categorical=True) # later can: XGBClassifier(**params)
    
    # Fit model on the entire dataset
    model.fit(X_train, y_train, sample_weight=sample_weights)

    return model
