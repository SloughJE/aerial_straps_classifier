import json
import os
from typing import Any, Dict

import numpy as np
import pandas as pd
from pandas import DataFrame
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def train_rf(X_train: DataFrame, y_train: np.ndarray, groups: np.ndarray, params: Dict[str, Any]) -> RandomForestClassifier:
    """
    Trains a RandomForestClassifier with optional hyperparameter optimization.

    Args:
    - X_train (DataFrame): The training features.
    - y_train (np.ndarray): The training target.
    - groups (np.ndarray): The groups for the training data.
    - params (dict): The parameters for the model.

    Returns:
    - model (RandomForestClassifier): The trained RandomForest model.
    """
    optimize_hyperparams = params.pop('optimize_hyperparams', False)
    
    if optimize_hyperparams:

        def objective(trial):
            # Hyperparameter search space
            n_estimators = trial.suggest_int('n_estimators', 100, 1000)
            max_depth = trial.suggest_int('max_depth', 1, 20)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
            max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
            
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                           min_samples_split=min_samples_split,
                                           min_samples_leaf=min_samples_leaf,
                                           max_features=max_features)
            
            score_metric = params.get('score_metric', 'accuracy')  # Defaulting to accuracy if score_metric isn't provided

            return -cross_val_score(model, X_train, y_train, cv=5, scoring=score_metric).mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=5)
        best_params = study.best_params
        print(f"Best hyperparameters found: {best_params}")
        model = RandomForestClassifier(**best_params)

        # Save the best hyperparameters
        models_dir = 'models/dev'
        model_dir = os.path.join(models_dir, 'rf')
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, 'best_hyperparameters.json'), 'w') as f:
            json.dump(best_params, f)

    else:
        model = RandomForestClassifier()  # You can provide more params here if needed

    # Fit the model on the entire dataset
    model.fit(X_train, y_train)

    return model
