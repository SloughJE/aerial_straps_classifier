import os
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score

from .evaluation_metrics import (
    generate_visualizations_and_save_metrics,
    generate_roc_curves_and_save,
    generate_pr_curves_and_save,
)

def predict_and_evaluate(model: BaseEstimator, X_train: DataFrame, y_train: pd.Series, 
                         X_test: DataFrame, y_test: pd.Series,
                         params: Dict[str, Any]) -> Tuple[float, float]:
  
    """
    Predicts and evaluates the model on the training and testing data.

    Args:
    - model (BaseEstimator): The trained model.
    - X_train (DataFrame): The training features.
    - y_train (DataFrame): The training target.
    - X_test (DataFrame): The testing features.
    - y_test (DataFrame): The testing target.
    - params (dict): The parameters for the model.

    Returns:
    - train_accuracy (float): The accuracy of the model on the training data.
    - test_accuracy (float): The accuracy of the model on the testing data.
    """
    predictions_dir = params['predictions_dir']
    model_type = params['model_type']

    # Helper function to predict, evaluate, and save predictions
    def predict_and_save(model: BaseEstimator, X: DataFrame, y: pd.Series, 
                        params: Dict[str, Any], data_type: str) -> float:
        """
        Predicts, evaluates, and saves the predictions of the model on the provided data.

        Args:
        - model (BaseEstimator): The trained model.
        - X (DataFrame): The features.
        - y (DataFrame): The target.
        - params (dict): The parameters for the model.
        - data_type (str): The type of the data ('train' or 'test').

        Returns:
        - accuracy (float): The accuracy of the model on the provided data.
        """
        label_encoder = params['label_encoder']

        save_path = os.path.join(predictions_dir, model_type, f"{data_type}_predictions.csv")
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        df = X.copy()
        X = X.drop(columns=['filename', 'frame_number'], errors='ignore')

        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)
        accuracy = accuracy_score(y, y_pred)

        # Adding original labels and decoded labels
        df['original_label'] = label_encoder.inverse_transform(y)
        df['predicted_label'] = label_encoder.inverse_transform(y_pred)

        # Keeping the numeric labels for further analysis
        df['numeric_label'] = y
        df['numeric_predicted_label'] = y_pred

        for i, class_name in enumerate(label_encoder.classes_):
            df[f'probability_{class_name}'] = y_prob[:, i]

        df.to_csv(save_path, index=False)
        # Generate and save visualizations and metrics
        generate_visualizations_and_save_metrics(predictions_dir, model_type,data_type, label_encoder, y, y_pred)
        generate_roc_curves_and_save(predictions_dir, model_type, data_type, label_encoder, y, y_prob)
        generate_pr_curves_and_save(predictions_dir, model_type, data_type, label_encoder, y, y_prob)

        return accuracy

    train_accuracy = predict_and_save(model, X_train, y_train, params, "train")
    test_accuracy = predict_and_save(model, X_test, y_test, params, "test")


    return train_accuracy, test_accuracy

