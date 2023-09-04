import os
import joblib
import json
import numpy as np
import pandas as pd
from pandas import DataFrame

import optuna
from typing import Tuple, Dict, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator
from xgboost import XGBClassifier

from .evaluation_metrics import (
    generate_visualizations_and_save_metrics,
    generate_roc_curves_and_save,
    generate_pr_curves_and_save,
    generate_feature_importance_visualization
)


# not implemented, need more data for this
class FileNameBasedKFold:
    def __init__(self, n_splits=5):
        self.n_splits = n_splits
    
    def split(self, X, y, groups):
        unique_files = groups.unique()
        kf = KFold(n_splits=self.n_splits)
        for train_files_idx, test_files_idx in kf.split(unique_files):
            train_files = unique_files[train_files_idx]
            test_files = unique_files[test_files_idx]
            train_idx = X.index[groups.isin(train_files)]
            test_idx = X.index[groups.isin(test_files)]
            yield train_idx, test_idx
    
    def get_n_splits(self, X, y=None, groups=None):
        return self.n_splits
    

def custom_cross_val_score(model, X, y, groups, cv, scoring_func):
    scores = []
    for train_idx, test_idx in cv.split(X, y, groups):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_test_fold = X.iloc[test_idx]
        y_test_fold = y.iloc[test_idx]
        
        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_test_fold)
        score = scoring_func(y_test_fold, y_pred)
        scores.append(score)
    return np.array(scores)


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
    
    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 1, 20),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1),
        }
        model = XGBClassifier(**param)
        
        score_metric = params.get('score_metric', 'accuracy')  # Defaulting to accuracy if score_metric isn't provided
        return -cross_val_score(model, X_train, y_train, cv=3, scoring=score_metric).mean()

    if optimize_hyperparams:
        print("Optimizing hyperparameters")
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=5)
        best_params = study.best_params
        print(f"Best hyperparameters found: {best_params}")
        model = XGBClassifier(**best_params)

        # Save the best hyperparameters
        models_dir = 'models/dev'
        model_dir = os.path.join(models_dir, 'xgb')
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, 'best_hyperparameters.json'), 'w') as f:
            json.dump(best_params, f)

    else:
        model = XGBClassifier() # later can: XGBClassifier(**params)
    
    # Fit model on the entire dataset
    model.fit(X_train, y_train)

    return model


def train_linear_regression(X_train, y_train, groups, params):
    model = LinearRegression(**params)
    model.fit(X_train, y_train)
    return model


# Define the model mapping
MODEL_MAPPER = {
    'rf': train_rf,
    'xgb': train_xgb,
    'linear_regression': train_linear_regression,
    # 'svm': train_svm,
    # 'neural_net': train_neural_net,
    # ... add other model functions here ...
}

def split_train_test(df: DataFrame, params: Dict[str, Any]) -> Tuple[DataFrame, DataFrame]:
    """
    Splits the data into train/test sets based on the filename.

    Args:
    - df (DataFrame): The main dataframe containing the data.
    - params (dict): Dictionary containing the following key-value pairs:
        - final_features_filepath (str): Path to the features CSV file.
        - test_size (float): Fraction of data to be used as the test set.
        - random_state (int, optional): Random seed for reproducibility. Defaults to None if not provided.

    Returns:
    - train_df (DataFrame): Training data.
    - test_df (DataFrame): Test data.
    """
    test_size = params['test_size']

    # Get unique filenames
    unique_files = df['filename'].unique()

    # Split the unique filenames into training and test sets
    train_files, test_files = train_test_split(unique_files, test_size=test_size, random_state=42)

    # Filter the main dataframe based on the train/test filenames
    train_df = df[df['filename'].isin(train_files)]
    test_df = df[df['filename'].isin(test_files)]
    
    # Ensure the index is from 0 to len(df), will need for CV splitting based on filename
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    # Print out the number of files and frames for both sets
    print(f"Training set: {len(train_files)} files with {len(train_df)} frames.")
    print(f"Test set: {len(test_files)} files with {len(test_df)} frames.")

    return train_df, test_df


def encode_labels(train_df: DataFrame, test_df: DataFrame, target_column: str) -> Tuple[DataFrame, DataFrame, LabelEncoder]:
    """
    Encodes the target labels to numerical values.
    
    Args:
    - train_df (DataFrame): Training data.
    - test_df (DataFrame): Test data.
    - target_column (str): The name of the target column in the dataframe.
    
    Returns:
    - train_df (DataFrame): Training data with encoded target.
    - test_df (DataFrame): Test data with encoded target.
    - le (LabelEncoder): Fitted label encoder.
    """
    le = LabelEncoder()
    train_df[target_column] = le.fit_transform(train_df[target_column])
    test_df[target_column] = le.transform(test_df[target_column])  # Use transform, not fit_transform for the test set
    return train_df, test_df, le


def preprocess_data(train_df: DataFrame, test_df: DataFrame, 
                    target_column: str) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame, DataFrame]:
    """
    Preprocesses the training and testing data by separating the features and target variable and extracting the groups.

    Args:
    - train_df (DataFrame): The training data.
    - test_df (DataFrame): The testing data.
    - target_column (str): The name of the target column.

    Returns:
    - X_train (DataFrame): The training features.
    - y_train (DataFrame): The training target.
    - X_test (DataFrame): The testing features.
    - y_test (DataFrame): The testing target.
    - groups (DataFrame): The groups for the training data.
    """
    X_train = train_df.drop(columns=[target_column, 'frame_number', 'filename'])
    y_train = train_df[target_column]
    X_test = test_df.drop(columns=[target_column, 'frame_number', 'filename'])
    y_test = test_df[target_column]
    groups = train_df['filename']
    return X_train, y_train, X_test, y_test, groups


def train_model(X_train: DataFrame, y_train: DataFrame, groups: DataFrame, 
                params: Dict[str, Any]) -> BaseEstimator:
    """
    Trains a model using the provided training data and parameters.

    Args:
    - X_train (DataFrame): The training features.
    - y_train (DataFrame): The training target.
    - groups (DataFrame): The groups for the training data.
    - params (dict): The parameters for the model.

    Returns:
    - model (BaseEstimator): The trained model.
    """
    model_type = params['model_type']
    model = MODEL_MAPPER[model_type](X_train, y_train, groups, params)
    return model


def save_model(model: BaseEstimator, params: Dict[str, Any], label_encoder: LabelEncoder) -> None:
    """
    Saves the trained model and label encoder to disk.

    Args:
    - model (BaseEstimator): The trained model.
    - params (dict): The parameters for the model.
    - label_encoder (LabelEncoder): The label encoder.

    Returns:
    - None
    """
    model_type = params['model_type']
    models_dir = 'models/dev'
    model_dir = os.path.join(models_dir, model_type)
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, f'{model_type}_model.pkl'))
    joblib.dump(label_encoder, os.path.join(model_dir, 'label_encoder.pkl'))


def predict_and_evaluate(model: BaseEstimator, X_train: DataFrame, y_train: DataFrame, X_test: DataFrame, 
                         y_test: DataFrame, params: Dict[str, Any]) -> Tuple[float, float]:
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
    target_column = params['target_column']

    # Helper function to predict, evaluate, and save predictions
    def predict_and_save(model: BaseEstimator, X: DataFrame, y: DataFrame, 
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

        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)
        accuracy = accuracy_score(y, y_pred)

        df = X.copy()
        df[target_column] = y
        df['predicted_label'] = y_pred
        df['predicted_label'] = label_encoder.inverse_transform(y_pred)

        for i, class_name in enumerate(label_encoder.classes_):
            df[f'probability_{class_name}'] = y_prob[:, i]
        
        df.to_csv(save_path, index=False)
        
        # Generate and save visualizations and metrics
        generate_visualizations_and_save_metrics(predictions_dir, model_type, label_encoder, y, y_pred)
        generate_roc_curves_and_save(predictions_dir, model_type, label_encoder, y, y_prob)
        generate_pr_curves_and_save(predictions_dir, model_type, label_encoder, y, y_prob)

        return accuracy

    train_accuracy = predict_and_save(model, X_train, y_train, params, "train")
    test_accuracy = predict_and_save(model, X_test, y_test, params, "test")

    return train_accuracy, test_accuracy


def train_and_evaluate_model(train_df: DataFrame, test_df: DataFrame, params: Dict[str, Any]) -> BaseEstimator:
    """
    Trains and evaluates a model using the provided training and testing data.

    Args:
    - train_df (DataFrame): The training data.
    - test_df (DataFrame): The testing data.
    - params (dict): The parameters for the model.

    Returns:
    - model (BaseEstimator): The trained model.
    """
    target_column = params['target_column']
    label_encoder = params['label_encoder']

    X_train, y_train, X_test, y_test, groups = preprocess_data(train_df, test_df, target_column)
    model = train_model(X_train, y_train, groups, params)
    train_accuracy, test_accuracy = predict_and_evaluate(model, X_train, y_train, X_test, y_test, params)
    save_model(model, params, label_encoder)

    feature_names = list(X_train.columns)
    save_path = os.path.join(params['predictions_dir'], params['model_type'], "feature_importance.png")
    generate_feature_importance_visualization(model, feature_names, save_path)

    print(f"Train accuracy: {train_accuracy:.2f}")
    print(f"Test accuracy: {test_accuracy:.2f}")
    return model

def train_model_pipeline(params: Dict[str, Any]) -> BaseEstimator:
    """
    Manages the process of splitting the data into train/test sets, encoding the target labels,
    and training a specified classifier.

    Args:
    - params (dict): Dictionary containing the following key-value pairs:
        - final_features_filepath (str): Path to the features CSV file.
        - test_size (float): Fraction of data to be used as the test set. (Optional. Defaults to None)
        - model_type (str): The type of classifier model to train (e.g., 'rf', 'xgb').
        - model_params (dict): Parameters specific to the chosen model.
        - predictions_dir (str): Path to save the predicted labels and probabilities.
        - target_column (str): The name of the target column in the dataframe.

    Returns:
    - model (BaseEstimator): The trained classifier model.
    """

    # load data
    final_features_filepath = params['final_features_filepath']
    df = pd.read_csv(final_features_filepath)

    # Split data into training and testing sets
    train_df, test_df = split_train_test(df, params)

    # Encode the target labels
    target_column = params['target_column']
    train_df, test_df, label_encoder = encode_labels(train_df, test_df, target_column)

    # Add the label encoder to the parameters to be accessible within train_and_evaluate_model
    params['label_encoder'] = label_encoder
    # Train the specified classifier and return the model
    print(f"training {params['model_type']} model")
    model = train_and_evaluate_model(train_df, test_df, params)
    print("Trained and saved model successfully!")

    return model