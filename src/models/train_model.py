import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import mlflow
from pandas import DataFrame
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from mlflow.data.pandas_dataset import PandasDataset

from .xgboost_model import train_xgb
from .label_encoder import CustomLabelEncoder

logger = logging.getLogger(__name__)


# Define the model mapping
MODEL_MAPPER = {
    #'rf': train_rf,
    'xgb': train_xgb,
    # 'svm': train_svm,
    # 'neural_net': train_neural_net,
}


def convert_spatial_features_to_categorical(df: DataFrame) -> DataFrame:
    """
    Convert all spatial features in the DataFrame to the 'categorical' data type.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame containing spatial features with "spatial_" prefix in the column names.
    
    Returns:
    - pd.DataFrame: The DataFrame with spatial features converted to 'categorical' data type.
    """
    logger.info("Converting spatial columns to categorical")
    
    # Find the spatial columns
    spatial_columns = df.filter(regex='^spatial_', axis=1).columns
    
    # Convert all the spatial columns to 'category' type using the apply method
    df[spatial_columns] = df[spatial_columns].apply(lambda col: col.astype('category'))
    
    return df


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

    logger.info(f"Training set: {len(train_files)} files with {len(train_df)} frames.")
    logger.info(f"Test set: {len(test_files)} files with {len(test_df)} frames.")

    return train_df, test_df


def encode_labels(train_df, test_df, target_column):
    """
    Encodes the target labels to numerical values.
    
    Args:
    - train_df (DataFrame): Training data.
    - test_df (DataFrame): Test data.
    - target_column (str): The name of the target column in the dataframe.
    
    Returns:
    - train_df (DataFrame): Training data with encoded target.
    - test_df (DataFrame): Test data with encoded target.
    - encoder (CustomLabelEncoder): Fitted label encoder.
    """
    encoder = CustomLabelEncoder()
    train_df[target_column] = encoder.fit_transform(train_df[target_column])
    test_df[target_column] = encoder.transform(test_df[target_column])
    return train_df, test_df, encoder



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
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]
    groups = train_df['filename']
    return X_train, y_train, X_test, y_test, groups


def train_model(X_train: DataFrame, y_train: DataFrame, X_test: DataFrame, y_test: DataFrame,groups: DataFrame, 
                params: Dict[str, Any]) -> BaseEstimator:
    """
    Trains a model using the provided training data and parameters.

    Args:
    - X_train (DataFrame): The training features.
    - y_train (DataFrame): The training target.
    - groups (DataFrame): The groups for the training data.
    - params (dict): The parameters for the model.

    Returns:
    - None
    """
    # Drop unnecessary columns before training
    X_train = X_train.drop(columns=['filename', 'frame_number'])
    model_type = params['model_type']
    MODEL_MAPPER[model_type](X_train, y_train, X_test,y_test, groups, params)


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

    X_train, y_train, X_test, y_test, groups = preprocess_data(train_df, test_df, target_column)
    train_model(X_train, y_train, X_test, y_test, groups, params)
    

def log_mlflow_metadata(params, df, train_df, test_df):

    # Log the model_dev parameters
    for key, value in params.items():
        mlflow.log_param(key, value)

    dataset: PandasDataset = mlflow.data.from_pandas(df, targets=params['target_column'], name="Dev Dataset")
    mlflow.log_input(dataset, context="Dev: Full Dataset")
    mlflow.set_tag("dataset.version", "v0.1")
    mlflow.set_tag("dataset.source", params['final_features_filepath'])
    test_size = params.get('test_size', 0.2)
    train_size = 1.0 - test_size
    split_info = f"{train_size*100:.0f}-{test_size*100:.0f}"
    mlflow.set_tag("dataset.split", split_info)
    mlflow.set_tag("dataset.preprocessing", "removing back lever and replacing l-hang")

    train_dataset: PandasDataset = mlflow.data.from_pandas(train_df, targets=params['target_column'], name="Train Dataset")
    test_dataset: PandasDataset = mlflow.data.from_pandas(test_df, targets=params['target_column'], name="Test Dataset")
    mlflow.log_input(train_dataset, context="Dev: train dataset")
    mlflow.log_input(test_dataset, context="Dev: test dataset")


def train_model_pipeline(params: Dict[str, Any]) -> BaseEstimator:
    """
    Manages the process of splitting the data into train/test sets, encoding the target labels,
    and training a specified classifier. Logs dataset and training details to MLflow.

    Args:
    - params (dict): Dictionary containing the following key-value pairs:
        - final_features_filepath (str): Path to the features CSV file.
        - test_size (float): Fraction of data to be used as the test set. (Optional. Defaults to None)
        - model_type (str): The type of classifier model to train (e.g., 'rf', 'xgb').
        - model_params (dict): Parameters specific to the chosen model.
        - predictions_dir (str): Path to save the predicted labels and probabilities.
        - target_column (str): The name of the target column in the dataframe.

    Returns:
    - None
    """
    mlflow.set_experiment(params['MLflow_config']['experiment_name']) 

    with mlflow.start_run(run_name=params['MLflow_config']['run_names']['main']):
        # load data
        final_features_filepath = params['final_features_filepath']
        df = pd.read_csv(final_features_filepath)

        # Preprocessing steps
        print(f"removing back lever and replacing l-hang")
        df = df[df.label!='back lever']
        df['label'] = df['label'].replace('l-hang', 'other pose or transition')
        df = convert_spatial_features_to_categorical(df)

        # Split data into training and testing sets
        train_df, test_df = split_train_test(df, params)

        log_mlflow_metadata(params, df, train_df, test_df)

        # Encode the target labels
        target_column = params['target_column']
        train_df, test_df, label_encoder = encode_labels(train_df, test_df, target_column)

        # Add the label encoder to the parameters to be accessible within train_and_evaluate_model
        params['label_encoder'] = label_encoder
        # Train the specified classifier and return the model
        logger.info(f"training {params['model_type']} model")
        train_and_evaluate_model(train_df, test_df, params)
        logger.info("Trained and saved model successfully!")

