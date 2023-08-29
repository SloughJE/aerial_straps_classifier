import os
import numpy as np
import pandas as pd
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

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


def train_rf(X_train, y_train, groups, params):
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model


def train_xgb(X_train, y_train, groups, params):

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
        
        custom_cv = FileNameBasedKFold(n_splits=2)
        scores = custom_cross_val_score(model, X_train, y_train, groups, cv=custom_cv, scoring_func=accuracy_score)
        return 1 - np.mean(scores)
    
    if optimize_hyperparams:
        print("optimizing hyperparameters")
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=3)
        best_params = study.best_params
        print(f"best hyperparameters found: {best_params}")
        model = XGBClassifier(**best_params)
    else:
        model = XGBClassifier(**params) # later can: XGBClassifier(**params)
    
    if not optimize_hyperparams:
        model.fit(X_train, y_train)
    return model



def train_linear_regression(X_train, y_train, groups, params):
    model = LinearRegression(**params)
    model.fit(X_train, y_train)
    return model


MODEL_MAPPER = {
    'rf': train_rf,
    'xgb': train_xgb,
    'linear_regression': train_linear_regression,
    # 'svm': train_svm,
    # 'neural_net': train_neural_net,
    #... add other model functions here ...
}

def split_train_test(df,params: dict):
    """
    Splits the data into train/test sets based on the filename.

    Args:
    - params (dict): Dictionary containing the following key-value pairs:
        - final_features_filepath (str): Path to the features CSV file.
        - test_size (float): Fraction of data to be used as the test set. (Optional. Defaults to None)
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
    
    # ensure the index is from 0 to len(df), will need for CV splitting based on filename
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    # Print out the number of files and frames for both sets
    print(f"Training set: {len(train_files)} files with {len(train_df)} frames.")
    print(f"Test set: {len(test_files)} files with {len(test_df)} frames.")

    return train_df, test_df


def train_and_evaluate_model(train_df: pd.DataFrame, test_df: pd.DataFrame, params: dict):
    """
    Train a classifier using the provided training data and evaluate it on the test data.
    
    Args:
    - train_df (DataFrame): Training data.
    - test_df (DataFrame): Test data.
    - params (dict): Dictionary containing model parameters and other configs.
    
    Returns:
    - model (Classifier): The trained classifier model.
    """
    target_column = params['target_column']
    model_type = params['model_type']
    model_params = params.get('model_params', {})
    predictions_dir = params['predictions_dir']
    
    # drop only video frame from train_df, also add filename to y_train
    # when implementing CV split based on filename, don't drop filename here
    X_train = train_df.drop(columns=[target_column, 'video_frame','filename'])
    y_train = train_df[target_column]
  
  
    # drop video_frame and filename from test
    X_test = test_df.drop(columns=[target_column, 'video_frame', 'filename'])
    y_test = test_df[target_column]

    # Initialize and train the classifier
    groups = train_df['filename']
    model = MODEL_MAPPER[model_type](X_train, y_train, groups, params)
    
    # Helper function to predict, evaluate, and save predictions
    def predict_and_save(model, X, y, params, data_type):
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
        return accuracy

    # Predict on training data for evaluation
    train_accuracy = predict_and_save(model, X_train, y_train, params, "train")
    print(f"Train accuracy: {train_accuracy:.2f}")

    # Predict and evaluate on test data
    test_accuracy = predict_and_save(model, X_test, y_test, params, "test")
    print(f"Test accuracy: {test_accuracy:.2f}")

    return model



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
    - le (LabelEncoder): Fitted label encoder.
    """
    le = LabelEncoder()
    train_df[target_column] = le.fit_transform(train_df[target_column])
    test_df[target_column] = le.transform(test_df[target_column])  # Use transform, not fit_transform for test set
    return train_df, test_df, le


def train_model_pipeline(params: dict):
    """
    Manages the process of splitting the data into train/test sets and training a specified classifier.

    Args:
    - params (dict): Dictionary containing the following key-value pairs:
        - final_features_filepath (str): Path to the features CSV file.
        - test_size (float): Fraction of data to be used as the test set. (Optional. Defaults to None)
        - model_type (str): The type of classifier model to train (e.g., 'rf', 'xgb').
        - model_params (dict): Parameters specific to the chosen model.
        - predictions_dir (str): Path to save the predicted labels and probabilities.
        - target_column (str): The name of the target column in the dataframe.

    Returns:
    - model (Classifier): The trained classifier model.
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
    print("Trained model successfully!")
    return model
