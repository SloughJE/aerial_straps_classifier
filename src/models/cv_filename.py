import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import optuna
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, StratifiedKFold

# if we get error related to target variable expected 0,1,2 but got 1,3,4, or something
# we just need more files, it's cause training CV has targets 0,1,2, best 'test' in CV fold has 1,3,4 in target.

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
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=3)
        best_params = study.best_params
        print(f"best hyperparameters found: {best_params}")
        model = XGBClassifier(**best_params)
        model.fit(X_train, y_train)

    else:
        model = XGBClassifier(**params)
    
    if not optimize_hyperparams:
        model.fit(X_train, y_train)
    return model



##########

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

