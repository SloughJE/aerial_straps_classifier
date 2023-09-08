import logging
import json
import os
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame, Series
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

logger = logging.getLogger(__name__)


def generate_roc_curves_and_save(predictions_dir: str, model_type: str, data_type: str, label_encoder: Any, y_test: Series, y_prob: np.ndarray) -> None:
    """
    Generates ROC curves for each class and saves the visualization.

    Args:
    - predictions_dir (str): Directory to save the visualization.
    - model_type (str): Type of the model.
    - label_encoder (LabelEncoder): The label encoder.
    - y_test (Series): True labels.
    - y_prob (np.ndarray): Predicted probabilities.

    Returns:
    - None
    """
    n_classes = len(label_encoder.classes_)
    plt.figure(figsize=(10, 8))
    
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    for i in range(n_classes):
        try:
            # Convert y_test to binary format for ROC curve calculation
            y_binary = (y_test == i).astype(int)
            fpr[i], tpr[i], _ = roc_curve(y_binary, y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        except ValueError:
            # Skip this class if it's not present in either predictions or test data
            logger.info(f"Skipping class {label_encoder.inverse_transform([i])[0]} for ROC curve calculation.")
    
    for i in range(n_classes):
        if i in fpr:
            plt.plot(fpr[i], tpr[i], lw=2, label=f'ROC curve (AUC = {roc_auc[i]:.2f}) for class {label_encoder.inverse_transform([i])[0]}')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves for {model_type} {data_type} (Multiclass)')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(predictions_dir, model_type, f"{data_type}_roc_curves.png"))
    plt.close()


def generate_pr_curves_and_save(predictions_dir: str, model_type: str, data_type: str, label_encoder: Any, y_test: DataFrame, y_prob: np.ndarray) -> None:
    """
    Generates precision-recall curves for each class and saves the visualization.

    Args:
    - predictions_dir (str): Directory to save the visualization.
    - model_type (str): Type of the model.
    - label_encoder (LabelEncoder): The label encoder.
    - y_test (DataFrame): True labels.
    - y_prob (np.ndarray): Predicted probabilities.

    Returns:
    - None
    """
    n_classes = len(label_encoder.classes_)
    plt.figure(figsize=(10, 8))
    
    precision = {}
    recall = {}
    pr_auc = {}
    
    for i in range(n_classes):
        try:
            precision[i], recall[i], _ = precision_recall_curve(y_test == i, y_prob[:, i])
            pr_auc[i] = auc(recall[i], precision[i])
        except ValueError:
            # Skip this class if it's not present in either predictions or test data
            logger.info(f"Skipping class {label_encoder.inverse_transform([i])[0]} for PR curve calculation.")
    
    for i in range(n_classes):
        if i in precision:
            plt.plot(recall[i], precision[i], lw=2, label=f'PR curve (AUC = {pr_auc[i]:.2f}) for class {label_encoder.inverse_transform([i])[0]}')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curves for {model_type} {data_type} (Multiclass)')
    plt.legend(loc='lower left')
    plt.savefig(os.path.join(predictions_dir, model_type, f"{data_type}_precision_recall_curves.png"))
    plt.close()


def generate_visualizations_and_save_metrics(predictions_dir: str, model_type: str, data_type: str, 
                                             label_encoder: Any, y_test: DataFrame, y_pred: np.ndarray) -> None:
    """
    Generates confusion matrix, classification report, and saves visualizations and metrics.

    Args:
    - predictions_dir (str): Directory to save visualizations and metrics.
    - model_type (str): Type of the model.
    - label_encoder (LabelEncoder): The label encoder.
    - y_test (DataFrame): True labels.
    - y_pred (np.ndarray): Predicted labels.

    Returns:
    - None
    """
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix for {model_type} {data_type} Predictions")
    plt.colorbar()
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.xticks(range(len(label_encoder.classes_)), label_encoder.classes_, rotation=45)
    plt.yticks(range(len(label_encoder.classes_)), label_encoder.classes_)

    # Annotate cells with counts and adjust text color based on background
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            cell_value = cm[i, j]
            text_color = "white" if cell_value > cm.max() / 2 else "black"
            plt.text(j, i, str(cell_value), ha="center", va="center", color=text_color)

    plt.tight_layout()
    plt.savefig(os.path.join(predictions_dir, model_type, f"{data_type}_confusion_matrix.png"))
    plt.close()

    # Save classification report
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True, 
                                   labels=label_encoder.transform(label_encoder.classes_))

    report_path = os.path.join(predictions_dir, model_type, f"{data_type}_classification_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f)

    logger.info(f"{data_type} metrics saved as JSON.")


def generate_feature_importance_visualization(model: Any, feature_names: List[str], save_path: str) -> None:
    """
    Generates and saves a feature importance visualization for the model.

    Args:
    - model (Any): The trained model that supports feature importance.
    - feature_names (List[str]): List of feature names used in the model.
    - save_path (str): Path to save the feature importance visualization.

    Returns:
    - None
    """
    if hasattr(model, "feature_importances_"):
        feature_importances = model.feature_importances_
        
        # Sort feature importances and feature names in descending order of importance
        sorted_indices = np.argsort(feature_importances)
        sorted_feature_importances = feature_importances[sorted_indices]
        sorted_feature_names = [feature_names[i] for i in sorted_indices]

        # Create a bar chart to visualize feature importance
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(sorted_feature_importances)), sorted_feature_importances, align="center")
        plt.yticks(range(len(sorted_feature_importances)), sorted_feature_names)
        plt.xlabel("Feature Importance")
        plt.ylabel("Feature")
        plt.title("Feature Importance")
        plt.tight_layout()

        # Save the visualization
        plt.savefig(save_path)
        plt.close()
        logger.info("Feature importance visualization saved.")

    else:
        logger.info("Feature importance visualization is not supported for this model.")
