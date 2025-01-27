import sys
sys.path.append(r"C:\Users\lilwa\OneDrive\Desktop\moringa\Edel\website-leads")

from pathlib import Path
from dataclasses import dataclass
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, auc,
                             precision_recall_curve, f1_score, precision_score,
                             recall_score, roc_auc_score, accuracy_score)
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import wandb


from src.lead_scoring.exception import CustomException
from src.lead_scoring.logger import logger
from src.lead_scoring.constants import *
from src.lead_scoring.utils.commons import *
wandb.require("core")



@dataclass
class ModelValidationConfig:
    root_dir: Path
    test_feature_path: Path
    test_targets_path: Path
    model_path: Path
    validation_scores_path: Path
    classification_report_path: Path
    confusion_matrix_path: Path
    roc_curve_path: Path
    pr_curve_path: Path
    precision_recall_path: Path
    optimal_threshold: float
    project_name: str


class ConfigurationManager:
    def __init__(self, model_validation_config: str = MODEL_VALIDATION_CONFIG_FILEPATH):
        self.validation_config = read_yaml(model_validation_config)
        artifacts_root = self.validation_config.artifacts_root
        create_directories([artifacts_root])

    def get_model_validation_config(self) -> ModelValidationConfig:
        logger.info("Getting model validation configuration")

        val_config = self.validation_config.model_validation
        create_directories([val_config.root_dir])

        return ModelValidationConfig(
            root_dir=Path(val_config.root_dir),
            test_feature_path=Path(val_config.test_feature_path),
            test_targets_path=Path(val_config.test_targets_path),
            model_path=Path(val_config.model_path),
            validation_scores_path=Path(val_config.validation_scores_path),
            classification_report_path=Path(val_config.classification_report_path),
            confusion_matrix_path=Path(val_config.confusion_matrix_path),
            roc_curve_path=Path(val_config.roc_curve_path),
            pr_curve_path=Path(val_config.pr_curve_path),
            precision_recall_path=Path(val_config.precision_recall_path),
            optimal_threshold=val_config.optimal_threshold,
            project_name = val_config.project_name
        )


class ModelValidation:
    def __init__(self, config: ModelValidationConfig):
        self.config = config
        self.model = None  # Initialize model attribute
        self.X_test = None
        self.y_test = None
        self.run = None

    def load_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """
        Logs the start of model validation data loading and retrieves
        paths for test features and test targets from the configuration.
        """
        logger.info("Loading model validation data")
        try:
            test_features = self.config.test_feature_path
            test_targets = self.config.test_targets_path

            X_test = joblib.load(test_features)
            y_test = pd.read_csv(test_targets)

            # Convert y_test to a Series if it's a DataFrame
            if isinstance(y_test, pd.DataFrame):
                y_test = y_test.squeeze()

            logger.info("Model validation data loaded successfully")
            return X_test, y_test

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise CustomException(f"Error loading data: {e}")

    def load_model(self) -> object:
        logger.info("Loading model")
        try:
            model = joblib.load(self.config.model_path)
            logger.info("Model loaded successfully")
            return model

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise CustomException(f"Error loading model: {e}")

    def initiate_stratified_cross_validation(self, X_test, y_test):
        """
        Performs stratified K-Fold cross-validation and returns the average F1 score.
        """
        logger.info("Initiating stratified cross-validation")
        # Initialize the Stratified K-Fold cross-validator
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Perform Stratified K-Fold Cross-Validation
        fold_f1_scores = []
        for fold, (train_index, val_index) in enumerate (skf.split(X_test, y_test)):
            X_train_fold, X_val_fold = X_test[train_index], X_test[val_index]
            y_train_fold, y_val_fold = y_test.iloc[train_index], y_test.iloc[val_index]
            
            # Train the model on the training fold
            self.model.fit(X_train_fold, y_train_fold)
            
            # Validate the model on the validation fold
            y_val_pred = self.model.predict(X_val_fold)
            
            # Evaluate the model
            fold_f1 = f1_score(y_val_fold, y_val_pred, average='macro')
            fold_f1_scores.append(fold_f1)
            logger.info(f"Fold {fold+1} Validation Macro F1-Score: {fold_f1}")

        # Print average Macro F1-Score across all folds
        avg_f1 = sum(fold_f1_scores) / len(fold_f1_scores)
        logger.info(f"Average Cross-Validation Macro F1-Score: {avg_f1}")
        
        # Use the whole X_test and y_test to predict with the model
        y_pred_all_folds = self.model.predict(X_test)
        class_report_all_folds = classification_report(y_test, y_pred_all_folds, zero_division=0)
        
        with open(self.config.classification_report_path, "w") as f:
           f.write(class_report_all_folds)

        if self.run:
            self.run.log({"avg_cv_f1": avg_f1})

    def make_probability_predictions(self, X_test, y_test):
        """Generates probability predictions on the test set and assigns lead scores."""

        logger.info("Making probability predictions and assigning lead scores")
        try:
            # Predict probabilities
            y_prob = self.model.predict_proba(X_test)[:, 1]

            # Scale lead scores between 0 and 100
            lead_scores = (y_prob * 100).round().astype(int)

            # Convert probabilities into binary predictions using the cutoff threshold
            y_pred_threshold = (y_prob >= self.config.optimal_threshold).astype(int)
        
            f1 = f1_score(y_test, y_pred_threshold)
            precision = precision_score(y_test, y_pred_threshold)
            recall = recall_score(y_test, y_pred_threshold)
            accuracy = accuracy_score(y_test, y_pred_threshold)

            # Save the metric in a json file
            metrics = {
              "f1_score": f1,
              "precision": precision,
              "recall": recall,
              "accuracy": accuracy
             }
            with open(self.config.validation_scores_path, 'w') as f:
                json.dump(metrics, f)
            logger.info("Probability predictions, lead scores, and evaluation metrics saved successfully.")
            
            if self.run:
                self.run.log(metrics)

            return y_test, y_prob


        except Exception as e:
            logger.error(f"Error generating probability predictions: {e}")
            raise CustomException(f"Error generating probability predictions: {e}")


    def evaluate_model(self, y_true, y_prob):
        """
        Calculates and saves evaluation metrics using a given threshold.
        """

        logger.info("Evaluating Model with custom Threshold")
        # Apply the threshold to get binary predictions
        y_pred_threshold = (np.array(y_prob) >= self.config.optimal_threshold).astype(int)

        # Calculate metrics
        conf_matrix = confusion_matrix(y_true, y_pred_threshold)
        class_report = classification_report(y_true, y_pred_threshold, zero_division = 0)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        prec, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, prec)

        # Save metrics
        self.save_metrics(conf_matrix, class_report, fpr, tpr, roc_auc, prec, recall, pr_auc, y_true, y_prob)

        logger.info("Evaluation metrics saved successfully")


    def save_metrics(self, conf_matrix, class_report, fpr, tpr, roc_auc, prec, recall, pr_auc, y_true, y_prob):
        """
        Saves evaluation metrics to specified paths and logs to W&B.
        """
        # Save confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.savefig(self.config.confusion_matrix_path)
        plt.close()

        # Save ROC Curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right")
        plt.savefig(self.config.roc_curve_path)
        plt.close()

        # Save precision-recall curve
        plt.figure(figsize=(8, 6))
        plt.plot(recall, prec, color="darkblue", lw=2, label=f"PR AUC = {pr_auc:.2f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend(loc="lower left")
        plt.savefig(self.config.pr_curve_path)
        plt.close()

        # Save precision and recall to json file
        precision_recall_data = {
            'precision': prec.tolist() if isinstance(prec, np.ndarray) else prec,
            'recall': recall.tolist() if isinstance(recall, np.ndarray) else recall,
            'y_true': y_true,  
            'y_prob': y_prob.tolist() if isinstance(y_prob, np.ndarray) else y_prob
        }

        with open(self.config.precision_recall_path, 'w') as f:
            json.dump(precision_recall_data, f)

        # Log to W&B if W&B is initialized
        if self.run:
            self.run.log({
              "confusion_matrix": wandb.Image(str(self.config.confusion_matrix_path)),
              "roc_curve": wandb.Image(str(self.config.roc_curve_path)),
              "pr_curve": wandb.Image(str(self.config.pr_curve_path)),
                "roc_auc": roc_auc,
                 "pr_auc": pr_auc
            })


    def run_validation(self):
        """
        Runs the complete model validation process.
        """
        try:
            # Initialize W&B run
            self.run = wandb.init(project=self.config.project_name, config=self.config.__dict__)

            X_test, y_test = self.load_data()
            model = self.load_model()
            self.model = model
            self.initiate_stratified_cross_validation(X_test, y_test)
            
            y_true, y_prob = self.make_probability_predictions(X_test, y_test)
            
            # Convert y_true to a list
            self.evaluate_model(y_true.tolist(), y_prob)

        except Exception as e:
            logger.error(f"Error during model validation: {e}")
            raise CustomException(f"Error during model validation: {e}", error_details=str(e))

        finally:
            if self.run:
                self.run.finish()


if __name__ == "__main__":
    try:
        config = ConfigurationManager()
        model_validation_config = config.get_model_validation_config()
        model_validation = ModelValidation(config=model_validation_config)
        model_validation.run_validation()
    except Exception as e:
        logger.error(f"Error during the model validation main process: {e}")
        raise CustomException(f"Error during the model validation main process: {e}", str(e))