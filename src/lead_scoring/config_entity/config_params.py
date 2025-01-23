from dataclasses import dataclass 
from pathlib import Path
from typing import Dict, List, Any


# Data ingestion 
@dataclass
class DataIngestionConfig:
    """
    Configuration for data ingestion process.

    """
    config_data: dict 

@dataclass
class DataValidationConfig:
    """
    Configuration for data validation process.

    """
    root_dir: Path
    val_status: str
    data_dir: Path
    all_schema: Dict[str, Any]
    critical_columns: List[str]

@dataclass
class DataTransformationConfig:
    """
    Configuration for data transformation process.

    """
    root_dir: Path
    data_path: Path
    numerical_cols: frozenset
    categorical_cols: frozenset
    target_col : str
    random_state: int  


@dataclass
class ModelTrainerConfig:
    root_dir: Path
    train_features_path: Path
    train_targets_path: Path
    model_name: str
    model_params: Dict[str, Any]
    project_name: str
    val_features_path: Path
    val_targets_path: Path


@dataclass
class ModelEvaluationConfig:
    root_dir: Path
    val_feature_path: Path
    val_targets_path: Path
    model_path: Path
    eval_scores_path: Path
    threshold_adjustment: Path
    precision_recall_path: Path


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