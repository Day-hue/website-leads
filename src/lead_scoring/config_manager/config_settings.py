
import pymongo
import os
import sys

from typing import Dict

from dotenv import load_dotenv
from src.lead_scoring.exception import CustomException
from src.lead_scoring.logger import logger
from src.lead_scoring.constants import *
from src.lead_scoring.utils.commons import read_yaml, create_directories
from src.lead_scoring.config_entity.config_params import DataIngestionConfig
from src.lead_scoring.config_entity.config_params import DataValidationConfig
from src.lead_scoring.config_entity.config_params import DataTransformationConfig
from src.lead_scoring.config_entity.config_params import ModelTrainerConfig
from src.lead_scoring.config_entity.config_params import ModelEvaluationConfig
from src.lead_scoring.config_entity.config_params import ModelValidationConfig

# Load environment variables from .env file
load_dotenv()

class ConfigurationManager:
    def __init__(
            self, 
            data_ingestion_config: str = DATA_INGESTION_CONFIG_FILEPATH,
            data_validation_config: Path = DATA_VALIDATION_CONFIG_FILEPATH,
            schema_config: Path = SCHEMA_CONFIG_FILEPATH,
            data_preprocessing_config: str = DATA_TRANSFORMATION_CONFIG_FILEPATH,
            model_training_config: Path = MODEL_TRAINER_CONFIG_FILEPATH, 
            model_params_config: Path = PARAMS_CONFIG_FILEPATH,
            hyperparameter_config: Path = HYPERPARAMETER_SEARCH_CONFIG_FILEPATH,
            model_evaluation_config: str = MODEL_EVALUATION_CONFIG_FILEPATH,
            model_validation_config: str = MODEL_VALIDATION_CONFIG_FILEPATH         
      
                 ) -> None:
        """
        Initializes the ConfigurationManager.

        Args:
            data_ingestion_config (str): Path to the data ingestion configuration file.
               
        """
        try:
            logger.info(f"Initializing ConfigurationManager with config files")
            
            self.ingestion_config = read_yaml(data_ingestion_config)
            self.data_val_config = read_yaml(data_validation_config)
            self.schema = read_yaml(schema_config) 
            self.preprocessing_config = read_yaml(data_preprocessing_config)
            self.training_config = read_yaml(model_training_config)
            self.model_params_config = read_yaml(model_params_config)
            self.wandb_config = read_yaml(hyperparameter_config)
            self.evaluation_config = read_yaml(model_evaluation_config)
            self.validation_config = read_yaml(model_validation_config)
            

            create_directories([self.ingestion_config.artifacts_root])
            create_directories([self.data_val_config.artifacts_root])
            create_directories([self.preprocessing_config.artifacts_root])
            create_directories([self.training_config.artifacts_root])
            create_directories([self.evaluation_config.artifacts_root])
            create_directories([self.validation_config.artifacts_root]) 

            
            logger.info("Configuration directories created successfully.")
        
        except Exception as e:
            logger.error(f"Error initializing ConfigurationManager: {e}")
            logger.error(f"Error creating directories")
            raise CustomException(e, sys)
        
# Data Ingestion Configuration
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """Returns the data ingestion configuration object."""
        try:
            data_config = self.ingestion_config.data_ingestion
            create_directories([data_config['root_dir']])
            logger.info(f"Data ingestion configuration loaded from: {DATA_INGESTION_CONFIG_FILEPATH}")
            data_config['mongo_uri'] = os.environ.get('MONGO_URI')
            return DataIngestionConfig(config_data=data_config)
        except Exception as e:
            logger.error(f"Error loading data ingestion configuration: {e}")
            raise CustomException(e, sys)
        


# Data Validation configuration
    def get_data_validation_config(self) -> DataValidationConfig:
        try:
            data_valid_config = self.data_val_config.data_validation 
            schema_dict = self._process_schema()

            create_directories([Path(data_valid_config.root_dir)]) 
            logger.info(f"Data Validation Config Loaded") 

            return DataValidationConfig(
                root_dir = Path(data_valid_config.root_dir), 
                val_status = data_valid_config.val_status, 
                data_dir = Path(data_valid_config.data_dir), 
                all_schema = schema_dict,
                critical_columns = data_valid_config.critical_columns
            )
        except Exception as e: 
            logger.exception(f"Error getting data validation configuration: {str(e)}") 
            raise CustomException(e, sys)

    def _process_schema(self) -> Dict[str, str]:
        schema_columns = self.schema.get("columns", {})
        target_column = self.schema.get("target_column", [])
        schema_dict = {col['name']: col['type'] for col in schema_columns}
        schema_dict.update({col['name']: col['type'] for col in target_column})
        return schema_dict
    

# Data Transformation 
    def get_data_transformation_config(self) -> DataTransformationConfig:
        logger.info("Getting data transformation configuration")

        transformation_config = self.preprocessing_config.data_transformation
        create_directories([transformation_config.root_dir])

        return DataTransformationConfig(
            root_dir = Path(transformation_config.root_dir),
            data_path = Path(transformation_config.data_path),
            numerical_cols = transformation_config.numerical_cols,
            categorical_cols = transformation_config.categorical_cols,
            target_col = transformation_config.target_col,
            random_state = transformation_config.random_state
        )



    def get_model_training_config(self) -> ModelTrainerConfig:
        logger.info("Getting model training configuration")
        try:
          trainer_config = self.training_config['model_trainer']
          model_params = self.model_params_config['XGBClassifier_params']

          # Creates all necessary directories
          create_directories([Path(trainer_config.root_dir)])
          
          return ModelTrainerConfig(
              root_dir = Path(trainer_config.root_dir),
              train_features_path = Path(trainer_config.train_features_path),
              train_targets_path = Path(trainer_config.train_targets_path),
              model_name = trainer_config.model_name,
              model_params = model_params

          )
        except Exception as e:
           logger.error(f"Error getting model training config: {str(e)}")
           raise CustomException(e, sys)
        
    
    def get_model_training_config(self) -> ModelTrainerConfig:
        logger.info("Getting model training configuration")
        try:
            trainer_config = self.training_config["model_trainer"]
            model_params = self.model_params_config["XGBClassifier_params"]

            # Creates all necessary directories
            create_directories([trainer_config.root_dir])

            return ModelTrainerConfig(
                root_dir=Path(trainer_config.root_dir),
                train_features_path=Path(trainer_config.train_features_path),
                train_targets_path=Path(trainer_config.train_targets_path),
                model_name=trainer_config.model_name,
                model_params=model_params,
                project_name=trainer_config.project_name,
                val_features_path=Path(trainer_config.val_features_path),
                val_targets_path=Path(trainer_config.val_targets_path),
            )
        except Exception as e:
            logger.error(f"Error getting model training config: {str(e)}")
            raise CustomException(e, sys)
        

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        logger.info("Getting model evaluation configuration")

        eval_config = self.evaluation_config.model_evaluation
        create_directories([eval_config.root_dir])

        return ModelEvaluationConfig(
            root_dir = Path(eval_config.root_dir),
            val_feature_path = Path(eval_config.val_feature_path),
            val_targets_path = Path(eval_config.val_targets_path),
            model_path = Path(eval_config.model_path),
            eval_scores_path = Path(eval_config.eval_scores_path),
            threshold_adjustment = Path(eval_config.threshold_adjustment),
            precision_recall_path = Path(eval_config.precision_recall_path),
        )
    

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
