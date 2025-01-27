import sys
sys.path.append(r"C:\Users\lilwa\OneDrive\Desktop\moringa\Edel\website-leads")
import pandas as pd
import joblib
from pathlib import Path
from typing import Any, Tuple
from xgboost import XGBClassifier
from sklearn.metrics import f1_score


# Local Modules
from src.lead_scoring.exception import CustomException
from src.lead_scoring.logger import logger
from src.lead_scoring.constants import *
from src.lead_scoring.utils.commons import *
from src.lead_scoring.config_entity.config_params import ModelTrainerConfig

# Wandb
import wandb
#wandb.require("core")


class DataManager:
    @staticmethod
    def load_data(
        train_features_path: Path,
        train_targets_path: Path,
        val_features_path: Path,
        val_targets_path: Path,
    ) -> Tuple[Any, pd.DataFrame, pd.DataFrame, Any]:
        """Loads the training and validation data from the given file paths."""
        try:
            with open(train_features_path, "rb") as f:
                X_train_transformed = joblib.load(f)
            y_train = pd.read_csv(train_targets_path)

            with open(val_features_path, "rb") as f:
                X_val_transformed = joblib.load(f)
            y_val = pd.read_csv(val_targets_path)

            logger.info("Training and validation data loaded successfully")
            return X_train_transformed, y_train, X_val_transformed, y_val

        except FileNotFoundError as fnf_error:
            logger.error(f"File not found: {str(fnf_error)}")
            raise CustomException(fnf_error, sys)
        except Exception as e:
            logger.error(f"Unexpected error loading data: {str(e)}")
            raise CustomException(e, sys)


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def _train(self, X_train_transformed, y_train):
        try:
            if not self.config.model_params:
                raise ValueError("Model parameters are empty.")

            run = wandb.init(
                project=self.config.project_name,
                config={**self.config.model_params},
            )
            xgb_model = XGBClassifier(**self.config.model_params)
            xgb_model.fit(X_train_transformed, y_train)

            # Save model
            model_path = Path(self.config.root_dir) / self.config.model_name
            joblib.dump(xgb_model, model_path)
            logger.info(f"Model trained and saved at: {model_path}")

            artifact = wandb.Artifact("model", type="model")
            artifact.add_file(model_path)
            run.log_artifact(artifact)

            run.finish()

            return xgb_model
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise CustomException(e, sys)

    def _train_and_evaluate(
        self, X_train_transformed, y_train, X_val_transformed, y_val
    ):
        with wandb.init() as run:
            config = wandb.config
            xgb_model = XGBClassifier(**config)
            xgb_model.fit(X_train_transformed, y_train)
            y_val_pred = xgb_model.predict(X_val_transformed)
            val_f1 = f1_score(y_val, y_val_pred, average="macro")
            wandb.log({"validation_f1_score": val_f1})

    # def train_with_sweep(self,
    #                      X_train_transformed,y_train,
    #                      X_val_transformed, y_val,
    #                      sweep_configuration, #Take sweep configuration as a parameter
    # ):

    #     sweep_id = wandb.sweep(
    #         sweep=sweep_configuration, project=self.config.project_name
    #     )
        
    #     wandb.agent(
    #         sweep_id,
    #         function=lambda: self._train_and_evaluate(
    #             X_train_transformed, y_train, X_val_transformed, y_val
    #         ),
    #         count=5,
    #     )
        
   

    def _train_and_evaluate(self, X_train, y_train, X_val, y_val):
        # Add your training and evaluation logic here
        pass

    def train_with_sweep(self, X_train_transformed, y_train, X_val_transformed, y_val, sweep_configuration):
        """
        This function starts a hyperparameter sweep with WandB.

        Parameters:
        - X_train_transformed, y_train: Training data and labels
        - X_val_transformed, y_val: Validation data and labels
        - sweep_configuration: Configuration for the sweep (from wandb)
        """
        # Initialize the sweep using wandb.sweep
        sweep_id = wandb.sweep(sweep=sweep_configuration, project=self.config.project_name)

        # Define the function to be called by each agent in the sweep
        def train_with_config():
            self._train_and_evaluate(X_train_transformed, y_train, X_val_transformed, y_val)

        # Start the sweep using wandb.agent
        wandb.agent(
            sweep_id,
            function=train_with_config,
            count=5,  # Define how many times you want to run the sweep
        )
    