import sys
sys.path.append(r"C:\Users\lilwa\OneDrive\Desktop\moringa\Edel\website-leads")
from src.lead_scoring.logger import logger
from src.lead_scoring.exception import CustomException
from src.lead_scoring.config_manager.config_settings import ConfigurationManager
from src.lead_scoring.components.c_04_model_trainer import ModelTrainer
from src.lead_scoring.components.c_04_model_trainer import DataManager
from typing import Callable, List

PIPELINE_NAME = "MODEL TRAINER PIPELINE"


class PipelineStep:
    """
    Represents a step in the model trainer pipeline.

    Attributes:
        name (str): The name of the pipeline step.
        step_function (Callable): The function to execute for this step.

    Methods:
        execute(**kwargs): Executes the pipeline step, logging its progress and handling exceptions.
    """

    def __init__(self, name: str, step_function: Callable):
        self.name = name
        self.step_function = step_function

    def execute(self, **kwargs):
        """Executes the pipeline step."""
        try:
            logger.info(f"Executing step: {self.name}")
            result = self.step_function(**kwargs)
            logger.info(f"Step {self.name} completed successfully.")
            return result
        except Exception as e:
            logger.error(f"Error executing step {self.name}: {e}")
            raise CustomException(e, sys)


class ModelTrainerPipeline:
    """
    Orchestrates the model trainer pipeline.

    Attributes:
        pipeline_name (str): The name of the pipeline.
        steps (List[PipelineStep]): A list of steps to be executed in the pipeline.

    Methods:
        add_step(step: PipelineStep):
            Adds a step to the pipeline.

        run():
            Executes all steps in the model trainer pipeline.
    """

    def __init__(self, pipeline_name: str = PIPELINE_NAME):
        self.pipeline_name = pipeline_name
        self.steps: List[PipelineStep] = []

    def add_step(self, step: PipelineStep):
        """Adds a step to the pipeline."""
        self.steps.append(step)

    def run(self):
        """Executes the model trainer pipeline."""
        try:
            logger.info(f"## ================ Starting {self.pipeline_name} pipeline =======================")

            config_manager = ConfigurationManager()
            model_training_config = config_manager.get_model_training_config()
            data_manager = DataManager()

            pipeline_data = {
                'model_training_config': model_training_config,
                'data_manager': data_manager,
            }

            for step in self.steps:
                pipeline_data = step.execute(**pipeline_data)

            logger.info(f"## ================ {self.pipeline_name} pipeline completed successfully =======================")

        except CustomException as e:
            logger.error(f"Error during {self.pipeline_name} pipeline execution: {e}")
            raise


def create_data_loading_step(name: str) -> PipelineStep:
    """
    Creates a pipeline step for loading training and validation data.

    Parameters:
    - name (str): The name of the pipeline step.

    Returns:
    - PipelineStep: An instance of PipelineStep configured for data loading.
    """
    def step_function(model_training_config, data_manager):
        logger.info("Loading training and validation data")
        X_train_transformed, y_train, y_val, X_val_transformed = data_manager.load_data(
            model_training_config.train_features_path,
            model_training_config.train_targets_path,
            model_training_config.val_features_path,
            model_training_config.val_targets_path,
        )
        logger.info("Data loading completed")
        return {
            'X_train_transformed': X_train_transformed,
            'y_train': y_train,
            'X_val_transformed': X_val_transformed,
            'y_val': y_val,
            'model_training_config': model_training_config,
        }

    return PipelineStep(name=name, step_function=step_function)


def create_model_training_step(name: str, mode: str) -> PipelineStep:
    """
    Creates a pipeline step for model training or hyperparameter sweeping.

    Parameters:
    - name (str): The name of the pipeline step.
    - mode (str): The mode of training - 'train' or 'sweep'.

    Returns:
    - PipelineStep: An instance of PipelineStep configured for model training or sweeping.
    """
    def step_function(model_training_config, X_train_transformed, y_train, X_val_transformed, y_val):
        logger.info("Initializing model training")
        model_trainer = ModelTrainer(config=model_training_config)

        if mode == "train":
            logger.info("Training model...")
            model = model_trainer._train(X_train_transformed, y_train)
            logger.info("Model training completed successfully.")
        elif mode == "sweep":
            logger.info("Starting hyperparameter search...")
            model_trainer.train_with_sweep(
                X_train_transformed,
                y_train,
                X_val_transformed,
                y_val,
                model_training_config.wandb_config.get("sweep", {}),
            )
            logger.info("Hyperparameter search completed successfully.")
        else:
            raise ValueError(f"Invalid mode '{mode}' specified. Use 'train' or 'sweep'.")

        return {}

    return PipelineStep(name=name, step_function=step_function)


if __name__ == "__main__":
    try:
        # Determine training mode from user input
        trigger = input("Enter 'train' for model training or 'sweep' for hyperparameter search: ").lower()
        if trigger not in ["train", "sweep"]:
            print("Invalid input. Please enter 'train' or 'sweep'.")
            sys.exit(1)

        # Instantiate pipeline
        model_trainer_pipeline = ModelTrainerPipeline()

        # Add steps to pipeline
        model_trainer_pipeline.add_step(create_data_loading_step("Load Data"))
        model_trainer_pipeline.add_step(create_model_training_step("Train Model" if trigger == "train" else "Sweep Model", mode=trigger))

        # Run pipeline
        model_trainer_pipeline.run()

    except Exception as e:
        logger.error(f"Error in {PIPELINE_NAME}: {e}")
        sys.exit(1)