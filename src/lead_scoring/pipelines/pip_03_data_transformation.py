import sys
sys.path.append(r"C:\Users\lilwa\OneDrive\Desktop\moringa\Edel\website-leads")
from src.lead_scoring.components.c_03_data_transformation import DataTransformation, DataTransformationConfig
from src.lead_scoring.logger import logger
from src.lead_scoring.config_manager.config_settings import ConfigurationManager
from src.lead_scoring.exception import CustomException
from typing import List, Callable

PIPELINE_NAME = "DATA TRANSFORMATION PIPELINE"


class PipelineStep:
    """
    Represents a step in the data transformation pipeline.

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


class DataTransformationPipeline:
    """
    Orchestrates the data transformation pipeline.

    Attributes:
        pipeline_name (str): The name of the pipeline.
        steps (List[PipelineStep]): A list of steps to be executed in the pipeline.

    Methods:
        add_step(step: PipelineStep):
            Adds a step to the pipeline.

        run():
            Executes all steps in the data transformation pipeline.
    """

    def __init__(self, pipeline_name: str = PIPELINE_NAME):
        self.pipeline_name = pipeline_name
        self.steps: List[PipelineStep] = []

    def add_step(self, step: PipelineStep):
        """Adds a step to the pipeline."""
        self.steps.append(step)

    def run(self):
        """Executes the data transformation pipeline."""
        try:
            logger.info(f"## ================ Starting {self.pipeline_name} pipeline =======================")

            config_manager = ConfigurationManager()
            data_transformation_config = config_manager.get_data_transformation_config()

            pipeline_data = {
                'data_transformation_config': data_transformation_config,
            }

            for step in self.steps:
                pipeline_data = step.execute(**pipeline_data)

            logger.info(f"## ================ {self.pipeline_name} pipeline completed successfully =======================")

        except CustomException as e:
            logger.error(f"Error during {self.pipeline_name} pipeline execution: {e}")
            raise


def create_data_transformation_step(name: str) -> PipelineStep:
    """
    Creates a pipeline step for data transformation.

    Parameters:
    - name (str): The name of the pipeline step.

    Returns:
    - PipelineStep: An instance of PipelineStep configured to perform data transformation.
    """
    def step_function(data_transformation_config: DataTransformationConfig):
        logger.info("Initializing Data Transformation")
        data_transformation = DataTransformation(config=data_transformation_config)

        logger.info("Starting Train-Validation-Test Split")
        X_train, X_val, X_test, y_train, y_val, y_test = data_transformation.train_val_test_split()
        logger.info("Data successfully split into train, validation, and test sets")

        logger.info("Applying data transformations")
        (
            preprocessor,
            X_train_transformed, X_val_transformed,
            X_test_transformed, y_train_transformed,
            y_val_transformed, y_test_transformed,
        ) = data_transformation.initiate_data_transformation(
            X_train, X_val, X_test, y_train, y_val, y_test
        )
        logger.info("Data transformation completed")

        return {}

    return PipelineStep(name=name, step_function=step_function)


if __name__ == "__main__":
    try:
        data_transformation_pipeline = DataTransformationPipeline()
        data_transformation_pipeline.add_step(create_data_transformation_step("Transform Data"))
        data_transformation_pipeline.run()
    except Exception as e:
        logger.error(f"Error during {PIPELINE_NAME} pipeline execution: {e}")
        raise CustomException(e, sys)