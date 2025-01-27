import sys
sys.path.append(r"C:\Users\lilwa\OneDrive\Desktop\moringa\Edel\website-leads")

from src.lead_scoring.logger import logger
from src.lead_scoring.exception import CustomException
from src.lead_scoring.config_manager.config_settings import ConfigurationManager
from src.lead_scoring.components.c_01_data_ingestion import DataIngestion
from typing import Callable, List

PIPELINE_NAME = "DATA INGESTION PIPELINE"


class PipelineStep:
    """
    Represents a step in the data ingestion pipeline.

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


class DataIngestionPipeline:
    """
    Orchestrates the data ingestion pipeline.

    Attributes:
        pipeline_name (str): The name of the pipeline.
        steps (List[PipelineStep]): A list of steps to be executed in the pipeline.

    Methods:
        add_step(step: PipelineStep):
            Adds a step to the pipeline.

        run():
            Executes all steps in the data ingestion pipeline.
    """

    def __init__(self, pipeline_name: str = PIPELINE_NAME):
        self.pipeline_name = pipeline_name
        self.steps: List[PipelineStep] = []

    def add_step(self, step: PipelineStep):
        """Adds a step to the pipeline."""
        self.steps.append(step)

    def run(self):
        """Executes the data ingestion pipeline."""
        try:
            logger.info(f"## ================ Starting {self.pipeline_name} pipeline =======================")

            config_manager = ConfigurationManager()
            data_ingestion_config = config_manager.get_data_ingestion_config()

            pipeline_data = {
                'data_ingestion_config': data_ingestion_config,
            }

            for step in self.steps:
                pipeline_data = step.execute(**pipeline_data)

            logger.info(f"## ================ {self.pipeline_name} pipeline completed successfully =======================")

        except CustomException as e:
            logger.error(f"Error during {self.pipeline_name} pipeline execution: {e}")
            raise


def create_data_ingestion_step(name: str) -> PipelineStep:
    """
    Creates a pipeline step for data ingestion.

    Parameters:
    - name (str): The name of the pipeline step.

    Returns:
    - PipelineStep: An instance of PipelineStep configured for data ingestion.
    """
    def step_function(data_ingestion_config):
        user_name = input("Enter your name: ")
        logger.info(f"Initializing data ingestion for user: {user_name}")
        
        data_ingestion = DataIngestion(config=data_ingestion_config, user_name=user_name)
        data_ingestion.import_data_from_mongodb()
        logger.info("Data ingestion completed successfully.")
        
        return {}

    return PipelineStep(name=name, step_function=step_function)


if __name__ == "__main__":
    try:
        # Instantiate pipeline
        data_ingestion_pipeline = DataIngestionPipeline()

        # Add steps to pipeline
        data_ingestion_pipeline.add_step(create_data_ingestion_step("Ingest Data"))

        # Run pipeline
        data_ingestion_pipeline.run()

    except Exception as e:
        logger.error(f"Error in {PIPELINE_NAME}: {e}")
        sys.exit(1)
