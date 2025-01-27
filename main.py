import sys
from typing import List, Callable
from src.lead_scoring.logger import logger
from src.lead_scoring.exception import CustomException
from src.lead_scoring.pipelines.pip_01_data_ingestion import DataIngestionPipeline
from src.lead_scoring.pipelines.pip_02_data_validation import DataValidationPipeline
from src.lead_scoring.pipelines.pip_03_data_transformation import DataTransformationPipeline
from src.lead_scoring.pipelines.pip_04_model_trainer import ModelTrainerPipeline
from src.lead_scoring.pipelines.pip_05_model_evaluation import ModelEvaluationPipeline
from src.lead_scoring.pipelines.pip_06_model_validation import ModelValidationPipeline


class PipelineOrchestrator:
    """Orchestrates the execution of multiple pipelines."""

    def __init__(self):
        self.pipelines: List[Callable] = []
    
    def add_pipeline(self, pipeline: Callable):
        """Adds a pipeline to the list of pipelines to be executed."""
        self.pipelines.append(pipeline)

    def run_all(self):
        """Executes all pipelines in sequence."""
        try:
            for pipeline in self.pipelines:
                logger.info(f"Starting {pipeline.__name__} pipeline")
                pipeline_result = pipeline()  # Execute the pipeline with no input
                logger.info(f"Completed {pipeline.__name__} pipeline")
        except Exception as e:
            logger.error(f"Error in orchestrator: {e}")
            raise CustomException(e, sys)

def create_data_ingestion_pipeline():
    """Creates a data ingestion pipeline."""
    ingestion_pipeline = DataIngestionPipeline()
    ingestion_pipeline.run()
    return {}

def create_data_validation_pipeline():
    """Creates a data validation pipeline."""
    validation_pipeline = DataValidationPipeline()
    validation_pipeline.run()
    return {}

def create_data_transformation_pipeline():
    """Creates a data transformation pipeline."""
    transformation_pipeline = DataTransformationPipeline()
    transformation_pipeline.run()
    return {}

def create_model_trainer_pipeline():
    """Creates a model trainer pipeline."""
    trainer_pipeline = ModelTrainerPipeline()
    trainer_pipeline.run()
    return {}

def create_model_evaluation_pipeline():
    """Creates a model evaluation pipeline."""
    evaluation_pipeline = ModelEvaluationPipeline()
    evaluation_pipeline.run()
    return {}

def create_model_validation_pipeline():
    """Creates a model validation pipeline."""
    validation_pipeline = ModelValidationPipeline()
    validation_pipeline.run()
    return {}


if __name__ == "__main__":
    try:
        logger.info("## ================ Starting Entire Model Pipeline =======================")
        orchestrator = PipelineOrchestrator()
        orchestrator.add_pipeline(create_data_ingestion_pipeline)
        orchestrator.add_pipeline(create_data_validation_pipeline)
        orchestrator.add_pipeline(create_data_transformation_pipeline)
        orchestrator.add_pipeline(create_model_trainer_pipeline)
        orchestrator.add_pipeline(create_model_evaluation_pipeline)
        orchestrator.add_pipeline(create_model_validation_pipeline)
        orchestrator.run_all()
        logger.info("## ================ Entire Model Pipeline completed successfully =======================")
    except Exception as e:
        logger.error(f"Error during entire pipeline execution: {e}")
        raise CustomException(e, sys)