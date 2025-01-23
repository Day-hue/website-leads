# Import the necessary libraries 
import os
from pathlib import Path

# Define the package name 
package_name = "lead_scoring"

# List the files to be created 
list_of_files = [
    Path(".github") / "workflows" / ".gitkeep", # github workflow directory
    f"src/__init__.py", 
    f"src/{package_name}/__init__.py", 
    f"src/{package_name}/components/__init__.py",
    f"src/{package_name}/components/c_01_data_ingestion.py",
    f"src/{package_name}/components/c_02_data_validation.py",
    f"src/{package_name}/components/c_03_data_transformation.py",
    f"src/{package_name}/components/c_04_model_trainer.py",
    f"src/{package_name}/components/c_05_model_evaluation.py", 
    f"src/{package_name}/components/c_05_model_validation.py", 
    f"src/{package_name}/utils/__init__.py", 
    f"src/{package_name}/utils/commons.py", 
    f"src/{package_name}/config_manager/__init__.py", 
    f"src/{package_name}/config_manager/config_settings.py",
    f"src/{package_name}/pipelines/__init__.py",
    f"src/{package_name}/pipelines/pip_01_data_ingestion.py",
    f"src/{package_name}/pipelines/pip_02_data_validation.py",
    f"src/{package_name}/pipelines/pip_03_data_transformation.py",
    f"src/{package_name}/pipelines/pip_04_model_trainer.py",
    f"src/{package_name}/pipelines/pip_05_model_evaluation.py",
    f"src/{package_name}/pipelines/pip_06_model_validation.py",
    f"src/{package_name}/pipelines/pip_07_prediction_pipeline.py",
    f"src/{package_name}/config_entity/__init__.py",
    f"src/{package_name}/config_entity/config_params.py",
    f"src/{package_name}/constants/__init__.py",
    f"src/{package_name}/exception.py",
    f"src/{package_name}/logger.py",
    
    "config/model-params.yaml",
    "config/data-schema.yaml",
    "config/data-ingestion.yaml",
    "config/data-validation.yaml",
    "config/data-transformation.yaml",
    "config/model-trainer.yaml",
    "config/model-evaluation.yaml",
    "config/model-validation.yaml",
    "config/logging.yaml",
    "config/prediction.yaml",
    "config/environment.yaml", 
    "tests/test_data_ingestion.py",
    "tests/test_model_trainer.py",
    "tests/integration/test_end_to_end.py",
    "tests/conftest.py",
    "CONTRIBUTING.md",
    "CHANGELOG.md",
    "dvc/dvc.yaml",
    "main.py",
    "app.py",
    "streamlit_app.py",
    "Dockerfile",
    "requirements.txt",
    "requirements_dev.txt",
    "experiment/trial_01_data_ingestion.py",
    "experiment/trial_02_data_validation.py",
    "experiment/trial_03_data_transformation.py",
    "experiment/trial_04_model_trainer.py",
    "experiment/trial_05_model_evaluation.py",
    "experiment/trial_06_model_validation.py",
    "templates/home.html",
    "templates/index.html",
    "templates/prediction.html",
    "templates/train.html",
    "static/styles.css"

]

# Loop through each file in the list 
for filepath in list_of_files:
    filepath = Path(filepath)

    # Split the filepath into a directory and filename 
    filedir, filename = os.path.split(filepath)

    # Create directories if they don't exist 
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)

    # Create an empty file if it doesn't exist or if empty
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open (filepath, 'w') as f:
            pass

