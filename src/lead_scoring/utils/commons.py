import os
from box.exceptions import BoxValueError
import yaml
import json
import joblib
import sys
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any, List
from pythonjsonlogger import jsonlogger
from src.lead_scoring.exception import CustomException
from src.lead_scoring.logger import logging

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads a YAML file and returns its contents as a ConfigBox object.

    Args:
        path_to_yaml (Path): Path to the YAML file.

    Returns:
        ConfigBox: Parsed YAML content as a ConfigBox object, which allows attribute-style access.
    
    Raises:
        CustomException: If the file is empty, not found, or an error occurs during reading.
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logging.info(f"YAML file: {path_to_yaml} loaded successfully.")
            return ConfigBox(content)
    except BoxValueError:
        logging.error("YAML file is empty.")
        raise CustomException("YAML file is empty")
    except FileNotFoundError:
        logging.error(f"YAML file not found: {path_to_yaml}.")
        raise CustomException(f"YAML file not found: {path_to_yaml}")
    except Exception as e:
        logging.error(f"Error loading YAML file: {path_to_yaml}, Error: {str(e)}")
        raise CustomException(f"Error loading YAML file: {path_to_yaml}, Error: {str(e)}")

@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """
    Creates directories specified in the list if they do not exist.

    Args:
        path_to_directories (list): List of directory paths to create.
        verbose (bool): If True, logs the directory creation.

    Raises:
        CustomException: If an error occurs while creating a directory.
    """
    for path in path_to_directories:
        try:
            # Create the directory if it doesn't exist
            os.makedirs(path, exist_ok=True)
            if verbose:
                logging.info(f"Created directory at: {path}")
        except Exception as e:
            logging.error(f"Error creating directory at: {path}, Error: {str(e)}")
            raise CustomException(f"Error creating directory at: {path}, Error: {str(e)}")

def save_object(obj, file_path):
    try:
        file_path = Path(file_path) # Added: Convert the path to Path object
        dir_path = file_path.parent
        os.makedirs(str(dir_path), exist_ok=True) # Modified: Pass a string in makedirs
        joblib.dump(obj, str(file_path)) # Modified: Pass a string in joblib
        logging.info(f"Object saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving object at: {file_path} exception: {str(e)}")
        raise CustomException(e, sys)
        

# def save_object(file_path: Path, obj: Any):
#     """
#     Saves a Python object to a file using joblib.

#     Args:
#         file_path (Path): Path where the object will be saved.
#         obj (Any): Python object to save.

#     Raises:
#         CustomException: If an error occurs during saving.
#     """

#     try:
#         dir_path = file_path.parent
#         os.makedirs(dir_path, exist_ok=True)
#         joblib.dump(obj, file_path)
#         logging.info(f"Object saved at: {file_path}")
#     except Exception as e:
#         logging.error(f"Error saving object at: {file_path}, Error: {str(e)}")
#         raise CustomException(f"Error saving object at: {file_path}, Error: {str(e)}")


def load_object(file_path: Path) -> Any:
    """
    Loads a Python object from a file using joblib.

    Args:
        file_path (Path): Path of the file to load the object from.

    Returns:
        Any: The loaded Python object.

    Raises:
        CustomException: If an error occurs during loading.
    """
    try:
        with open(file_path, 'rb') as file_obj:
            obj = joblib.load(file_obj)
            logging.info(f"Object loaded from: {file_path}")
            return obj
    except Exception as e:
        logging.error(f"Error loading object from: {file_path}, Error: {str(e)}")
        raise CustomException(f"Error loading object from: {file_path}, Error: {str(e)}")


def save_json(path: Path, data: dict):
    """
    Saves a dictionary as a JSON file.

    Args:
        path (Path): Path where the JSON file will be saved.
        data (dict): Dictionary to save in JSON format.

    Raises:
        CustomException: If an error occurs during saving.
    """
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        logging.info(f"JSON file saved at: {path}")
    except Exception as e:
        logging.error(f"Error saving JSON file at: {path}, Error: {str(e)}")
        raise CustomException(f"Error saving JSON file at: {path}, Error: {str(e)}")

# def save_json(path: Path, data: dict):
#     """
#     Saves a dictionary to a JSON file.

#     Args:
#         path (Path): Path where the JSON file will be saved.
#         data (dict): Dictionary to save as JSON.
#     """
#     try:
#         # Check if the path is of type Path or str
#         if not isinstance(path, (str, Path, os.PathLike)):
#             raise TypeError(f"Invalid path type: {type(path)}. Expected str, bytes, or os.PathLike object.")
        
#         with open(path, "w") as f:
#             json.dump(data, f, indent=4)
#         logging.info(f"JSON file saved at: {path}")
#     except Exception as e:
#         logging.error(f"Error saving JSON file at: {path}, Error: {str(e)}")
#         raise CustomException(f"Error saving JSON file at: {path}, Error: {str(e)}")


def load_json(path: Path) -> ConfigBox:
    """
    Loads a JSON file and returns its contents as a ConfigBox object.

    Args:
        path (Path): Path of the JSON file to load.

    Returns:
        ConfigBox: Parsed JSON content as a ConfigBox object, which allows attribute-style access.
    
    Raises:
        CustomException: If an error occurs during loading.
    """
    try:
        with open(path) as f:
            content = json.load(f)
        logging.info(f"JSON file loaded successfully from: {path}")
        return ConfigBox(content)
    except Exception as e:
        logging.error(f"Error loading JSON file from: {path}, Error: {str(e)}")
        raise CustomException(f"Error loading JSON file from: {path}, Error: {str(e)}")


def save_bin(data: Any, path: Path):
    """
    Saves data to a binary file using joblib.

    Args:
        data (Any): Data to save.
        path (Path): Path where the binary file will be saved.

    Raises:
        CustomException: If an error occurs during saving.
    """
    try:
        joblib.dump(value=data, filename=path)
        logging.info(f"Binary file saved at: {path}")
    except Exception as e:
        logging.error(f"Error saving binary file at: {path}, Error: {str(e)}")
        raise CustomException(f"Error saving binary file at: {path}, Error: {str(e)}")


def load_bin(path: Path) -> Any:
    """
    Loads data from a binary file using joblib.

    Args:
        path (Path): Path of the binary file to load.

    Returns:
        Any: The loaded data.

    Raises:
        CustomException: If an error occurs during loading.
    """
    try:
        data = joblib.load(path)
        logging.info(f"Binary file loaded from: {path}")
        return data
    except Exception as e:
        logging.error(f"Error loading binary file from: {path}, Error: {str(e)}")
        raise CustomException(f"Error loading binary file from: {path}, Error: {str(e)}")


def get_size(path: Path) -> str:
    """
    Gets the size of the file at the given path in kilobytes.

    Args:
        path (Path): Path of the file.

    Returns:
        str: Size of the file in kilobytes, rounded to the nearest whole number.
    """
    try:
        if os.path.isfile(path):  # Check if the file exists
            size_in_kb = round(os.path.getsize(path) / 1024)
            return f"~ {size_in_kb} KB"
        else:
            logging.error(f"File not found at: {path}")
            raise CustomException(f"File not found at: {path}")
    except Exception as e:
        logging.error(f"Error getting size for file at: {path}, Error: {str(e)}")
        raise CustomException(f"Error getting size for file at: {path}, Error: {str(e)}")
