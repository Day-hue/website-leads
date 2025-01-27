import sys
sys.path.append(r"C:\Users\lilwa\OneDrive\Desktop\moringa\Edel\website-leads")

import pandas as pd
import numpy as np
import joblib 

from pathlib import Path 
from dataclasses import dataclass
from typing import Tuple


from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.lead_scoring.exception import CustomException
from src.lead_scoring.logger import logger
from src.lead_scoring.constants import *
from src.lead_scoring.utils.commons import *


@dataclass
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    numerical_cols: frozenset
    categorical_cols: frozenset
    target_col : str
    random_state: int  


class ConfigurationManager:
    def __init__(self, data_preprocessing_config: str = DATA_TRANSFORMATION_CONFIG_FILEPATH):
        self.preprocessing_config = read_yaml(data_preprocessing_config)
        artifacts_root = self.preprocessing_config.artifacts_root
        create_directories([artifacts_root])

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
    

class DataTransformation:
    def __init__(self, config:DataTransformationConfig):
        self.config = config


    def get_transformer_object(self) -> ColumnTransformer:
        logger.info("Creating transformer object")

        try:
            if not hasattr(self.config, 'numerical_cols') or not hasattr(self.config, 'categorical_cols'):
                raise ValueError("Numerical and categorical columns must be defined in the configuration.")

            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, self.config.numerical_cols),
                    ('cat', categorical_transformer, self.config.categorical_cols),
                ], remainder='passthrough'
            )

            return preprocessor
        
        except Exception as e:
            logger.exception(f"Error creating transformer object: {str(e)}")
            raise CustomException(e, sys)
        

    def train_val_test_split(self):
        try:
            logger.info("Splitting data into train, validation and test sets")

            if not os.path.exists(self.config.data_path):
                raise FileNotFoundError(f"Data path {self.config.data_path} does not exist.")
            df = pd.read_csv(self.config.data_path)

            if self.config.target_col not in df.columns:
                raise ValueError(f"Target column {self.config.target_col} not found in data.")
            
            X = df.drop(columns=[self.config.target_col])
            y = df[self.config.target_col]

            logger.info("Splitting data into train and test sets")

            # Split into training and temporary sets (70% train, 30% temp)
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=0.30, stratify=y, random_state=self.config.random_state
            )

            # Split the temporary set into validation and test sets (50% each)
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=self.config.random_state
            ) 

            logger.info(f"Saving the training, validation and testing data in artifacts file")

            # Save the target variables for each set 
            y_train.to_frame().to_csv(self.config.root_dir / 'y_train.csv', index=False)
            y_val.to_frame().to_csv(self.config.root_dir / 'y_val.csv', index=False)
            y_test.to_frame().to_csv(self.config.root_dir / 'y_test.csv', index=False)

            return X_train, X_val, X_test, y_train, y_val, y_test
        
        except Exception as e:
            logger.exception(f"Error splitting data: {str(e)}")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, 
                                     X_train: pd.DataFrame, 
                                     X_val: pd.DataFrame, 
                                     X_test: pd.DataFrame, 
                                     y_train: pd.Series, 
                                     y_val: pd.Series, 
                                     y_test: pd.Series) -> Tuple[ColumnTransformer, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        try:
            logger.info("Initiating data transformation")

            # Get the preprocessor object
            preprocessor_obj = self.get_transformer_object()
            if not isinstance(preprocessor_obj, ColumnTransformer):
                raise TypeError("Expected a ColumnTransformer object")

            # Transform the training testing and validation data
            X_train_transformed = preprocessor_obj.fit_transform(X_train)
            X_val_transformed = preprocessor_obj.transform(X_val)
            X_test_transformed = preprocessor_obj.transform(X_test)

            # Save the preprocessing object to the artifacts file
            preprocessor_path = Path(self.config.root_dir) / 'preprocessor.joblib'
            save_object(obj=preprocessor_obj, file_path=preprocessor_path)

            # Save the transformed data to the artifacts file
            X_train_transformed_path = Path(self.config.root_dir) / 'X_train_transformed.joblib'
            X_val_transformed_path = Path(self.config.root_dir) / 'X_val_transformed.joblib'
            X_test_transformed_path = Path(self.config.root_dir) / 'X_test_transformed.joblib'

            joblib.dump(X_train_transformed, X_train_transformed_path)
            logger.info(f"X_train_transformed saved to {X_train_transformed_path}")
            joblib.dump(X_val_transformed, X_val_transformed_path)
            logger.info(f"X_val_transformed saved to {X_val_transformed_path}")
            joblib.dump(X_test_transformed, X_test_transformed_path)
            logger.info(f"X_test_transformed saved to {X_test_transformed_path}")

            logger.info("Data transformation completed")

            return preprocessor_obj, X_train_transformed, X_val_transformed, X_test_transformed, y_train, y_val, y_test

        except Exception as e:
            logger.error(f"Error transforming data: {str(e)}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        config_manager = ConfigurationManager()
        data_transformation_config = config_manager.get_data_transformation_config()
        data_transformation = DataTransformation(config = data_transformation_config)
        X_train, X_val, X_test, y_train, y_val, y_test = data_transformation.train_val_test_split()
        preprocessor, X_train_transformed, X_val_transformed, X_test_transformed, y_train, y_val, y_test = \
            data_transformation.initiate_data_transformation(X_train, X_val, X_test, y_train, y_val, y_test) 

    except CustomException as e:
        logger.error(f"Error in data transformation: {str(e)}")
        sys.exit(1)