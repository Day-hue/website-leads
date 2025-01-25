import sys
sys.path.append(r"C:\Users\lilwa\OneDrive\Desktop\moringa\Edel\website-leads")

from dataclasses import dataclass
from pathlib import Path
import pymongo
import pandas as pd
import numpy as np
import os
import json
import time
import logging
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient
from src.lead_scoring.exception import CustomException
from src.lead_scoring.logger import logger
from src.lead_scoring.constants import DATA_INGESTION_CONFIG_FILEPATH
from src.lead_scoring.utils.commons import read_yaml, create_directories

# Load environment variables from .env file
load_dotenv()

@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion."""
    config_data: dict

class ConfigurationManager:
    """Manages the data ingestion configuration."""
    def __init__(self, data_ingestion_config: str = DATA_INGESTION_CONFIG_FILEPATH):
        """
        Initializes the ConfigurationManager.

        Args:
            data_ingestion_config (str): Path to the data ingestion configuration file.
        """
        try:
            logger.info(f"Initializing ConfigurationManager with config file: {data_ingestion_config}")
            self.ingestion_config = read_yaml(data_ingestion_config)
            create_directories([self.ingestion_config.artifacts_root])
            logger.info("Configuration directories created successfully.")
        except Exception as e:
            logger.error(f"Error initializing ConfigurationManager: {e}")
            raise CustomException(e, sys.exc_info())
    
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
            raise CustomException(e, sys.exc_info())
        
class DataIngestion:
    """Handles data ingestion from MongoDB to Parquet."""
    def __init__(self, config: DataIngestionConfig, user_name: str):
        """
        Initializes DataIngestion with the configuration.

        Args:
            config (DataIngestionConfig): Data ingestion configuration.
             user_name (str): The name of the person doing ingestion
        """
        self.config = config
        self.user_name = user_name
        config_data = self.config.config_data
        try:
            logger.info("Connecting to MongoDB using URI from env")
            self.client = pymongo.MongoClient(config_data['mongo_uri'])
            self.db = self.client[config_data['database_name']]
            self.collection = self.db[config_data['collection_name']]
            logger.info(f"Connected to MongoDB: database={config_data['database_name']}, collection={config_data['collection_name']}")
        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {e}")
            raise CustomException(e, sys.exc_info())

    def import_data_from_mongodb(self):
        """Imports data from MongoDB to a Pandas DataFrame."""
        start_time = time.time()
        start_timestamp = datetime.now()
        try:
            logger.info("Starting data ingestion from MongoDB.")
            all_data = self.fetch_all_data()
            output_path = self.save_data(all_data)
            total_records = len(all_data)
            logger.info(f"Total records fetched: {total_records}")
            self._save_metadata(start_time, start_timestamp, total_records, output_path)
            logger.info("Data ingestion completed successfully.")
        except Exception as e:
            logger.error(f"Error during data ingestion: {e}")
            raise CustomException(e, sys.exc_info())

    def fetch_all_data(self) -> pd.DataFrame:
        """Fetches all the data in batches from MongoDB collection."""
        try:
            logger.info("Fetching all data from MongoDB in batches.")
            all_data = []
            batch_size = self.config.config_data['batch_size']
            skip_count = 0

            while True:
                logger.debug(f"Fetching batch with skip: {skip_count}, limit: {batch_size}")
                cursor = self.collection.find({}, {'_id': 0}).skip(skip_count).limit(batch_size)
                batch = list(cursor)
                if not batch:
                   logger.info("No more data found")
                   break
                
                df = pd.DataFrame(batch)
                df.replace([np.inf, -np.inf], np.nan, inplace=True)
                if df.isnull().values.any():
                    logger.warning("Data contains NaN values. Dropping rows with NaN values.")
                    df = df.dropna()
                    
                logger.debug(f"Batch DataFrame shape: {df.shape}")
                all_data.append(df)
                skip_count += batch_size
           
            final_df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
            logger.info(f"Data fetched successfully. Total records: {len(final_df)}")
            return final_df
        except Exception as e:
            logger.error(f"Error fetching data from MongoDB: {e}")
            raise CustomException(e, sys.exc_info())


    def save_data(self, all_data):
        """Saves data to a csv file."""
        try:
            output_path = Path(self.config.config_data['root_dir']) / "website_visitors.csv"
            all_data.to_csv(output_path, index=False)
            logger.info(f"Data saved to csv file: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error saving data to csv: {e}")
            raise CustomException(e, sys.exc_info())

    def _save_metadata(self, start_time, start_timestamp, total_records, output_path):
        """Saves metadata about data ingestion to a JSON file."""
        try:
            end_time = time.time()
            end_timestamp = datetime.now()
            duration = end_time - start_time
            metadata = {
                'start_time': start_timestamp.isoformat(),
                'end_time': end_timestamp.isoformat(),
                'duration': duration,
                'total_records': total_records,
                'data_source': self.config.config_data['collection_name'],
                'output_path': str(output_path),
                'ingested_by': self.user_name 
            }
            metadata_path = Path(self.config.config_data['root_dir']) / "data-ingestion-metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)
            logger.info("Metadata saved successfully")
            logger.debug(f"Metadata content: {metadata}")
        except Exception as e:
            logger.error(f"Error during metadata saving: {e}")
            raise CustomException(e, sys.exc_info())


if __name__ == "__main__":
    try:
        user_name = input("Enter your name: ")
        config_manager = ConfigurationManager()
        data_ingestion_config = config_manager.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config, user_name=user_name)
        data_ingestion.import_data_from_mongodb()
        logger.info("Data ingestion process completed successfully.")
    except CustomException as e:
        logger.error(f"Error during data ingestion: {e}")
        logger.info("Data ingestion process failed.")