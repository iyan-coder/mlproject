
# Required imports
import os
import sys
import pandas as pd
import logging
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.exception import CustomException  # Custom error handler
from src.logger import logging  # For tracking steps

# Configuration using dataclass to store file paths
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")

# Class to handle data ingestion
class DataIngestion:
    def __init__(self):
        # Load config with paths when this class is called
        self.ingestion_congfig = DataIngestionConfig()

    # Main method to run the data ingestion pipeline
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion component.")

        try:
            # Read the raw CSV dataset
            df = pd.read_csv(r"D:\Projects\mlproject\notebook\data\stud.csv")
            logging.info("Dataset loaded into DataFrame.")

            # Create directory for saving artifacts if not already present
            os.makedirs(os.path.dirname(self.ingestion_congfig.train_data_path), exist_ok=True)

            # Save the original/raw data for record-keeping
            df.to_csv(self.ingestion_congfig.raw_data_path, index=False, header=True)

            logging.info("Train-Test split started.")
            # Split the dataset into training and testing sets (80-20 split)
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save the train and test datasets to files
            train_set.to_csv(self.ingestion_congfig.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_congfig.test_data_path, index=False, header=True)

            logging.info("Data ingestion completed successfully.")

            # Return paths so that the next stage can use the files
            return (
                self.ingestion_congfig.train_data_path,
                self.ingestion_congfig.test_data_path
            )

        # If any error occurs, raise a custom error with traceback info
        except Exception as e:
            raise CustomException(e, sys)

# This ensures the ingestion runs when this file is executed directly
if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
