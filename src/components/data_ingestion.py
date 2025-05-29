
# Required imports
import os
import sys
import pandas as pd
import logging
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.exception import CustomException  # Custom error handler
from src.logger import logging  # For tracking steps


@dataclass
class DataIngestionConfig:
    """
    Configuration class for storing file paths used in the data ingestion process.
    """
    train_data_path: str = os.path.join("artifacts", "train.csv")  # Path to save the training dataset
    test_data_path: str = os.path.join("artifacts", "test.csv")    # Path to save the testing dataset
    raw_data_path: str = os.path.join("artifacts", "data.csv")     # Path to save the raw/original dataset


class DataIngestion:
    """
    Class to handle the data ingestion pipeline.
    Reads raw data, performs a train-test split, and saves the resulting files.
    """

    def __init__(self):
        """
        Initialize the DataIngestion class and load configuration paths.
        """
        self.ingestion_congfig = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Executes the data ingestion steps:
        1. Reads the raw data CSV file.
        2. Saves the raw data into the artifacts directory.
        3. Splits the data into train and test sets.
        4. Saves the train and test data into the artifacts directory.

        Returns:
            Tuple[str, str]: Paths to the train and test datasets.

        Raises:
            CustomException: If any error occurs during data ingestion.
        """
        logging.info("Entered the data ingestion component.")

        try:
            # Step 1: Read the raw dataset CSV
            df = pd.read_csv(r"D:\Projects\mlproject\notebook\data\stud.csv")
            logging.info("Dataset loaded into DataFrame.")

            # Step 2: Ensure the artifacts directory exists
            os.makedirs(os.path.dirname(self.ingestion_congfig.train_data_path), exist_ok=True)

            # Step 3: Save the raw dataset for future reference or auditing
            df.to_csv(self.ingestion_congfig.raw_data_path, index=False, header=True)
            logging.info("Raw data saved at: %s", self.ingestion_congfig.raw_data_path)

            # Step 4: Perform train-test split (80% train, 20% test)
            logging.info("Train-Test split started.")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Step 5: Save the training and testing datasets
            train_set.to_csv(self.ingestion_congfig.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_congfig.test_data_path, index=False, header=True)
            logging.info("Train and Test data saved successfully.")

            # Step 6: Return file paths to be used by downstream components
            logging.info("Data ingestion completed successfully.")
            return (
                self.ingestion_congfig.train_data_path,
                self.ingestion_congfig.test_data_path
            )

        except Exception as e:
            # Raise a custom exception with traceback for debugging
            raise CustomException(e, sys)
