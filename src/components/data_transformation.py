import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

# Dataclass to hold configuration for data transformation
@dataclass
class DataTransformationconfig:
    # Path to save the serialized preprocessor object
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        # Initialize configuration object
        self.data_transformation_config = DataTransformationconfig()
    
    def get_data_transformer_object(self):
        '''
        Creates and returns a preprocessing pipeline object that:
        - imputes missing values,
        - scales numerical features,
        - encodes categorical features,
        - scales encoded features.
        
        Returns:
            preprocessor: sklearn ColumnTransformer object
        '''
        try:
            # Define numerical and categorical columns based on dataset
            numerical_columns = ['reading_score', 'writing_score']
            categorical_columns = [
                'gender',
                'race_ethnicity', 
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]

            # Pipeline for numerical features:
            # 1. Impute missing values with median
            # 2. Scale features using StandardScaler
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]      
            )

            # Pipeline for categorical features:
            # 1. Impute missing values with most frequent category
            # 2. One-hot encode categorical variables (non-sparse output for compatibility)
            # 3. Scale encoded features with StandardScaler
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(sparse_output=False)),
                    ("scaler", StandardScaler())
                ]
            )

            # Log column types for debugging
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Combine numerical and categorical pipelines using ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipelines", cat_pipeline, categorical_columns)
                ]
            )

            # Return the full preprocessing pipeline
            return preprocessor
        
        except Exception as e:
            # Wrap and raise any exceptions as CustomException with system info
            raise CustomException(sys, e)
        

    def initiate_data_transformation(self, train_path, test_path):
        '''
        Executes data transformation pipeline:
        - Loads train and test datasets,
        - Applies preprocessing pipeline,
        - Returns transformed train/test arrays and path to saved preprocessor object.
        
        Args:
            train_path: filepath to training CSV
            test_path: filepath to testing CSV
        
        Returns:
            train_arr: transformed training data as numpy array (features + target)
            test_arr: transformed testing data as numpy array (features + target)
            preprocessor_obj_file_path: path where the preprocessor is saved
        '''
        try:
            # Load datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            # Get the preprocessing pipeline object
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_columns = ['reading_score', 'writing_score']

            # Separate input features and target in training set
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            # Separate input features and target in testing set
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframes")

            # Fit and transform training input features
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            # Only transform testing input features
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine transformed features with target array for train and test sets
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save the preprocessor object to disk for reuse during inference
            logging.info("Saving preprocessing object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            # Return transformed data arrays and preprocessor path
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            # Raise custom exception with error details and system info
            raise CustomException(e, sys)
