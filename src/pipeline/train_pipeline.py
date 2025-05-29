import os
import sys
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, load_data, preprocess_data
from src.components.model_trainer import ModelTrainer


class TrainPipeline:
    """
    Pipeline to train and evaluate machine learning models.

    Steps:
    - Load dataset from a CSV file.
    - Preprocess the dataset (impute, scale, encode).
    - Split the data into training and test sets.
    - Train and evaluate multiple regression models using GridSearchCV.
    - Save the best model and the preprocessor as artifacts.
    """

    def __init__(self, raw_data_path, model_save_path, preprocessor_save_path):
        self.raw_data_path = raw_data_path
        self.model_save_path = model_save_path
        self.preprocessor_save_path = preprocessor_save_path

    def run(self):
        try:
            logging.info("Loading raw data...")
            data = load_data(self.raw_data_path)

            logging.info("Preprocessing data...")
            X, y, preprocessor = preprocess_data(data)

            logging.info("Splitting data into train and test sets...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Save preprocessor for reuse in prediction pipeline
            save_object(file_path=self.preprocessor_save_path, obj=preprocessor)

            # Define models to evaluate
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=0),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            # Define hyperparameter grid for each model
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "K-Neighbors Regressor": {},  # No hyperparams for now
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},  # No hyperparams
                "XGBRegressor": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }

            # Initialize ModelTrainer
            model_trainer = ModelTrainer(model_save_path=self.model_save_path)

            # Train and evaluate all models, returning best model and R2 score
            best_model, best_r2_score = model_trainer.train_and_evaluate(
                X_train, y_train, X_test, y_test, models, params
            )

            logging.info(f"Training pipeline completed successfully.")
            logging.info(f"Best model saved. Best R2 score: {best_r2_score}")

            # Return both best model and R2 score for main.py to use
            return best_model, best_r2_score

        except Exception as e:
            raise CustomException(e, sys)
