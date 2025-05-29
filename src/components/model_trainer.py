# Standard libraries
import os 
import sys
from dataclasses import dataclass

# ML models
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier  # ❌ Should be KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# Custom modules for logging and exceptions
from src.exception import CustomException
from src.logger import logging

# Utilities to save objects and evaluate models
from src.utils import save_object, evaluate_models

# Configuration class to hold model path
@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

# Main class for model training
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    # Method to train models and select the best one
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting the training and test input data")

            # Separate features and target from train and test arrays
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],   # all columns except last
                train_array[:, -1],    # last column
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Dictionary of regression models to evaluate
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsClassifier(),  # ❌ Should use KNeighborsRegressor
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=0),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            # Evaluate models using the utility function
            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models
            )

            # Select the model with the best R² score
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            # Raise error if all models perform poorly
            if best_model_score < 0.6:
                raise CustomException("No suitable model found (R² < 0.6)")

            logging.info(f"Best model found: {best_model_name} with R²: {best_model_score}")

            # Save the best model as a pickle file
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Predict and calculate R² score for final evaluation
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            raise CustomException(sys, e)
