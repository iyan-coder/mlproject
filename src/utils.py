import os
import sys
import numpy as np
import pandas as pd
import dill

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

from src.exception import CustomException

# -----------------------------
# Save any Python object using dill
# -----------------------------
def save_object(file_path, obj):
    try:
        # Create directory if it doesn't exist
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Save the object to the specified path
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        # Raise a custom exception with full traceback info
        raise CustomException(e, sys)


# -----------------------------
# Train, tune, and evaluate multiple ML models
# -----------------------------
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for model_name, model in models.items():
            # Safely get hyperparameters for the current model
            model_params = param.get(model_name, {})

            # Perform hyperparameter tuning using GridSearchCV
            gs = GridSearchCV(model, model_params, cv=3)
            gs.fit(X_train, y_train)

            # Set best hyperparameters to model and train again
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Predictions for train and test
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # R² scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Store only test R² score for reporting
            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
