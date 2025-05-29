import os
import sys
import numpy as np
import pandas as pd
import dill

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.exception import CustomException

# ---------------------------------------------------------
# Save a Python object to disk using dill
# ---------------------------------------------------------
def save_object(file_path, obj):
    """
    Save a Python object to a specified file path using dill serialization.

    Args:
        file_path (str): Path where the object will be saved.
        obj (Any): Python object to be saved.

    Raises:
        CustomException: If saving fails due to any reason.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)  # Ensure directory exists

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)  # Save object using dill

    except Exception as e:
        raise CustomException(e, sys)


# ---------------------------------------------------------
# Load a saved Python object using dill
# ---------------------------------------------------------
def load_object(file_path):
    """
    Load a Python object from a dill-serialized file.

    Args:
        file_path (str): Path to the file containing the saved object.

    Returns:
        Any: Deserialized Python object.

    Raises:
        CustomException: If loading fails.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


# ---------------------------------------------------------
# Load data from CSV into a pandas DataFrame
# ---------------------------------------------------------
def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.

    Raises:
        CustomException: If loading fails.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        raise CustomException(e, sys)


# ---------------------------------------------------------
# Preprocess the data: imputation, encoding, and scaling
# ---------------------------------------------------------
def preprocess_data(df: pd.DataFrame):
    """
    Preprocess input DataFrame by:
    - Separating target from features
    - Identifying numerical and categorical columns
    - Applying appropriate transformers (imputation, scaling, encoding)
    
    Args:
        df (pd.DataFrame): Raw input data including target column.

    Returns:
        Tuple: (X_processed, y, preprocessor)
            - X_processed (np.ndarray): Preprocessed features
            - y (np.ndarray): Target values
            - preprocessor (ColumnTransformer): Fitted preprocessing pipeline

    Raises:
        CustomException: If preprocessing fails.
    """
    try:
        target_column = "math_score"  # <-- Replace with your actual target column name
        y = df[target_column].values
        X = df.drop(columns=[target_column])

        # Identify column types
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

        # Pipeline for numerical columns
        numerical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        # Pipeline for categorical columns
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Combine pipelines into a column transformer
        preprocessor = ColumnTransformer([
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

        # Fit and transform data
        X_processed = preprocessor.fit_transform(X)

        return X_processed, y, preprocessor

    except Exception as e:
        raise CustomException(e, sys)


# ---------------------------------------------------------
# Train and evaluate multiple models using GridSearchCV
# ---------------------------------------------------------
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Train and evaluate multiple machine learning models using hyperparameter tuning (GridSearchCV).

    Args:
        X_train (np.ndarray): Training feature matrix.
        y_train (np.ndarray): Training target vector.
        X_test (np.ndarray): Test feature matrix.
        y_test (np.ndarray): Test target vector.
        models (dict): Dictionary of {model_name: model_object}.
        param (dict): Dictionary of {model_name: hyperparameter_grid}.

    Returns:
        dict: Dictionary of {model_name: test_R2_score}.

    Raises:
        CustomException: If training or evaluation fails.
    """
    try:
        report = {}

        for model_name, model in models.items():
            model_params = param.get(model_name, {})

            # Perform grid search for best hyperparameters
            gs = GridSearchCV(model, model_params, cv=3)
            gs.fit(X_train, y_train)

            # Retrain model using best params
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Predict on both train and test
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Evaluate using R² score
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score  # Only storing test performance

        return report

    except Exception as e:
        raise CustomException(e, sys)
