from sklearn.metrics import r2_score
from src.utils import save_object, evaluate_models
from src.exception import CustomException
from src.logger import logging
import sys


class ModelTrainer:
    """
    Handles training, hyperparameter tuning, evaluating,
    and returning/saving the best model.
    """
    def __init__(self, model_save_path):
        self.model_save_path = model_save_path

    def train_and_evaluate(self, X_train, y_train, X_test, y_test, models, params):
        try:
            logging.info("Starting model training and evaluation...")

            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params
            )

            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No suitable model found with R² score >= 0.6")

            logging.info(f"Best model: {best_model_name} with score: {best_model_score}")

            # Save the best model
            save_object(file_path=self.model_save_path, obj=best_model)

            # Predict and return R² score
            y_pred = best_model.predict(X_test)
            r2_square = r2_score(y_test, y_pred)

            return best_model, r2_square
    
        except Exception as e:
            raise CustomException(e, sys)

