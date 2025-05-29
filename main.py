from src.components.data_ingestion import DataIngestion 
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationconfig
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig


# Entry point for the end-to-end Machine Learning pipeline
if __name__ == "__main__":
    # Step 1: Initialize Data Ingestion and retrieve train and test data paths
    data_ingestion = DataIngestion()
    train_data, test_data = data_ingestion.initiate_data_ingestion()

    # Step 2: Initialize Data Transformation and transform the data into arrays suitable for ML models
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    # Step 3: Initialize Model Trainer, train models, evaluate them, and print the best model's R² score
    model_trainer = ModelTrainer()
    r2_score_result = model_trainer.initiate_model_trainer(train_arr, test_arr)
    print(f"Best model R² score: {r2_score_result}")
