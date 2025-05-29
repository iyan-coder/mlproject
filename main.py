from src.pipeline.train_pipeline import TrainPipeline

if __name__ == "__main__":
    RAW_DATA_PATH = "artifacts/data.csv"
    MODEL_SAVE_PATH = "artifacts/model.pkl"
    PREPROCESSOR_SAVE_PATH = "artifacts/preprocessor.pkl"

    pipeline = TrainPipeline(
        raw_data_path=RAW_DATA_PATH,
        model_save_path=MODEL_SAVE_PATH,
        preprocessor_save_path=PREPROCESSOR_SAVE_PATH
    )
    r2_square = pipeline.run()

    # ✅ Display the result
    print(f"Best model R² score: {r2_square}")

