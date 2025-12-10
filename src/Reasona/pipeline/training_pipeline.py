from Reasona.utils.logger import setup_logger
from Reasona.config.config_manager import ConfigurationManager

import os
import joblib
import pandas as pd


logger = setup_logger("logs/pipeline/training_pipeline.log")


class TrainingPipeline:
    def __init__(self):
        logger.info("Initializing TrainingPipeline()")

        cfg = ConfigurationManager()
        self.train_cfg = cfg.get_training_config()

    # --------------------------------
    # 1. LOAD PREPROCESSED DATA
    # --------------------------------
    def load_data(self):
        logger.info("Loading transformed dataset...")

        data_path = self.train_cfg.transformed_data_path  # Path from config
        if not os.path.exists(data_path):
            logger.error(f"Transformed dataset not found: {data_path}")
            return None

        df = pd.read_csv(data_path)
        logger.info(f"Loaded dataset with {len(df)} rows")

        return df

    # --------------------------------
    # 2. TRAIN MODEL
    # --------------------------------
    def train_model(self, df):

        # Dummy model placeholder for DVC correctness
        logger.info("Training model (placeholder)...")

        model = {"model": "dummy", "status": "ok"}

        return model

    # --------------------------------
    # 3. SAVE MODEL
    # --------------------------------
    def save_model(self, model):
        logger.info("Saving model...")

        out_dir = self.train_cfg.output_dir
        os.makedirs(out_dir, exist_ok=True)

        model_path = os.path.join(out_dir, "model.pkl")
        joblib.dump(model, model_path)

        logger.info(f"Model saved to: {model_path}")

    # --------------------------------
    # MAIN PIPELINE
    # --------------------------------
    def run(self):
        logger.info("=== TRAINING PIPELINE STARTED ===")

        df = self.load_data()
        if df is None:
            logger.error("Stopping training: no dataset found.")
            return

        model = self.train_model(df)
        self.save_model(model)

        logger.info("=== TRAINING PIPELINE FINISHED ===")


if __name__ == "__main__":
    TrainingPipeline().run()
