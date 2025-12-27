import os
import joblib
import pandas as pd
from Reasona.utils.logger import setup_logger
from Reasona.config.config_manager import ConfigurationManager

logger = setup_logger("training_pipeline", "logs/pipeline/training_pipeline.json")

class TrainingPipeline:
    def __init__(self):
        logger.info("Initializing TrainingPipeline()")
        cfg = ConfigurationManager()
        self.train_cfg = cfg.get_training_config()

    def load_data(self):
        logger.info("Loading transformed dataset...")

        data_path = self.train_cfg.transformed_data_path
        if not os.path.exists(data_path):
            logger.error(f"Transformed dataset not found: {data_path}")
            return None

        df = pd.read_json(data_path, lines=True)
        logger.info(f"Loaded dataset with {len(df)} rows")
        return df

    def train_model(self, df):
        logger.info("Training model (placeholder)...")
        model = {"model": "dummy", "status": "ok"}
        return model

    def save_model(self, model):
        logger.info("Saving model...")
        os.makedirs(self.train_cfg.output_dir, exist_ok=True)
        model_path = os.path.join(self.train_cfg.output_dir, "model.pkl")
        joblib.dump(model, model_path)
        logger.info(f"Model saved to: {model_path}")

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
