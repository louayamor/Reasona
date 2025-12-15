from Reasona.utils.logger import setup_logger
from Reasona.pipeline.preprocess_pipeline import PreprocessPipeline
from Reasona.pipeline.training_pipeline import TrainingPipeline
##from Reasona.pipeline.inference_pipeline import InferencePipeline

logger = setup_logger("main_pipeline", "logs/pipeline/main_pipeline.log")

def run_stage(stage_name, pipeline_cls):
    stage_logger = setup_logger(f"{stage_name.lower()}_stage", f"logs/pipeline/{stage_name.lower()}_stage.log")
    stage_logger.info(f"===== Starting: {stage_name} =====")
    try:
        pipeline = pipeline_cls()
        pipeline.run()  
        stage_logger.info(f"===== Completed: {stage_name} =====\n")
    except Exception as e:
        stage_logger.exception(f"Error during {stage_name}: {e}")
        raise e

if __name__ == "__main__":
    run_stage("Preprocessing", PreprocessPipeline)
    run_stage("Training", TrainingPipeline)
    ##run_stage("Inference", InferencePipeline)
