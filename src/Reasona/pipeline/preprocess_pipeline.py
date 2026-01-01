from typing import Iterator, Dict, Any
from Reasona.utils.logger import setup_logger
from Reasona.data.loader import StreamingDatasetProcessor
from Reasona.data.formatter import DataFormatter
from Reasona.entities.config_entity import PreprocessConfig  

logger = setup_logger(__name__, "logs/pipeline/preprocess_pipeline.json")


class PreprocessPipeline:
    def __init__(self, cfg: PreprocessConfig = None):
        logger.info("Initializing PreprocessPipeline")

        # Use config from ConfigurationManager if not passed explicitly
        if cfg is None:
            from Reasona.config.config_manager import ConfigurationManager
            cfg = ConfigurationManager().get_preprocess_config()
        self.cfg = cfg

        self.processor = StreamingDatasetProcessor(
            dataset_name=self.cfg.dataset_name,
            revision=self.cfg.revision,
        )

        self.formatter = DataFormatter()

    def run(self) -> Iterator[Dict[str, Any]]:
        logger.info("=== PREPROCESSING PIPELINE STARTED ===")
        logger.info("Starting stream")

        stream = self.processor.stream_samples(
            split=self.cfg.split,
            max_samples=self.cfg.max_samples or float("inf"),
        )

        for i, sample in enumerate(stream, start=1):
            if i == 1:
                logger.info("Streaming started")

            formatted = self.formatter.format_sample(sample)
            yield formatted
