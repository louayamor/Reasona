from typing import Iterator, Dict, Any
from Reasona.utils.logger import setup_logger
from Reasona.data.loader import StreamingDatasetProcessor
from Reasona.data.formatter import DataFormatter
from Reasona.entities.config_entity import PreprocessConfig

logger = setup_logger(__name__, "logs/pipeline/preprocess_pipeline.json")


class PreprocessPipeline:
    """
    Streaming data PRODUCER.
    Yields preprocessed samples one-by-one.
    """

    def __init__(self, cfg: PreprocessConfig):
        logger.info("Initializing PreprocessPipeline (stream producer)")
        self.cfg = cfg

        self.loader = StreamingDatasetProcessor(
            dataset_name=cfg.dataset_name,
            revision=cfg.revision,
        )

        self.formatter = DataFormatter()

    def stream(self) -> Iterator[Dict[str, Any]]:
        logger.info("=== PREPROCESS STREAM STARTED ===")

        stream = self.loader.stream_samples(
            split=self.cfg.split,
            max_samples=self.cfg.max_samples,
        )

        for idx, raw_sample in enumerate(stream, start=1):
            if idx == 1:
                logger.info("Streaming first sample")

            yield self.formatter.format_sample(raw_sample)

        logger.info("=== PREPROCESS STREAM ENDED ===")
