from typing import Dict, Optional
import time
import uuid

from Reasona.pipeline.inference_pipeline import InferencePipeline
from Reasona.utils.logger import setup_logger

logger = setup_logger(
    "reasona_service",
    "logs/service/reasona_service.json",
)


class ReasonaService:

    def __init__(self, inference_pipeline: InferencePipeline):
        self.inference_pipeline = inference_pipeline
        logger.info("ReasonaService initialized with InferencePipeline")

    def answer(self, query: str) -> Dict[str, object]:

        request_id = uuid.uuid4().hex[:10]
        start_time = time.time()
        logger.info("New request started | id=%s | query='%s'", request_id, query)

        try:
            self._validate_query(query)
            logger.debug("Query validated | id=%s", request_id)

            result = self.inference_pipeline.execute(query)
            logger.info(
                "InferencePipeline executed | id=%s | chunks=%d | total_latency=%.3fs",
                request_id,
                len(result["chunks"]),
                result["stats"]["latency_total"],
            )

            response = self._build_response(
                request_id=request_id,
                result=result,
                latency=time.time() - start_time,
            )

            logger.info(
                "Request completed successfully | id=%s | total_latency=%.3fs",
                request_id,
                response["stats"]["latency_total"],
            )

            return response

        except Exception as e:
            logger.exception("Request failed | id=%s", request_id)
            return self._error_response(
                request_id=request_id,
                error=str(e),
                latency=time.time() - start_time,
            )

    @staticmethod
    def _validate_query(query: str) -> None:
        if not query or not isinstance(query, str):
            raise ValueError("query must be a non-empty string")
        if len(query.strip()) < 3:
            raise ValueError("query too short")

    @staticmethod
    def _build_response(
        request_id: str,
        result: Dict[str, object],
        latency: float,
    ) -> Dict[str, object]:
        return {
            "id": request_id,
            "query": result["query"],
            "answer": result["answer"],
            "chunks": result["chunks"],
            "prompt": result.get("prompt", ""),
            "stats": {
                **result.get("stats", {}),
                "latency_total": latency,
            },
        }

    @staticmethod
    def _error_response(
        request_id: str,
        error: str,
        latency: float,
    ) -> Dict[str, object]:
        return {
            "id": request_id,
            "error": error,
            "stats": {
                "latency_total": latency,
            },
        }
