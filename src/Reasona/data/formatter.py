from typing import Dict, Any
from Reasona.utils.logger import setup_logger

logger = setup_logger(__name__, "logs/data/formatter.json")


class DataFormatter:
    def format_sample(self, sample: dict) -> dict:
        
        query = (sample.get("query") or "").strip()
        reasoning = (sample.get("synthetic_reasoning") or "").strip()
        answer = (sample.get("synthetic_answer") or "").strip()

        text_parts = [part for part in [query, reasoning, answer] if part]
        text = " ".join(text_parts)

        if not text:
            logger.warning(
                "Formatted text is empty | sample keys=%s",
                list(sample.keys())
            )

        return {
            "text": text,
            "query": query,
            "reasoning": reasoning,
            "answer": answer,
            "metadata": {
                "source": "PleIAs/SYNTH",
                "lang": sample.get("language"),
                "exercise": sample.get("exercise"),
                "model": sample.get("model"),
                "synth_id": sample.get("synth_id"),
            },
        }
