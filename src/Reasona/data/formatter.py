from typing import Dict, Any
from Reasona.utils.logger import setup_logger

logger = setup_logger(__name__, "logs/data/formatter.json")


class DataFormatter:
    def format_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        text_parts = []

        exercise = sample.get("exercise")
        if isinstance(exercise, dict):
            for key in ("instruction", "input", "output"):
                val = exercise.get(key)
                if isinstance(val, str) and val.strip():
                    text_parts.append(val.strip())

        if not text_parts:
            raw_text = sample.get("text")
            if isinstance(raw_text, str) and raw_text.strip():
                text_parts.append(raw_text.strip())

        final_text = "\n\n".join(text_parts)

        if not final_text:
            logger.warning(
                "Formatted text is empty | sample keys=%s",
                list(sample.keys()),
            )

        return {
            "text": final_text,
            "language": sample.get("language"),
            "synth_id": sample.get("synth_id"),
            "_metadata": sample,  
        }
