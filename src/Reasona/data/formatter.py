from typing import Dict, Any
from Reasona.utils.logger import setup_logger

logger = setup_logger(__name__, "logs/data/formatter.json")


class DataFormatter:
    def format_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:

        return {
            "instruction": sample.get("instruction") or sample.get("prompt"),
            "input": sample.get("input", ""),
            "output": sample.get("output") or sample.get("completion"),
            "metadata": {
                "source": "PleIAs/SYNTH",
                "lang": sample.get("lang"),
            },
        }
