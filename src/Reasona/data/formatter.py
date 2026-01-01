from typing import Dict, Any
from Reasona.utils.logger import setup_logger

logger = setup_logger(__name__, "logs/data/formatter.json")


class DataFormatter:
    def format_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        instruction = sample.get("instruction") or sample.get("prompt")
        input_text = sample.get("input", "")
        output = sample.get("output") or sample.get("completion")

        text = " ".join(
            part for part in [instruction, input_text, output] if part
        )

        return {
            "text": text,                      
            "instruction": instruction,
            "input": input_text,
            "output": output,
            "metadata": {
                "source": "PleIAs/SYNTH",
                "lang": sample.get("lang"),
            },
        }
