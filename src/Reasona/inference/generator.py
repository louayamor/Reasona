from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class Generator:
    """
    Text generation wrapper (HF-based)
    """

    def __init__(self, cfg):
        self.cfg = cfg

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model,
            use_fast=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
        ).to(self.model.device)

        output = self.model.generate(
            **inputs,
            max_new_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            top_p=getattr(self.cfg, "top_p", 0.9),
            repetition_penalty=getattr(
                self.cfg, "repetition_penalty", 1.0
            ),
            do_sample=True,
        )

        return self.tokenizer.decode(
            output[0],
            skip_special_tokens=True,
        )
