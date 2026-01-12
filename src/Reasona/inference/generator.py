import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
)
from Reasona.entities.config_entity import GeneratorConfig
from Reasona.utils.logger import setup_logger

logger = setup_logger("generator", "logs/inference/generator.json")


class Generator:
    def __init__(self, cfg: GeneratorConfig):
        self.cfg = cfg
        self.model = None
        self.tokenizer = None
        self.pipeline = None

    def load(self):
        logger.info("Loading generator model | model=%s", self.cfg.model)

        # ---- Quantization config (modern, non-deprecated)
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        # ---- Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model,
            use_fast=True,
            trust_remote_code=True,
        )

        # Qwen requires this
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Tokenizer loaded successfully.")

        # ---- Model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model,
            device_map="auto",
            quantization_config=quant_config,
            dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        self.model.eval()
        logger.info("Model loaded successfully (4-bit NF4).")

        # ---- Pipeline (NO device/device_map here)
        self.pipeline = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            repetition_penalty=self.cfg.repetition_penalty,
            do_sample=True,
        )

        logger.info("Generator pipeline ready.")

    def generate(self, prompt: str) -> str:
        if self.pipeline is None:
            self.load()

        with torch.inference_mode():
            output = self.pipeline(
                prompt,
                truncation=True,
                return_full_text=False,
            )

        return output[0]["generated_text"]

    def unload(self):
        """Explicitly free GPU memory if needed"""
        if self.pipeline:
            del self.pipeline
        if self.model:
            del self.model
        torch.cuda.empty_cache()
