from __future__ import annotations

from typing import Optional


class LocalQwenLLM:
    def __init__(
        self,
        model_id: str,
        quantization: str = "int4",
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ) -> None:
        self.model_id = model_id
        self.quantization = quantization
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self._model = None
        self._tokenizer = None

    def _ensure_model(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # type: ignore

        quant = (self.quantization or "").lower()
        bnb_config: Optional[BitsAndBytesConfig] = None
        if quant in {"int4", "4bit", "4-bit"}:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif quant in {"int8", "8bit", "8-bit"}:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
            trust_remote_code=True,
        )
        self._model.eval()

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self._ensure_model()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        tokenizer = self._tokenizer
        model = self._model
        if tokenizer is None or model is None:
            return ""
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        input_ids = input_ids.to(model.device)
        outputs = model.generate(
            input_ids,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=self.temperature,
            top_p=self.top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated = outputs[0][input_ids.shape[-1]:]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()
