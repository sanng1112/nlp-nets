"""
Inference engine for running trained models on new text.
"""

from typing import Any, Dict, List, Optional, Union

import torch
from torch import nn, Tensor

from utils import logger


class InferenceEngine:
    """
    Inference engine for trained NLP models.

    Supports:
    - Text generation (autoregressive)
    - Encoder-only inference (BERT-style)
    - Encoder-decoder inference (T5-style)
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        Run inference on a single text.

        Args:
            text: Input text string.
            **kwargs: Additional generation parameters.

        Returns:
            Dict with 'logits', 'predictions', and optionally 'generated_text'.
        """
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        logits = outputs.get("logits")

        result = {"logits": logits}

        if logits is not None:
            predictions = logits.argmax(dim=-1)
            result["predictions"] = predictions

        return result

    @torch.no_grad()
    def generate(
        self,
        text: str,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
    ) -> str:
        """
        Autoregressive text generation.

        Args:
            text: Prompt text.
            max_length: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k filtering.
            top_p: Nucleus (top-p) filtering.
            do_sample: Whether to sample (vs greedy).
            eos_token_id: End-of-sequence token ID.

        Returns:
            Generated text string.
        """
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(self.device)

        generated = input_ids
        past_key_values = None

        for _ in range(max_length):
            with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
                outputs = self.model(
                    input_ids=generated if past_key_values is None else generated[:, -1:],
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            logits = outputs["logits"][:, -1, :]
            past_key_values = outputs.get("past_key_values")

            # Apply temperature
            logits = logits / temperature

            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float("-inf")

            # Sample or greedy
            if do_sample:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)

            # Stop at EOS
            if eos_token_id is not None and (next_token == eos_token_id).any():
                break

        # Decode
        generated_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        return generated_text
