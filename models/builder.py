"""
Model builder — constructs NLP models from config dicts using the MODEL_REGISTRY.
"""

from typing import Dict, Any

from utils.registry import Registry
from models.base_model import BaseNLPModel
from models.transformers import (
    BertForMLM,
    BertForSequenceClassification,
    GPTForCausalLM,
    T5ForConditionalGeneration,
)

# Global model registry
MODEL_REGISTRY = Registry(
    registry_name="models",
    base_class=BaseNLPModel,
    lazy_load_dirs=["models/transformers"],
    internal_dirs=["internal", "internal/projects/*"],
)


def build_model(opts: Dict[str, Any]) -> BaseNLPModel:
    """
    Build a model from configuration.

    Args:
        opts: Configuration dictionary (typically loaded from YAML).

    Returns:
        An initialized model instance.
    """
    model_config = opts.get("model", {})
    model_name = model_config.get("name", "").lower()

    if not model_name:
        raise ValueError("Model name must be specified in config under 'model.name'")

    # Map common model names to registry keys
    model_registry_map = {
        "bert-base": "bert_mlm",
        "bert": "bert_mlm",
        "bert-large": "bert_mlm",
        "bert-tiny": "bert_mlm",
        "bert-mini": "bert_mlm",
        "bert-small": "bert_mlm",
        "bert-medium": "bert_mlm",
        "bert-seq-class": "bert_seq_class",
        "gpt": "gpt_causal_lm",
        "gpt-small": "gpt_causal_lm",
        "gpt-medium": "gpt_causal_lm",
        "gpt-large": "gpt_causal_lm",
        "gpt2": "gpt_causal_lm",
        "t5": "t5_seq2seq",
        "t5-small": "t5_seq2seq",
        "t5-base": "t5_seq2seq",
        "t5-large": "t5_seq2seq",
    }

    registry_key = model_registry_map.get(model_name, model_name)

    try:
        model = MODEL_REGISTRY[registry_key](opts)
    except (KeyError, AssertionError) as e:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available models: {', '.join(MODEL_REGISTRY.keys())}. "
            f"Error: {e}"
        )

    return model
