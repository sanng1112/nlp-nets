"""
NLP collator functions for dynamic batching.

Provides padding, masking, and data preparation utilities
compatible with BERT/GPT/T5 model inputs.
"""

from typing import Any, Dict, List, Optional, Union

import torch
from torch.nn.utils.rnn import pad_sequence


def nlpcollate(batch: List[Dict[str, Any]], padding_value: int = 0) -> Dict[str, torch.Tensor]:
    """Collate a batch of NLP samples with dynamic padding.

    Handles variable-length sequences by padding to the longest
    sequence in the batch. Produces attention masks automatically.

    Args:
        batch: List of dicts, each containing ``input_ids`` and optionally
            ``attention_mask``, ``token_type_ids``, ``labels``.
        padding_value: Token ID used for padding (default: 0).

    Returns:
        Padded batch dict with keys:
            ``input_ids``: (batch, max_len)
            ``attention_mask``: (batch, max_len)
            ``token_type_ids``: (batch, max_len), optional
            ``labels``: (batch, max_len), optional
    """
    input_ids = [torch.tensor(item["input_ids"], dtype=torch.long) for item in batch]

    # Pad sequences to max length in batch
    padded = pad_sequence(input_ids, batch_first=True, padding_value=padding_value)

    # Attention mask: 1 for real tokens, 0 for padding
    attention_mask = (padded != padding_value).long()

    result: Dict[str, torch.Tensor] = {
        "input_ids": padded,
        "attention_mask": attention_mask,
    }

    # Optional token_type_ids (BERT-style segment embeddings)
    if "token_type_ids" in batch[0]:
        type_ids = [torch.tensor(item["token_type_ids"], dtype=torch.long) for item in batch]
        padded_types = pad_sequence(type_ids, batch_first=True, padding_value=0)
        result["token_type_ids"] = padded_types

    # Optional labels (for MLM / NER / classification)
    if "labels" in batch[0]:
        labels = [torch.tensor(item["labels"], dtype=torch.long) for item in batch]
        # For sequence-level tasks, keep labels as-is
        if labels[0].dim() == 0:
            result["labels"] = torch.stack(labels)
        else:
            # Pad labels with -100 so they are ignored in loss computation
            padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)
            result["labels"] = padded_labels

    return result


def causal_lm_collate(batch: List[Dict[str, Any]], padding_value: int = 0) -> Dict[str, torch.Tensor]:
    """Collate for causal language modeling (GPT-style).

    Shifts labels so the model predicts the next token.

    Args:
        batch: List of dicts with ``input_ids``.
        padding_value: Token ID for padding.

    Returns:
        dict with ``input_ids`` and ``labels``.
    """
    result = nlpcollate(batch, padding_value=padding_value)
    # For causal LM, labels = input_ids (shifted inside model or loss)
    if "labels" not in result:
        result["labels"] = result["input_ids"].clone()
    return result


def mlm_collate(batch: List[Dict[str, Any]], padding_value: int = 0) -> Dict[str, torch.Tensor]:
    """Collate for masked language modeling (BERT-style).

    Expects ``input_ids`` and ``labels`` already prepared by the
    tokenizer (with ``[MASK]`` tokens applied).

    Args:
        batch: List of dicts with ``input_ids`` and ``labels``.
        padding_value: Token ID for padding.

    Returns:
        dict with ``input_ids``, ``attention_mask``, ``labels``.
    """
    return nlpcollate(batch, padding_value=padding_value)
