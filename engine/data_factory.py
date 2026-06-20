"""
Data Factory — builds dataloaders for NLP datasets.

Automatically:
1. Loads tokenized datasets from HuggingFace datasets or local files
2. Creates PyTorch DataLoaders
3. Handles DDP (DistributedDataParallel) transparently
"""

from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from utils import logger


class TextDataset(Dataset):
    """Wrapper for text datasets with tokenization on-the-fly."""

    def __init__(
        self,
        texts: list,
        labels: Optional[list] = None,
        tokenizer: Any = None,
        max_length: int = 512,
        task_type: str = "mlm",
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_type = task_type

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Any:
        text = self.texts[idx]

        if self.tokenizer is not None:
            # Use provided tokenizer
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].squeeze(0)
            attention_mask = encoding.get("attention_mask", torch.ones_like(input_ids)).squeeze(0)
        else:
            # Fallback: character-level tokenization
            input_ids = torch.tensor([ord(c) for c in text[:self.max_length]], dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)

        # For MLM, labels are input_ids (model handles masking internally)
        if self.task_type in ("mlm",):
            labels = input_ids.clone()
        elif self.task_type in ("causal_lm",):
            labels = input_ids.clone()
        elif self.labels is not None:
            labels = torch.tensor(self.labels[idx], dtype=torch.long)
        else:
            labels = torch.zeros(1, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def collate_fn(batch: list) -> Dict[str, torch.Tensor]:
    """Collate function for variable-length sequences."""
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]

    # Pad sequences
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def build_dataloaders(
    opts: Dict[str, Any],
    tokenizer: Any = None,
) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
    """
    Build training and validation dataloaders from configuration.

    Args:
        opts: Configuration dictionary.
        tokenizer: Tokenizer instance.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    dataset_config = opts.get("dataset", {})
    train_config = opts.get("train", {})

    dataset_name = dataset_config.get("name", "wikitext-2").lower()
    batch_size = train_config.get("batch_size", 32)
    max_length = dataset_config.get("max_seq_length", 512)
    task_type = opts.get("task", {}).get("type", "mlm")
    num_workers = dataset_config.get("num_workers", 2)
    split = dataset_config.get("split", "train")

    # Try HuggingFace datasets first
    try:
        from datasets import load_dataset

        if dataset_name == "wikitext-2":
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
            train_texts = dataset["text"]
            train_labels = None
            # Use same for validation
            val_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
            val_texts = val_dataset["text"]
            val_labels = None
        elif dataset_name == "wikitext-103":
            dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
            train_texts = dataset["text"]
            train_labels = None
            val_dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
            val_texts = val_dataset["text"]
            val_labels = None
        elif dataset_name == "imdb":
            dataset = load_dataset("imdb", split=split)
            train_texts = dataset["text"]
            train_labels = dataset["label"]
            val_dataset = load_dataset("imdb", split="test")
            val_texts = val_dataset["text"]
            val_labels = val_dataset["label"]
        elif dataset_name == "sst2":
            dataset = load_dataset("glue", "sst2", split=split)
            train_texts = dataset["sentence"]
            train_labels = dataset["label"]
            val_dataset = load_dataset("glue", "sst2", split="validation")
            val_texts = val_dataset["sentence"]
            val_labels = val_dataset["label"]
        else:
            # Try loading by exact name
            try:
                dataset = load_dataset(dataset_name, split=split)
                # Assume 'text' and optionally 'label' columns
                if "text" in dataset.features:
                    train_texts = dataset["text"]
                elif "sentence" in dataset.features:
                    train_texts = dataset["sentence"]
                else:
                    # Use first column
                    first_col = list(dataset.features.keys())[0]
                    train_texts = dataset[first_col]
                train_labels = dataset.get("label", None) if "label" in dataset.features else None

                val_dataset = load_dataset(dataset_name, split="validation")
                if "text" in val_dataset.features:
                    val_texts = val_dataset["text"]
                elif "sentence" in val_dataset.features:
                    val_texts = val_dataset["sentence"]
                else:
                    first_col = list(val_dataset.features.keys())[0]
                    val_texts = val_dataset[first_col]
                val_labels = val_dataset.get("label", None) if "label" in val_dataset.features else None
            except Exception as e:
                raise ValueError(f"Unknown dataset '{dataset_name}': {e}")

    except ImportError:
        logger.warning("datasets package not installed. Using fallback data.")
        # Fallback: create dummy data
        train_texts = ["dummy text for training"] * 100
        val_texts = ["dummy text for validation"] * 20
        train_labels = [0] * 100 if task_type == "seq_class" else None
        val_labels = [0] * 20 if task_type == "seq_class" else None

    # Filter empty texts
    train_texts = [t for t in train_texts if t and t.strip()]
    val_texts = [t for t in val_texts if t and t.strip()]

    if train_labels is not None:
        train_labels = [l for t, l in zip(train_texts, train_labels) if t and t.strip()]
    if val_labels is not None:
        val_labels = [l for t, l in zip(val_texts, val_labels) if t and t.strip()]

    logger.log(f"Dataset '{dataset_name}': {len(train_texts)} train, {len(val_texts)} validation samples")

    # Create datasets
    train_ds = TextDataset(
        texts=train_texts,
        labels=train_labels,
        tokenizer=tokenizer,
        max_length=max_length,
        task_type=task_type,
    )
    val_ds = TextDataset(
        texts=val_texts,
        labels=val_labels,
        tokenizer=tokenizer,
        max_length=max_length,
        task_type=task_type,
    )

    # DDP support
    import torch.distributed as dist
    is_ddp = dist.is_available() and dist.is_initialized()

    if is_ddp:
        from torch.utils.data.distributed import DistributedSampler

        train_sampler = DistributedSampler(train_ds)
        val_sampler = DistributedSampler(val_ds, shuffle=False)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        logger.log("[Data Factory] DistributedSampler enabled for DDP.")
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    return train_loader, val_loader
