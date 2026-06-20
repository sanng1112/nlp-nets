"""
Unit tests for nlp-nets trainer and data factory.
"""

import torch
import pytest


class TestTextDataset:
    """Tests for TextDataset in data_factory."""

    def test_with_tokenizer(self):
        from engine.data_factory import TextDataset
        from transformers import AutoTokenizer

        try:
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        except Exception:
            pytest.skip("HuggingFace tokenizer not available")

        texts = ["hello world", "test sentence here"]
        ds = TextDataset(texts=texts, tokenizer=tokenizer, max_length=16, task_type="mlm")
        item = ds[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item

    def test_without_tokenizer(self):
        from engine.data_factory import TextDataset
        texts = ["hello", "world"]
        ds = TextDataset(texts=texts, tokenizer=None, max_length=16, task_type="mlm")
        item = ds[0]
        assert "input_ids" in item


class TestCollateFn:
    """Tests for collate_fn."""

    def test_collate_variable_length(self):
        from engine.data_factory import collate_fn
        import torch

        batch = [
            {"input_ids": torch.tensor([1, 2, 3]), "attention_mask": torch.tensor([1, 1, 1]), "labels": torch.tensor([1, 2, 3])},
            {"input_ids": torch.tensor([4, 5]), "attention_mask": torch.tensor([1, 1]), "labels": torch.tensor([4, 5])},
        ]

        result = collate_fn(batch)
        assert result["input_ids"].shape == (2, 3)
        assert result["attention_mask"].shape == (2, 3)
        assert result["labels"].shape == (2, 3)


class TestTrainer:
    """Tests for the Trainer class."""

    @pytest.fixture
    def opts(self):
        return {
            "train": {
                "epochs": 1,
                "batch_size": 4,
                "device": "cpu",
                "use_amp": False,
                "output_dir": "/tmp/test_nlp_nets",
                "gradient_accumulation_steps": 1,
                "max_grad_norm": 1.0,
                "log_interval": 10,
                "save_every": 100,
            },
            "model": {"name": "bert-tiny", "vocab_size": 100, "hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4},
        }

    def test_trainer_init(self, opts):
        from engine.trainer import Trainer
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()

        trainer = Trainer(
            opts=opts,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
        )
        assert trainer.epochs == 1
        assert trainer.device.type == "cpu"

    def test_prepare_batch_list(self, opts):
        from engine.trainer import Trainer
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()

        trainer = Trainer(
            opts=opts,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
        )

        batch = [torch.randint(0, 100, (2, 16)), torch.randint(0, 100, (2, 16))]
        inputs, targets = trainer._prepare_batch(batch)
        assert "input_ids" in inputs
        assert targets is not None

    def test_prepare_batch_dict(self, opts):
        from engine.trainer import Trainer
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()

        trainer = Trainer(
            opts=opts,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
        )

        batch = {"input_ids": torch.randint(0, 100, (2, 16)), "labels": torch.randint(0, 100, (2, 16))}
        inputs, targets = trainer._prepare_batch(batch)
        assert "input_ids" in inputs
        assert targets is not None
