"""
Unit tests for nlp-nets transformer models.
"""

import torch
import pytest


class TestBertModel:
    """Tests for BERT model."""

    @pytest.fixture
    def config(self):
        return {
            "model": {
                "name": "bert-tiny",
                "vocab_size": 100,
                "hidden_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "intermediate_size": 64,
                "max_position_embeddings": 64,
                "pad_token_id": 0,
            }
        }

    def test_bert_mlm_forward(self, config):
        from models.transformers.bert import BertForMLM
        model = BertForMLM(config)
        input_ids = torch.randint(0, 100, (2, 16))
        output = model(input_ids=input_ids)
        assert "logits" in output
        assert output["logits"].shape == (2, 16, 100)

    def test_bert_mlm_with_loss(self, config):
        from models.transformers.bert import BertForMLM
        model = BertForMLM(config)
        input_ids = torch.randint(0, 100, (2, 16))
        labels = input_ids.clone()
        output = model(input_ids=input_ids, labels=labels)
        assert "loss" in output
        assert output["loss"].item() > 0

    def test_bert_seq_class_forward(self, config):
        from models.transformers.bert import BertForSequenceClassification
        config["model"]["num_labels"] = 2
        model = BertForSequenceClassification(config)
        input_ids = torch.randint(0, 100, (2, 16))
        output = model(input_ids=input_ids)
        assert "logits" in output
        assert output["logits"].shape == (2, 2)


class TestGPTModel:
    """Tests for GPT model."""

    @pytest.fixture
    def config(self):
        return {
            "model": {
                "name": "gpt-small",
                "vocab_size": 100,
                "hidden_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "intermediate_size": 64,
                "max_position_embeddings": 64,
                "pad_token_id": 99,
            }
        }

    def test_gpt_forward(self, config):
        from models.transformers.gpt import GPTForCausalLM
        model = GPTForCausalLM(config)
        input_ids = torch.randint(0, 100, (2, 16))
        output = model(input_ids=input_ids)
        assert "logits" in output
        assert output["logits"].shape == (2, 16, 100)

    def test_gpt_with_loss(self, config):
        from models.transformers.gpt import GPTForCausalLM
        model = GPTForCausalLM(config)
        input_ids = torch.randint(0, 100, (2, 16))
        output = model(input_ids=input_ids, labels=input_ids)
        assert "loss" in output
        assert output["loss"].item() > 0


class TestT5Model:
    """Tests for T5 model."""

    @pytest.fixture
    def config(self):
        return {
            "model": {
                "name": "t5-small",
                "vocab_size": 100,
                "hidden_size": 32,
                "num_hidden_layers": 2,
                "num_decoder_layers": 2,
                "num_attention_heads": 4,
                "intermediate_size": 64,
                "max_position_embeddings": 64,
                "pad_token_id": 0,
            }
        }

    def test_t5_forward(self, config):
        from models.transformers.t5 import T5ForConditionalGeneration
        model = T5ForConditionalGeneration(config)
        input_ids = torch.randint(0, 100, (2, 16))
        decoder_input_ids = torch.randint(0, 100, (2, 8))
        output = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        assert "logits" in output
        assert output["logits"].shape == (2, 8, 100)

    def test_t5_with_loss(self, config):
        from models.transformers.t5 import T5ForConditionalGeneration
        model = T5ForConditionalGeneration(config)
        input_ids = torch.randint(0, 100, (2, 16))
        labels = torch.randint(0, 100, (2, 8))
        output = model(input_ids=input_ids, labels=labels)
        assert "loss" in output
        assert output["loss"].item() > 0
