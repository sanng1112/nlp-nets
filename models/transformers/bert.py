"""
BERT model implementation (encoder-only transformer).

Supports:
- Masked Language Modeling (BertForMLM)
- Sequence Classification (BertForSequenceClassification)
"""

from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from models.base_model import BaseNLPModel
from utils.registry import Registry
from utils import logger

from layers.normalization import LayerNorm
from layers.embeddings import BERTEmbeddings
from layers.attention import MultiHeadAttention
from layers.feedforward import PositionwiseFeedForward


class BertConfig:
    """Configuration holder for BERT models."""

    def __init__(self, opts: Dict[str, Any]) -> None:
        model_cfg = opts.get("model", {})
        self.vocab_size = model_cfg.get("vocab_size", 30522)
        self.hidden_size = model_cfg.get("hidden_size", 768)
        self.num_hidden_layers = model_cfg.get("num_hidden_layers", 12)
        self.num_attention_heads = model_cfg.get("num_attention_heads", 12)
        self.intermediate_size = model_cfg.get("intermediate_size", self.hidden_size * 4)
        self.hidden_dropout_prob = model_cfg.get("hidden_dropout_prob", 0.1)
        self.attention_probs_dropout_prob = model_cfg.get("attention_probs_dropout_prob", 0.1)
        self.max_position_embeddings = model_cfg.get("max_position_embeddings", 512)
        self.type_vocab_size = model_cfg.get("type_vocab_size", 2)
        self.pad_token_id = model_cfg.get("pad_token_id", 0)
        self.layer_norm_eps = model_cfg.get("layer_norm_eps", 1e-12)
        self.num_labels = model_cfg.get("num_labels", 2)


class BertSelfAttention(nn.Module):
    """BERT self-attention layer."""

    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        output, _ = self.attention(hidden_states, attention_mask=attention_mask)
        return output


class BertLayer(nn.Module):
    """Single BERT transformer layer: Self-Attention + FFN with residual connections."""

    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.attention = BertSelfAttention(config)
        self.attention_output = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_layer_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn = PositionwiseFeedForward(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            dropout=config.hidden_dropout_prob,
            activation="gelu",
        )
        self.ffn_output = nn.Linear(config.hidden_size, config.hidden_size)
        self.ffn_layer_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # Self-attention block
        attn_output = self.attention(hidden_states, attention_mask)
        attn_output = self.attention_output(attn_output)
        attn_output = self.dropout(attn_output)
        hidden_states = self.attention_layer_norm(hidden_states + attn_output)

        # FFN block
        ffn_output = self.ffn(hidden_states)
        ffn_output = self.ffn_output(ffn_output)
        ffn_output = self.dropout(ffn_output)
        hidden_states = self.ffn_layer_norm(hidden_states + ffn_output)

        return hidden_states


class BertEncoder(nn.Module):
    """Stack of BERT transformer layers."""

    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        output_hidden_states: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, ...]]]:
        all_hidden_states = () if output_hidden_states else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            hidden_states = layer(hidden_states, attention_mask)

        return hidden_states, all_hidden_states


class BertModel(BaseNLPModel):
    """
    BERT encoder model (without task-specific head).

    Registry name: "bert"
    """

    def __init__(self, opts: Dict[str, Any]) -> None:
        super().__init__(opts)
        self.config = BertConfig(opts)
        self.embeddings = BERTEmbeddings(
            vocab_size=self.config.vocab_size,
            hidden_size=self.config.hidden_size,
            max_seq_length=self.config.max_position_embeddings,
            num_segments=self.config.type_vocab_size,
            dropout=self.config.hidden_dropout_prob,
            padding_idx=self.config.pad_token_id,
        )
        self.encoder = BertEncoder(self.config)
        self.pooler = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.Tanh(),
        )
        self.init_weights()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        output_hidden_states: bool = False,
    ) -> Dict[str, Tensor]:
        if attention_mask is None:
            attention_mask = (input_ids != self.config.pad_token_id).long()

        # Expand attention mask for multi-head attention: (B, 1, 1, S)
        extended_mask = attention_mask[:, None, None, :].float()
        extended_mask = (1.0 - extended_mask) * torch.finfo(extended_mask.dtype).min

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoder_output, hidden_states = self.encoder(
            embedding_output,
            attention_mask=extended_mask,
            output_hidden_states=output_hidden_states,
        )

        # Pooled output: use [CLS] token
        pooled = self.pooler(encoder_output[:, 0, :])

        return {
            "last_hidden_state": encoder_output,
            "pooler_output": pooled,
            "hidden_states": hidden_states,
        }

    def init_weights(self) -> None:
        self.embeddings.init_weights()
        for module in self.modules():
            if isinstance(module, nn.Linear) and module not in [self.pooler[0]]:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def get_input_embeddings(self) -> nn.Module:
        return self.embeddings.token.embedding

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.embeddings.token.embedding = value


class BertPredictionHead(nn.Module):
    """BERT MLM prediction head (tied with input embeddings)."""

    def __init__(self, config: BertConfig, embedding_weight: Optional[Tensor] = None) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.activation = nn.GELU()
        if embedding_weight is not None:
            self.decoder.weight = embedding_weight

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        logits = self.decoder(hidden_states)
        return logits


# Register models
bert_registry = Registry("bert_models", base_class=BaseNLPModel)


@bert_registry.register(name="bert_mlm")
class BertForMLM(BaseNLPModel):
    """
    BERT for Masked Language Modeling.
    Registry keys: "bert_mlm", "bert"

    Config example:
        model:
          name: "bert-base"
          vocab_size: 30522
          hidden_size: 768
          num_hidden_layers: 12
          num_attention_heads: 12
    """

    def __init__(self, opts: Dict[str, Any]) -> None:
        super().__init__(opts)
        self.config = BertConfig(opts)
        self.bert = BertModel(opts)
        self.cls = BertPredictionHead(self.config, self.bert.get_input_embeddings().weight)
        self.init_weights()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        sequence_output = outputs["last_hidden_state"]
        logits = self.cls(sequence_output)

        result = {"logits": logits}
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            result["loss"] = loss_fn(logits.view(-1, self.config.vocab_size), labels.view(-1))
        return result

    def init_weights(self) -> None:
        self.bert.init_weights()
        nn.init.xavier_uniform_(self.cls.dense.weight)
        if self.cls.dense.bias is not None:
            nn.init.zeros_(self.cls.dense.bias)


@bert_registry.register(name="bert_seq_class")
class BertForSequenceClassification(BaseNLPModel):
    """
    BERT for sequence classification.
    Registry key: "bert_seq_class"
    """

    def __init__(self, opts: Dict[str, Any]) -> None:
        super().__init__(opts)
        self.config = BertConfig(opts)
        self.bert = BertModel(opts)
        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        pooled_output = outputs["pooler_output"]
        logits = self.classifier(pooled_output)

        result = {"logits": logits}
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            result["loss"] = loss_fn(logits.view(-1, self.config.num_labels), labels.view(-1))
        return result

    def init_weights(self) -> None:
        self.bert.init_weights()
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)
