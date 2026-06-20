"""
T5 model implementation (encoder-decoder transformer).

Supports:
- Conditional Generation (T5ForConditionalGeneration)
"""

from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from models.base_model import BaseNLPModel
from utils.registry import Registry

from layers.normalization import RMSLayerNorm
from layers.embeddings import TokenEmbedding
from layers.attention import MultiHeadAttention
from layers.feedforward import GatedFeedForward
from layers.positional_encoding import SinusoidalPositionalEncoding
from utils.tokenizer_utils import create_causal_mask


class T5Config:
    """Configuration holder for T5 models."""

    def __init__(self, opts: Dict[str, Any]) -> None:
        model_cfg = opts.get("model", {})
        self.vocab_size = model_cfg.get("vocab_size", 32128)
        self.hidden_size = model_cfg.get("hidden_size", 768)
        self.num_hidden_layers = model_cfg.get("num_hidden_layers", 12)
        self.num_decoder_layers = model_cfg.get("num_decoder_layers", self.num_hidden_layers)
        self.num_attention_heads = model_cfg.get("num_attention_heads", 12)
        self.intermediate_size = model_cfg.get("intermediate_size", self.hidden_size * 4)
        self.hidden_dropout_prob = model_cfg.get("hidden_dropout_prob", 0.1)
        self.attention_probs_dropout_prob = model_cfg.get("attention_probs_dropout_prob", 0.1)
        self.max_position_embeddings = model_cfg.get("max_position_embeddings", 512)
        self.pad_token_id = model_cfg.get("pad_token_id", 0)
        self.layer_norm_eps = model_cfg.get("layer_norm_eps", 1e-6)


class T5Attention(nn.Module):
    """T5 attention layer (supports both self-attention and cross-attention)."""

    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False) -> None:
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
        key_value_states: Optional[Tensor] = None,
    ) -> Tensor:
        output, _ = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            key_value_states=key_value_states,
        )
        return output


class T5Block(nn.Module):
    """Single T5 block (self-attention + optional cross-attention + FFN)."""

    def __init__(self, config: T5Config, has_cross_attention: bool = False) -> None:
        super().__init__()
        self.has_cross_attention = has_cross_attention

        self.self_attention = T5Attention(config)
        self.self_attention_layer_norm = RMSLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attention_dropout = nn.Dropout(config.hidden_dropout_prob)

        if has_cross_attention:
            self.cross_attention = T5Attention(config)
            self.cross_attention_layer_norm = RMSLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.cross_attention_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.ffn = GatedFeedForward(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            dropout=config.hidden_dropout_prob,
        )
        self.ffn_layer_norm = RMSLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn_dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # Self-attention (Pre-Norm)
        normed = self.self_attention_layer_norm(hidden_states)
        attn_output = self.self_attention(normed, attention_mask=attention_mask)
        hidden_states = hidden_states + self.self_attention_dropout(attn_output)

        # Cross-attention (for decoder blocks)
        if self.has_cross_attention and encoder_hidden_states is not None:
            normed = self.cross_attention_layer_norm(hidden_states)
            cross_output = self.cross_attention(
                normed,
                attention_mask=encoder_attention_mask,
                key_value_states=encoder_hidden_states,
            )
            hidden_states = hidden_states + self.cross_attention_dropout(cross_output)

        # FFN (Pre-Norm)
        normed = self.ffn_layer_norm(hidden_states)
        ffn_output = self.ffn(normed)
        hidden_states = hidden_states + self.ffn_dropout(ffn_output)

        return hidden_states


class T5Stack(nn.Module):
    """Stack of T5 blocks (encoder or decoder)."""

    def __init__(self, config: T5Config, is_decoder: bool = False) -> None:
        super().__init__()
        self.is_decoder = is_decoder
        self.embed_tokens = TokenEmbedding(config.vocab_size, config.hidden_size)
        self.embed_positions = SinusoidalPositionalEncoding(config.hidden_size, config.max_position_embeddings)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        num_layers = config.num_decoder_layers if is_decoder else config.num_hidden_layers
        self.layers = nn.ModuleList([
            T5Block(config, has_cross_attention=is_decoder)
            for _ in range(num_layers)
        ])
        self.final_layer_norm = RMSLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.embed_positions(hidden_states)
        hidden_states = self.dropout(hidden_states)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            )

        hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states


# Register models
t5_registry = Registry("t5_models", base_class=BaseNLPModel)


@t5_registry.register(name="t5_seq2seq")
class T5ForConditionalGeneration(BaseNLPModel):
    """
    T5 for conditional sequence-to-sequence generation.
    Registry keys: "t5_seq2seq", "t5"

    Config example:
        model:
          name: "t5-base"
          vocab_size: 32128
          hidden_size: 768
          num_hidden_layers: 12
          num_attention_heads: 12
    """

    def __init__(self, opts: Dict[str, Any]) -> None:
        super().__init__(opts)
        self.config = T5Config(opts)

        self.shared = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.encoder = T5Stack(self.config, is_decoder=False)
        self.decoder = T5Stack(self.config, is_decoder=True)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

        # Share weights
        self.encoder.embed_tokens.embedding = self.shared
        self.decoder.embed_tokens.embedding = self.shared
        self.lm_head.weight = self.shared.weight

        self.init_weights()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        decoder_input_ids: Optional[Tensor] = None,
        decoder_attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        # Encoder
        encoder_hidden_states = self.encoder(input_ids, attention_mask)

        # Decoder (teacher forcing)
        if decoder_input_ids is None and labels is not None:
            decoder_input_ids = self._shift_right(labels)

        decoder_output = self.decoder(
            decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
        )

        logits = self.lm_head(decoder_output)

        result = {"logits": logits, "encoder_last_hidden_state": encoder_hidden_states}

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            result["loss"] = loss_fn(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
            )

        return result

    def _shift_right(self, input_ids: Tensor) -> Tensor:
        """Shift decoder input IDs right by one (insert start token)."""
        # Use pad_token_id as start token
        start_tokens = input_ids.new_full((input_ids.size(0), 1), self.config.pad_token_id)
        return torch.cat([start_tokens, input_ids[:, :-1]], dim=-1)

    def init_weights(self) -> None:
        nn.init.normal_(self.shared.weight, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear) and module.weight is not self.lm_head.weight:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def get_input_embeddings(self) -> nn.Module:
        return self.shared

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.shared = value
        self.encoder.embed_tokens.embedding = value
        self.decoder.embed_tokens.embedding = value
