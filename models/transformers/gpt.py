"""
GPT model implementation (decoder-only transformer).

Supports:
- Causal Language Modeling (GPTForCausalLM)
"""

from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from models.base_model import BaseNLPModel
from utils.registry import Registry

from layers.normalization import LayerNorm
from layers.embeddings import TokenEmbedding
from layers.attention import MultiHeadAttention
from layers.feedforward import PositionwiseFeedForward
from layers.positional_encoding import LearnablePositionalEncoding
from utils.tokenizer_utils import create_causal_mask


class GPTConfig:
    """Configuration holder for GPT models."""

    def __init__(self, opts: Dict[str, Any]) -> None:
        model_cfg = opts.get("model", {})
        self.vocab_size = model_cfg.get("vocab_size", 50257)
        self.hidden_size = model_cfg.get("hidden_size", 768)
        self.num_hidden_layers = model_cfg.get("num_hidden_layers", 12)
        self.num_attention_heads = model_cfg.get("num_attention_heads", 12)
        self.intermediate_size = model_cfg.get("intermediate_size", self.hidden_size * 4)
        self.hidden_dropout_prob = model_cfg.get("hidden_dropout_prob", 0.1)
        self.attention_probs_dropout_prob = model_cfg.get("attention_probs_dropout_prob", 0.1)
        self.max_position_embeddings = model_cfg.get("max_position_embeddings", 1024)
        self.pad_token_id = model_cfg.get("pad_token_id", 50256)
        self.layer_norm_eps = model_cfg.get("layer_norm_eps", 1e-5)


class GPTAttention(nn.Module):
    """GPT self-attention layer with causal masking."""

    def __init__(self, config: GPTConfig) -> None:
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
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        return self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )


class GPTBlock(nn.Module):
    """Single GPT decoder block: LayerNorm → Self-Attention → LayerNorm → FFN."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = GPTAttention(config)
        self.ln_2 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn = PositionwiseFeedForward(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            dropout=config.hidden_dropout_prob,
            activation="gelu",
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        # Pre-LN attention
        attn_input = self.ln_1(hidden_states)
        attn_output, present_kv = self.attention(
            attn_input,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = hidden_states + attn_output

        # Pre-LN FFN
        ffn_input = self.ln_2(hidden_states)
        ffn_output = self.ffn(ffn_input)
        hidden_states = hidden_states + ffn_output

        return hidden_states, present_kv


class GPTModel(BaseNLPModel):
    """
    GPT decoder-only model (without LM head).

    Registry name: "gpt"
    """

    def __init__(self, opts: Dict[str, Any]) -> None:
        super().__init__(opts)
        self.config = GPTConfig(opts)
        self.token_embedding = TokenEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            padding_idx=None,
        )
        self.position_embedding = LearnablePositionalEncoding(
            self.config.max_position_embeddings,
            self.config.hidden_size,
        )
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.layers = nn.ModuleList([GPTBlock(self.config) for _ in range(self.config.num_hidden_layers)])
        self.ln_f = LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)
        self.init_weights()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[Tensor, Tensor], ...]] = None,
        use_cache: bool = False,
    ) -> Dict[str, Any]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Token + positional embeddings
        hidden_states = self.token_embedding(input_ids)
        hidden_states = self.position_embedding(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Causal attention mask
        if attention_mask is None:
            causal_mask = create_causal_mask(seq_len).to(device)
            attention_mask = causal_mask
        else:
            # Combine padded mask with causal mask
            causal_mask = create_causal_mask(seq_len).to(device)
            pad_mask = (attention_mask[:, None, None, :] == 0).float() * torch.finfo(hidden_states.dtype).min
            attention_mask = causal_mask + pad_mask

        # Past key values for incremental decoding
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        presents = () if use_cache else None
        new_past_key_values = []

        for layer, past_kv in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_kv,
                use_cache=use_cache,
            )
            if use_cache:
                new_past_key_values.append(present)

        hidden_states = self.ln_f(hidden_states)

        return {
            "last_hidden_state": hidden_states,
            "past_key_values": tuple(new_past_key_values) if use_cache else None,
        }

    def init_weights(self) -> None:
        self.token_embedding.init_weights()
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def get_input_embeddings(self) -> nn.Module:
        return self.token_embedding.embedding

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.token_embedding.embedding = value


# Register models
gpt_registry = Registry("gpt_models", base_class=BaseNLPModel)


@gpt_registry.register(name="gpt_causal_lm")
class GPTForCausalLM(BaseNLPModel):
    """
    GPT for Causal Language Modeling.
    Registry keys: "gpt_causal_lm", "gpt"

    Config example:
        model:
          name: "gpt"
          vocab_size: 50257
          hidden_size: 768
          num_hidden_layers: 12
          num_attention_heads: 12
    """

    def __init__(self, opts: Dict[str, Any]) -> None:
        super().__init__(opts)
        self.config = GPTConfig(opts)
        self.transformer = GPTModel(opts)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.transformer.get_input_embeddings().weight
        self.init_weights()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[Tensor, Tensor], ...]] = None,
        use_cache: bool = False,
    ) -> Dict[str, Any]:
        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        hidden_states = outputs["last_hidden_state"]
        logits = self.lm_head(hidden_states)

        result = {"logits": logits, "past_key_values": outputs.get("past_key_values")}

        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fn = nn.CrossEntropyLoss()
            result["loss"] = loss_fn(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )

        return result

    def init_weights(self) -> None:
        self.transformer.init_weights()
        nn.init.xavier_uniform_(self.lm_head.weight)
