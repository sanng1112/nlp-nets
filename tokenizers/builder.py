"""
Tokenizer builder — constructs tokenizers from config dicts.

Supports HuggingFace tokenizers and HuggingFace AutoTokenizer.
"""

from typing import Any, Dict, Optional


def build_tokenizer(opts: Dict[str, Any]) -> Any:
    """
    Build a tokenizer from configuration.

    Args:
        opts: Configuration dictionary (typically loaded from YAML).

    Returns:
        A tokenizer instance with encode/decode methods.
    """
    tokenizer_config = opts.get("tokenizer", {})
    dataset_config = opts.get("dataset", {})

    # Priority: explicit tokenizer config > dataset config
    pretrained_name = tokenizer_config.get(
        "pretrained_model_name",
        dataset_config.get("tokenizer_name", None),
    )

    if pretrained_name:
        return _build_hf_tokenizer(pretrained_name)

    # Fall back to building from config
    return _build_from_config(tokenizer_config)


def _build_hf_tokenizer(pretrained_model_name: str) -> Any:
    """
    Build a tokenizer from a HuggingFace pretrained model name.

    Args:
        pretrained_model_name: Name or path of the pretrained tokenizer.

    Returns:
        HuggingFace tokenizer.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError(
            "transformers package is required to use pretrained tokenizers. "
            "Install with: pip install transformers"
        )

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, use_fast=True)
    return tokenizer


def _build_from_config(tokenizer_config: Dict[str, Any]) -> Any:
    """
    Build a tokenizer from configuration parameters.
    Used when no pretrained model name is specified.

    Args:
        tokenizer_config: Tokenizer configuration dict.

    Returns:
        A tokenizer instance.
    """
    tokenizer_type = tokenizer_config.get("type", "wordpiece").lower()

    if tokenizer_type == "wordpiece":
        return _build_wordpiece_tokenizer(tokenizer_config)
    elif tokenizer_type == "bpe":
        return _build_bpe_tokenizer(tokenizer_config)
    elif tokenizer_type == "sentencepiece":
        return _build_sentencepiece_tokenizer(tokenizer_config)
    else:
        raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")


def _build_wordpiece_tokenizer(config: Dict[str, Any]) -> Any:
    """Build a WordPiece tokenizer (like BERT)."""
    try:
        from tokenizers import Tokenizer as HFTokenizer
        from tokenizers.models import WordPiece
        from tokenizers.trainers import WordPieceTrainer
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.processors import TemplateProcessing
        from tokenizers.decoders import WordPiece as WordPieceDecoder
    except ImportError:
        raise ImportError("tokenizers package required. Install: pip install tokenizers")

    vocab_size = config.get("vocab_size", 30522)
    unk_token = config.get("unk_token", "[UNK]")
    sep_token = config.get("sep_token", "[SEP]")
    pad_token = config.get("pad_token", "[PAD]")
    cls_token = config.get("cls_token", "[CLS]")
    mask_token = config.get("mask_token", "[MASK]")

    tokenizer = HFTokenizer(WordPiece(unk_token=unk_token))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.decoder = WordPieceDecoder()

    tokenizer.post_processor = TemplateProcessing(
        single=f"{cls_token}:0 $A:0 {sep_token}:0",
        pair=f"{cls_token}:0 $A:0 {sep_token}:0 $B:1 {sep_token}:1",
        special_tokens=[
            (cls_token, 0),
            (sep_token, 1),
        ],
    )

    tokenizer.enable_padding(pad_token=pad_token, pad_id=0)
    tokenizer.enable_truncation(max_length=config.get("max_length", 512))

    return tokenizer


def _build_bpe_tokenizer(config: Dict[str, Any]) -> Any:
    """Build a BPE tokenizer (like GPT)."""
    try:
        from tokenizers import Tokenizer as HFTokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        from tokenizers.processors import TemplateProcessing
    except ImportError:
        raise ImportError("tokenizers package required. Install: pip install tokenizers")

    tokenizer = HFTokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.post_processor = None
    tokenizer.enable_truncation(max_length=config.get("max_length", 1024))

    return tokenizer


def _build_sentencepiece_tokenizer(config: Dict[str, Any]) -> Any:
    """Build a SentencePiece tokenizer (like T5, LLaMA)."""
    try:
        from transformers import T5Tokenizer
    except ImportError:
        raise ImportError("transformers package required. Install: pip install transformers")

    vocab_file = config.get("vocab_file", "")
    if vocab_file:
        return T5Tokenizer(vocab_file=vocab_file)
    else:
        # Use default T5 tokenizer
        return T5Tokenizer.from_pretrained("t5-small")
