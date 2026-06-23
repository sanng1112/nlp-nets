# nlp-nets

> **A modular PyTorch NLP library** for building, training, and evaluating transformer-based language models.
> Inspired by the architecture of the nlp-nets library for **Natural Language Processing**.

---

## ✨ Features

- **🧱 Modular Components** — Attention, FeedForward, Embeddings, Positional Encoding as pluggable building blocks
- **🤖 Transformer Zoo** — BERT, GPT, T5 encoders built from the same core layers
- **📦 Registry-based Model Discovery** — Register new models/layers/losses without touching core code
- **⚙️ YAML-driven Config** — Full experiment configuration via YAML files (no hardcoded args)
- **🚀 Production Training Loop** — AMP, gradient accumulation, DDP, logging, checkpoints
- **📊 Metric Tracking** — Perplexity, Accuracy, F1, and custom metrics
- **🔌 HuggingFace Integration** — Use 🤗 Tokenizers, Datasets, and Pretrained weights seamlessly
- **🧪 Sanity Check** — Pre-training dry-run to catch OOM, shape mismatches, and gradient issues

---

## 🗂️ Project Structure

```
nlp-nets/
├── main.py                  # Entry point: CLI → config → train / eval
├── configs/                 # YAML experiment configurations
├── models/                  # Transformer model definitions (BERT, GPT, T5...)
│   └── transformers/
├── layers/                  # Reusable neural network building blocks
├── tokenizers/              # Tokenizer builder & utilities
├── loss_fn/                 # Loss functions (CrossEntropy, MLM, etc.)
├── optim/                   # Optimizer & LR scheduler builders
├── engine/                  # Training loop, data factory, sanity check
│   └── metrics_modules/     # Metric collections (PPL, Acc, F1)
├── data/                    # Dataset definitions & collators
└── utils/                   # Registry, logger, seed, config helpers
```

---

## 🚀 Quick Start

### 1. Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/nlp-nets.git
cd nlp-nets

# Create conda environment
conda env create -f environment.yml
conda activate nlp-nets

# Or install via pip
pip install -e .
```

### 2. Train a model

```bash
# Train with a YAML config
python main.py --common.config-file configs/demo.yaml

# Override config via CLI
python main.py --common.config-file configs/demo.yaml \
    --model.name bert-base \
    --optim.lr 2e-5 \
    --dataset.name wikitext-2
```

### 3. Run sanity check

```bash
python main.py --common.config-file configs/demo.yaml --common.sanity-check
```

---

## 📖 Configuration

All experiments are driven by YAML configuration files. See [`configs/demo.yaml`](configs/demo.yaml) for a complete example.

**Key config sections:**

```yaml
model:
  name: "bert-base"
  vocab_size: 30522
  hidden_size: 768

task:
  type: "mlm"               # mlm | causal_lm | seq_class | seq2seq
  loss: "cross_entropy"

dataset:
  name: "wikitext-2"
  max_seq_length: 128

optim:
  name: "adamw"
  lr: 2e-5
  scheduler: "linear_warmup"
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 🧩 Extending the Library

### Register a new model

```python
from utils.registry import Registry
from models.base_model import BaseNLPModel

MODEL_REGISTRY = Registry("models", base_class=BaseNLPModel)

@MODEL_REGISTRY.register(name="my_model")
class MyModel(BaseNLPModel):
    ...
```

### Register a new loss

```python
from loss_fn.base_criteria import BaseCriteria

LOSS_REGISTRY = Registry("loss_functions", base_class=BaseCriteria)

@LOSS_REGISTRY.register(name="my_loss")
class MyLoss(BaseCriteria):
    ...
```

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
