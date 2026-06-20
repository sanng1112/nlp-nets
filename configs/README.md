# Configuration Files

This directory contains YAML configuration files for nlp-nets experiments.

## Usage

```bash
python main.py --common.config-file configs/demo.yaml
```

## Structure

Each config file defines:

| Section | Description |
|---------|-------------|
| `seed` | Random seed for reproducibility |
| `task` | Task type and loss function |
| `model` | Model architecture parameters |
| `dataset` | Dataset name and processing params |
| `tokenizer` | Tokenizer configuration |
| `train` | Training hyperparameters |
| `optim` | Optimizer and scheduler settings |

## Available Configs

- `demo.yaml` — Minimal BERT MLM training on WikiText-2

## Creating Custom Configs

Copy `demo.yaml` and modify the sections as needed. CLI overrides take
precedence over YAML values:

```bash
python main.py --common.config-file my_config.yaml --model.name gpt --optim.lr 3e-4
```
