# Repository Guidelines

## Project Structure & Module Organization

```
nlp-nets/
├── main.py                  # CLI entry point → YAML config → train/eval
├── configs/                 # YAML experiment configuration files
│   ├── demo.yaml            # Full training config example
│   └── compiler_example.yaml# Compilation pipeline config example
├── config/                  # Configuration system (shared standard with cv-nets)
│   ├── resolver.py          # ConfigResolver — dotted-path access, deep merge, export
│   ├── schema.py            # ConfigSchema — model/train/optim validation
│   └── __init__.py          # Public API: ConfigResolver, ConfigSchema, ConfigValidationError
├── models/                  # Model definitions (BERT, GPT, T5)
│   ├── base_model.py        # Abstract base class for all models
│   ├── builder.py           # YAML-to-model factory
│   └── transformers/        # Transformer architectures
│       ├── bert.py          # BERT (encoder-only)
│       ├── gpt.py           # GPT (decoder-only, causal LM)
│       └── t5.py            # T5 (encoder-decoder)
├── layers/                  # Pluggable building blocks
│   ├── base_layer.py        # Base class for all layers
│   ├── attention.py         # Multi-head & self-attention
│   ├── embeddings.py        # Token, positional, segment embeddings
│   ├── feedforward.py       # Position-wise FFN & gated FFN
│   ├── normalization.py     # LayerNorm & RMSLayerNorm
│   └── positional_encoding.py # Sinusoidal, learnable, RoPE, ALiBi
├── loss_fn/                 # Loss functions
│   ├── base_criteria.py     # Base criteria class
│   ├── cross_entropy.py     # Cross-entropy with label smoothing
│   └── mlm_loss.py          # Masked language modeling loss
├── optim/                   # Optimizer & LR scheduler builders
│   ├── optimizer_builder.py # AdamW, SGD with configurable params
│   └── scheduler_builder.py # Warmup, cosine, linear, polynomial
├── engine/                  # Training loop, data, inference, utilities
│   ├── trainer.py           # Full training loop with logging
│   ├── data_factory.py      # Dataset creation from YAML config
│   ├── inference.py         # Text generation pipeline
│   ├── sanity_check.py      # Dry-run for OOM/shape/gradient check
│   ├── ema.py               # Exponential Moving Average (from cv-nets)
│   ├── loggers.py           # CSVLogger for training metrics
│   └── metrics_modules/     # Metric collections (PPL, accuracy, F1)
│       └── builder.py       # torchmetrics-based metric factory
├── data/                    # Dataset definitions & collators
│   ├── datasets.py          # TextClassificationDataset, MaskedLMDataset, etc.
│   └── collator.py          # NLP collation utilities (dynamic padding)
├── tokenizer_factory/       # Tokenizer builder (HuggingFace, WordPiece, BPE)
│   ├── builder.py           # Factory selecting backend from config
│   └── __init__.py          # Public API
├── compiler/                # Model compilation & optimization
│   ├── base_compiler.py     # Abstract compiler interface
│   ├── builder.py           # Factory: selects backend from config
│   ├── torchscript_compiler.py # TorchScript tracing/scripting
│   ├── onnx_compiler.py     # ONNX export (dynamic axes support)
│   ├── torch_compile_compiler.py # torch.compile graph optimization
│   └── quantizer.py         # Dynamic/static/fp16 post-training quantization
├── visualization/           # Post-training model inspection & plotting
│   ├── model_viewer.py      # High-level combined inspection API
│   ├── weight_visualizer.py # Weight distribution histograms & heatmaps
│   ├── gradient_visualizer.py # Gradient distribution & flow tracking
│   └── attention_visualizer.py # Attention head patterns & entropy
├── utils/                   # Registry, logger, seed, config helpers
│   ├── registry.py          # Generic Registry pattern (decorator-based)
│   ├── logger.py            # Logger factory (console + file)
│   ├── seed.py              # Global random seed for reproducibility
│   ├── config_helper.py     # YAML/argparse config utilities (legacy wrapper for ``config`` module)
│   ├── import_utils.py      # Lazy import helpers
│   └── tokenizer_utils.py   # Tokenizer helper functions
├── tests/                   # pytest test suite
│   ├── test_layers.py       # Layer unit tests
│   ├── test_models.py       # Model integration tests
│   ├── test_trainer.py      # Trainer regression tests
│   ├── test_loss_fn.py      # Loss function tests
│   ├── test_optim.py        # Optimizer & scheduler tests
│   ├── test_compiler.py     # Compilation pipeline tests
│   └── test_visualization.py# Visualizer output tests
└── .github/workflows/       # CI pipeline (from cv-nets)
    └── ci.yml               # GitHub Actions: test across 3.10-3.12
```

All source code lives under the project root. YAML configs in `configs/` drive experiments. Tests mirror the structure of `tests/`.

---

## Build, Test, and Development Commands

| Command | Description |
|---|---|
| `pip install -e .` | Editable install of the package and its dependencies |
| `conda env create -f environment.yml` | Create the full Conda environment |
| `python main.py --common.config-file configs/demo.yaml` | Run a training/eval experiment from a YAML config |
| `python main.py --common.config-file configs/demo.yaml --common.sanity-check` | Dry-run to catch OOM, shape mismatches, or gradient issues |
| `python -c "from visualization import ModelViewer; v = ModelViewer(model); v.inspect_weights('histogram')"` | Plot per-layer weight distribution histograms |
| `python -c "from visualization import ModelViewer; v.summary_report('report.txt')"` | Generate parameter statistics report |
| `pytest tests/ -v` | Run the full test suite verbosely |
| `pytest tests/test_layers.py -v` | Run a single test file |
| `python -c "from compiler.builder import build_compiler; ..."` | Compile an NLP model using the compiler module |
| `python main.py --common.config-file configs/compiler_example.yaml --common.sanity-check` | Dry-run compilation pipeline without model weights |

**Formatting & linting** (defined in `pyproject.toml`):

```bash
black .                     # Auto-format (line-length 120)
isort .                     # Sort imports (black-compatible)
flake8 .                    # Lint check
mypy .                      # Type check
```

---

## Coding Style & Naming Conventions

- **Language:** Python 3.10+
- **Formatter:** [Black](https://github.com/psf/black) with `line-length = 120` (see `pyproject.toml`).
- **Import sorter:** [isort](https://pycqa.github.io/isort/) with `profile = "black"`.
- **Linter:** [Flake8](https://flake8.pycqa.org/) (configurable via `.flake8` or `pyproject.toml`).
- **Type checker:** [mypy](https://mypy-lang.org/) (`python_version = "3.10"`, `warn_return_any = true`).

**Naming patterns:**

| Element | Convention | Example |
|---|---|---|
| Modules/packages | `snake_case` | `loss_fn/`, `tokenizers/` |
| Classes | `PascalCase` | `BaseNLPModel`, `CrossEntropyLoss` |
| Functions/methods | `snake_case` | `build_model()`, `forward()` |
| Constants | `UPPER_SNAKE_CASE` | `DEFAULT_VOCAB_SIZE` |
| Registry decorators | lowercase | `@MODEL_REGISTRY.register(name="bert-base")` |

---

## Testing Guidelines

- **Framework:** [pytest](https://docs.pytest.org/) 7+ with `pytest-cov`.
- **Coverage threshold:** Aim for ≥80% on new code.
- **Test file naming:** `test_<module>.py` (e.g., `test_layers.py`, `test_models.py`, `test_trainer.py`).
- **Test function naming:** `test_<feature>` (e.g., `test_attention_output_shape`).
- **Running tests:**

```bash
pytest tests/ -v --tb=short                     # Full suite, short traceback
pytest tests/ --cov=nlp-nets --cov-report=term  # With coverage report
```

---

## Commit & Pull Request Guidelines

**Commit messages** follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <short summary>

[optional body]
```

The project currently uses a single commit (`feat: initial nlp-nets library`). Moving forward, use these types:

| Type | Usage |
|---|---|
| `feat` | A new feature or component |
| `fix` | A bug fix |
| `docs` | Documentation changes |
| `refactor` | Code restructuring without behavior change |
| `test` | Adding or updating tests |
| `chore` | Build, CI, or tooling changes |

**Pull request requirements:**

- Title mirrors the commit message format (e.g., `feat: add RoPE positional encoding`).
- Description must explain **what** and **why** the change is made.
- Link related issues (e.g., `Closes #42`).
- Attach screenshots or terminal logs for visual or metric-affecting changes.
- Ensure all CI checks pass (tests, lint, mypy).
- Squash-merge commits into a single conventional commit.

---

## Registry Pattern & Extending the Library

New components (models, loss functions, etc.) are discovered automatically via the central registry system in `utils/registry.py`:

```python
from utils.registry import Registry
from models.base_model import BaseNLPModel

MODEL_REGISTRY = Registry("models", base_class=BaseNLPModel)

@MODEL_REGISTRY.register(name="my_model")
class MyModel(BaseNLPModel):
    ...
```

The same pattern applies to loss functions (`loss_fn/base_criteria.py`) and any future pluggable component. This keeps the core code decoupled from user extensions.

---

## Post-Training Visualization

The `visualization/` package provides tools to inspect a trained model:

| Tool | API | Purpose |
|---|---|---|
| `ModelViewer` | `viewer = ModelViewer(model)` | Unified entry point for all inspections |
| `WeightVisualizer` | `viewer.inspect_weights("histogram")` | Weight distribution histograms, heatmaps, comparison KDE |
| `GradientVisualizer` | `viewer.inspect_gradients("flow")` | Gradient flow tracking, per-layer histograms, timeline |
| `AttentionVisualizer` | `viewer.inspect_attention(attn_w, "heads")` | Per-head attention heatmaps, entropy analysis, rollout |

```python
from visualization import ModelViewer
viewer = ModelViewer(model)

viewer.inspect_weights("histogram")           # grid of all layer histograms
viewer.inspect_weights("heatmap", layer_name="fc1")  # single weight matrix heatmap
viewer.summary_report("report.txt")           # text statistics report
viewer.inspect_gradients("flow")              # gradient magnitude across layers
viewer.inspect_attention(attn, "entropy")     # attention head confidence
```

---

## Model Compilation & Optimization

The `compiler/` package provides tools to compile, quantize, and export trained models for production inference:

| Component | API | Purpose |
|---|---|---|
| `TorchScriptCompiler` | `compiler.compile(mode="trace", example_input=x)` | Trace/script models via `torch.jit` for deployment |
| `ONNXCompiler` | `compiler.compile(example_input=x)` | Export to ONNX for cross-platform (CPU/GPU/mobile) inference |
| `TorchCompileCompiler` | `compiler.compile()` | Apply `torch.compile` (PyTorch 2.0+) graph optimization |
| `Quantizer` | `quantizer.quantize(mode="dynamic")` | Reduce precision: dynamic/static quantization or fp16 conversion |
| `build_compiler` | `compiler = build_compiler(opts, model)` | Factory function selecting backend from config |

```python
from compiler import build_compiler

# Build from config (backend selected by opts["compiler"]["backend"])
compiler = build_compiler(opts, model)

# Compile with an example input
compiler.compile(example_input=torch.randint(0, 100, (1, 64)))

# Save the artifact
compiler.save("compiled_model.onnx")
```

**Config-driven compilation** (see `configs/compiler_example.yaml`):

```yaml
compiler:
  backend: "onnx"              # torchscript | onnx | torch_compile | quantize
  onnx:
    opset_version: 17
    input_names: ["input_ids", "attention_mask"]
    output_names: ["logits"]
    dynamic_axes:
      0: "batch_size"
      1: "seq_length"
```
