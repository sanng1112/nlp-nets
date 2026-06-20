# Repository Guidelines

## Project Structure & Module Organization

```
nlp-nets/
├── main.py                  # CLI entry point → YAML config → train/eval
├── configs/                 # YAML experiment configuration files
├── models/                  # Model definitions (BERT, GPT, T5)
│   └── transformers/        # Transformer architectures
├── layers/                  # Pluggable building blocks (attention, FFN, embeddings)
├── loss_fn/                 # Loss functions (cross-entropy, MLM, etc.)
├── optim/                   # Optimizer & LR scheduler builders
├── engine/                  # Training loop, data factory, sanity check
│   └── metrics_modules/     # Metric collections (PPL, accuracy, F1)
├── data/                    # Dataset definitions & collators
├── tokenizer_factory/       # Tokenizer builder (HuggingFace, WordPiece, BPE)
├── visualization/           # Post-training model inspection & plotting
│   ├── weight_visualizer    # Weight distribution histograms & heatmaps
│   ├── gradient_visualizer  # Gradient distribution & flow tracking
│   ├── attention_visualizer # Attention head patterns & entropy
│   └── model_viewer         # High-level combined inspection API
├── utils/                   # Registry, logger, seed, config helpers
└── tests/                   # pytest test suite
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
