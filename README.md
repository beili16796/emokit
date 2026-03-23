# EmoKit

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/emokit.svg)](https://pypi.org/project/emokit/)
[![Docs](https://readthedocs.org/projects/emokit/badge/?version=latest)](https://emokit.readthedocs.io)
[![CI](https://github.com/emokit/emokit/actions/workflows/ci.yml/badge.svg)](https://github.com/emokit/emokit/actions)

**Modular Physiological Signal Analysis & Benchmarking Toolkit** for EEG-based
emotion recognition research. EmoKit provides unified dataset loaders, feature
extraction pipelines, deep-learning models, and reproducible evaluation
protocols — all wired together through a single YAML config.

> **论文级可复现性**：当前仓库提供接口与单元测试；真实数据集上的 LOSO 数字、统计检验与消融仍需按
> [`docs/PAPER_ROADMAP.md`](docs/PAPER_ROADMAP.md) 推进（含 `scripts/verify_deap_pipeline.py` 等）。

---

## Installation

**From PyPI:**

```bash
pip install emokit
```

**Editable install with dev dependencies (for contributors):**

```bash
git clone https://github.com/emokit/emokit.git
cd emokit
pip install -e ".[dev]"
```

## Quickstart

```python
from emokit.datasets import load_dataset
from emokit.features.eeg import DEExtractor, EEGNormalizer
from emokit.features.base import FeaturePipeline
from emokit.evaluation.protocols import LOSOEvaluator

ds = load_dataset("DEAP", root="data/DEAP", modalities=["eeg"])
pipeline = FeaturePipeline([
    ("de", DEExtractor(fs=128)),
    ("norm", EEGNormalizer()),
])
evaluator = LOSOEvaluator(
    dataset=ds, feature_pipeline=pipeline,
    model_config={"n_channels": 32, "n_bands": 5, "n_epochs": 50},
    model_name="DGCNN", seed=42,
)
results = evaluator.run()
print(f"Mean accuracy: {results['mean']['accuracy']:.4f}")
```

## YAML Configuration

All experiments can be driven by a single YAML file (see
[`configs/deap_loso_dgcnn.yaml`](configs/deap_loso_dgcnn.yaml)):

```yaml
experiment:
  name: deap_loso_dgcnn
  seed: 42
  device: cpu

dataset:
  name: DEAP
  root: data/DEAP
  modalities: [eeg, gsr]
  label_axis: valence

feature_pipeline:
  steps:
    - name: DEExtractor
      params: { fs: 128 }
    - name: EEGNormalizer
      params: {}

model:
  name: DGCNN
  params:
    n_channels: 32
    n_bands: 5
    hidden_dim: 64
    n_epochs: 50

evaluation:
  protocol: loso
  val_fraction: 0.1

output:
  results_dir: results/
```

Run it:

```bash
python -m emokit --config configs/deap_loso_dgcnn.yaml
```

## Supported Datasets

| Name       | Channels | Subjects | Classes | Modalities           |
|------------|----------|----------|---------|----------------------|
| DEAP       | 32       | 32       | 2       | EEG, GSR, ECG        |
| SEED       | 62       | 15       | 3       | EEG                  |
| SEED-V     | 62       | 16       | 5       | EEG, Eye tracking    |
| MAHNOB-HCI | 32       | 27       | 2       | EEG, ECG, GSR, Video |
| DREAMER    | 14       | 23       | 2       | EEG, ECG             |

## Supported Models

| Name           | Paradigm        | Input Type   | Reference                               |
|----------------|-----------------|--------------|---------------------------------------- |
| CNN-LSTM       | Supervised      | Raw / DE     | Yang et al., 2018                       |
| DGCNN          | Graph Neural    | DE           | Song et al., *IEEE TAFFC*, 2020         |
| Transformer-MM | Multi-modal     | DE + Periph. | Tao et al., 2023                        |
| BiDAE          | Semi-supervised | DE           | Li et al., *IEEE TAFFC*, 2022           |
| DGCCA-AM       | Domain Adapt.   | DE           | Chen et al., 2023                       |
| PR-PL          | Prompt Learning | DE           | Zhang et al., 2024                      |

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository.
2. **Create a feature branch:** `git checkout -b feat/my-feature`
3. **Commit** your changes with a clear message.
4. **Run tests:** `pytest`
5. **Open a Pull Request** against `main`.

Please make sure all tests pass and code follows the project's linting rules
(`ruff`, `black`, `isort`) before submitting.

## Citation

If you use EmoKit in your research, please cite:

```bibtex
@software{emokit2024,
  title   = {EmoKit: Modular Physiological Signal Analysis \& Benchmarking Toolkit},
  author  = {EmoKit Contributors},
  year    = {2024},
  url     = {https://github.com/emokit/emokit},
  license = {MIT}
}
```

## License

This project is licensed under the [MIT License](LICENSE).
