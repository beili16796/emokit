# EmoKit

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/emokit/emokit/actions/workflows/ci.yml/badge.svg)](https://github.com/emokit/emokit/actions)

**Modular Physiological Signal Analysis & Benchmarking Toolkit** for EEG-based
emotion recognition research. EmoKit provides unified dataset loaders, feature
extraction pipelines, deep-learning models, and reproducible evaluation
protocols — all wired together through a single YAML config.

---

## Quickstart (< 60 seconds, no dataset required)

```bash
pip install -e ".[dev]"
python -m emokit.run configs/quick_demo.yaml --dry-run
# Expected: Mean accuracy ~0.50 (random, synthetic data)
```

## Run with real data (DEAP)

```bash
# After downloading DEAP preprocessed data:
export EMOKIT_DATA_ROOT=/path/to/datasets
python -m emokit.run configs/deap_loso_dgcnn.yaml
```

## Reproduce paper results

```bash
python -m emokit.scripts.reproduce_baselines \
    --deap-root $EMOKIT_DATA_ROOT/DEAP \
    --seedv-root $EMOKIT_DATA_ROOT/SEED-V
```

## Installation

**From source (editable):**

```bash
git clone https://github.com/emokit/emokit.git
cd emokit
pip install -e ".[dev]"
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
python -m emokit.run configs/deap_loso_dgcnn.yaml
```

## Supported datasets

| Dataset    | Subjects | Modalities        | Labels     | Trials |
|------------|----------|-------------------|------------|--------|
| DEAP       | 32       | EEG+GSR+ECG+EMG   | V/A binary | 1,280  |
| SEED-V     | 16       | EEG+EOG           | 5-class    | 3,000  |
| SEED       | 15       | EEG               | 3-class    | 675    |
| MAHNOB-HCI | 27       | EEG+ECG+GSR+video | V/A        | 527    |
| DREAMER    | 23       | EEG+ECG           | V/A scale  | 414    |

## Supported models

| Model         | Paradigm            | Ref        |
|---------------|---------------------|------------|
| CNN-LSTM      | Spatio-temporal     | Li 2016    |
| BiDAE         | Multi-view fusion   | Zheng 2019 |
| DGCNN         | Dynamic graph       | Song 2020  |
| Transformer-MM| Attention-based     | Wu 2024    |
| DGCCA-AM      | Adaptive fusion     | Lan 2020   |
| PR-PL         | Prototypical align. | Zhou 2024  |

## Python API

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
