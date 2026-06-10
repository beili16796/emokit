# EmoKit

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/beili16796/emokit/actions/workflows/ci.yml/badge.svg)](https://github.com/beili16796/emokit/actions)

**LOSO-first benchmark and toolkit** for multimodal physiological emotion
recognition. EmoKit provides unified dataset loaders, feature extraction
pipelines, deep-learning models, and reproducible evaluation protocols, all
wired together through a single YAML config.

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

## Cross-corpus benchmark

EmoKit treats cross-dataset transfer as a first-class protocol. The source
dataset is declared in `dataset`, the held-out target dataset is declared in
`target_dataset`, and the evaluator aligns shared EEG channels by 10-20 names.

```bash
export EMOKIT_DATA_ROOT=/path/to/datasets
python -m emokit.run configs/cross_corpus_seed_to_dreamer_dgcnn.yaml
```

The run writes JSON, CSV, and `results_db.csv` summaries under
`results/cross_corpus_seed_to_dreamer_dgcnn/`.

## Augmentation ablation

Two LOSO-oriented training transforms are available through YAML:
`FeatureMixup` and `TemporalSegmentPermutation`. To run the configured
DGCNN valence ablation:

```bash
python -m emokit.run configs/deap_loso_dgcnn_valence_augmented.yaml
```

For the full four-condition ablation used in development:

```bash
python -m emokit.scripts.augmentation_ablation \
    --data-root $EMOKIT_DATA_ROOT/DEAP \
    --output results/augmentation_ablation
```

## Installation

**From source (editable):**

```bash
git clone https://github.com/beili16796/emokit.git
cd emokit
pip install -e ".[dev]"
```

**Docker smoke test:**

```bash
docker build -t emokit:dev .
docker run --rm emokit:dev
```

See [`examples/`](examples/) for reviewer-oriented commands.

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
  protocol: loso  # also supports subject_dependent, session, cross_corpus
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

## Evaluation protocols

| Protocol | YAML value | Output |
|----------|------------|--------|
| Leave-One-Subject-Out | `loso` | per-subject folds, mean/std, raw predictions |
| Subject-dependent | `subject_dependent` | within-subject train/test summaries |
| Cross-session | `session` | train earlier sessions, test final session |
| Cross-corpus | `cross_corpus` | source-to-target transfer with channel alignment |

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

## Dataset Verification

Verify your dataset setup (see [`docs/dataset_setup.md`](docs/dataset_setup.md)):

```bash
python -m emokit.scripts.verify_dreamer_pipeline --root $EMOKIT_DATA_ROOT/DREAMER
python scripts/verify_hci_pipeline.py --root $EMOKIT_DATA_ROOT/MAHNOB-HCI
python -m emokit.scripts.verify_deap_pipeline --root /path/to/DEAP
python -m emokit.scripts.verify_seedv_pipeline --root /path/to/SEED-V
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
  author  = {Xu, Wentian and Shen, Jian},
  year    = {2026},
  url     = {https://github.com/beili16796/emokit},
  license = {MIT}
}
```

## License

This project is licensed under the [MIT License](LICENSE).
