# AGENTS.md

## Cursor Cloud specific instructions

**EmoKit** is a pure Python library (no servers/databases) for EEG-based emotion recognition research. Development setup is `pip install -e ".[dev]"` from the repo root.

### Running / Testing

- **Lint**: `python3 -m ruff check emokit/ tests/` and `python3 -m black --check emokit/ tests/`
  Note: `ruff` / `black` / `isort` must be invoked via `python3 -m <tool>` as they may not be on `$PATH`.
- **Tests**: `python3 -m pytest tests/ -v --tb=short` — all 150 tests should pass.
- **CLI**: `python -m emokit configs/quick_demo.yaml` — runs LOSO with SyntheticDataset + PR-PL (~3s).
- **Dry-run all experiments**: `python scripts/run_all_experiments.py --dry-run --output-dir /tmp/dryrun` — 9 experiments in ~6s.
- **Modality ablation dry-run**: `python scripts/modality_ablation.py --config configs/deap_loso_dgcnn.yaml --dry-run`

### Architecture notes

- Registry-based design: datasets, models, features, and evaluation protocols each have a base class + registry.
- Experiments are configured via YAML and assembled at runtime by `emokit/run.py`.
- No external services (databases, Docker, etc.) are needed — everything runs in-process.
- Multimodal models (BiDAE, DGCCA-AM) have `multimodal = True` flag; `LOSOEvaluator` keeps per-modality dicts for these models instead of concatenating.
- `FeaturePipeline` auto-applies transforms per-modality when input is a dict, except for `ModalityFusionTransform` which handles dicts natively.
- Real dataset verification scripts (`verify_deap_pipeline.py`, `verify_dreamer_pipeline.py`, `verify_hci_pipeline.py`) require the actual datasets; they exit gracefully with instructions when data is not available.
