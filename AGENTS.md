# AGENTS.md

## Cursor Cloud specific instructions

**EmoKit** is a pure Python library (no servers/databases) for EEG-based emotion recognition research. Development setup is `pip install -e ".[dev]"` from the repo root.

### Running / Testing

- **Lint**: `python3 -m ruff check emokit/ tests/` and `python3 -m black --check emokit/ tests/`  
  Note: `ruff` / `black` / `isort` must be invoked via `python3 -m <tool>` as they may not be on `$PATH`.
- **Tests**: `python3 -m pytest tests/ -v --tb=short` (115 pass, 6 fail due to pre-existing scikit-learn `multi_class` compat issue)
- **CLI**: `python -m emokit --config configs/<yaml>` — requires a valid YAML config. `configs/quick_demo.yaml` has a known bug (`root: null` fails Pydantic validation); use `root: /tmp/synthetic` or similar instead.

### Known issues

- 6 test failures in `tests/test_evaluation.py` are caused by `LogisticRegression(multi_class=...)` — the `multi_class` param was removed in scikit-learn ≥ 1.7. These are pre-existing.
- The `quick_demo.yaml` config sets `root: null` but `DatasetConfig.root` is typed as `str` (not `Optional[str]`), so config loading fails.

### Architecture notes

- Registry-based design: datasets, models, features, and evaluation protocols each have a base class + registry.
- Experiments are configured via YAML and assembled at runtime by `emokit/run.py`.
- No external services (databases, Docker, etc.) are needed — everything runs in-process.
