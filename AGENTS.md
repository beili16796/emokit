# AGENTS.md

Guidance for cloud agents working in this repository.

## Project overview

EmoKit is a **Python CLI/library** for multimodal physiological emotion recognition (EEG benchmarks). There is no web server, database, or long-running service — experiments run as one-shot `python3 -m emokit.run` processes.

## Cursor Cloud specific instructions

### PATH and Python

- Use **`python3`** (not `python`); the `python` shim may be missing.
- Dev tools install to `~/.local/bin`. Either `export PATH="$HOME/.local/bin:$PATH"` or invoke via modules: `python3 -m pytest`, `python3 -m ruff`, `python3 -m black`.

### Install (also handled by VM update script)

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -e ".[dev]"
```

### Lint and test (matches `.github/workflows/ci.yml`)

```bash
export PATH="$HOME/.local/bin:$PATH"
ruff check emokit/ tests/
black --check emokit/ tests/
pytest tests/ -v --tb=short
```

### Smoke / hello-world (no dataset download)

```bash
python3 -m emokit.run configs/quick_demo.yaml --dry-run
```

Writes results under `results/quick_demo/`. Uses built-in synthetic data; mean accuracy is not meaningful on random labels.

### Real datasets

Set `EMOKIT_DATA_ROOT` to a directory containing downloaded DEAP, SEED-V, etc. See `docs/dataset_setup.md` and `README.md`.

### Optional

- **GPU:** configs default to `cpu`; set `experiment.device: cuda` in YAML when CUDA is available.
- **Docker:** `docker build -t emokit:dev .` then `docker run --rm emokit:dev` (image runs the same quick demo).
- **Docs:** `pip install -e ".[docs]"` then build from `docs/`.
