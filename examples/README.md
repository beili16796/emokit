# EmoKit Examples

These examples are designed for reviewers and new users who want one command
per benchmark path.

## Synthetic smoke test

```bash
python -m emokit.run configs/quick_demo.yaml --dry-run
```

## DEAP LOSO baseline

```bash
export EMOKIT_DATA_ROOT=/path/to/datasets
python -m emokit.run configs/deap_loso_dgcnn.yaml
```

## SEED to DREAMER cross-corpus transfer

```bash
export EMOKIT_DATA_ROOT=/path/to/datasets
python -m emokit.run configs/cross_corpus_seed_to_dreamer_dgcnn.yaml
```

## Augmentation ablation

```bash
python -m emokit.run configs/deap_loso_dgcnn_valence_augmented.yaml
python -m emokit.scripts.augmentation_ablation \
    --data-root "$EMOKIT_DATA_ROOT/DEAP" \
    --output results/augmentation_ablation
```
