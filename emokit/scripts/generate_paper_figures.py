"""Generate paper-ready figures from EmoKit experiment outputs.

Usage::

    python -m emokit.scripts.generate_paper_figures \\
        --results results/paper_experiments --output figures/

    python -m emokit.scripts.generate_paper_figures --dry-run \\
        --output /tmp/figures_dryrun
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _load_results(results_dir: Path) -> dict:
    p = results_dir / "results_all.json"
    return json.loads(p.read_text(encoding="utf-8"))


def _find_median_checkpoint(results_dir: Path, exp_key: str) -> Path | None:
    results = _load_results(results_dir)
    result = results.get(exp_key)
    if not result:
        return None
    per_subject = result.get("per_subject", {})
    if not per_subject:
        return None
    mean_acc = result.get("mean", {}).get("accuracy", 0.0)
    sid, _metrics = min(
        per_subject.items(),
        key=lambda item: abs(item[1].get("accuracy", 0.0) - mean_acc),
    )
    ckpt = (
        results_dir
        / exp_key.replace("/", "_")
        / "checkpoints"
        / f"subject_{int(sid):02d}_best.pt"
    )
    return ckpt if ckpt.exists() else None


def fig2_dgcnn_adjacency(
    results_dir: Path | None,
    figures_dir: Path,
    dry_run: bool = False,
) -> None:
    """DGCNN learned adjacency matrix heatmap."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if dry_run:
        A = np.random.rand(32, 32)
        A = (A + A.T) / 2
        ch_names = [f"ch{i}" for i in range(32)]
    else:
        ckpt = _find_median_checkpoint(
            results_dir, "DGCNN/DEAP/valence"  # type: ignore[arg-type]
        )
        if ckpt is None:
            logger.warning("No DGCNN checkpoint found; skipping adjacency figure")
            return
        from emokit.datasets.deap import _EEG_CHANNELS
        from emokit.models.dgcnn import DGCNNModel

        model = DGCNNModel(n_classes=2, n_channels=32, n_bands=5)
        model.load(str(ckpt))
        A = model.get_adjacency_matrix()
        ch_names = list(_EEG_CHANNELS)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    im = ax.imshow(A, cmap="RdBu_r")
    title = "DGCNN learned adjacency matrix"
    if dry_run:
        title += " (placeholder)"
    ax.set_title(title)
    ax.set_xticks(range(len(ch_names)))
    ax.set_xticklabels(ch_names, rotation=90, fontsize=6)
    ax.set_yticks(range(len(ch_names)))
    ax.set_yticklabels(ch_names, fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = figures_dir / "fig2_dgcnn_adjacency.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def fig3_modality_ablation(
    ablation_json: Path | None,
    figures_dir: Path,
    dry_run: bool = False,
) -> None:
    """Modality ablation bar chart."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if dry_run:
        labels = ["EEG+GSR+ECG", "EEG+GSR", "EEG+ECG", "EEG only"]
        val_means = [73.1, 71.5, 70.2, 68.0]
        val_stds = [2.1, 2.5, 2.3, 3.0]
        arr_means = [72.5, 70.8, 69.5, 67.3]
        arr_stds = [2.3, 2.7, 2.5, 3.2]
    else:
        if ablation_json is None or not ablation_json.exists():
            logger.warning("Ablation JSON not found: %s", ablation_json)
            return
        data = json.loads(ablation_json.read_text(encoding="utf-8"))
        labels = list(data.keys())
        val_means = [data[k]["valence_mean"] for k in labels]
        val_stds = [data[k]["valence_std"] for k in labels]
        arr_means = [data[k]["arousal_mean"] for k in labels]
        arr_stds = [data[k]["arousal_std"] for k in labels]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=150)
    ax.bar(
        x - width / 2,
        val_means,
        width,
        yerr=val_stds,
        label="Valence",
        capsize=3,
    )
    ax.bar(
        x + width / 2,
        arr_means,
        width,
        yerr=arr_stds,
        label="Arousal",
        capsize=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Accuracy (%)")
    title = "Modality ablation on DEAP"
    if dry_run:
        title += " (placeholder)"
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = figures_dir / "fig3_modality_ablation.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def fig4_per_subject_boxplot(
    results_json: Path | None,
    figures_dir: Path,
    dry_run: bool = False,
) -> None:
    """Per-subject accuracy box plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = [
        "CNN-LSTM",
        "BiDAE",
        "DGCNN",
        "Transformer-MM",
        "DGCCA-AM",
        "PR-PL",
    ]

    if dry_run:
        rng = np.random.default_rng(42)
        values = [rng.normal(70, 8, 32).tolist() for _ in models]
    else:
        if results_json is None or not results_json.exists():
            logger.warning("Results JSON not found: %s", results_json)
            return
        data = json.loads(results_json.read_text(encoding="utf-8"))
        values = []
        for model in models:
            ps = data.get(f"{model}/DEAP/valence", {}).get("per_subject", {})
            values.append([m.get("accuracy", 0.0) * 100 for m in ps.values()])

    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    ax.boxplot(values, tick_labels=models, patch_artist=True)
    ax.set_ylabel("Per-subject accuracy (%)")
    title = "Cross-subject variability on DEAP valence"
    if dry_run:
        title += " (placeholder)"
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = figures_dir / "fig4_per_subject_boxplot.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate placeholder figures with random data.",
    )
    args = parser.parse_args()

    if not args.dry_run and args.results is None:
        parser.error("--results is required unless --dry-run is set")

    args.output.mkdir(parents=True, exist_ok=True)

    fig2_dgcnn_adjacency(args.results, args.output, args.dry_run)
    ablation = args.results / "modality_ablation.json" if args.results else None
    fig3_modality_ablation(ablation, args.output, args.dry_run)
    results_json = args.results / "results_all.json" if args.results else None
    fig4_per_subject_boxplot(results_json, args.output, args.dry_run)


if __name__ == "__main__":
    main()
