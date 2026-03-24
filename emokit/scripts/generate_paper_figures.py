"""Generate paper-ready figures from EmoKit experiment outputs."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _load_results(results_dir: Path) -> dict:
    return json.loads((results_dir / "results_all.json").read_text(encoding="utf-8"))


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
    ckpt = results_dir / exp_key.replace("/", "_") / "checkpoints" / f"subject_{int(sid):02d}_best.pt"
    return ckpt if ckpt.exists() else None


def fig2_dgcnn_adjacency(results_dir: Path, figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    ckpt = _find_median_checkpoint(results_dir, "DGCNN/DEAP/valence")
    if ckpt is None:
        logger.warning("No DGCNN checkpoint found; skipping adjacency figure")
        return

    from emokit.datasets.deap import DEAP_EEG_CHANNELS  # noqa: WPS433
    from emokit.models.dgcnn import DGCNNModel  # noqa: WPS433

    model = DGCNNModel(n_classes=2, n_channels=32, n_bands=5)
    model.load(str(ckpt))
    A = model.get_adjacency_matrix()

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    im = ax.imshow(A, cmap="RdBu_r")
    ax.set_title("DGCNN learned adjacency matrix")
    ax.set_xticks(range(len(DEAP_EEG_CHANNELS)))
    ax.set_xticklabels(DEAP_EEG_CHANNELS, rotation=90, fontsize=6)
    ax.set_yticks(range(len(DEAP_EEG_CHANNELS)))
    ax.set_yticklabels(DEAP_EEG_CHANNELS, fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = figures_dir / "fig2_dgcnn_adjacency.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def fig3_modality_ablation(ablation_json: Path, figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    if not ablation_json.exists():
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
    ax.bar(x - width / 2, val_means, width, yerr=val_stds, label="Valence", capsize=3)
    ax.bar(x + width / 2, arr_means, width, yerr=arr_stds, label="Arousal", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Modality ablation on DEAP")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = figures_dir / "fig3_modality_ablation.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def fig4_per_subject_boxplot(results_json: Path, figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    data = json.loads(results_json.read_text(encoding="utf-8"))
    models = ["CNN-LSTM", "BiDAE", "DGCNN", "Transformer-MM", "DGCCA-AM", "PR-PL"]
    values = []
    for model in models:
        per_subject = data.get(f"{model}/DEAP/valence", {}).get("per_subject", {})
        values.append([metrics.get("accuracy", 0.0) * 100 for metrics in per_subject.values()])

    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    ax.boxplot(values, labels=models, patch_artist=True)
    ax.set_ylabel("Per-subject accuracy (%)")
    ax.set_title("Cross-subject variability on DEAP valence")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = figures_dir / "fig4_per_subject_boxplot.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    fig2_dgcnn_adjacency(args.results, args.output)
    fig3_modality_ablation(args.results / "modality_ablation.json", args.output)
    fig4_per_subject_boxplot(args.results / "results_all.json", args.output)


if __name__ == "__main__":
    main()
