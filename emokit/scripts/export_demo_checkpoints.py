"""Export representative EmoKit checkpoints for the EmoSense demo."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

CHECKPOINT_MAP = {
    "cnn_lstm_deap_valence_demo.pt": "CNN-LSTM/DEAP/valence",
    "dgcnn_deap_valence_demo.pt": "DGCNN/DEAP/valence",
    "bidae_deap_demo.pt": "BiDAE/DEAP/valence",
    "transformer_mm_seedv_demo.pt": "ALL/SEED-V/5class",
    "dgcca_deap_demo.pt": "DGCCA-AM/DEAP/arousal",
    "prpl_deap_valence_demo.pt": "PR-PL/DEAP/valence",
}


def _pick_subject_id(result: dict) -> int | None:
    per_subject = result.get("per_subject", {})
    if not per_subject:
        return None
    mean_acc = result.get("mean", {}).get("accuracy", 0.0)
    chosen_sid, _ = min(
        per_subject.items(),
        key=lambda item: abs(item[1].get("accuracy", 0.0) - mean_acc),
    )
    return int(chosen_sid)


def export(results_dir: Path, emosense_dir: Path) -> None:
    emosense_dir.mkdir(parents=True, exist_ok=True)
    results_all = json.loads(
        (results_dir / "results_all.json").read_text(encoding="utf-8")
    )

    for demo_name, exp_key in CHECKPOINT_MAP.items():
        if exp_key.startswith("ALL/SEED-V"):
            seedv = (
                results_all.get(exp_key, {})
                .get("per_model", {})
                .get("Transformer-MM", {})
            )
            per_subject = seedv.get("per_subject", {})
            if not per_subject:
                logger.warning("Missing SEED-V per-subject results for %s", exp_key)
                continue
            mean_acc = seedv.get("mean_acc", 0.0)
            sid, metrics = min(
                per_subject.items(),
                key=lambda item: abs(item[1].get("accuracy", 0.0) - mean_acc),
            )
            src = (
                results_dir
                / "ALL_SEED-V_5class"
                / "checkpoints"
                / f"subject_{int(sid):02d}_best.pt"
            )
            chosen_sid = int(sid)
            chosen_acc = metrics.get("accuracy", 0.0)
        else:
            result = results_all.get(exp_key)
            if result is None:
                logger.warning("Missing experiment: %s", exp_key)
                continue
            chosen_sid = _pick_subject_id(result)
            if chosen_sid is None:
                logger.warning("No per-subject results for %s", exp_key)
                continue
            chosen_acc = (
                result["per_subject"][str(chosen_sid)]["accuracy"]
                if str(chosen_sid) in result["per_subject"]
                else result["per_subject"][chosen_sid]["accuracy"]
            )
            src = (
                results_dir
                / exp_key.replace("/", "_")
                / "checkpoints"
                / f"subject_{chosen_sid:02d}_best.pt"
            )

        if not src.exists():
            logger.warning("Checkpoint not found: %s", src)
            continue
        dst = emosense_dir / demo_name
        shutil.copy(src, dst)
        print(
            f"Copied {demo_name} from {src.name} "
            f"(subject {chosen_sid}, acc={chosen_acc * 100:.1f}%)"
        )

    print(f"\nExported checkpoints to {emosense_dir}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--emosense-dir", required=True, type=Path)
    args = parser.parse_args()
    export(args.results, args.emosense_dir)


if __name__ == "__main__":
    main()
