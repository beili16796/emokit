#!/usr/bin/env python3
"""Master paper experiment runner for EmoKit."""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from emokit.evaluation.config import (
    ConfigLoader,
    FeaturePipelineConfig,
    FeatureStepConfig,
)
from emokit.evaluation.protocols import LOSOEvaluator
from emokit.scripts.reproduce_baselines import PAPER_NUMBERS, TOLERANCE

logger = logging.getLogger(__name__)

EXPERIMENTS = [
    (
        "configs/deap_loso_cnnlstm_valence.yaml",
        "DEAP",
        "valence",
        "CNN-LSTM/DEAP/valence",
    ),
    (
        "configs/deap_loso_cnnlstm_arousal.yaml",
        "DEAP",
        "arousal",
        "CNN-LSTM/DEAP/arousal",
    ),
    ("configs/deap_loso_bidae_valence.yaml", "DEAP", "valence", "BiDAE/DEAP/valence"),
    ("configs/deap_loso_bidae_arousal.yaml", "DEAP", "arousal", "BiDAE/DEAP/arousal"),
    ("configs/deap_loso_dgcnn_valence.yaml", "DEAP", "valence", "DGCNN/DEAP/valence"),
    ("configs/deap_loso_dgcnn_arousal.yaml", "DEAP", "arousal", "DGCNN/DEAP/arousal"),
    (
        "configs/deap_loso_transformer_valence.yaml",
        "DEAP",
        "valence",
        "Transformer-MM/DEAP/valence",
    ),
    (
        "configs/deap_loso_transformer_arousal.yaml",
        "DEAP",
        "arousal",
        "Transformer-MM/DEAP/arousal",
    ),
    (
        "configs/deap_loso_dgcca_valence.yaml",
        "DEAP",
        "valence",
        "DGCCA-AM/DEAP/valence",
    ),
    (
        "configs/deap_loso_dgcca_arousal.yaml",
        "DEAP",
        "arousal",
        "DGCCA-AM/DEAP/arousal",
    ),
    ("configs/deap_loso_prpl_valence.yaml", "DEAP", "valence", "PR-PL/DEAP/valence"),
    ("configs/deap_loso_prpl_arousal.yaml", "DEAP", "arousal", "PR-PL/DEAP/arousal"),
    ("configs/seedv_loso_all_models.yaml", "SEED-V", "five_class", "ALL/SEED-V/5class"),
    (
        "configs/dreamer_loso_dgcnn_valence.yaml",
        "DREAMER",
        "valence",
        "DGCNN/DREAMER/valence",
    ),
    (
        "configs/dreamer_loso_dgcnn_arousal.yaml",
        "DREAMER",
        "arousal",
        "DGCNN/DREAMER/arousal",
    ),
    (
        "configs/dreamer_loso_cnnlstm_valence.yaml",
        "DREAMER",
        "valence",
        "CNN-LSTM/DREAMER/valence",
    ),
    (
        "configs/dreamer_loso_transformer_valence.yaml",
        "DREAMER",
        "valence",
        "Transformer-MM/DREAMER/valence",
    ),
]


def _make_dry_run_cfg(cfg: Any, dataset_key: str, *, mock_run: bool = False) -> Any:
    """Convert a real-data config into a fast synthetic dry-run config.

    When *mock_run* is ``True``, further shrink all model dimensions to
    the absolute minimum (1 epoch, tiny hidden dims) so the full
    pipeline completes in seconds on CPU.
    """
    n_ch = 62 if dataset_key == "SEED-V" else 14 if dataset_key == "DREAMER" else 32
    dataset_update = {
        "name": "SYNTHETIC",
        "root": None,
        "params": {
            "n_subjects": 3,
            "n_trials": 8 if not mock_run else 6,
            "n_channels": n_ch,
            "n_classes": 5 if dataset_key == "SEED-V" else 2,
        },
    }
    feature_pipeline = cfg.feature_pipeline
    if dataset_key == "SEED-V":
        feature_pipeline = FeaturePipelineConfig(
            steps=[
                FeatureStepConfig(name="DEExtractor", params={"fs": 128}),
                FeatureStepConfig(name="EEGNormalizer", params={}),
            ]
        )
    if cfg.model is not None:
        params = dict(cfg.model.params or {})
        max_ep = 2 if not mock_run else 1
        params["n_epochs"] = min(int(params.get("n_epochs", max_ep)), max_ep)
        if mock_run:
            params["batch_size"] = min(int(params.get("batch_size", 8)), 8)
            for k in ("hidden_dim", "hidden_size", "d_model"):
                if k in params:
                    params[k] = min(int(params[k]), 16)
        cfg = cfg.model_copy(
            update={
                "dataset": cfg.dataset.model_copy(update=dataset_update),
                "feature_pipeline": feature_pipeline,
                "model": cfg.model.model_copy(update={"params": params}),
            }
        )
    if getattr(cfg, "models_to_run", None):
        dry_models = [
            model.model_copy(update={"params": {**(model.params or {}), "n_epochs": 1}})
            for model in cfg.models_to_run
        ]
        cfg = cfg.model_copy(
            update={
                "dataset": cfg.dataset.model_copy(update=dataset_update),
                "feature_pipeline": feature_pipeline,
                "models_to_run": dry_models,
            }
        )
    return cfg


_DATASET_ALIASES: dict[str, str] = {
    "deap": "DEAP",
    "seedv": "SEED-V",
    "seed-v": "SEED-V",
    "seed_v": "SEED-V",
    "seed": "SEED",
    "dreamer": "DREAMER",
}


def _override_roots(cfg: Any, args: argparse.Namespace, dataset_key: str) -> Any:
    mock = getattr(args, "mock_run", False)
    if args.dry_run or mock:
        return _make_dry_run_cfg(cfg, dataset_key, mock_run=mock)

    dataset_root = None
    if dataset_key == "DEAP":
        dataset_root = args.deap_root
    elif dataset_key == "SEED-V":
        dataset_root = args.seedv_root
    elif dataset_key == "SEED":
        dataset_root = args.seed_root
    elif dataset_key == "DREAMER":
        dataset_root = args.dreamer_root

    updates: dict[str, Any] = {}
    if dataset_root:
        updates["dataset"] = cfg.dataset.model_copy(update={"root": dataset_root})
    if args.device:
        updates["experiment"] = cfg.experiment.model_copy(update={"device": args.device})
    if updates:
        return cfg.model_copy(update=updates)
    return cfg


def _fmt_result(result: dict[str, Any] | None) -> str:
    if not result:
        return "-"
    mean = result.get("mean", {}).get("accuracy", 0.0) * 100
    std = result.get("std", {}).get("accuracy", 0.0) * 100
    return f"{mean:.1f} +- {std:.1f}"


def _print_paper_tables(results: dict[str, Any]) -> None:
    models = ["CNN-LSTM", "BiDAE", "DGCNN", "Transformer-MM", "DGCCA-AM", "PR-PL"]
    print("\n" + "=" * 65)
    print("TABLE 2 - DEAP LOSO Accuracy (%)")
    print("=" * 65)
    print(f"{'Model':20s} {'Valence':>15s} {'Arousal':>15s}")
    print("-" * 65)
    for model in models:
        print(
            f"{model:20s} "
            f"{_fmt_result(results.get(f'{model}/DEAP/valence')):>15s} "
            f"{_fmt_result(results.get(f'{model}/DEAP/arousal')):>15s}"
        )

    print("\n" + "=" * 65)
    print("TABLE 3 - SEED-V 5-class LOSO Accuracy (%)")
    print("=" * 65)
    seedv = results.get("ALL/SEED-V/5class", {}).get("per_model", {})
    for model in models:
        stats = seedv.get(model, {})
        print(
            f"{model:20s}: "
            f"{stats.get('mean_acc', 0.0) * 100:.1f} +- "
            f"{stats.get('std_acc', 0.0) * 100:.1f}"
        )


def _save_latex_tables(results: dict[str, Any], out_dir: Path) -> None:
    models = [
        ("CNN-LSTM", "CNN-LSTM"),
        ("BiDAE", "BiDAE"),
        ("DGCNN", r"DGCNN$\dagger$"),
        ("Transformer-MM", "Transformer-MM"),
        ("DGCCA-AM", "DGCCA-AM"),
        ("PR-PL", r"PR-PL$\dagger$"),
    ]
    lines = [
        r"\begin{table}[t]",
        r"  \caption{LOSO accuracy (\%) on DEAP.}",
        r"  \label{tab:deap}",
        r"  \small",
        r"  \begin{tabular}{lcc}",
        r"    \toprule",
        r"    Model & Valence & Arousal \\",
        r"    \midrule",
    ]
    for key, label in models:
        val = results.get(f"{key}/DEAP/valence")
        aro = results.get(f"{key}/DEAP/arousal")
        val_str = _latex_metric(val)
        aro_str = _latex_metric(aro)
        lines.append(f"    {label} & {val_str} & {aro_str} \\\\")
    lines.extend([r"    \bottomrule", r"  \end{tabular}", r"\end{table}"])
    (out_dir / "table2_deap.tex").write_text("\n".join(lines), encoding="utf-8")

    seedv = results.get("ALL/SEED-V/5class", {}).get("per_model", {})
    lines = [
        r"\begin{table}[t]",
        r"  \caption{LOSO accuracy (\%) on SEED-V.}",
        r"  \label{tab:seedv}",
        r"  \small",
        r"  \begin{tabular}{lc}",
        r"    \toprule",
        r"    Model & Accuracy \\",
        r"    \midrule",
    ]
    for key, label in models:
        stats = seedv.get(key, {})
        mean_acc = stats.get("mean_acc", 0.0) * 100
        std_acc = stats.get("std_acc", 0.0) * 100
        lines.append(f"    {label} & " f"${mean_acc:.1f} \\pm {std_acc:.1f}$ \\\\")
    lines.extend([r"    \bottomrule", r"  \end{tabular}", r"\end{table}"])
    (out_dir / "table3_seedv.tex").write_text("\n".join(lines), encoding="utf-8")


def _latex_metric(result: dict[str, Any] | None) -> str:
    if not result:
        return "--"
    mean = result.get("mean", {}).get("accuracy", 0.0) * 100
    std = result.get("std", {}).get("accuracy", 0.0) * 100
    return f"${mean:.1f} \\pm {std:.1f}$"


def _run_statistical_analysis(results: dict[str, Any], out_dir: Path) -> None:
    from emokit.scripts.statistical_analysis import run_pairwise_wilcoxon

    per_model = {}
    for exp_key, result in results.items():
        if "/DEAP/valence" not in exp_key:
            continue
        model_name = exp_key.split("/")[0]
        per_model[model_name] = {
            str(sid): {"accuracy": metrics.get("accuracy", 0.0)}
            for sid, metrics in result.get("per_subject", {}).items()
        }
    if len(per_model) < 2:
        return

    tmp_path = out_dir / "deap_valence_per_subject.json"
    tmp_path.write_text(json.dumps(per_model, indent=2), encoding="utf-8")
    stats = run_pairwise_wilcoxon(str(tmp_path), alpha=0.05)
    (out_dir / "statistical_tests_deap_valence.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )


def _check_paper_claims(results: dict[str, Any]) -> None:
    print("\n" + "=" * 65)
    print(f"PAPER CLAIM VERIFICATION (+/-{TOLERANCE:.1f}%)")
    print("=" * 65)
    passed = failed = 0
    for model, metrics in PAPER_NUMBERS.items():
        for metric_key, paper_val in metrics.items():
            dataset, axis = metric_key.split("_", 1)
            exp_key = f"{model}/{dataset.replace('SEEDV', 'SEED-V')}/{axis}"
            result = results.get(exp_key)
            if result is None and dataset == "SEEDV":
                result = {
                    "mean": {
                        "accuracy": results.get("ALL/SEED-V/5class", {})
                        .get("per_model", {})
                        .get(model, {})
                        .get("mean_acc", 0.0)
                    },
                    "config": results.get("ALL/SEED-V/5class", {}).get("config", {}),
                }
            if result is None:
                print(f"MISSING {exp_key}")
                continue
            if dataset == "SEEDV" and result.get("config", {}).get("protocol") == "loso":
                print(
                    f"{model:15s} {metric_key:18s} "
                    "[SKIP: protocol mismatch, paper uses subject-dependent split]"
                )
                continue
            ours = result.get("mean", {}).get("accuracy", 0.0) * 100
            delta = abs(ours - paper_val)
            ok = delta <= TOLERANCE
            passed += int(ok)
            failed += int(not ok)
            marker = "PASS" if ok else "FAIL"
            print(
                f"{model:15s} {metric_key:18s} "
                f"ours={ours:5.1f} paper={paper_val:5.1f} delta={delta:4.1f} [{marker}]"
            )
    print(f"\nSummary: {passed}/{passed + failed} within tolerance")


def run_all(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run_log.jsonl"
    results_path = out_dir / "results_all.json"
    results_all: dict[str, Any] = {}
    if args.resume and results_path.exists():
        results_all = json.loads(results_path.read_text(encoding="utf-8"))
        logger.info("Resuming with %d finished experiments", len(results_all))

    experiments = EXPERIMENTS
    if args.datasets:
        allowed = {_DATASET_ALIASES.get(d.lower(), d.upper()) for d in args.datasets}
        experiments = [exp for exp in EXPERIMENTS if exp[1] in allowed]
        logger.info("Filtered to datasets %s (%d experiments)", allowed, len(experiments))

    for cfg_path, dataset_key, _axis, exp_key in experiments:
        if exp_key in results_all:
            logger.info("Skipping completed run: %s", exp_key)
            continue

        started = time.time()
        cfg = ConfigLoader.load(cfg_path)
        cfg = _override_roots(cfg, args, dataset_key)
        run_dir = out_dir / exp_key.replace("/", "_")
        cfg = cfg.model_copy(
            update={
                "output": cfg.output.model_copy(update={"results_dir": str(run_dir)})
            }
        )
        logger.info("Running %s", exp_key)

        try:
            result = LOSOEvaluator.run_from_config(cfg)
            elapsed = round(time.time() - started, 1)
            result["elapsed_seconds"] = elapsed
            result["timestamp"] = datetime.now().isoformat()
            results_all[exp_key] = result
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        {
                            "exp": exp_key,
                            "status": "ok",
                            "mean_acc": result.get("mean", {}).get("accuracy", 0.0),
                            "elapsed_seconds": elapsed,
                        }
                    )
                    + "\n"
                )
            results_path.write_text(
                json.dumps(results_all, indent=2, default=float),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.exception("Experiment failed: %s", exp_key)
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps({"exp": exp_key, "status": "error", "error": str(exc)})
                    + "\n"
                )
            if not args.skip_errors:
                raise

    _print_paper_tables(results_all)
    _save_latex_tables(results_all, out_dir)
    _run_statistical_analysis(results_all, out_dir)
    _check_paper_claims(results_all)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deap-root", default=None)
    parser.add_argument("--seedv-root", default=None)
    parser.add_argument("--seed-root", default=None)
    parser.add_argument("--dreamer-root", default=None)
    parser.add_argument("--output-dir", default="results/paper_experiments")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Filter experiments to specific datasets (e.g. deap seedv dreamer)",
    )
    parser.add_argument("--device", default=None, help="Override device (e.g. cuda, cpu)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--mock-run",
        action="store_true",
        help="Ultra-minimal CPU run: 1 epoch, tiny dims, for pipeline validation.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-errors", action="store_true")
    run_all(parser.parse_args())


if __name__ == "__main__":
    main()
