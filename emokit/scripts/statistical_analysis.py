# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Statistical significance analysis — pairwise Wilcoxon signed-rank tests.

Usage::

    python -m emokit.scripts.statistical_analysis \\
        results/loso_all_models_DEAP_valence.json \\
        --alpha 0.05 --output results/statistical_tests_DEAP_valence.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import wilcoxon

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def sig_marker(p_corrected: float) -> str:
    if p_corrected < 0.001:
        return "***"
    if p_corrected < 0.01:
        return "**"
    if p_corrected < 0.05:
        return "*"
    return "ns"


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    diff = a - b
    pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    if pooled_std < 1e-12:
        return 0.0
    return float(np.mean(diff) / pooled_std)


def run_pairwise_wilcoxon(
    json_path: str,
    alpha: float = 0.05,
    metric: str = "accuracy",
) -> dict[str, dict[str, Any]]:
    """Run pairwise Wilcoxon tests from a multi-model results JSON.

    The JSON should be a dict mapping model_name -> {subject_id: {metric: value}}.

    Args:
        json_path: Path to results JSON.
        alpha: Significance level before correction.
        metric: Metric key to compare.

    Returns:
        Dict mapping "ModelA vs ModelB" -> test results.
    """
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))

    per_model: dict[str, dict[str, float]] = {}
    for model_name, subjects in data.items():
        per_model[model_name] = {}
        for sid, metrics in subjects.items():
            if isinstance(metrics, dict) and metric in metrics:
                per_model[model_name][sid] = float(metrics[metric])
            elif isinstance(metrics, (int, float)):
                per_model[model_name][sid] = float(metrics)

    models = sorted(per_model.keys())
    n_tests = max(1, len(list(combinations(models, 2))))
    alpha_corrected = alpha / n_tests

    results: dict[str, dict[str, Any]] = {}

    for m_a, m_b in combinations(models, 2):
        common = sorted(set(per_model[m_a].keys()) & set(per_model[m_b].keys()))
        if len(common) < 2:
            continue

        vec_a = np.array([per_model[m_a][s] for s in common])
        vec_b = np.array([per_model[m_b][s] for s in common])

        diffs = vec_a - vec_b
        if np.all(diffs == 0):
            w_stat, p_val = float("nan"), 1.0
        else:
            res = wilcoxon(vec_a, vec_b, alternative="two-sided")
            w_stat, p_val = float(res.statistic), float(res.pvalue)

        p_corr = min(p_val * n_tests, 1.0)
        d = cohens_d(vec_a, vec_b)

        key = f"{m_a} vs {m_b}"
        results[key] = {
            "W": w_stat,
            "p": p_val,
            "p_corrected": p_corr,
            "d": d,
            "significant": sig_marker(p_corr),
            "n_subjects": len(common),
            "mean_a": float(np.mean(vec_a)),
            "mean_b": float(np.mean(vec_b)),
        }

        logger.info(
            "  %s | W=%.1f | p=%.6f | p_corr=%.6f | d=%.3f | %s",
            key, w_stat, p_val, p_corr, d, sig_marker(p_corr),
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_file", help="Multi-model results JSON.")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--metric", type=str, default="accuracy")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    results = run_pairwise_wilcoxon(args.json_file, args.alpha, args.metric)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        logger.info("Saved to %s", out)

    n_sig = sum(1 for r in results.values() if r["significant"] != "ns")
    logger.info("%d / %d significant at α=%.4f (Bonferroni)", n_sig, len(results), args.alpha)


if __name__ == "__main__":
    main()
