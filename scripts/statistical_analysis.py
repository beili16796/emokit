# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Statistical analysis: Wilcoxon signed-rank tests with Bonferroni correction.

Compares per-subject accuracy vectors from two or more evaluation result JSON
files and reports whether differences are statistically significant.

Usage::

    python scripts/statistical_analysis.py \
        results/model_a.json results/model_b.json

    python scripts/statistical_analysis.py \
        results/model_a.json results/model_b.json results/model_c.json \
        --metric accuracy --alpha 0.05
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
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def load_per_subject_metric(
    json_path: str,
    metric: str = "accuracy",
) -> tuple[str, dict[int, float]]:
    """Extract per-subject metric values from an evaluator JSON file.

    Args:
        json_path: Path to evaluator output JSON.
        metric: Metric key inside per-subject dicts.

    Returns:
        ``(label, {subject_id: value})`` pair.
    """
    path = Path(json_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    per_subject = data.get("per_subject", {})
    config = data.get("config", {})
    label = config.get("model_name", path.stem)

    values: dict[int, float] = {}
    for sid_str, metrics in per_subject.items():
        sid = int(sid_str)
        if metric not in metrics:
            logger.warning(
                "Metric '%s' missing for subject %s in %s",
                metric,
                sid_str,
                json_path,
            )
            continue
        values[sid] = float(metrics[metric])

    return label, values


def wilcoxon_test(
    values_a: list[float],
    values_b: list[float],
) -> tuple[float, float]:
    """Run a two-sided Wilcoxon signed-rank test.

    Args:
        values_a: Per-subject metric values for model A.
        values_b: Per-subject metric values for model B.

    Returns:
        ``(statistic, p_value)`` tuple.  Returns ``(nan, nan)`` when all
        differences are zero or fewer than 6 paired samples exist.
    """
    diffs = np.array(values_a) - np.array(values_b)
    if np.all(diffs == 0):
        return float("nan"), 1.0
    if len(diffs) < 6:
        logger.warning(
            "Fewer than 6 paired samples (%d); Wilcoxon results unreliable.",
            len(diffs),
        )
    result = stats.wilcoxon(values_a, values_b, alternative="two-sided")
    return float(result.statistic), float(result.pvalue)


def run_pairwise_analysis(
    json_paths: list[str],
    metric: str = "accuracy",
    alpha: float = 0.05,
) -> list[dict[str, Any]]:
    """Run all pairwise Wilcoxon tests with Bonferroni correction.

    Args:
        json_paths: Paths to evaluator output JSONs.
        metric: Metric to compare.
        alpha: Family-wise significance level.

    Returns:
        List of comparison result dicts.
    """
    loaded: list[tuple[str, dict[int, float]]] = []
    for jp in json_paths:
        loaded.append(load_per_subject_metric(jp, metric))

    n_comparisons = max(1, len(list(combinations(range(len(loaded)), 2))))
    corrected_alpha = alpha / n_comparisons
    logger.info(
        "Bonferroni correction: %d comparisons, α=%.4f → corrected α=%.6f",
        n_comparisons,
        alpha,
        corrected_alpha,
    )

    results: list[dict[str, Any]] = []

    for (label_a, vals_a), (label_b, vals_b) in combinations(loaded, 2):
        common_sids = sorted(set(vals_a.keys()) & set(vals_b.keys()))
        if len(common_sids) < 2:
            logger.warning(
                "Skipping %s vs %s — only %d common subjects.",
                label_a,
                label_b,
                len(common_sids),
            )
            continue

        vec_a = [vals_a[s] for s in common_sids]
        vec_b = [vals_b[s] for s in common_sids]

        stat, pval = wilcoxon_test(vec_a, vec_b)
        significant = pval < corrected_alpha

        row = {
            "model_a": label_a,
            "model_b": label_b,
            "metric": metric,
            "n_subjects": len(common_sids),
            "mean_a": float(np.mean(vec_a)),
            "mean_b": float(np.mean(vec_b)),
            "statistic": stat,
            "p_value": pval,
            "corrected_alpha": corrected_alpha,
            "significant": significant,
        }
        results.append(row)

        sig_str = "***" if significant else "n.s."
        logger.info(
            "  %s vs %s  mean=%.4f vs %.4f  W=%.1f  p=%.6f  %s",
            label_a,
            label_b,
            row["mean_a"],
            row["mean_b"],
            stat,
            pval,
            sig_str,
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pairwise Wilcoxon signed-rank tests with Bonferroni correction.",
    )
    parser.add_argument(
        "json_files",
        nargs="+",
        help="Two or more evaluator result JSON files.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="accuracy",
        help="Per-subject metric key to compare (default: accuracy).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Family-wise significance level (default: 0.05).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save results JSON.",
    )
    args = parser.parse_args()

    if len(args.json_files) < 2:
        logger.error("At least 2 result files are required.")
        sys.exit(1)

    results = run_pairwise_analysis(
        args.json_files,
        metric=args.metric,
        alpha=args.alpha,
    )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        logger.info("Results saved to %s", out)

    if not results:
        logger.info("No valid comparisons performed.")
        sys.exit(0)

    n_sig = sum(1 for r in results if r["significant"])
    logger.info(
        "\nDone. %d / %d comparisons significant at α=%.4f (Bonferroni-corrected).",
        n_sig,
        len(results),
        args.alpha,
    )


if __name__ == "__main__":
    main()
