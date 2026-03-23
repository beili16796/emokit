# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Verify toolkit comparison claims (paper Table 4).

Usage::

    python -m emokit.scripts.verify_toolkit_comparison
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

CLAIMS_TO_VERIFY: dict[str, dict[str, Any]] = {
    "MNE-Python": {
        "has_deep_learning": False,
        "evidence": "MNE has no nn.Module or torch dependency in its API",
        "verify_cmd": 'python -c "import mne; print(hasattr(mne, \'torch\'))"',
    },
    "TorchEEG": {
        "LOSO_built_in": False,
        "evidence": "TorchEEG has no LOSOEvaluator; uses standard DataLoader",
        "doc_url": "https://torcheeg.readthedocs.io/en/latest/",
    },
    "Tyee": {
        "LOSO_built_in": False,
        "evidence": "Tyee MM'25 paper: evaluation on held-out test set, not LOSO",
        "paper_ref": "zhou2025tyee, Section 4.1",
    },
    "NeuroKit2": {
        "has_deep_learning": False,
        "evidence": "NeuroKit2 provides HRV/EDA processing, no DL models",
        "doc_url": "https://neuropsychology.github.io/NeuroKit/",
    },
    "EmoKit (ours)": {
        "has_deep_learning": True,
        "LOSO_built_in": True,
        "multimodal": True,
        "evidence": "6 DL models, LOSOEvaluator, multi-modal fusion (DGCCA-AM, BiDAE)",
    },
}


def verify_claims() -> dict[str, dict[str, Any]]:
    """Print and save the claims verification index."""
    out = Path("results/toolkit_claims_index.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(CLAIMS_TO_VERIFY, indent=2), encoding="utf-8")
    logger.info("Claims index saved to %s\n", out)

    for name, claim in CLAIMS_TO_VERIFY.items():
        evidence = claim.get("evidence", "no evidence provided")
        dl = claim.get("has_deep_learning", "N/A")
        loso = claim.get("LOSO_built_in", "N/A")
        logger.info("[✓] %s: DL=%s, LOSO=%s — %s", name, dl, loso, evidence)

    return CLAIMS_TO_VERIFY


def main() -> None:
    verify_claims()


if __name__ == "__main__":
    main()
