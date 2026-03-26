# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Evidence index for toolkit comparison tables — fill URLs and verification notes.

Run::

    python scripts/verify_toolkit_comparison.py

Outputs a checklist reminder; does **not** crawl the web automatically.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

CLAIMS: dict[str, dict[str, Any]] = {
    "MNE-Python": {
        "has_deep_learning": False,
        "supports_MPER": False,
        "has_LOSO": False,
        "evidence_url": "https://mne.tools/stable/api/index.html",
        "verification_notes": "Confirm via docs + optional local pip show mne",
    },
    "NeuroKit2": {
        "has_deep_learning": False,
        "supports_MPER": "indirect",
        "has_LOSO": False,
        "evidence_url": "https://neuropsychology.github.io/NeuroKit/",
        "verification_notes": "HRV/EDA pipelines; not an EEG-DL benchmark toolkit",
    },
    "TorchEEG": {
        "has_deep_learning": True,
        "supports_MPER": "partial",
        "has_LOSO": "user-implemented",
        "evidence_url": "https://torcheeg.readthedocs.io",
        "verification_notes": "Verify multimodal + subject protocols in docs",
    },
}


def main() -> None:
    """Print and optionally save the claims registry."""
    out = Path("results/toolkit_claims_index.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(CLAIMS, indent=2), encoding="utf-8")
    logger.info(
        "Wrote %s — replace booleans with cited evidence before submission.",
        out,
    )
    for name, row in CLAIMS.items():
        logger.info("%s: %s", name, row.get("evidence_url"))


if __name__ == "__main__":
    main()
