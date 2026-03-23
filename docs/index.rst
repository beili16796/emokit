.. MIT License
   Copyright (c) 2024 EmoKit Contributors
   See LICENSE for full text.

EmoKit Documentation
====================

**Modular Physiological Signal Analysis & Benchmarking Toolkit** for EEG-based
emotion recognition research.

EmoKit provides unified dataset loaders, feature extraction pipelines,
deep-learning models, and reproducible evaluation protocols — all driven by a
single YAML configuration file.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   quickstart

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/datasets
   api/features
   api/models
   api/evaluation

Introduction
------------

EmoKit is designed to accelerate affective computing research by providing:

- **Unified dataset loaders** for DEAP, SEED, SEED-V, MAHNOB-HCI, and DREAMER.
- **Feature extraction** pipelines including Differential Entropy (DE), band
  power, and peripheral signal features.
- **Deep-learning models** spanning CNN-LSTM, DGCNN, Transformer-MM, BiDAE,
  DGCCA-AM, and PR-PL.
- **Evaluation protocols**: Leave-One-Subject-Out (LOSO), subject-dependent, and
  cross-session evaluation with automatic metric logging.

Installation
------------

From PyPI::

    pip install emokit

Editable install for development::

    git clone https://github.com/emokit/emokit.git
    cd emokit
    pip install -e ".[dev]"

Build the docs locally::

    pip install -e ".[docs]"
    cd docs && make html


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
