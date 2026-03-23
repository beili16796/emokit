.. MIT License
   Copyright (c) 2024 EmoKit Contributors
   See LICENSE for full text.

Quickstart
==========

This guide walks you through running your first emotion recognition experiment
with EmoKit in under five minutes.

Prerequisites
-------------

Install EmoKit and its dependencies::

    pip install emokit

Download at least one dataset (e.g. DEAP) and place the files under a data
directory such as ``data/DEAP/``.

Option 1 — Python API
----------------------

Load a dataset, build a feature pipeline, and evaluate with LOSO:

.. code-block:: python

    from emokit.datasets import load_dataset
    from emokit.features.eeg import DEExtractor, EEGNormalizer
    from emokit.features.base import FeaturePipeline
    from emokit.evaluation.protocols import LOSOEvaluator

    # 1. Load the DEAP dataset (binary valence classification)
    ds = load_dataset("DEAP", root="data/DEAP", modalities=["eeg"])

    # 2. Define the feature pipeline
    pipeline = FeaturePipeline([
        ("de", DEExtractor(fs=128)),
        ("norm", EEGNormalizer()),
    ])

    # 3. Configure and run LOSO evaluation with a DGCNN model
    evaluator = LOSOEvaluator(
        dataset=ds,
        feature_pipeline=pipeline,
        model_config={"n_channels": 32, "n_bands": 5, "hidden_dim": 64, "n_epochs": 50},
        model_name="DGCNN",
        seed=42,
    )
    results = evaluator.run()

    # 4. Inspect results
    print(f"Mean accuracy: {results['mean']['accuracy']:.4f}")
    print(f"Std accuracy:  {results['std']['accuracy']:.4f}")

Option 2 — YAML Config
-----------------------

Create an experiment configuration file (or use the provided
``configs/deap_loso_dgcnn.yaml``):

.. code-block:: yaml

    experiment:
      name: deap_loso_dgcnn
      seed: 42
      device: cpu

    dataset:
      name: DEAP
      root: data/DEAP
      modalities: [eeg]
      label_axis: valence

    feature_pipeline:
      steps:
        - name: DEExtractor
          params: { fs: 128 }
        - name: EEGNormalizer
          params: {}

    model:
      name: DGCNN
      params:
        n_channels: 32
        n_bands: 5
        hidden_dim: 64
        n_epochs: 50

    evaluation:
      protocol: loso
      val_fraction: 0.1

    output:
      results_dir: results/

Then run the experiment from the command line::

    python -m emokit --config configs/deap_loso_dgcnn.yaml

Results are saved to ``results/`` as JSON and CSV files.

Understanding the Results
-------------------------

The evaluator returns a dictionary with four keys:

``per_subject``
    A mapping from subject ID to that fold's classification metrics
    (accuracy, F1-macro, F1-weighted).

``mean``
    Mean of each metric across all subjects.

``std``
    Standard deviation of each metric across all subjects.

``config``
    Metadata about the experiment (dataset name, model, seed, protocol).

Next Steps
----------

- Explore other datasets: ``SEED``, ``SEED-V``, ``MAHNOB-HCI``, ``DREAMER``.
- Try different models: ``CNN-LSTM``, ``Transformer-MM``, ``BiDAE``,
  ``DGCCA-AM``, ``PR-PL``.
- Add peripheral features with :class:`~emokit.features.peripheral` extractors.
- Write custom transforms by subclassing
  :class:`~emokit.features.base.BaseTransform`.
