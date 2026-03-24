from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from emokit.evaluation.config import ConfigLoader
from emokit.evaluation.result_logger import ResultLogger


def test_run_all_experiments_dry_run(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "emokit.scripts.run_all_experiments",
            "--dry-run",
            "--output-dir",
            str(tmp_path / "paper_runs"),
            "--skip-errors",
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert "TABLE 2" in result.stdout
    assert "TABLE 3" in result.stdout


def test_config_env_expansion_and_base(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMOKIT_DATA_ROOT", str(tmp_path))
    base = tmp_path / "base.yaml"
    base.write_text(
        "experiment:\n  name: base\n"
        "dataset:\n  name: DEAP\n  root: ${EMOKIT_DATA_ROOT}/DEAP\n"
        "model:\n  name: DGCNN\n",
        encoding="utf-8",
    )
    child = tmp_path / "child.yaml"
    child.write_text(
        "_base_: base.yaml\n"
        "experiment:\n  name: child\n",
        encoding="utf-8",
    )
    cfg = ConfigLoader.load(str(child))
    assert cfg.experiment.name == "child"
    assert "${" not in str(cfg.dataset.root)
    assert str(tmp_path) in str(cfg.dataset.root)


def test_all_yaml_configs_are_valid() -> None:
    configs = sorted(Path("configs").glob("*.yaml"))
    assert len(configs) >= 13
    for path in configs:
        if path.name == "standard_protocol.yaml":
            continue
        cfg = ConfigLoader.load(str(path))
        assert cfg.model is not None or cfg.models_to_run


def test_result_logger_leaderboard_append(tmp_path: Path) -> None:
    logger = ResultLogger(tmp_path)
    mock_result = {
        "mean": {"accuracy": 0.731},
        "std": {"accuracy": 0.051},
        "config": {"model": "DGCNN", "dataset": "DEAP"},
        "per_subject": {},
    }
    leaderboard = tmp_path / "leaderboard.csv"
    logger.append_to_leaderboard(mock_result, leaderboard)
    logger.append_to_leaderboard(mock_result, leaderboard)
    df = pd.read_csv(leaderboard)
    assert len(df) == 2
    assert "model" in df.columns
    assert "accuracy" in df.columns
