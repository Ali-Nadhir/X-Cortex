from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest
from PIL import Image

from xcortex.config import Settings


def make_settings(tmp_path: Path, mode: str = "mock") -> Settings:
    settings = Settings()
    settings.app.mode = mode
    settings.paths.data = str(tmp_path / "data")
    settings.paths.db = str(tmp_path / "db")
    settings.paths.artifacts = str(tmp_path / "artifacts")
    settings.paths.state = str(tmp_path / "state/sessions.json")
    return settings


@pytest.fixture
def settings(tmp_path):
    return make_settings(tmp_path)


@pytest.fixture
def study_png(tmp_path) -> Path:
    gradient = np.tile(np.linspace(20, 235, 320, dtype=np.uint8), (320, 1))
    path = tmp_path / "study.png"
    Image.fromarray(gradient).save(path)
    return path
