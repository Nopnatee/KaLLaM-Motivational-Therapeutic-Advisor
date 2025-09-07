import json
import os
import shutil
from pathlib import Path
import pytest

# Ensure a clean data dir per run
@pytest.fixture(autouse=True)
def clean_export_dir(tmp_path, monkeypatch):
    export_dir = tmp_path / "exported_sessions"
    export_dir.mkdir()
    monkeypatch.setenv("EXPORT_FOLDER", str(export_dir))
    yield
    shutil.rmtree(export_dir, ignore_errors=True)

@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "chatbot_data.db")
