"""Tests for environment capture."""

import pytest
from pathlib import Path

from labelforge.core.env_capture import (
    EnvironmentSnapshot,
    capture_environment,
)


class TestEnvironmentSnapshot:
    """Tests for EnvironmentSnapshot."""

    def test_capture_environment(self):
        """Should capture environment successfully."""
        snapshot = capture_environment()
        
        assert snapshot.python_version is not None
        assert snapshot.platform_system is not None
        assert snapshot.platform_release is not None

    def test_snapshot_has_pip_packages(self):
        """Should capture pip packages."""
        snapshot = capture_environment()
        
        assert snapshot.pip_freeze is not None
        assert len(snapshot.pip_freeze) > 0
        # pytest should be in the list
        assert any("pytest" in pkg.lower() for pkg in snapshot.pip_freeze)

    def test_snapshot_has_timestamp(self):
        """Should have capture timestamp."""
        snapshot = capture_environment()
        
        assert snapshot.captured_at is not None

    def test_snapshot_serialization(self):
        """Should serialize to dict."""
        snapshot = capture_environment()
        data = snapshot.to_dict()
        
        assert "python_version" in data
        assert "platform_system" in data
        assert "pip_freeze" in data

    def test_snapshot_json_serialization(self):
        """Should serialize to JSON."""
        snapshot = capture_environment()
        json_str = snapshot.to_json()
        
        assert "python_version" in json_str
        assert isinstance(json_str, str)

    def test_snapshot_save_load(self, tmp_path):
        """Should save and load snapshot."""
        snapshot = capture_environment()
        path = tmp_path / "env_snapshot.json"
        
        snapshot.save(path)
        assert path.exists()

        loaded = EnvironmentSnapshot.load(path)
        assert loaded.python_version == snapshot.python_version
