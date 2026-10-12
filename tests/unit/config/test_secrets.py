"""Process-environment secret access contracts."""

import pytest

from src.config.errors import ConfigError
from src.config.secrets import environment_snapshot, read_secret


def test_environment_snapshot_is_a_detached_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KSE_SNAPSHOT_PROBE", "value-123")
    snapshot = environment_snapshot()
    assert snapshot["KSE_SNAPSHOT_PROBE"] == "value-123"
    snapshot["KSE_SNAPSHOT_PROBE"] = "changed"
    assert environment_snapshot()["KSE_SNAPSHOT_PROBE"] == "value-123"


def test_read_secret_default_and_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KSE_SECRET_PROBE", raising=False)
    assert read_secret("KSE_SECRET_PROBE", default="fallback") == "fallback"
    with pytest.raises(ConfigError, match="KSE_SECRET_PROBE"):
        read_secret("KSE_SECRET_PROBE")
    with pytest.raises(ConfigError):
        read_secret(" ")
