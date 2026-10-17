"""Adapter invariants for host_broker."""
from __future__ import annotations

import ast
import hashlib
import logging
import sys
from pathlib import Path

import pytest

from src.config.errors import ConfigError

from src.integrations import host_broker as hb


def test_vendored_module_is_stdlib_only() -> None:
    tree = ast.parse(Path("src/integrations/host_broker_contract.py").read_text(encoding="utf-8"))
    mods: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.append(node.module)
    assert mods
    for name in mods:
        top = name.split(".")[0]
        assert not top.startswith("src")
        assert top in sys.stdlib_module_names


def test_only_adapter_imports_vendored_module() -> None:
    importers: list[str] = []
    for path in Path("src").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(a.name == "src.integrations.host_broker_contract" for a in node.names):
                    importers.append(str(path))
            elif isinstance(node, ast.ImportFrom) and (
                node.module == "src.integrations.host_broker_contract" or (
                    node.module == "src.integrations"
                    and any(a.name == "host_broker_contract" for a in node.names)
                ) or (
                    node.level == 1 and node.module == "host_broker_contract"
                    and path.parent == Path("src/integrations")
                )
            ):
                importers.append(str(path))
    assert sorted(set(importers)) == ["src/integrations/host_broker.py"]


def test_broker_dir_from_env(tmp_path, monkeypatch) -> None:
    marked = tmp_path / "marked"
    marked.mkdir()
    (marked / ".host-admission").write_text("", encoding="utf-8")
    monkeypatch.setenv("X_DIR", str(marked))
    assert hb.resolve_broker_dir("X_DIR") == marked
    monkeypatch.delenv("X_DIR")
    with pytest.raises(ConfigError, match="X_DIR"):
        hb.resolve_broker_dir("X_DIR")
    monkeypatch.setenv("X_DIR", "   ")
    with pytest.raises(ConfigError, match="X_DIR"):
        hb.resolve_broker_dir("X_DIR")


def test_undeclared_rate_fails_closed() -> None:
    with pytest.raises(ValueError, match=r"."):
        hb.contract_rate("dbsec", "orders")


def test_dbsec_rates_match_vendor_limits() -> None:
    assert hb.contract_rate("dbsec") == 20.0
    assert hb.contract_rate("dbsec", "chart") == 4.0
    assert hb.contract_rate("dbsec", "quote") == 3.0


def test_lock_timeout_must_exceed_issue_interval(tmp_path) -> None:
    broker = tmp_path / "b"
    broker.mkdir()
    with pytest.raises(ValueError, match=r"."):
        hb.open_token_store(broker, "dbsec", "K", lock_timeout_seconds=61.0, expiry_margin_seconds=0.0)
    store = hb.open_token_store(broker, "dbsec", "K", lock_timeout_seconds=240.0, expiry_margin_seconds=0.0)
    assert store._path.name == f"token_dbsec_{hashlib.sha256(b'K').hexdigest()[:12]}.json"


def test_admission_slot_names_contract_file(tmp_path) -> None:
    broker = tmp_path / "b"
    broker.mkdir()
    sha = hashlib.sha256(b"K").hexdigest()[:12]
    scoped = hb.AdmissionSlot(broker, "dbsec", "K", "chart")
    assert scoped.state_path.name == f"admission_dbsec_{sha}_chart.state"
    unscoped = hb.AdmissionSlot(broker, "dbsec", "K")
    assert unscoped.state_path.name == f"admission_dbsec_{sha}.state"


def test_secrets_never_logged(tmp_path, caplog) -> None:
    broker = tmp_path / "b"
    broker.mkdir()
    with caplog.at_level(logging.DEBUG):
        store = hb.open_token_store(
            broker, "dbsec", "SECRETKEY", lock_timeout_seconds=240.0, expiry_margin_seconds=0.0
        )
        store.get_or_issue(
            lambda: hb.IssuedToken("SECRETTOKEN", 3600.0),
        )
    text = caplog.text
    assert "SECRETKEY" not in text
    assert "SECRETTOKEN" not in text
