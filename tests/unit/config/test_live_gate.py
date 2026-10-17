"""Live gate policy loading invariants."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.config import ConfigError


def test_loads_valid_policy() -> None:
    from src.config.live_gate import load_live_gate_policy

    policy = load_live_gate_policy(Path("config/research/live_gate.toml"))
    assert 0.0 <= policy.max_missing_flow_fraction < 1.0


def test_rejects_bad_files(tmp_path: Path) -> None:
    from src.config.live_gate import load_live_gate_policy

    with pytest.raises(ConfigError):
        load_live_gate_policy(tmp_path / "absent.toml")
    broken = tmp_path / "broken.toml"
    broken.write_text("flow = [\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_live_gate_policy(broken)
    for name, text in (
        ("unknown-top.toml", '[flow]\nmax_missing_flow_fraction = 0.02\n[surprise]\nx = 1\n'),
        ("unknown-key.toml", '[flow]\nmax_missing_flow_fraction = 0.02\nunlimited = true\n'),
        ("negative.toml", "[flow]\nmax_missing_flow_fraction = -0.1\n"),
        ("one.toml", "[flow]\nmax_missing_flow_fraction = 1.0\n"),
        ("missing-section.toml", ""),
        ("flow-not-table.toml", "flow = 0.02\n"),
        ("missing-field.toml", "[flow]\n"),
    ):
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ConfigError):
            load_live_gate_policy(path)
