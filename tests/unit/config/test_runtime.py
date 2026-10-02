"""Runtime path configuration invariants."""
from __future__ import annotations

from pathlib import Path

import pytest


def test_paths_resolve_against_repository_root_not_working_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.config import load_runtime_config

    monkeypatch.chdir(tmp_path)
    config = load_runtime_config()

    repo = Path(__file__).resolve().parents[3]
    assert config.repo_root == repo
    assert config.data_root == repo / "data"
    assert config.default_scope == repo / "config" / "research" / "kr_swing_2019_v1.toml"
    assert config.logs_root == repo / "logs"
    assert config.market_rules == repo / "config" / "market" / "krx_market_rules.toml"
    assert config.reference_benchmarks == repo / "config" / "data" / "reference_benchmarks.toml"
    assert config.hedge_series == repo / "config" / "data" / "hedge_series.toml"
    assert config.cash_series == repo / "config" / "data" / "cash_series.toml"
    assert config.engine == repo / "config" / "backtest" / "default_engine.toml"
    assert config.strategies_root == repo / "config" / "research" / "strategies"
    for path in (
        config.data_root,
        config.default_scope,
        config.logs_root,
        config.market_rules,
        config.reference_benchmarks,
        config.hedge_series,
        config.cash_series,
        config.engine,
        config.research_protocol,
        config.strategies_root,
    ):
        assert path.is_absolute()


def test_research_protocol_path_resolved_and_exists() -> None:
    from src.config import load_runtime_config

    config = load_runtime_config()
    assert config.research_protocol.is_absolute()
    assert config.research_protocol.is_file()
    assert config.hedge_series.is_absolute()
    assert config.hedge_series.is_file()
    assert config.cash_series.is_absolute()
    assert config.cash_series.is_file()


def test_unknown_key_fails_closed(tmp_path: Path) -> None:
    from src.config import ConfigError, load_runtime_config

    source = Path("config/runtime.toml").read_text(encoding="utf-8")
    extra = tmp_path / "runtime.toml"
    extra.write_text(source + '\nunknown_key = "nope"\n', encoding="utf-8")
    with pytest.raises(ConfigError):
        load_runtime_config(extra)


def test_missing_referenced_file_fails_closed_with_key_name(tmp_path: Path) -> None:
    from src.config import ConfigError, load_runtime_config

    source = Path("config/runtime.toml").read_text(encoding="utf-8")
    (tmp_path / "config").mkdir()
    broken = tmp_path / "config" / "runtime.toml"
    repo = Path("config/runtime.toml").resolve().parent.parent
    document = source
    for key, relative in (
        ("data_root", "data"),
        ("default_scope", "config/research/kr_swing_2019_v1.toml"),
        ("logs_root", "logs"),
        ("reference_benchmarks", "config/data/reference_benchmarks.toml"),
        ("hedge_series", "config/data/hedge_series.toml"),
        ("cash_series", "config/data/cash_series.toml"),
        ("engine", "config/backtest/default_engine.toml"),
        ("research_protocol", "config/research/protocol.toml"),
        ("strategies_root", "config/research/strategies"),
    ):
        document = document.replace(f'{key} = "{relative}"', f'{key} = "{repo / relative}"')
    document = document.replace(
        "config/market/krx_market_rules.toml", "config/market/does_not_exist.toml"
    )
    broken.write_text(document, encoding="utf-8")
    with pytest.raises(ConfigError, match="market_rules"):
        load_runtime_config(broken)


def test_blank_path_and_missing_file_fail_closed(tmp_path: Path) -> None:
    import pytest

    from src.config.runtime import ConfigError, _resolve, load_runtime_config

    with pytest.raises(ConfigError, match="non-empty path string"):
        _resolve(tmp_path, "data_root", "  ")
    with pytest.raises(ConfigError, match="missing"):
        load_runtime_config(tmp_path / "absent.toml")
    broken = tmp_path / "broken.toml"
    broken.write_text("data_root = [\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_runtime_config(broken)


def test_missing_strategies_dir_fails_closed(tmp_path: Path) -> None:
    from src.config import ConfigError, load_runtime_config

    source = Path("config/runtime.toml").read_text(encoding="utf-8")
    (tmp_path / "config").mkdir()
    broken = tmp_path / "config" / "runtime.toml"
    repo = Path("config/runtime.toml").resolve().parent.parent
    document = source
    for key, relative in (
        ("data_root", "data"),
        ("default_scope", "config/research/kr_swing_2019_v1.toml"),
        ("logs_root", "logs"),
        ("reference_benchmarks", "config/data/reference_benchmarks.toml"),
        ("hedge_series", "config/data/hedge_series.toml"),
        ("cash_series", "config/data/cash_series.toml"),
        ("engine", "config/backtest/default_engine.toml"),
        ("research_protocol", "config/research/protocol.toml"),
        ("strategies_root", "config/research/strategies"),
    ):
        document = document.replace(f'{key} = "{relative}"', f'{key} = "{repo / relative}"')
    document = document.replace(f'"{repo / "config/research/strategies"}"', f'"{tmp_path / "absent-strategies"}"')
    (tmp_path / "config" / "market").mkdir(parents=True)
    (tmp_path / "config" / "market" / "krx_market_rules.toml").write_bytes(
        (repo / "config/market/krx_market_rules.toml").read_bytes()
    )
    document = document.replace(
        f'"{repo / "config/market/krx_market_rules.toml"}"',
        f'"{tmp_path / "config/market/krx_market_rules.toml"}"',
    )
    broken.write_text(document, encoding="utf-8")
    with pytest.raises(ConfigError, match="strategies_root"):
        load_runtime_config(broken)
