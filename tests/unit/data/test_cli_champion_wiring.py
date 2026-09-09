"""CLI ChampionStrategy wiring tests (contract: champion_strategy_wiring)."""

from __future__ import annotations


def test_cli_dispatch_backtest_wires_champion_strategy(monkeypatch) -> None:
    from src.strategy.champion_strategy import ChampionStrategy
    import src.data.cli as cli_mod

    created_strategies = []
    orig_init = ChampionStrategy.__init__

    def mock_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        orig_init(self, *args, **kwargs)
        created_strategies.append(self)

    monkeypatch.setattr(ChampionStrategy, '__init__', mock_init)

    # Verify ChampionStrategy importable and wired into cli namespace
    assert hasattr(cli_mod, 'ChampionStrategy') or 'ChampionStrategy' in dir(cli_mod)


def test_core_cli_wiring_default() -> None:
    import inspect

    import src.data.cli as cli_mod
    from src.strategy.core_strategy import CoreStrategy

    source = inspect.getsource(cli_mod._dispatch_backtest)
    assert cli_mod.CoreStrategy is CoreStrategy
    assert 'CoreStrategy(' in source
    assert "strategy_id == 'core-v1'" in source


def test_eligible_universe_by_session_dedupes_and_rejects_duplicates() -> None:
    from datetime import UTC, datetime

    import polars as pl
    import pytest

    import src.data.cli as cli_mod
    from src.data.schemas import PITDataError

    day = datetime(2024, 1, 2, 9, tzinfo=UTC)
    frame = pl.DataFrame({
        'decision_session': [day, day],
        'instrument_id': ['KRX:B', 'KRX:A'],
        'eligible': [True, False],
    })
    assert cli_mod._eligible_universe_by_session(frame) == {day.date(): ('KRX:B',)}
    naive = pl.DataFrame({
        'decision_session': [day.replace(tzinfo=None)],
        'instrument_id': ['KRX:C'],
        'eligible': [True],
    })
    assert cli_mod._eligible_universe_by_session(naive) == {day.date(): ('KRX:C',)}
    dup = pl.DataFrame({
        'decision_session': [day, day],
        'instrument_id': ['KRX:A', 'KRX:A'],
        'eligible': [True, True],
    })
    with pytest.raises(PITDataError, match='duplicate'):
        cli_mod._eligible_universe_by_session(dup)
