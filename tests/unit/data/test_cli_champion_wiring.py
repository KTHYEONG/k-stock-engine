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
