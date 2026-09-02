def test_construct_champion_portfolio_equal_volatility_allocates_equal_weights() -> None:
    from datetime import UTC, datetime
    import pytest
    from src.core.instruments import AssetKind, Instrument
    from src.core.portfolio import PortfolioSnapshot
    from src.strategy.portfolio import PortfolioConstructionStatus, PortfolioSecurityInput, construct_champion_portfolio
    from src.strategy.selection import ChampionSelectionResult

    now = datetime(2024, 1, 3, tzinfo=UTC)
    instruments = tuple(Instrument(f'KRX:{i:06d}', AssetKind.STOCK, 'KRX', f'{i:06d}', 'KRW') for i in range(20))
    selection = ChampionSelectionResult(now, 'account-1', tuple(x.instrument_id for x in instruments), (), 0, 'champion-v1-selection-v1')
    inputs = tuple(PortfolioSecurityInput(x, f'Sector-{i}', 0.20, 100_000_000.0) for i, x in enumerate(instruments))
    result = construct_champion_portfolio(selection, inputs, PortfolioSnapshot('account-1', now, 1_000_000.0, 0.0, ()), {x.instrument_id: 10_000.0 for x in instruments}, 0.10, decision_time=now)

    assert result.status is PortfolioConstructionStatus.ALLOCATED
    assert result.gross_exposure == pytest.approx(1.0)
    assert result.residual_cash == pytest.approx(0.0)
    assert [x.target_weight for x in result.targets] == pytest.approx([0.05] * 20)


def test_construct_champion_portfolio_redistributes_security_cap() -> None:
    from datetime import UTC, datetime
    import pytest
    from src.core.instruments import AssetKind, Instrument
    from src.core.portfolio import PortfolioSnapshot
    from src.strategy.portfolio import PortfolioSecurityInput, construct_champion_portfolio
    from src.strategy.selection import ChampionSelectionResult

    now = datetime(2024, 1, 3, tzinfo=UTC)
    instruments = tuple(Instrument(f'KRX:{i:06d}', AssetKind.STOCK, 'KRX', f'{i:06d}', 'KRW') for i in range(5))
    selection = ChampionSelectionResult(now, 'account-2', tuple(x.instrument_id for x in instruments), (), 15, 'champion-v1-selection-v1')
    inputs = tuple(PortfolioSecurityInput(x, f'Sector-{i}', 0.10 if i == 0 else 0.20, 100_000_000.0) for i, x in enumerate(instruments))
    result = construct_champion_portfolio(selection, inputs, PortfolioSnapshot('account-2', now, 1_000_000.0, 0.0, ()), {x.instrument_id: 10_000.0 for x in instruments}, 0.50, decision_time=now)

    weights = {x.allocation.instrument.instrument_id: x.target_weight for x in result.targets}
    assert weights[instruments[0].instrument_id] == pytest.approx(0.075)
    assert sum(weights.values()) == pytest.approx(0.30)
    assert all(value <= 0.075 for value in weights.values())


def test_construct_champion_portfolio_enforces_sector_cap() -> None:
    from datetime import UTC, datetime
    import pytest
    from src.core.instruments import AssetKind, Instrument
    from src.core.portfolio import PortfolioSnapshot
    from src.strategy.portfolio import PortfolioConstraint, PortfolioSecurityInput, construct_champion_portfolio
    from src.strategy.selection import ChampionSelectionResult

    now = datetime(2024, 1, 3, tzinfo=UTC)
    instruments = tuple(Instrument(f'KRX:{i:06d}', AssetKind.STOCK, 'KRX', f'{i:06d}', 'KRW') for i in range(20))
    selection = ChampionSelectionResult(now, 'account-3', tuple(x.instrument_id for x in instruments), (), 0, 'champion-v1-selection-v1')
    inputs = tuple(PortfolioSecurityInput(x, 'Technology' if i < 6 else f'Sector-{i}', 0.20, 100_000_000.0) for i, x in enumerate(instruments))
    result = construct_champion_portfolio(selection, inputs, PortfolioSnapshot('account-3', now, 1_000_000.0, 0.0, ()), {x.instrument_id: 10_000.0 for x in instruments}, 0.10, decision_time=now)

    technology = sum(x.target_weight for x in result.targets if x.sector == 'Technology')
    assert technology == pytest.approx(0.25)
    assert max(x.target_weight for x in result.targets if x.sector != 'Technology') > 0.05
    assert PortfolioConstraint.SECTOR_CAP in result.binding_constraints


def test_construct_champion_portfolio_scales_market_exposure_without_leverage() -> None:
    from datetime import UTC, datetime
    import pytest
    from src.core.instruments import AssetKind, Instrument
    from src.core.portfolio import PortfolioSnapshot
    from src.strategy.portfolio import PortfolioSecurityInput, construct_champion_portfolio
    from src.strategy.selection import ChampionSelectionResult

    now = datetime(2024, 1, 3, tzinfo=UTC)
    instruments = tuple(Instrument(f'KRX:{i:06d}', AssetKind.STOCK, 'KRX', f'{i:06d}', 'KRW') for i in range(20))
    selection = ChampionSelectionResult(now, 'account-4', tuple(x.instrument_id for x in instruments), (), 0, 'champion-v1-selection-v1')
    inputs = tuple(PortfolioSecurityInput(x, f'Sector-{i}', 0.20, 100_000_000.0) for i, x in enumerate(instruments))
    snapshot = PortfolioSnapshot('account-4', now, 1_000_000.0, 0.0, ())
    marks = {x.instrument_id: 10_000.0 for x in instruments}
    exposures = [construct_champion_portfolio(selection, inputs, snapshot, marks, vol, decision_time=now).gross_exposure for vol in (0.10, 0.15, 0.30)]

    assert exposures == pytest.approx([1.0, 1.0, 0.5])


def test_construct_champion_portfolio_applies_capacity_and_residual_cash() -> None:
    from datetime import UTC, datetime
    import pytest
    from src.core.instruments import AssetKind, Instrument
    from src.core.portfolio import PortfolioSnapshot
    from src.strategy.portfolio import PortfolioConstraint, PortfolioSecurityInput, construct_champion_portfolio
    from src.strategy.selection import ChampionSelectionResult

    now = datetime(2024, 1, 3, tzinfo=UTC)
    instruments = tuple(Instrument(f'KRX:{i:06d}', AssetKind.STOCK, 'KRX', f'{i:06d}', 'KRW') for i in range(20))
    selection = ChampionSelectionResult(now, 'account-5', tuple(x.instrument_id for x in instruments), (), 0, 'champion-v1-selection-v1')
    inputs = tuple(PortfolioSecurityInput(x, f'Sector-{i}', 0.20, 15_000_000.0) for i, x in enumerate(instruments))
    result = construct_champion_portfolio(selection, inputs, PortfolioSnapshot('account-5', now, 1_000_000.0, 0.0, ()), {x.instrument_id: 10_000.0 for x in instruments}, 0.10, decision_time=now)

    assert all(x.participation <= 0.0025 for x in result.targets)
    assert sum(x.allocation.target_value for x in result.targets) == pytest.approx(750_000.0)
    assert result.residual_cash == pytest.approx(250_000.0)
    assert PortfolioConstraint.TARGET_PARTICIPATION in result.binding_constraints


def test_construct_champion_portfolio_hard_capacity_returns_no_trade() -> None:
    from datetime import UTC, datetime
    from src.core.instruments import AssetKind, Instrument
    from src.core.portfolio import PortfolioSnapshot
    from src.strategy.portfolio import PortfolioConstructionStatus, PortfolioSecurityInput, construct_champion_portfolio
    from src.strategy.selection import ChampionSelectionResult

    now = datetime(2024, 1, 3, tzinfo=UTC)
    instruments = tuple(Instrument(f'KRX:{i:06d}', AssetKind.STOCK, 'KRX', f'{i:06d}', 'KRW') for i in range(20))
    selection = ChampionSelectionResult(now, 'account-6', tuple(x.instrument_id for x in instruments), (), 0, 'champion-v1-selection-v1')
    inputs = tuple(PortfolioSecurityInput(x, f'Sector-{i}', 0.20, 5_000_000.0) for i, x in enumerate(instruments))
    result = construct_champion_portfolio(selection, inputs, PortfolioSnapshot('account-6', now, 1_000_000.0, 0.0, ()), {x.instrument_id: 10_000.0 for x in instruments}, 0.10, decision_time=now)

    assert result.status is PortfolioConstructionStatus.NO_TRADE
    assert result.targets == ()


def test_construct_champion_portfolio_property_sample_preserves_all_caps() -> None:
    from datetime import UTC, datetime
    from math import isfinite
    from random import Random
    from src.core.instruments import AssetKind, Instrument
    from src.core.portfolio import PortfolioSnapshot
    from src.strategy.portfolio import PortfolioConstructionStatus, PortfolioSecurityInput, construct_champion_portfolio
    from src.strategy.selection import ChampionSelectionResult

    now = datetime(2024, 1, 3, tzinfo=UTC)
    random = Random(7)  # noqa: S311
    for case in range(32):
        instruments = tuple(Instrument(f'KRX:{case:02d}{i:04d}', AssetKind.STOCK, 'KRX', f'{case:02d}{i:04d}', 'KRW') for i in range(20))
        selection = ChampionSelectionResult(now, f'account-{case}', tuple(x.instrument_id for x in instruments), (), 0, 'champion-v1-selection-v1')
        inputs = tuple(PortfolioSecurityInput(x, f'Sector-{i % 7}', random.uniform(0.05, 0.80), 1_000_000_000.0) for i, x in enumerate(instruments))
        volatility = random.uniform(0.05, 0.60)
        result = construct_champion_portfolio(selection, inputs, PortfolioSnapshot(f'account-{case}', now, 1_000_000.0, 0.0, ()), {x.instrument_id: 10_000.0 for x in instruments}, volatility, decision_time=now)

        assert result.status is PortfolioConstructionStatus.ALLOCATED
        assert all(isfinite(x.target_weight) and x.target_weight >= 0.0 for x in result.targets)
        assert all(isfinite(x.allocation.target_value) and x.allocation.target_value >= 0.0 for x in result.targets)
        assert all(x.target_weight <= 0.075 for x in result.targets)
        assert result.gross_exposure <= min(1.0, 0.15 / volatility)
        sector_weights = {}
        for target in result.targets:
            sector_weights[target.sector] = sector_weights.get(target.sector, 0.0) + target.target_weight
        assert all(weight <= 0.25 for weight in sector_weights.values())
        assert all(x.participation <= 0.0025 for x in result.targets)


def test_construct_champion_portfolio_invalid_market_volatility_fails_closed() -> None:
    from datetime import UTC, datetime
    from math import inf, nan
    from src.core.instruments import AssetKind, Instrument
    from src.core.portfolio import PortfolioSnapshot
    from src.strategy.portfolio import PortfolioConstructionStatus, PortfolioSecurityInput, construct_champion_portfolio
    from src.strategy.selection import ChampionSelectionResult

    now = datetime(2024, 1, 3, tzinfo=UTC)
    instrument = Instrument('KRX:000001', AssetKind.STOCK, 'KRX', '000001', 'KRW')
    selection = ChampionSelectionResult(now, 'account-7', (instrument.instrument_id,), (), 19, 'champion-v1-selection-v1')
    inputs = (PortfolioSecurityInput(instrument, 'Technology', 0.20, 100_000_000.0),)
    snapshot = PortfolioSnapshot('account-7', now, 1_000_000.0, 0.0, ())
    for volatility in (0.0, -0.01, nan, inf):
        result = construct_champion_portfolio(selection, inputs, snapshot, {instrument.instrument_id: 10_000.0}, volatility, decision_time=now)
        assert result.status is PortfolioConstructionStatus.NO_TRADE
        assert result.targets == ()


def test_construct_champion_portfolio_excluded_held_security_emits_exit_target() -> None:
    from datetime import UTC, datetime

    from src.core.instruments import AssetKind, Instrument
    from src.core.portfolio import PortfolioSnapshot, Position
    from src.strategy.portfolio import PortfolioSecurityInput, construct_champion_portfolio
    from src.strategy.selection import ChampionSelectionResult

    now = datetime(2024, 1, 3, tzinfo=UTC)
    held = Instrument("KRX:000001", AssetKind.STOCK, "KRX", "000001", "KRW")
    entrant = Instrument("KRX:000002", AssetKind.STOCK, "KRX", "000002", "KRW")
    selection = ChampionSelectionResult(now, "account-8", (held.instrument_id, entrant.instrument_id), (), 18, "champion-v1-selection-v1")
    inputs = (
        PortfolioSecurityInput(held, "Technology", 0.0, 100_000_000.0),
        PortfolioSecurityInput(entrant, "Industrials", 0.20, 100_000_000.0),
    )
    snapshot = PortfolioSnapshot("account-8", now, 1_000_000.0, 0.0, (Position(held, 10, 10_000.0),))
    result = construct_champion_portfolio(
        selection,
        inputs,
        snapshot,
        {held.instrument_id: 10_000.0, entrant.instrument_id: 10_000.0},
        0.10,
        decision_time=now,
    )

    held_target = next(target for target in result.targets if target.allocation.instrument.instrument_id == held.instrument_id)
    assert held_target.allocation.target_value == 0.0
