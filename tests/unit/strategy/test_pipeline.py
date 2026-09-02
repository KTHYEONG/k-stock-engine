def test_build_champion_portfolio_wires_construction_entrypoint() -> None:
    from datetime import UTC, datetime

    from src.core.instruments import AssetKind, Instrument
    from src.core.portfolio import PortfolioSnapshot
    from src.strategy.pipeline import build_champion_portfolio
    from src.strategy.portfolio import PortfolioConstructionStatus, PortfolioSecurityInput
    from src.strategy.selection import ChampionSelectionResult

    now = datetime(2024, 1, 3, tzinfo=UTC)
    instrument = Instrument("KRX:000001", AssetKind.STOCK, "KRX", "000001", "KRW")
    selection = ChampionSelectionResult(
        now,
        "account-pipeline",
        (instrument.instrument_id,),
        (),
        19,
        "champion-v1-selection-v1",
    )
    inputs = (PortfolioSecurityInput(instrument, "Technology", 0.20, 100_000_000.0),)

    result = build_champion_portfolio(
        selection,
        inputs,
        PortfolioSnapshot("account-pipeline", now, 1_000_000.0, 0.0, ()),
        {instrument.instrument_id: 10_000.0},
        0.15,
        decision_time=now,
    )

    assert result.status is PortfolioConstructionStatus.ALLOCATED
    assert result.account_snapshot_id == "account-pipeline"
    assert len(result.targets) == 1
