def test_portfolio_constructor_reexport_preserves_prepared_market_type() -> None:
    from legacy.stocks.trading import portfolio_allocation as allocation
    from legacy.stocks.trading import portfolio_constructor as constructor
    from legacy.stocks.trading import portfolio_market as market

    assert allocation.PreparedAllocationMarket is market.PreparedAllocationMarket
    assert constructor.PreparedAllocationMarket is market.PreparedAllocationMarket
