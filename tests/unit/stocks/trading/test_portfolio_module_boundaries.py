def test_portfolio_constructor_reexport_preserves_prepared_market_type() -> None:
    from src.stocks.trading import portfolio_allocation as allocation
    from src.stocks.trading import portfolio_constructor as constructor
    from src.stocks.trading import portfolio_market as market

    assert allocation.PreparedAllocationMarket is market.PreparedAllocationMarket
    assert constructor.PreparedAllocationMarket is market.PreparedAllocationMarket
