def test_portfolio_constraints_leaf_exists() -> None:
    from legacy.stocks.trading import portfolio_constraints
    assert hasattr(portfolio_constraints, "_apply_constraints")
