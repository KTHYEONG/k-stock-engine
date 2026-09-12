def test_capped_inverse_volatility_weights_are_deterministic_and_capped() -> None:
    import pytest
    from src.strategy.compounding_strategy import CompoundingStrategyPolicy, capped_inverse_volatility_weights

    ids = tuple(f"KRX:{i:06d}" for i in range(15))
    vols = {iid: 0.10 + index * 0.01 for index, iid in enumerate(ids)}
    weights = capped_inverse_volatility_weights(instrument_ids=tuple(reversed(ids)), volatilities=vols, policy=CompoundingStrategyPolicy())
    assert tuple(iid for iid, _ in weights) == ids
    assert sum(weight for _, weight in weights) == pytest.approx(1.0, abs=1e-12)
    assert max(weight for _, weight in weights) <= 0.075 + 1e-12
    with pytest.raises(ValueError, match="duplicate"):
        capped_inverse_volatility_weights(instrument_ids=(ids[0],) * 15, volatilities=vols, policy=CompoundingStrategyPolicy())
    bad = dict(vols)
    bad[ids[0]] = float("nan")
    with pytest.raises(ValueError, match="volatility"):
        capped_inverse_volatility_weights(instrument_ids=ids, volatilities=bad, policy=CompoundingStrategyPolicy())


def test_compounding_policy_constants_are_immutable() -> None:
    import pytest
    from src.strategy.compounding_strategy import CompoundingStrategyPolicy

    policy = CompoundingStrategyPolicy()
    assert (policy.max_positions, policy.selection_rebalance_sessions, policy.risk_rebalance_sessions) == (15, 40, 10)
    assert (policy.short_sma_sessions, policy.long_sma_sessions, policy.security_weight_cap) == (100, 200, 0.075)
    with pytest.raises(ValueError, match="immutable"):
        CompoundingStrategyPolicy(max_positions=20)
    with pytest.raises(ValueError, match="immutable"):
        CompoundingStrategyPolicy(risk_rebalance_sessions=11)


def test_compounding_strategy_separates_selection_and_risk_cadence() -> None:
    from datetime import datetime, timedelta
    from src.core.instruments import AssetKind, Instrument
    from src.core.portfolio import PortfolioSnapshot
    from src.core.time import KRX_TZ, SessionCalendar
    from src.engine.decision import DecisionContext
    from src.strategy.compounding_strategy import CompoundingStrategy

    sessions = tuple(datetime(2024, 1, 2, 9, tzinfo=KRX_TZ) + timedelta(days=i) for i in range(42))
    calendar = SessionCalendar(sessions)
    ids = tuple(f"KRX:{i:06d}" for i in range(16))
    eligible = {session.date(): ids for session in sessions}
    strategy = CompoundingStrategy(eligible_by_session=eligible, calendar=calendar)
    instruments = {iid: Instrument(iid, AssetKind.STOCK, "KRX", iid.split(":")[1], "KRW") for iid in ids}
    def context(index: int) -> DecisionContext:
        decision = sessions[index].replace(hour=15, minute=30)
        snapshot = {"mark_prices": dict.fromkeys(ids, 10000.0), "market_caps": {iid: float(1000 - pos) for pos, iid in enumerate(ids)}, "volatilities": {iid: 0.10 + pos / 1000 for pos, iid in enumerate(ids)}, "instruments": instruments, "market_index_level": 2.0, "market_index_sma100": 1.5, "market_index_sma200": 1.4}
        portfolio = PortfolioSnapshot("acct", decision, 1_000_000.0, 0.0, ())
        return DecisionContext(decision, portfolio, snapshot)
    first = strategy.decide(context(0))
    assert len(first) == 15
    assert all(intent.execution_time == sessions[1] for intent in first)
    assert all(intent.strategy_id == "compounding-v1" for intent in first)
    assert sum(intent.target_value for intent in first) <= 1_000_000.0 + 1e-6
    assert max(intent.target_value for intent in first) <= 75_000.0 + 1e-6
    assert strategy.decide(context(1)) == ()
    assert len(strategy.decide(context(10))) == 15
    assert len(strategy.decide(context(20))) == 15
    assert len(strategy.decide(context(30))) == 15
    changed = strategy.decide(context(40))
    assert len(changed) == 15
    assert {intent.instrument_id for intent in changed} == set(ids[:15])


def test_compounding_strategy_fails_closed_to_cash() -> None:
    from datetime import datetime, timedelta
    from src.core.instruments import AssetKind, Instrument
    from src.core.portfolio import PortfolioSnapshot, Position
    from src.core.time import KRX_TZ, SessionCalendar
    from src.engine.decision import DecisionContext
    from src.strategy.compounding_strategy import CompoundingStrategy

    sessions = tuple(datetime(2024, 1, 2, 9, tzinfo=KRX_TZ) + timedelta(days=i) for i in range(12))
    calendar = SessionCalendar(sessions)
    ids = tuple(f"KRX:{i:06d}" for i in range(15))
    instruments = {iid: Instrument(iid, AssetKind.STOCK, "KRX", iid.split(":")[1], "KRW") for iid in ids}
    strategy = CompoundingStrategy(eligible_by_session={s.date(): ids for s in sessions}, calendar=calendar)
    decision = sessions[0].replace(hour=15, minute=30)
    held = Position(instruments[ids[0]], 10.0, 9000.0)
    portfolio = PortfolioSnapshot("acct", decision, 900_000.0, 0.0, (held,))
    snapshot = {"mark_prices": dict.fromkeys(ids, 10000.0), "market_caps": dict.fromkeys(ids, 1000000.0), "volatilities": dict.fromkeys(ids, 0.2), "instruments": instruments, "market_index_level": 1.0, "market_index_sma100": None, "market_index_sma200": None}
    intents = strategy.decide(DecisionContext(decision, portfolio, snapshot))
    assert len(intents) == 1
    assert intents[0].instrument_id == ids[0]
    assert intents[0].target_value == 0.0
    assert intents[0].execution_time == sessions[1]
