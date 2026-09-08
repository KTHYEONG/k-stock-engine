"""ChampionStrategy decision tests (contract: champion_strategy_wiring)."""

from __future__ import annotations


def test_champion_strategy_initialization() -> None:
    from src.strategy.champion_strategy import ChampionStrategy
    from src.strategy.selection import ChampionSelectionPolicy
    from src.strategy.portfolio import ChampionPortfolioPolicy

    strategy = ChampionStrategy(
        scores_by_session={},
        selection_policy=ChampionSelectionPolicy(),
        portfolio_policy=ChampionPortfolioPolicy(),
        rebalance_frequency='monthly',
    )
    assert strategy is not None
    assert strategy.rebalance_frequency == 'monthly'


def test_champion_strategy_decide_skips_non_rebalance_session() -> None:
    from datetime import UTC, datetime
    from src.core.portfolio import PortfolioSnapshot
    from src.engine.decision import DecisionContext
    from src.strategy.champion_strategy import ChampionStrategy

    strategy = ChampionStrategy(scores_by_session={}, rebalance_frequency='monthly')
    strategy._last_rebalance_month = (2024, 1)

    ctx = DecisionContext(
        decision_time=datetime(2024, 1, 15, 15, 30, tzinfo=UTC),
        portfolio=PortfolioSnapshot('test-acc', datetime(2024, 1, 15, 15, 30, tzinfo=UTC), 100_000_000.0, 0.0, ()),
        market_snapshot={},
    )
    intents = strategy.decide(ctx)
    assert intents == ()


def _make_ranked_scores(d_time: object, instruments: object, sector: str) -> tuple[object, ...]:
    from src.strategy.scoring import ChampionScoreRow

    rows: list[object] = []
    for i, inst in enumerate(instruments):  # type: ignore[union-attr]
        rows.append(
            ChampionScoreRow(
                decision_session=d_time,  # type: ignore[arg-type]
                instrument_id=inst.instrument_id,  # type: ignore[union-attr]
                eligible=True,
                champion_score=float(20 - i),
                rank=i + 1,
                exclusion_reasons=(),
                feature_policy_version='champion-v1-qvef-v1',
                score_policy_version='champion-v1-scoring-v1',
            )
        )
    _ = sector
    return tuple(rows)


def test_champion_strategy_decide_rebalances_targets() -> None:
    from datetime import UTC, datetime
    from src.core.instruments import AssetKind, Instrument
    from src.core.portfolio import PortfolioSnapshot
    from src.engine.decision import DecisionContext
    from src.execution.domain.intents import TradeIntent
    from src.strategy.champion_strategy import ChampionStrategy

    d_time = datetime(2024, 1, 2, 15, 30, tzinfo=UTC)
    instruments = [Instrument(f'KRX:{i:06d}', AssetKind.STOCK, 'KRX', f'{i:06d}', 'KRW') for i in range(20)]
    scores = _make_ranked_scores(d_time, instruments, 'Technology')

    market_snapshot = {
        'mark_prices': {inst.instrument_id: 50_000.0 for inst in instruments},
        'volatilities': {inst.instrument_id: 0.15 for inst in instruments},
        'adtv20': {inst.instrument_id: 10_000_000_000.0 for inst in instruments},
        'sectors': {inst.instrument_id: 'Technology' for inst in instruments},
        'instruments': {inst.instrument_id: inst for inst in instruments},
        'market_volatility': 0.15,
    }

    strategy = ChampionStrategy(scores_by_session={d_time.date(): scores}, rebalance_frequency='monthly')  # type: ignore[dict-item]
    ctx = DecisionContext(
        decision_time=d_time,
        portfolio=PortfolioSnapshot('test-acc', d_time, 100_000_000.0, 0.0, ()),
        market_snapshot=market_snapshot,
    )
    intents = strategy.decide(ctx)
    assert len(intents) > 0
    assert all(isinstance(intent, TradeIntent) for intent in intents)
    assert all(intent.target_value > 0 for intent in intents)


def test_champion_strategy_decide_emits_exit_intents() -> None:
    from datetime import UTC, datetime
    from src.core.instruments import AssetKind, Instrument
    from src.core.portfolio import PortfolioSnapshot, Position
    from src.engine.decision import DecisionContext
    from src.strategy.champion_strategy import ChampionStrategy

    d_time = datetime(2024, 2, 1, 15, 30, tzinfo=UTC)
    old_inst = Instrument('KRX:999999', AssetKind.STOCK, 'KRX', '999999', 'KRW')
    new_inst = [Instrument(f'KRX:{i:06d}', AssetKind.STOCK, 'KRX', f'{i:06d}', 'KRW') for i in range(20)]
    scores = _make_ranked_scores(d_time, new_inst, 'Finance')

    market_snapshot = {
        'mark_prices': {inst.instrument_id: 10_000.0 for inst in [*new_inst, old_inst]},
        'volatilities': {inst.instrument_id: 0.15 for inst in [*new_inst, old_inst]},
        'adtv20': {inst.instrument_id: 10_000_000_000.0 for inst in [*new_inst, old_inst]},
        'sectors': {inst.instrument_id: 'Finance' for inst in [*new_inst, old_inst]},
        'instruments': {inst.instrument_id: inst for inst in [*new_inst, old_inst]},
        'market_volatility': 0.15,
    }

    strategy = ChampionStrategy(scores_by_session={d_time.date(): scores}, rebalance_frequency='monthly')  # type: ignore[dict-item]
    held_pos = Position(instrument=old_inst, quantity=100.0, average_cost=10_000.0)
    ctx = DecisionContext(
        decision_time=d_time,
        portfolio=PortfolioSnapshot('test-acc', d_time, 99_000_000.0, 0.0, (held_pos,)),
        market_snapshot=market_snapshot,
    )
    intents = strategy.decide(ctx)
    exit_intent = next((it for it in intents if it.instrument_id == old_inst.instrument_id), None)
    assert exit_intent is not None
    assert exit_intent.target_value == 0.0


def test_champion_strategy_receives_complete_market_snapshot_from_session(tmp_path) -> None:
    from datetime import UTC, datetime
    import polars as pl
    from src.core.portfolio import PortfolioSnapshot
    from src.core.time import SessionCalendar
    from src.data.backtest_sessions import build_backtest_sessions
    from src.data.schemas import SilverTable
    from src.data.snapshot import PITSnapshotRepository
    from src.engine.decision import DecisionContext
    from src.strategy.champion_strategy import ChampionStrategy
    from src.strategy.scoring import ChampionScoreRow

    d1 = datetime(2024, 1, 2, 9, 0, tzinfo=UTC)
    d2 = datetime(2024, 1, 3, 9, 0, tzinfo=UTC)
    frame = pl.DataFrame({
        'session': [d1, d2],
        'instrument_id': ['KRX:005930', 'KRX:005930'],
        'open': [70000.0, 71000.0],
        'high': [71000.0, 72000.0],
        'low': [69500.0, 70500.0],
        'close': [70500.0, 71500.0],
        'volume': [1000000.0, 1100000.0],
        'trading_value': [70500000000.0, 78650000000.0],
        'available_at': [
            d1.replace(hour=15, minute=30),
            d2.replace(hour=15, minute=30),
        ],
    })
    repo = PITSnapshotRepository.from_frames({SilverTable.DAILY_MARKET: frame}, root=tmp_path)
    calendar = SessionCalendar((d1, d2))
    sessions = build_backtest_sessions(
        snapshot_repository=repo,
        calendar=calendar,
        start=d1,
        end=d1,
        decision_time_of=lambda s: s.replace(hour=15, minute=30),
    )
    assert len(sessions) == 1
    session = sessions[0]
    scores = (
        ChampionScoreRow(
            decision_session=d1,
            instrument_id='KRX:005930',
            eligible=True,
            champion_score=0.95,
            rank=1,
            exclusion_reasons=(),
            feature_policy_version='champion-v1-qvef-v1',
            score_policy_version='champion-v1-scoring-v1',
        ),
    )
    strategy = ChampionStrategy(scores_by_session={d1.date(): scores})
    context = DecisionContext(
        decision_time=session.decision_time,
        portfolio=PortfolioSnapshot('test-acc', session.decision_time, 100_000_000.0, 0.0, ()),
        market_snapshot=session.market_snapshot,
    )
    intents = strategy.decide(context)
    assert len(intents) == 1
    assert intents[0].instrument_id == 'KRX:005930'
