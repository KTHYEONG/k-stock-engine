def test_build_historical_universe_preserves_delisted_historical_membership_without_lookahead() -> None:
    from datetime import UTC, datetime, timedelta

    import polars as pl

    from src.core.time import SessionCalendar
    from src.strategy.universe import ExclusionReason, build_historical_universe

    sessions = tuple(datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(253))
    calendar = SessionCalendar(sessions)
    delisting_session = sessions[-1]
    master = pl.DataFrame({
        'instrument_id': ['KRX:DELISTED', 'KRX:DELISTED'],
        'ticker': ['DELISTED', 'DELISTED'],
        'company_id': ['C1', 'C1'],
        'market': ['KOSPI', 'KOSPI'],
        'sector': ['Industrials', 'Industrials'],
        'listing_date': [sessions[0], sessions[0]],
        'delisting_date': [None, delisting_session],
        'share_class': ['common', 'common'],
        'status': ['listed', 'liquidation'],
        'valid_from': [sessions[0], delisting_session],
        'valid_to': [sessions[-2], None],
        'available_at': [sessions[0], delisting_session],
    })
    daily = pl.DataFrame({
        'session': list(sessions[-61:]),
        'instrument_id': ['KRX:DELISTED'] * 61,
        'trading_value': [3_000_000_000.0] * 61,
        'available_at': list(sessions[-61:]),
    })

    before = build_historical_universe(
        decision_session=sessions[-2], decision_time=sessions[-2], calendar=calendar,
        security_master=master, daily_market=daily,
    )
    after = build_historical_universe(
        decision_session=delisting_session, decision_time=delisting_session, calendar=calendar,
        security_master=master, daily_market=daily,
    )

    assert before[0].eligible is True
    assert after[0].eligible is False
    assert ExclusionReason.DELISTED in after[0].exclusion_reasons


def test_materialize_historical_universe_writes_immutable_reason_coded_gold_dataset(tmp_path) -> None:
    from datetime import UTC, datetime

    import json
    import pytest

    from src.core.datasets import DatasetCertification
    from src.strategy.universe import (
        ExclusionReason, UniverseDecision, UniversePolicy, materialize_historical_universe,
    )

    decision_time = datetime(2024, 1, 3, tzinfo=UTC)
    policy = UniversePolicy()
    decisions = (
        UniverseDecision(decision_time, 'KRX:ELIGIBLE', True, (), 252, 2_000_000_000.0),
        UniverseDecision(decision_time, 'KRX:EXCLUDED', False, (ExclusionReason.FINANCIAL_SECTOR,), 252, 3_000_000_000.0),
    )
    kwargs = dict(  # noqa: C408
        root=tmp_path, dataset_id='universe-v1', decision_time=decision_time, policy=policy,
        provider_version='fixture', calendar_hash='calendar', master_hash='master',
        quality_report_hash='quality', certification=DatasetCertification.RESEARCH,
    )

    path = materialize_historical_universe(decisions, **kwargs)
    manifest = json.loads((path / 'dataset_manifest.json').read_text(encoding='utf-8'))

    assert path.exists()
    assert manifest['feature_set'] == 'stock_historical_eligible_universe_v1'
    assert manifest['universe_policy_version'] == policy.version
    with pytest.raises(ValueError, match='already exists'):
        materialize_historical_universe(decisions, **kwargs)
