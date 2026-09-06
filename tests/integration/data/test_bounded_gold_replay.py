def test_bounded_replay_excludes_future_master_and_fact_corrections() -> None:
    from datetime import UTC, datetime, timedelta
    import polars as pl
    from src.core.time import SessionCalendar
    from src.data.replay import PITReplayReader
    from src.features.contracts import QvefFeaturePolicy
    from src.strategy.universe import UniversePolicy

    sessions = tuple(datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(70))
    decision = sessions[-1]
    master = pl.DataFrame([{'instrument_id': 'KRX:1', 'valid_from': sessions[0], 'valid_to': None, 'available_at': sessions[0]}, {'instrument_id': 'KRX:F', 'valid_from': sessions[0], 'valid_to': None, 'available_at': decision + timedelta(days=1)}])
    facts = pl.DataFrame([{'company_id': 'C1', 'available_at': decision, 'restatement_id': 'r0'}, {'company_id': 'C1', 'available_at': decision + timedelta(days=1), 'restatement_id': 'r1'}])
    reader = PITReplayReader.from_frames_for_test(calendar=SessionCalendar(sessions), security_master=master, daily_market=pl.DataFrame(), investor_flow=pl.DataFrame(), financial_facts=facts, corporate_actions=pl.DataFrame())

    replay = reader.session_input(session=decision, decision_time=decision, universe_policy=UniversePolicy(), qvef_policy=QvefFeaturePolicy())

    assert replay.security_master['instrument_id'].to_list() == ['KRX:1']
    assert replay.financial_facts['restatement_id'].to_list() == ['r0']
