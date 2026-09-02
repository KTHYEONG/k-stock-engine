def test_build_qvef_features_uses_ttm_and_ignores_future_correction() -> None:
    from datetime import UTC, datetime, timedelta

    import polars as pl

    from src.core.time import SessionCalendar
    from src.features.qvef import build_qvef_features
    from src.strategy.universe import UniverseDecision

    sessions = tuple(datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(70))
    decision = sessions[-1]
    identifiers = [f'KRX:{i:06d}' for i in range(10)]
    universe = tuple(UniverseDecision(decision, instrument_id, True, (), 252, 2_000_000_000.0) for instrument_id in identifiers)
    master = pl.DataFrame({'instrument_id': identifiers, 'company_id': identifiers, 'sector': ['Technology'] * 10, 'valid_from': [sessions[0]] * 10, 'valid_to': [None] * 10, 'available_at': [sessions[0]] * 10})
    market = pl.DataFrame([{'session': session, 'instrument_id': instrument_id, 'trading_value': 100.0 + index, 'market_cap': 200.0 + index, 'available_at': session} for index, instrument_id in enumerate(identifiers) for session in sessions[-20:]])
    flow = pl.DataFrame([{'session': session, 'instrument_id': instrument_id, 'foreign_net_value': float(index + 1), 'available_at': session} for index, instrument_id in enumerate(identifiers) for session in sessions[-20:]])
    facts = []
    for index, company_id in enumerate(identifiers):
        for quarter in range(1, 5):
            for fact, value in {'gross_profit': 10.0 + index, 'net_income': 5.0 + index, 'operating_cash_flow': 4.0 + index, 'assets': 100.0, 'equity': 50.0, 'operating_profit': 20.0 + index, 'sales': 120.0 + index}.items():
                facts.append({'company_id': company_id, 'fiscal_period': f'2024Q{quarter}', 'filing_id': f'{company_id}-24-{quarter}', 'fact': fact, 'consolidated': True, 'value': value, 'unit': 'KRW', 'restatement_id': 'r0', 'available_at': decision})
        for fact, value in {'assets': 100.0, 'operating_profit': 10.0 + index, 'sales': 100.0 + index}.items():
            facts.append({'company_id': company_id, 'fiscal_period': '2023Q4', 'filing_id': f'{company_id}-23-4', 'fact': fact, 'consolidated': True, 'value': value, 'unit': 'KRW', 'restatement_id': 'r0', 'available_at': sessions[0]})
    facts.append({'company_id': identifiers[0], 'fiscal_period': '2024Q4', 'filing_id': 'future-correction', 'fact': 'net_income', 'consolidated': True, 'value': 999.0, 'unit': 'KRW', 'restatement_id': 'r1', 'available_at': decision + timedelta(days=1)})

    rows = build_qvef_features(decision_session=decision, decision_time=decision, calendar=SessionCalendar(sessions), universe=universe, security_master=master, daily_market=market, investor_flow=flow, financial_facts=pl.DataFrame(facts))

    first = next(row for row in rows if row.instrument_id == identifiers[0])
    assert first.gross_profitability == 0.4
    assert first.roe == 0.4
    assert first.cfo_to_assets == 0.16
    assert first.earnings_to_price == 0.1
    assert first.operating_income_change == 0.1
    assert first.sales_growth == 0.2
    assert round(first.operating_margin_change, 12) == round(20.0 / 120.0 - 10.0 / 100.0, 12)
    assert first.foreign_flow_5 == 5.0 / 100.0
    assert first.foreign_flow_20 == 20.0 / 100.0
    assert first.quality_score == -1.0
    assert first.source_available_at[0][1] <= decision


def test_build_qvef_features_neutralizes_negative_earnings_and_rejects_incomplete_flow() -> None:
    from datetime import UTC, datetime, timedelta

    import polars as pl

    from src.core.time import SessionCalendar
    from src.features.qvef import build_qvef_features
    from src.strategy.universe import UniverseDecision

    sessions = tuple(datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(70))
    decision = sessions[-1]
    identifiers = [f'KRX:{i:06d}' for i in range(10)]
    universe = tuple(UniverseDecision(decision, instrument_id, True, (), 252, 2_000_000_000.0) for instrument_id in identifiers)
    master = pl.DataFrame({'instrument_id': identifiers, 'company_id': identifiers, 'sector': ['Technology'] * 10, 'valid_from': [sessions[0]] * 10, 'valid_to': [None] * 10, 'available_at': [sessions[0]] * 10})
    market = pl.DataFrame([{'session': session, 'instrument_id': instrument_id, 'trading_value': 100.0, 'market_cap': 200.0, 'available_at': session} for instrument_id in identifiers for session in sessions[-20:]])
    flow = pl.DataFrame([{'session': session, 'instrument_id': instrument_id, 'foreign_net_value': 1.0, 'available_at': session} for instrument_id in identifiers for session in sessions[-20:] if not (instrument_id == identifiers[1] and session == sessions[-2])])
    facts = []
    for index, company_id in enumerate(identifiers):
        for quarter in range(1, 5):
            values = {'gross_profit': 10.0 + index, 'net_income': -1.0 if index == 0 else 5.0 + index, 'operating_cash_flow': 4.0 + index, 'assets': 100.0, 'equity': 50.0, 'operating_profit': 20.0 + index, 'sales': 120.0 + index}
            facts.extend({'company_id': company_id, 'fiscal_period': f'2024Q{quarter}', 'filing_id': f'{company_id}-{quarter}', 'fact': fact, 'consolidated': True, 'value': value, 'unit': 'KRW', 'restatement_id': 'r0', 'available_at': decision} for fact, value in values.items())
        facts.extend({'company_id': company_id, 'fiscal_period': '2023Q4', 'filing_id': f'{company_id}-old', 'fact': fact, 'consolidated': True, 'value': value, 'unit': 'KRW', 'restatement_id': 'r0', 'available_at': sessions[0]} for fact, value in {'assets': 100.0, 'operating_profit': 10.0 + index, 'sales': 100.0 + index}.items())

    rows = build_qvef_features(decision_session=decision, decision_time=decision, calendar=SessionCalendar(sessions), universe=universe, security_master=master, daily_market=market, investor_flow=flow, financial_facts=pl.DataFrame(facts))

    negative_earnings = next(row for row in rows if row.instrument_id == identifiers[0])
    incomplete_flow = next(row for row in rows if row.instrument_id == identifiers[1])
    assert negative_earnings.value_score is not None
    assert 'earnings_to_price_neutral' in negative_earnings.component_presence
    assert incomplete_flow.foreign_flow_score is None
    assert 'foreign_flow_incomplete' in incomplete_flow.component_presence
