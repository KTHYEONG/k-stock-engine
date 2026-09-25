import pytest


def test_normalize_financial_fact_uses_next_krx_session_when_intraday_unknown() -> None:
    from datetime import datetime

    from src.core.time import KRX_TZ, SessionCalendar
    from src.data.silver import next_krx_session_open

    published = datetime(2024, 1, 2, 18, 0, tzinfo=KRX_TZ)
    calendar = SessionCalendar((datetime(2024, 1, 2, 9, 0, tzinfo=KRX_TZ), datetime(2024, 1, 3, 9, 0, tzinfo=KRX_TZ)))

    assert next_krx_session_open(published, calendar) == datetime(2024, 1, 3, 9, 0, tzinfo=KRX_TZ)


def canonical_page(*, filing_id: str, fiscal_period: str, published_at, value: float) -> dict:
    return {
        "company_id": "001",
        "fiscal_period": fiscal_period,
        "filing_id": filing_id,
        "fact": "sales",
        "published_at": published_at,
        "available_at": published_at,
        "value": value,
        "unit": "KRW",
        "consolidated": True,
        "restatement_id": "r0",
        "source_kind": "opendart_standard",
        "mapping_version": "v1",
        "raw_document_hash": "d" * 64,
    }


def test_normalize_dart_facts_preserves_all_filings_and_pit_excludes_later_correction() -> None:
    from datetime import UTC, datetime
    from src.core.time import SessionCalendar
    from src.data.normalization import normalize_dart_financial_facts

    cutoff = datetime(2016, 2, 1, 9, tzinfo=UTC)
    pages = (canonical_page(filing_id="Q1", fiscal_period="2015Q1", published_at=datetime(2015, 5, 15, tzinfo=UTC), value=10.0), canonical_page(filing_id="Q2", fiscal_period="2015Q2", published_at=datetime(2015, 8, 17, tzinfo=UTC), value=20.0), canonical_page(filing_id="CORR", fiscal_period="2015Q1", published_at=datetime(2016, 3, 30, tzinfo=UTC), value=99.0))
    calendar = SessionCalendar((datetime(2015, 5, 18, 9, tzinfo=UTC), datetime(2015, 8, 18, 9, tzinfo=UTC), datetime(2016, 3, 31, 9, tzinfo=UTC)))
    frame = normalize_dart_financial_facts(pages=pages, disclosure_rows=(), source_hash="c" * 64, calendar=calendar, decision_time=cutoff)
    assert frame.height == 2
    assert set(frame["filing_id"].to_list()) == {"Q1", "Q2"}
    assert set(frame["source_kind"].to_list()) == {"opendart_standard"}


def test_normalize_dart_facts_uses_frozen_ticker_and_rejects_unmapped_corp_code() -> None:
    from datetime import UTC, datetime
    from src.core.time import SessionCalendar
    from src.data.normalization import normalize_dart_financial_facts

    pages = [{"records": [{"corp_code": "00126380", "ticker": "005930", "fiscal_period": "2015Q3", "filing_id": "F1", "fact": "sales", "published_at": datetime(2015, 11, 16, 9, tzinfo=UTC), "value": 1.0, "unit": "KRW", "consolidated": True}, {"corp_code": "00999999", "fiscal_period": "2015Q3", "filing_id": "F2", "fact": "sales", "published_at": datetime(2015, 11, 16, 9, tzinfo=UTC), "value": 2.0, "unit": "KRW", "consolidated": True}]}]
    frame = normalize_dart_financial_facts(pages=pages, disclosure_rows=(), source_hash="b" * 64, calendar=SessionCalendar((datetime(2015, 11, 17, 9, tzinfo=UTC),)), decision_time=datetime(2016, 1, 4, 9, tzinfo=UTC))

    assert frame.select(["company_id", "dart_corp_code", "ticker"]).to_dicts() == [{"company_id": "005930", "dart_corp_code": "00126380", "ticker": "005930"}]


def test_normalize_dart_facts_inherits_frozen_page_ticker_bridge() -> None:
    from datetime import UTC, datetime
    from src.core.time import SessionCalendar
    from src.data.normalization import normalize_dart_financial_facts

    page = {
        "corp_code": "00126380",
        "ticker": "005930",
        "records": [{
            "company_id": "00126380", "fiscal_period": "2015Q3", "filing_id": "F1",
            "fact": "sales", "published_at": datetime(2015, 11, 16, 9, tzinfo=UTC),
            "value": 1.0, "unit": "KRW", "consolidated": True,
        }],
    }
    frame = normalize_dart_financial_facts(
        pages=[page], disclosure_rows=(), source_hash="d" * 64,
        calendar=SessionCalendar((datetime(2015, 11, 17, 9, tzinfo=UTC),)),
        decision_time=datetime(2016, 1, 4, 9, tzinfo=UTC),
    )
    assert frame.select(["company_id", "dart_corp_code", "ticker"]).to_dicts() == [
        {"company_id": "005930", "dart_corp_code": "00126380", "ticker": "005930"}
    ]


def test_normalize_dart_facts_maps_corp_code_only_record_with_frozen_bridge() -> None:
    from datetime import UTC, datetime

    from src.core.time import SessionCalendar
    from src.data.normalization import normalize_dart_financial_facts

    frame = normalize_dart_financial_facts(
        pages=[{'records': [{
            'company_id': '00126380',
            'corp_code': '00126380',
            'fiscal_period': '2015Q3',
            'filing_id': 'F1',
            'fact': 'sales',
            'published_at': datetime(2015, 11, 16, tzinfo=UTC),
            'value': 1.0,
            'unit': 'KRW',
        }]}],
        disclosure_rows=(),
        source_hash='b' * 64,
        calendar=SessionCalendar((datetime(2015, 11, 17, tzinfo=UTC),)),
        decision_time=datetime(2016, 1, 4, tzinfo=UTC),
        ticker_by_corp_code={'00126380': '005930'},
        bridge_receipt_hash='c' * 64,
    )

    assert frame.select(['company_id', 'ticker', 'dart_corp_code']).to_dicts() == [
        {'company_id': '005930', 'ticker': '005930', 'dart_corp_code': '00126380'}
    ]
    assert frame.item(0, 'mapping_version').endswith('bridge:' + ('c' * 64))


def test_normalize_corporate_action_records_preserves_settlement_and_rejects_legacy_bonus() -> None:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import pytest

    from src.data.normalization import normalize_corporate_action_records
    from src.data.schemas import PITDataError

    tz = ZoneInfo('Asia/Seoul')
    price_session = datetime(2016, 11, 23, 9, tzinfo=tz)
    listing_session = datetime(2016, 12, 14, 9, tzinfo=tz)
    record = {'instrument_id': 'KRX:027410', 'effective_session': price_session, 'share_listing_date': listing_session, 'share_delta': 24773661, 'action_id': '20161107000214', 'type': 'bonus_issue', 'factor': 2.0, 'cash_amount': 0.0, 'source': 'opendart_structured_decisions', 'available_at': datetime(2016, 11, 8, 9, tzinfo=tz), 'evidence_status': 'verified', 'evidence_reason': None}

    frame = normalize_corporate_action_records(action_records=[record], calendar_sessions=(price_session, listing_session), corporate_action_available_at=price_session, corporate_action_source_hash='a' * 64)

    assert frame.select(['effective_date', 'share_listing_date', 'share_delta']).to_dicts() == [{'effective_date': price_session, 'share_listing_date': listing_session, 'share_delta': 24773661}]
    legacy = dict(record)
    del legacy['share_delta']
    with pytest.raises(PITDataError, match='requires rebuild from raw OpenDART Bronze'):
        normalize_corporate_action_records(action_records=[legacy], calendar_sessions=(price_session, listing_session), corporate_action_available_at=price_session, corporate_action_source_hash='a' * 64)


def test_normalize_stock_evidence_wires_bonus_settlement_normalizer(monkeypatch, tmp_path) -> None:
    from datetime import UTC, datetime

    import polars as pl

    import src.data.normalization as module
    import src.data.silver as silver
    from src.data.schemas import BronzeReceipt, EvidenceKind, SilverTable

    session = datetime(2024, 1, 2, 9, tzinfo=UTC)
    receipts = {}
    for kind in EvidenceKind:
        payload_path = tmp_path / f'{kind.value}.json'
        metadata_path = tmp_path / f'{kind.value}.receipt.json'
        payload_path.write_text('{}', encoding='utf-8')
        metadata_path.write_text('{}', encoding='utf-8')
        receipts[kind] = BronzeReceipt(kind=kind, content_hash=kind.value.ljust(64, '0'), source_path=str(payload_path), retrieved_at=session, ingested_at=session, payload_path=payload_path, metadata_path=metadata_path)
    payloads = {
        EvidenceKind.CALENDAR: {'sessions': [session]},
        EvidenceKind.SECURITY_MASTER: {'records': [{'ticker': '027410', 'market': 'KOSPI', 'is_common_stock': True}]},
        EvidenceKind.DAILY_MARKET: {'records': [{'session': session, 'ticker': '027410', 'open': 100.0, 'high': 100.0, 'low': 100.0, 'close': 100.0, 'volume': 1.0, 'trading_value': 100.0, 'market_cap': 1000.0, 'shares_outstanding': 10.0}]},
        EvidenceKind.INVESTOR_FLOW: {'records': [{'session': session, 'ticker': '027410', '_source_provider': 'LS', 'foreign_buy_value': 1.0, 'foreign_sell_value': 0.0, 'foreign_net_value': 1.0, 'institution_net_value': 0.0, 'retail_net_value': -1.0}]},
        EvidenceKind.DISCLOSURES: {'records': [{'company_id': '027410', 'filing_id': 'f1', 'filing_type': 'annual', 'published_at': session}]},
        EvidenceKind.FINANCIAL_FACTS: {'records': [{'ignored': True}]},
        EvidenceKind.CORPORATE_ACTIONS: {'records': []},
        EvidenceKind.HISTORICAL_COSTS: {'commission': 0.00015},
    }
    monkeypatch.setattr(module, '_load_payload', lambda receipt, kind: payloads[kind])
    monkeypatch.setattr(module, 'normalize_dart_financial_facts', lambda **_kwargs: pl.DataFrame({'company_id': ['027410'], 'fiscal_period': ['2024Q1'], 'filing_id': ['f1'], 'fact': ['sales'], 'published_at': [session], 'available_at': [session], 'value': [1.0], 'unit': ['KRW'], 'consolidated': [True], 'restatement_id': ['r0'], 'source_hash': ['f' * 64], 'source_kind': ['fixture'], 'mapping_version': ['fixture'], 'raw_document_hash': [None]}))
    monkeypatch.setattr(silver, 'certify_silver', lambda **_kwargs: object())
    action = {'instrument_id': 'KRX:027410', 'effective_session': session, 'share_listing_date': session, 'share_delta': 10, 'action_id': 'a1', 'type': 'bonus_issue', 'factor': 2.0, 'cash_amount': 0.0, 'source': 'opendart', 'available_at': session, 'evidence_status': 'verified', 'evidence_reason': None}

    tables, _report = module.normalize_stock_evidence(receipts, decision_time=session, streamed_corporate_actions=[action])

    assert tables[SilverTable.CORPORATE_ACTIONS].select(['share_listing_date', 'share_delta']).to_dicts() == [{'share_listing_date': session, 'share_delta': 10}]


def test_normalize_corporate_action_records_requires_evidence_status_migration() -> None:
    from datetime import datetime
    from zoneinfo import ZoneInfo
    import pytest
    from src.data.normalization import normalize_corporate_action_records
    from src.data.schemas import PITDataError

    session = datetime(2024, 1, 2, 9, tzinfo=ZoneInfo('Asia/Seoul'))
    with pytest.raises(PITDataError, match=r'evidence status.*rebuild'):
        normalize_corporate_action_records(action_records=[{'instrument_id': 'KRX:A', 'action_id': 'legacy', 'type': 'no_action'}], calendar_sessions=(session,), corporate_action_available_at=session, corporate_action_source_hash='h')
    frame = normalize_corporate_action_records(action_records=[{'instrument_id': 'KRX:A', 'action_id': 'u1', 'type': 'unresolved', 'effective_session': session, 'factor': 1.0, 'cash_amount': 0.0, 'available_at': session, 'evidence_status': 'unresolved', 'evidence_reason': 'unsupported_merger'}], calendar_sessions=(session,), corporate_action_available_at=session, corporate_action_source_hash='h')
    assert frame.row(0, named=True)['evidence_reason'] == 'unsupported_merger'


def test_verified_flow_selection_rejects_kis_only_and_conflicting_overlap() -> None:
    import pytest
    from src.data.normalization import select_verified_investor_flow_records
    from src.data.schemas import PITDataError

    base = {'session': '2016-12-29', 'ticker': '000020', 'foreign_buy_value': 100.0, 'foreign_sell_value': 40.0, 'foreign_net_value': 60.0, 'institution_net_value': -20.0, 'retail_net_value': -40.0}
    with pytest.raises(PITDataError, match='unsupported investor-flow provider'):
        select_verified_investor_flow_records([{**base, '_source_provider': 'KIS'}])
    assert select_verified_investor_flow_records([{**base, '_source_provider': 'KIWOOM'}, {**base, '_source_provider': 'LS'}]) == [{**base, '_source_provider': 'LS'}]
    with pytest.raises(PITDataError, match='conflicting investor_flow primary key'):
        select_verified_investor_flow_records([{**base, '_source_provider': 'LS'}, {**base, 'foreign_net_value': 61.0, '_source_provider': 'KIWOOM'}])


def test_verified_flow_selection_edge_cases() -> None:
    import pytest
    from src.data.normalization import _raw_isin, select_verified_investor_flow_records
    from src.data.schemas import PITDataError

    assert _raw_isin({"ISU_CD": "KR1234567890"}) == "KR1234567890"
    assert _raw_isin({"ISU_CD": "KYG210AT1036"}) == "KYG210AT1036"
    assert _raw_isin({"ISU_CD": "nope"}) is None

    base = {
        "session": "2016-12-29",
        "ticker": "000020",
        "foreign_buy_value": 100.0,
        "foreign_sell_value": 40.0,
        "foreign_net_value": 60.0,
        "institution_net_value": -20.0,
        "retail_net_value": -40.0,
    }
    only_kiwoom = select_verified_investor_flow_records(
        [{**base, "_source_provider": "KIWOOM"}]
    )
    assert only_kiwoom == [{**base, "_source_provider": "KIWOOM"}]

    with pytest.raises(PITDataError, match="investor-flow provider"):
        select_verified_investor_flow_records(["not-a-mapping"])
    with pytest.raises(PITDataError, match="session/ticker"):
        select_verified_investor_flow_records(
            [{**base, "session": "", "_source_provider": "LS"}]
        )
    incomplete = dict(base)
    del incomplete["foreign_net_value"]
    with pytest.raises(PITDataError, match="conflicting investor_flow primary key"):
        select_verified_investor_flow_records(
            [{**incomplete, "_source_provider": "LS"}]
        )


def test_verified_flow_selection_ignores_kis_when_accepted_provider_covers_key() -> None:
    from src.data.normalization import select_verified_investor_flow_records

    base = {
        "session": "2016-12-29",
        "ticker": "000020",
        "foreign_buy_value": 100.0,
        "foreign_sell_value": 40.0,
        "foreign_net_value": 60.0,
        "institution_net_value": -20.0,
        "retail_net_value": -40.0,
    }
    selected = select_verified_investor_flow_records(
        [{**base, "_source_provider": "KIS"}, {**base, "_source_provider": "LS"}]
    )
    assert selected == [{**base, "_source_provider": "LS"}]


def test_normalize_stock_evidence_covers_calendar_branch(tmp_path, monkeypatch) -> None:
    from datetime import UTC, datetime

    import polars as pl

    import src.data.normalization as module
    import src.data.silver as silver
    from src.core.time import SessionCalendar
    from src.data.schemas import BronzeReceipt, EvidenceKind, SilverTable

    session = datetime(2024, 1, 2, 9, tzinfo=UTC)
    receipts = {}
    for kind in EvidenceKind:
        payload_path = tmp_path / f"cal-{kind.value}.json"
        metadata_path = tmp_path / f"cal-{kind.value}.receipt.json"
        payload_path.write_text("{}", encoding="utf-8")
        metadata_path.write_text("{}", encoding="utf-8")
        receipts[kind] = BronzeReceipt(
            kind=kind,
            content_hash=kind.value.ljust(64, "0"),
            source_path=str(payload_path),
            retrieved_at=session,
            ingested_at=session,
            payload_path=payload_path,
            metadata_path=metadata_path,
        )
    payloads = {
        EvidenceKind.CALENDAR: {"sessions": [session]},
        EvidenceKind.SECURITY_MASTER: {
            "records": [{"ticker": "027410", "market": "KOSPI", "is_common_stock": True}]
        },
        EvidenceKind.DAILY_MARKET: {
            "records": [
                {
                    "session": session,
                    "ticker": "027410",
                    "open": 100.0,
                    "high": 100.0,
                    "low": 100.0,
                    "close": 100.0,
                    "volume": 1.0,
                    "trading_value": 100.0,
                    "market_cap": 1000.0,
                    "shares_outstanding": 10.0,
                }
            ]
        },
        EvidenceKind.INVESTOR_FLOW: {
            "records": [
                {
                    "session": session,
                    "ticker": "027410",
                    "_source_provider": "LS",
                    "foreign_buy_value": 1.0,
                    "foreign_sell_value": 0.0,
                    "foreign_net_value": 1.0,
                    "institution_net_value": 0.0,
                    "retail_net_value": -1.0,
                }
            ]
        },
        EvidenceKind.DISCLOSURES: {
            "records": [
                {
                    "company_id": "027410",
                    "filing_id": "f1",
                    "filing_type": "annual",
                    "published_at": session,
                }
            ]
        },
        EvidenceKind.FINANCIAL_FACTS: {"records": [{"ignored": True}]},
        EvidenceKind.CORPORATE_ACTIONS: {"records": []},
        EvidenceKind.HISTORICAL_COSTS: {"commission": 0.00015},
    }
    monkeypatch.setattr(
        module, "_load_payload", lambda receipt, kind: payloads[kind]
    )
    monkeypatch.setattr(
        module,
        "normalize_dart_financial_facts",
        lambda **_kwargs: pl.DataFrame(
            {
                "company_id": ["027410"],
                "fiscal_period": ["2024Q1"],
                "filing_id": ["f1"],
                "fact": ["sales"],
                "published_at": [session],
                "available_at": [session],
                "value": [1.0],
                "unit": ["KRW"],
                "consolidated": [True],
                "restatement_id": ["r0"],
                "source_hash": ["f" * 64],
                "source_kind": ["fixture"],
                "mapping_version": ["fixture"],
                "raw_document_hash": [None],
            }
        ),
    )
    monkeypatch.setattr(silver, "certify_silver", lambda **_kwargs: object())
    calendar = SessionCalendar((session,))
    action = {
        "instrument_id": "KRX:027410",
        "effective_session": session,
        "share_listing_date": session,
        "share_delta": 10,
        "action_id": "a1",
        "type": "bonus_issue",
        "factor": 2.0,
        "cash_amount": 0.0,
        "source": "opendart",
        "available_at": session,
        "evidence_status": "verified",
        "evidence_reason": None,
    }
    tables, _report = module.normalize_stock_evidence(
        receipts,
        calendar=calendar,
        decision_time=session,
        streamed_corporate_actions=[action],
    )
    assert tables[SilverTable.INVESTOR_FLOW].height >= 0


def test_verified_flow_selection_rejects_missing_provider() -> None:
    import pytest

    from src.data.normalization import select_verified_investor_flow_records
    from src.data.schemas import PITDataError

    base = {
        "session": "2016-12-29",
        "ticker": "000020",
        "foreign_buy_value": 100.0,
        "foreign_sell_value": 40.0,
        "foreign_net_value": 60.0,
        "institution_net_value": -20.0,
        "retail_net_value": -40.0,
    }
    untagged = dict(base)
    assert "_source_provider" not in untagged
    with pytest.raises(PITDataError, match="unsupported investor-flow provider"):
        select_verified_investor_flow_records([untagged])
    with pytest.raises(PITDataError, match="unsupported investor-flow provider"):
        select_verified_investor_flow_records([{**base, "_source_provider": "  "}])


def test_normalize_dart_facts_rejects_corp_code_only_company_id_without_bridge() -> None:
    from datetime import UTC, datetime
    from src.core.time import SessionCalendar
    from src.data.normalization import normalize_dart_financial_facts

    frame = normalize_dart_financial_facts(pages=[{'company_id': '00126380', 'fiscal_period': '2015Q3', 'filing_id': 'F1', 'fact': 'sales', 'published_at': datetime(2015, 11, 16, tzinfo=UTC), 'value': 1.0, 'unit': 'KRW', 'consolidated': True}], disclosure_rows=(), source_hash='a' * 64, calendar=SessionCalendar((datetime(2015, 11, 17, tzinfo=UTC),)), decision_time=datetime(2016, 1, 4, tzinfo=UTC))

    assert frame.is_empty()


def test_normalize_dart_facts_maps_corp_code_only_company_id_with_bridge() -> None:
    from datetime import UTC, datetime
    from src.core.time import SessionCalendar
    from src.data.normalization import normalize_dart_financial_facts

    frame = normalize_dart_financial_facts(pages=[{'company_id': '00126380', 'fiscal_period': '2015Q3', 'filing_id': 'F1', 'fact': 'sales', 'published_at': datetime(2015, 11, 16, tzinfo=UTC), 'value': 1.0, 'unit': 'KRW', 'consolidated': True}], disclosure_rows=(), source_hash='a' * 64, calendar=SessionCalendar((datetime(2015, 11, 17, tzinfo=UTC),)), decision_time=datetime(2016, 1, 4, tzinfo=UTC), ticker_by_corp_code={'00126380': '005930'}, bridge_receipt_hash='b' * 64)

    assert frame.select(['company_id', 'ticker', 'dart_corp_code']).to_dicts() == [{'company_id': '005930', 'ticker': '005930', 'dart_corp_code': '00126380'}]
    assert frame.item(0, 'mapping_version').endswith('bridge:' + ('b' * 64))


def test_normalize_financial_facts_keeps_consolidated_and_separate_rows() -> None:
    """One filing reporting both 연결 and 별도 must survive as two distinct rows."""
    from datetime import UTC, datetime

    from src.core.time import SessionCalendar
    from src.data.normalization import normalize_dart_financial_facts

    # Given: the same (company, period, filing, fact) reported on both bases.
    base = {
        'ticker': '005930',
        'corp_code': '00126380',
        'fiscal_period': '2015Q3',
        'filing_id': 'F1',
        'fact': 'sales',
        'published_at': datetime(2015, 11, 16, tzinfo=UTC),
        'unit': 'KRW',
    }
    pages = [
        {**base, 'value': 100.0, 'consolidated': True},
        {**base, 'value': 70.0, 'consolidated': False},
    ]

    # When
    frame = normalize_dart_financial_facts(
        pages=pages,
        disclosure_rows=(),
        source_hash='a' * 64,
        calendar=SessionCalendar((datetime(2015, 11, 17, tzinfo=UTC),)),
        decision_time=datetime(2016, 1, 4, tzinfo=UTC),
    )

    # Then: the accounting basis participates in the dedup key.
    assert frame.height == 2
    assert sorted(frame['consolidated'].to_list()) == [False, True]
    by_basis = dict(zip(frame['consolidated'].to_list(), frame['value'].to_list(), strict=True))
    assert by_basis[True] == 100.0
    assert by_basis[False] == 70.0

    # And: a genuine duplicate on the same basis is still collapsed.
    deduped = normalize_dart_financial_facts(
        pages=[pages[0], dict(pages[0])],
        disclosure_rows=(),
        source_hash='a' * 64,
        calendar=SessionCalendar((datetime(2015, 11, 17, tzinfo=UTC),)),
        decision_time=datetime(2016, 1, 4, tzinfo=UTC),
    )
    assert deduped.height == 1


def _kst_open(day: str):
    from datetime import datetime, date
    from src.core.time import KRX_TZ

    parsed = date.fromisoformat(day)
    return datetime(parsed.year, parsed.month, parsed.day, 9, 0, tzinfo=KRX_TZ)


def _fact_record(*, filing_id: str, published_at, extra: dict | None = None) -> dict:
    record = {
        'ticker': '005930',
        'corp_code': '00126380',
        'fiscal_period': '2024Q3',
        'filing_id': filing_id,
        'fact': 'sales',
        'published_at': published_at,
        'value': 10.0,
        'unit': 'KRW',
        'consolidated': True,
    }
    if extra:
        record.update(extra)
    return record


def test_normalize_dart_facts_available_at_next_session_open() -> None:
    from datetime import UTC, datetime

    from src.core.time import SessionCalendar
    from src.data.normalization import normalize_dart_financial_facts

    frame = normalize_dart_financial_facts(
        pages=[_fact_record(filing_id='F-W13', published_at=datetime(2024, 11, 13, 0, 0, tzinfo=UTC))],
        disclosure_rows=(),
        source_hash='a' * 64,
        calendar=SessionCalendar((_kst_open('2024-11-13'), _kst_open('2024-11-14'))),
        decision_time=datetime(2024, 11, 20, tzinfo=UTC),
    )

    assert frame.height == 1
    assert frame.item(0, 'published_at') == datetime(2024, 11, 12, 15, 0, tzinfo=UTC)
    assert frame.item(0, 'available_at') == datetime(2024, 11, 14, 0, 0, tzinfo=UTC)


def test_normalize_dart_facts_friday_filing_available_monday() -> None:
    from datetime import UTC, datetime

    from src.core.time import SessionCalendar
    from src.data.normalization import normalize_dart_financial_facts

    frame = normalize_dart_financial_facts(
        pages=[_fact_record(filing_id='F-FRI', published_at=datetime(2024, 11, 15, 0, 0, tzinfo=UTC))],
        disclosure_rows=(),
        source_hash='a' * 64,
        calendar=SessionCalendar((_kst_open('2024-11-15'), _kst_open('2024-11-18'))),
        decision_time=datetime(2024, 11, 20, tzinfo=UTC),
    )

    assert frame.height == 1
    assert frame.item(0, 'available_at') == datetime(2024, 11, 18, 0, 0, tzinfo=UTC)


def test_normalize_dart_facts_restated_record_uses_own_receipt_date() -> None:
    from datetime import UTC, datetime

    from src.core.time import SessionCalendar
    from src.data.normalization import normalize_dart_financial_facts

    frame = normalize_dart_financial_facts(
        pages=[_fact_record(
            filing_id='20260701000123',
            published_at=datetime(2024, 11, 13, 0, 0, tzinfo=UTC),
            extra={'rcept_no': '20260701000123'},
        )],
        disclosure_rows=(),
        source_hash='a' * 64,
        calendar=SessionCalendar((_kst_open('2026-07-01'), _kst_open('2026-07-02'))),
        decision_time=datetime(2026, 12, 30, tzinfo=UTC),
    )

    assert frame.height == 1
    assert frame.item(0, 'filing_id') == '20260701000123'
    assert frame.item(0, 'published_at') == datetime(2026, 6, 30, 15, 0, tzinfo=UTC)
    assert frame.item(0, 'available_at') == datetime(2026, 7, 2, 0, 0, tzinfo=UTC)


def test_normalize_dart_facts_restated_record_excluded_before_receipt() -> None:
    from datetime import UTC, datetime

    from src.core.time import SessionCalendar
    from src.data.normalization import normalize_dart_financial_facts

    frame = normalize_dart_financial_facts(
        pages=[_fact_record(
            filing_id='20260701000123',
            published_at=datetime(2024, 11, 13, 0, 0, tzinfo=UTC),
            extra={'rcept_no': '20260701000123'},
        )],
        disclosure_rows=(),
        source_hash='a' * 64,
        calendar=SessionCalendar((_kst_open('2024-11-14'),)),
        decision_time=datetime(2025, 6, 30, tzinfo=UTC),
    )

    assert frame.is_empty()


def test_normalize_dart_facts_empty_calendar_fails_closed() -> None:
    from datetime import UTC, datetime

    import pytest

    from src.core.time import SessionCalendar
    from src.data.normalization import normalize_dart_financial_facts
    from src.data.schemas import PITDataError

    with pytest.raises(PITDataError, match='calendar'):
        normalize_dart_financial_facts(
            pages=[_fact_record(filing_id='F1', published_at=datetime(2024, 11, 13, 0, 0, tzinfo=UTC))],
            disclosure_rows=(),
            source_hash='a' * 64,
            calendar=SessionCalendar(()),
            decision_time=datetime(2024, 11, 20, tzinfo=UTC),
        )


def test_normalize_dart_facts_calendar_ending_before_receipt_fails_closed() -> None:
    from datetime import UTC, datetime

    import pytest

    from src.core.time import SessionCalendar
    from src.data.normalization import normalize_dart_financial_facts
    from src.data.schemas import PITDataError

    with pytest.raises(PITDataError, match='no next KRX session'):
        normalize_dart_financial_facts(
            pages=[_fact_record(filing_id='F1', published_at=datetime(2024, 11, 13, 0, 0, tzinfo=UTC))],
            disclosure_rows=(),
            source_hash='a' * 64,
            calendar=SessionCalendar((_kst_open('2024-11-12'),)),
            decision_time=datetime(2024, 11, 20, tzinfo=UTC),
        )


def test_normalize_dart_facts_next_session_after_decision_time_excluded() -> None:
    from datetime import UTC, datetime

    from src.core.time import SessionCalendar
    from src.data.normalization import normalize_dart_financial_facts

    frame = normalize_dart_financial_facts(
        pages=[_fact_record(filing_id='F1', published_at=datetime(2024, 11, 14, 0, 0, tzinfo=UTC))],
        disclosure_rows=(),
        source_hash='a' * 64,
        calendar=SessionCalendar((_kst_open('2024-11-14'), _kst_open('2024-11-15'))),
        decision_time=datetime(2024, 11, 14, 12, 0, tzinfo=UTC),
    )

    assert frame.is_empty()


def test_normalize_dart_facts_malformed_receipt_number_skipped() -> None:
    from datetime import UTC, datetime

    from src.core.time import SessionCalendar
    from src.data.normalization import normalize_dart_financial_facts

    frame = normalize_dart_financial_facts(
        pages=[
            _fact_record(
                filing_id='BAD',
                published_at=datetime(2024, 11, 13, 0, 0, tzinfo=UTC),
                extra={'rcept_no': '2024111'},
            ),
            _fact_record(filing_id='GOOD', published_at=datetime(2024, 11, 13, 0, 0, tzinfo=UTC)),
        ],
        disclosure_rows=(),
        source_hash='a' * 64,
        calendar=SessionCalendar((_kst_open('2024-11-13'), _kst_open('2024-11-14'))),
        decision_time=datetime(2024, 11, 20, tzinfo=UTC),
    )

    assert frame['filing_id'].to_list() == ['GOOD']


def test_normalize_dart_facts_receipt_date_from_filing_id_and_invalid_skipped() -> None:
    from datetime import UTC, datetime

    from src.core.time import SessionCalendar
    from src.data.normalization import normalize_dart_financial_facts

    frame = normalize_dart_financial_facts(
        pages=[
            _fact_record(
                filing_id='20241113000123',
                published_at=datetime(2024, 11, 13, 0, 0, tzinfo=UTC),
            ),
            _fact_record(
                filing_id='BAD-DATE',
                published_at=datetime(2024, 11, 13, 0, 0, tzinfo=UTC),
                extra={'rcept_no': '20261301000123'},
            ),
        ],
        disclosure_rows=(),
        source_hash='a' * 64,
        calendar=SessionCalendar((_kst_open('2024-11-13'), _kst_open('2024-11-14'))),
        decision_time=datetime(2024, 11, 20, tzinfo=UTC),
    )

    assert frame['filing_id'].to_list() == ['20241113000123']
    assert frame.item(0, 'published_at') == datetime(2024, 11, 12, 15, 0, tzinfo=UTC)
    assert frame.item(0, 'available_at') == datetime(2024, 11, 14, 0, 0, tzinfo=UTC)


def test_normalize_dart_facts_naive_session_fails_closed() -> None:
    from datetime import UTC, datetime

    import pytest

    from src.core.time import SessionCalendar
    from src.data.normalization import normalize_dart_financial_facts
    from src.data.schemas import PITDataError

    with pytest.raises(PITDataError, match='timezone-aware'):
        normalize_dart_financial_facts(
            pages=[_fact_record(filing_id='F1', published_at=datetime(2024, 11, 13, 0, 0, tzinfo=UTC))],
            disclosure_rows=(),
            source_hash='a' * 64,
            calendar=SessionCalendar((datetime(2024, 11, 14, 9, 0),)),
            decision_time=datetime(2024, 11, 20, tzinfo=UTC),
        )


def test_normalize_dart_facts_non_numeric_value_skipped() -> None:
    from datetime import UTC, datetime

    from src.core.time import SessionCalendar
    from src.data.normalization import normalize_dart_financial_facts

    frame = normalize_dart_financial_facts(
        pages=[
            {**_fact_record(filing_id='BAD-VALUE', published_at=datetime(2024, 11, 13, 0, 0, tzinfo=UTC)), 'value': 'not-a-number'},
            _fact_record(filing_id='GOOD', published_at=datetime(2024, 11, 13, 0, 0, tzinfo=UTC)),
        ],
        disclosure_rows=(),
        source_hash='a' * 64,
        calendar=SessionCalendar((_kst_open('2024-11-13'), _kst_open('2024-11-14'))),
        decision_time=datetime(2024, 11, 20, tzinfo=UTC),
    )

    assert frame['filing_id'].to_list() == ['GOOD']


def _legacy_record(*, filing_id: str, published_at, fiscal_period: str = '2024Q3', extra: dict | None = None) -> dict:
    record = {
        'ticker': '005930',
        'corp_code': '00126380',
        'fiscal_period': fiscal_period,
        'filing_id': filing_id,
        'fact': 'sales',
        'published_at': published_at,
        'value': 10.0,
        'unit': 'KRW',
        'consolidated': True,
        'source_kind': 'legacy_document',
    }
    if extra:
        record.update(extra)
    return record


def _standard_page(*, filing_id: str, published_at, fiscal_period: str = '2024Q3') -> dict:
    return {
        'source_kind': 'opendart_standard',
        'records': [_fact_record(filing_id=filing_id, published_at=published_at, extra={'fiscal_period': fiscal_period})],
    }


def _legacy_page(*, filing_id: str, published_at, fiscal_period: str = '2024Q3') -> dict:
    return {
        'source_kind': 'legacy_document',
        'records': [_legacy_record(filing_id=filing_id, published_at=published_at, fiscal_period=fiscal_period)],
    }


def test_quarantine_withholds_legacy_page() -> None:
    from datetime import UTC, datetime

    from src.core.time import SessionCalendar
    from src.data.normalization import normalize_dart_financial_facts_with_quarantine

    calendar = SessionCalendar((_kst_open('2024-11-13'), _kst_open('2024-11-14')))
    decision_time = datetime(2024, 11, 20, tzinfo=UTC)
    pages = [
        _standard_page(filing_id='STD', published_at=datetime(2024, 11, 13, 0, 0, tzinfo=UTC)),
        _legacy_page(filing_id='LEG', published_at=datetime(2024, 11, 13, 0, 0, tzinfo=UTC)),
    ]
    frame, quarantined = normalize_dart_financial_facts_with_quarantine(
        pages=pages, disclosure_rows=(), source_hash='a' * 64,
        calendar=calendar, decision_time=decision_time,
    )

    assert frame['filing_id'].to_list() == ['STD']
    assert len(quarantined) == 1
    assert quarantined[0].filing_id == 'LEG'
    assert quarantined[0].company_id == '005930'
    assert quarantined[0].source_kind == 'legacy_document'


def test_quarantine_uses_fact_availability_rule() -> None:
    from datetime import UTC, datetime

    from src.core.time import SessionCalendar
    from src.data.normalization import normalize_dart_financial_facts_with_quarantine

    calendar = SessionCalendar((_kst_open('2024-11-15'), _kst_open('2024-11-18')))
    _, quarantined = normalize_dart_financial_facts_with_quarantine(
        pages=[_legacy_page(filing_id='F-FRI', published_at=datetime(2024, 11, 15, 0, 0, tzinfo=UTC))],
        disclosure_rows=(), source_hash='a' * 64,
        calendar=calendar, decision_time=datetime(2024, 11, 20, tzinfo=UTC),
    )

    assert len(quarantined) == 1
    assert quarantined[0].available_at == datetime(2024, 11, 18, 0, 0, tzinfo=UTC)
    assert quarantined[0].published_at == datetime(2024, 11, 14, 15, 0, tzinfo=UTC)


def test_quarantine_trusted_set_configurable() -> None:
    from datetime import UTC, datetime

    from src.core.time import SessionCalendar
    from src.data.normalization import normalize_dart_financial_facts_with_quarantine

    calendar = SessionCalendar((_kst_open('2024-11-13'), _kst_open('2024-11-14')))
    frame, quarantined = normalize_dart_financial_facts_with_quarantine(
        pages=[_legacy_page(filing_id='LEG', published_at=datetime(2024, 11, 13, 0, 0, tzinfo=UTC))],
        disclosure_rows=(), source_hash='a' * 64,
        calendar=calendar, decision_time=datetime(2024, 11, 20, tzinfo=UTC),
        trusted_source_kinds=frozenset({'opendart_standard', 'legacy_document'}),
    )

    assert frame['filing_id'].to_list() == ['LEG']
    assert quarantined == ()


def test_quarantine_skips_unresolvable_company() -> None:
    from datetime import UTC, datetime

    from src.core.time import SessionCalendar
    from src.data.normalization import normalize_dart_financial_facts_with_quarantine

    calendar = SessionCalendar((_kst_open('2024-11-13'), _kst_open('2024-11-14')))
    page = {
        'source_kind': 'legacy_document',
        'records': [{
            'corp_code': '00999999', 'fiscal_period': '2024Q3', 'filing_id': 'GHOST',
            'fact': 'sales', 'published_at': datetime(2024, 11, 13, 0, 0, tzinfo=UTC),
            'value': 1.0, 'unit': 'KRW',
        }],
    }
    frame, quarantined = normalize_dart_financial_facts_with_quarantine(
        pages=[page], disclosure_rows=(), source_hash='a' * 64,
        calendar=calendar, decision_time=datetime(2024, 11, 20, tzinfo=UTC),
    )

    assert frame.is_empty()
    assert quarantined == ()


def test_quarantine_ignores_empty_legacy_page() -> None:
    from datetime import UTC, datetime

    from src.core.time import SessionCalendar
    from src.data.normalization import normalize_dart_financial_facts_with_quarantine

    calendar = SessionCalendar((_kst_open('2024-11-13'), _kst_open('2024-11-14')))
    frame, quarantined = normalize_dart_financial_facts_with_quarantine(
        pages=[{'source_kind': 'legacy_document', 'records': []}],
        disclosure_rows=(), source_hash='a' * 64,
        calendar=calendar, decision_time=datetime(2024, 11, 20, tzinfo=UTC),
    )

    assert frame.is_empty()
    assert quarantined == ()


def test_quarantine_excludes_future_filing() -> None:
    from datetime import UTC, datetime

    from src.core.time import SessionCalendar
    from src.data.normalization import normalize_dart_financial_facts_with_quarantine

    calendar = SessionCalendar((_kst_open('2024-11-13'), _kst_open('2024-11-14')))
    frame, quarantined = normalize_dart_financial_facts_with_quarantine(
        pages=[_legacy_page(filing_id='FUT', published_at=datetime(2024, 11, 13, 0, 0, tzinfo=UTC))],
        disclosure_rows=(), source_hash='a' * 64,
        calendar=calendar, decision_time=datetime(2024, 11, 13, 12, 0, tzinfo=UTC),
    )

    assert frame.is_empty()
    assert quarantined == ()


def test_quarantine_mixed_record_kinds() -> None:
    from datetime import UTC, datetime

    from src.core.time import SessionCalendar
    from src.data.normalization import normalize_dart_financial_facts_with_quarantine

    published = datetime(2024, 11, 13, 0, 0, tzinfo=UTC)
    calendar = SessionCalendar((_kst_open('2024-11-13'), _kst_open('2024-11-14')))
    kwargs = {
        'disclosure_rows': (), 'source_hash': 'a' * 64,
        'calendar': calendar, 'decision_time': datetime(2024, 11, 20, tzinfo=UTC),
    }
    mixed_trusted = {
        'source_kind': 'opendart_standard',
        'records': [
            _fact_record(filing_id='MIX', published_at=published),
            _legacy_record(filing_id='MIX', published_at=published),
        ],
    }
    frame, quarantined = normalize_dart_financial_facts_with_quarantine(pages=[mixed_trusted], **kwargs)
    assert frame['filing_id'].to_list() == ['MIX']
    assert quarantined == ()

    only_legacy = {
        'source_kind': 'opendart_standard',
        'records': [_legacy_record(filing_id='ONLY', published_at=published)],
    }
    frame2, quarantined2 = normalize_dart_financial_facts_with_quarantine(pages=[only_legacy], **kwargs)
    assert frame2.is_empty()
    assert [q.filing_id for q in quarantined2] == ['ONLY']


def test_normalize_wrapper_returns_trusted_frame() -> None:
    from datetime import UTC, datetime

    from src.core.time import SessionCalendar
    from src.data.normalization import (
        normalize_dart_financial_facts,
        normalize_dart_financial_facts_with_quarantine,
    )

    calendar = SessionCalendar((_kst_open('2024-11-13'), _kst_open('2024-11-14')))
    pages = [
        _standard_page(filing_id='STD', published_at=datetime(2024, 11, 13, 0, 0, tzinfo=UTC)),
        _legacy_page(filing_id='LEG', published_at=datetime(2024, 11, 13, 0, 0, tzinfo=UTC)),
    ]
    kwargs = {
        'disclosure_rows': (), 'source_hash': 'a' * 64,
        'calendar': calendar, 'decision_time': datetime(2024, 11, 20, tzinfo=UTC),
    }
    assert normalize_dart_financial_facts(pages=pages, **kwargs).equals(
        normalize_dart_financial_facts_with_quarantine(pages=pages, **kwargs)[0]
    )


def test_quarantine_ordering_deterministic() -> None:
    from datetime import UTC, datetime

    from src.core.time import SessionCalendar
    from src.data.normalization import normalize_dart_financial_facts_with_quarantine

    calendar = SessionCalendar((_kst_open('2024-11-13'), _kst_open('2024-11-14')))
    decision_time = datetime(2024, 11, 20, tzinfo=UTC)
    pages = [
        _legacy_page(filing_id='B', published_at=datetime(2024, 11, 13, 0, 0, tzinfo=UTC)),
        _legacy_page(filing_id='A', published_at=datetime(2024, 11, 13, 0, 0, tzinfo=UTC)),
    ]
    kwargs = {'disclosure_rows': (), 'source_hash': 'a' * 64, 'calendar': calendar, 'decision_time': decision_time}
    _, first = normalize_dart_financial_facts_with_quarantine(pages=pages, **kwargs)
    _, second = normalize_dart_financial_facts_with_quarantine(pages=list(reversed(pages)), **kwargs)

    assert first == second
    assert [q.filing_id for q in first] == ['A', 'B']


def test_quarantine_page_identity_fallback() -> None:
    from datetime import UTC, datetime

    from src.core.time import SessionCalendar
    from src.data.normalization import normalize_dart_financial_facts_with_quarantine

    calendar = SessionCalendar((_kst_open('2024-11-13'), _kst_open('2024-11-14')))
    page = {
        'source_kind': 'legacy_document',
        'identity': {'filing_id': 'ID-F', 'corp_code': '00126380'},
        'records': [{
            'ticker': '005930', 'corp_code': '00126380', 'fiscal_period': '2024Q3',
            'fact': 'sales', 'published_at': datetime(2024, 11, 13, 0, 0, tzinfo=UTC),
            'value': 1.0, 'unit': 'KRW',
        }],
    }
    frame, quarantined = normalize_dart_financial_facts_with_quarantine(
        pages=[page], disclosure_rows=(), source_hash='a' * 64,
        calendar=calendar, decision_time=datetime(2024, 11, 20, tzinfo=UTC),
        ticker_by_corp_code={'00126380': '005930'}, bridge_receipt_hash='b' * 64,
    )

    assert frame.is_empty()
    assert [q.filing_id for q in quarantined] == ['ID-F']


def test_quarantine_skips_unresolvable_ticker_shapes() -> None:
    from datetime import UTC, datetime

    from src.core.time import SessionCalendar
    from src.data.normalization import normalize_dart_financial_facts_with_quarantine

    calendar = SessionCalendar((_kst_open('2024-11-13'), _kst_open('2024-11-14')))
    published = datetime(2024, 11, 13, 0, 0, tzinfo=UTC)
    base = {'fiscal_period': '2024Q3', 'fact': 'sales', 'published_at': published, 'value': 1.0, 'unit': 'KRW'}
    pages = [
        {'source_kind': 'legacy_document', 'records': [{**base, 'ticker': 'ABC1234', 'corp_code': '00126380', 'filing_id': 'BAD-T'}]},
        {'source_kind': 'legacy_document', 'records': [{**base, 'ticker': '005930', 'filing_id': 'NO-CORP'}]},
        {'source_kind': 'legacy_document', 'records': [{**base, 'company_id': 'UNKNOWN-CO', 'corp_code': '00999999', 'filing_id': 'NO-BRIDGE'}]},
    ]
    frame, quarantined = normalize_dart_financial_facts_with_quarantine(
        pages=pages, disclosure_rows=(), source_hash='a' * 64,
        calendar=calendar, decision_time=datetime(2024, 11, 20, tzinfo=UTC),
    )

    assert frame.is_empty()
    assert quarantined == ()


def test_trusted_company_passthrough_shapes() -> None:
    from datetime import UTC, datetime

    from src.core.time import SessionCalendar
    from src.data.normalization import normalize_dart_financial_facts_with_quarantine

    calendar = SessionCalendar((_kst_open('2024-11-13'), _kst_open('2024-11-14')))
    published = datetime(2024, 11, 13, 0, 0, tzinfo=UTC)
    base = {'fiscal_period': '2024Q3', 'fact': 'sales', 'published_at': published, 'value': 1.0, 'unit': 'KRW'}
    pages = [
        {**base, 'company_id': '005930', 'filing_id': 'SIX'},
        {**base, 'company_id': 'CUSTOM-CO', 'filing_id': 'RAW'},
    ]
    frame, quarantined = normalize_dart_financial_facts_with_quarantine(
        pages=pages, disclosure_rows=(), source_hash='a' * 64,
        calendar=calendar, decision_time=datetime(2024, 11, 20, tzinfo=UTC),
    )

    assert sorted(frame['filing_id'].to_list()) == ['RAW', 'SIX']
    assert quarantined == ()


def test_company_bridge_and_disclosure_fallback_shapes() -> None:
    from datetime import UTC, datetime

    from src.core.time import SessionCalendar
    from src.data.normalization import normalize_dart_financial_facts_with_quarantine

    calendar = SessionCalendar((_kst_open('2024-11-13'), _kst_open('2024-11-14')))
    published = datetime(2024, 11, 13, 0, 0, tzinfo=UTC)
    pages = [
        {
            'company_id': '005930',
            'corp_code': '00126380',
            'fiscal_period': '2024Q3',
            'filing_id': 'BRIDGED',
            'fact': 'sales',
            'published_at': published,
            'value': 1.0,
            'unit': 'KRW',
        },
        {
            'company_id': '005930',
            'fiscal_period': '2024Q3',
            'filing_id': 'FROM-DISC',
            'fact': 'sales',
            'value': 2.0,
            'unit': 'KRW',
        },
    ]
    frame, quarantined = normalize_dart_financial_facts_with_quarantine(
        pages=pages,
        disclosure_rows=[{'filing_id': 'FROM-DISC', 'published_at': published}],
        source_hash='a' * 64,
        calendar=calendar,
        decision_time=datetime(2024, 11, 20, tzinfo=UTC),
        ticker_by_corp_code={'00126380': '005930'},
    )

    assert sorted(frame['filing_id'].to_list()) == ['BRIDGED', 'FROM-DISC']
    assert quarantined == ()


def test_non_mapping_and_keyless_pages_skipped() -> None:
    from datetime import UTC, datetime

    from src.core.time import SessionCalendar
    from src.data.normalization import normalize_dart_financial_facts_with_quarantine

    calendar = SessionCalendar((_kst_open('2024-11-13'), _kst_open('2024-11-14')))
    published = datetime(2024, 11, 13, 0, 0, tzinfo=UTC)
    pages = [
        'not-a-mapping',
        {'fiscal_period': '2024Q3', 'filing_id': 'NO-IDS', 'fact': 'sales', 'published_at': published, 'value': 1.0, 'unit': 'KRW'},
        {'ticker': '005930', 'corp_code': '00126380', 'fact': 'sales', 'published_at': published, 'value': 1.0, 'unit': 'KRW'},
        {**_fact_record(filing_id='GOOD', published_at=published)},
    ]
    frame, quarantined = normalize_dart_financial_facts_with_quarantine(
        pages=pages, disclosure_rows=(), source_hash='a' * 64,
        calendar=calendar, decision_time=datetime(2024, 11, 20, tzinfo=UTC),
    )

    assert frame['filing_id'].to_list() == ['GOOD']
    assert quarantined == ()


def _attested_page(*, filing_id: str, published_at, verification) -> dict:
    record = _legacy_record(
        filing_id=filing_id, published_at=published_at,
        extra={'source_kind': 'legacy_document_verified', 'verification': verification},
    )
    return {'source_kind': 'legacy_document_verified', 'records': [record]}


_ATTESTATION = {'benchmark_id': 'b' * 64, 'fact_class': 'balance_sheet', 'checks': ['identity']}


def test_attested_page_enters_silver() -> None:
    from datetime import UTC, datetime

    from src.core.time import SessionCalendar
    from src.data.normalization import normalize_dart_financial_facts_with_quarantine

    published = datetime(2024, 11, 13, 0, 0, tzinfo=UTC)
    frame, quarantined = normalize_dart_financial_facts_with_quarantine(
        pages=[_attested_page(filing_id='ATT', published_at=published, verification=_ATTESTATION)],
        disclosure_rows=(), source_hash='a' * 64,
        calendar=SessionCalendar((_kst_open('2024-11-13'), _kst_open('2024-11-14'))),
        decision_time=datetime(2024, 11, 20, tzinfo=UTC),
    )

    assert frame['source_kind'].to_list() == ['legacy_document_verified']
    assert quarantined == ()


@pytest.mark.parametrize(
    'verification',
    [
        None,
        {},
        {'benchmark_id': ' ', 'fact_class': 'balance_sheet', 'checks': ['identity']},
        {'benchmark_id': 'b' * 64, 'fact_class': '', 'checks': ['identity']},
        {'benchmark_id': 'b' * 64, 'fact_class': 'balance_sheet', 'checks': []},
        {'benchmark_id': 'b' * 64, 'fact_class': 'balance_sheet', 'checks': 'identity'},
    ],
)
def test_attested_page_without_complete_attestation_fails_closed(verification) -> None:
    from datetime import UTC, datetime

    from src.core.time import SessionCalendar
    from src.data.normalization import normalize_dart_financial_facts_with_quarantine

    published = datetime(2024, 11, 13, 0, 0, tzinfo=UTC)
    frame, quarantined = normalize_dart_financial_facts_with_quarantine(
        pages=[_attested_page(filing_id='ATT', published_at=published, verification=verification)],
        disclosure_rows=(), source_hash='a' * 64,
        calendar=SessionCalendar((_kst_open('2024-11-13'), _kst_open('2024-11-14'))),
        decision_time=datetime(2024, 11, 20, tzinfo=UTC),
    )

    assert frame.is_empty()
    assert [q.filing_id for q in quarantined] == ['ATT']


def test_attested_page_level_verification_is_inherited_by_records() -> None:
    from datetime import UTC, datetime

    from src.core.time import SessionCalendar
    from src.data.normalization import normalize_dart_financial_facts_with_quarantine

    published = datetime(2024, 11, 13, 0, 0, tzinfo=UTC)
    record = _legacy_record(filing_id='ATT', published_at=published, extra={'source_kind': 'legacy_document_verified'})
    page = {'source_kind': 'legacy_document_verified', 'verification': _ATTESTATION, 'records': [record]}
    frame, quarantined = normalize_dart_financial_facts_with_quarantine(
        pages=[page], disclosure_rows=(), source_hash='a' * 64,
        calendar=SessionCalendar((_kst_open('2024-11-13'), _kst_open('2024-11-14'))),
        decision_time=datetime(2024, 11, 20, tzinfo=UTC),
    )

    assert frame['filing_id'].to_list() == ['ATT']
    assert quarantined == ()
