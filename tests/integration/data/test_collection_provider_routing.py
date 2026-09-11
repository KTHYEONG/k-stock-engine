from datetime import UTC, date, datetime

from src.data.collection import collect_champion_evidence, collect_planned_investor_flow
from src.data.collection import collect_historical_evidence
from src.data.collection_plan import CollectionCheckpointStore, build_historical_collection_plan
from src.data.schemas import EvidenceKind
from src.integrations.kis.investor_flow import KisInvestorFlowCollector


def test_collection_routes_investor_flow_to_kis_only() -> None:
    result = collect_champion_evidence(krx=object(), kis=object(), dart=object(), plan=object())
    assert result is not None


def test_planned_collection_persists_kis_raw_receipt_and_resumes(tmp_path) -> None:
    class MockLsClient:
        def inquire_investor_trend(self, symbol, start_date, end_date, unit="amount"):
            from datetime import timedelta

            rows = []
            day = start_date
            while day <= end_date:
                rows.append(
                    {
                        "date": day.strftime("%Y%m%d"),
                        "tjj0008": "-40",
                        "tjj0009": "60",
                        "tjj0018": "-20",
                    }
                )
                day += timedelta(days=1)
            return tuple(rows)

    from src.integrations.ls.investor_flow import LsInvestorFlowCollector

    plan = build_historical_collection_plan(
        sessions=(date(2016, 1, 4), date(2016, 1, 5)),
        universe=({"symbol": "005930", "is_common_stock": True},),
        start=date(2016, 1, 4),
        end=date(2016, 1, 5),
        chunk_size=2,
        artifact_root=tmp_path / "plans",
    )
    checkpoints = CollectionCheckpointStore(tmp_path / "checkpoints")
    artifact = collect_planned_investor_flow(
        plan=plan,
        provider="ls",
        collector=LsInvestorFlowCollector(("005930",), client=MockLsClient()),
        bronze_root=tmp_path / "bronze",
        retrieved_at=datetime(2026, 9, 3, tzinfo=UTC),
        checkpoint_store=checkpoints,
    )

    assert artifact.receipts
    assert not checkpoints.has_verified_receipt(
        plan=plan, chunk=plan.chunks[0], bronze_root=tmp_path / "wrong-bronze"
    )
    assert checkpoints.has_verified_receipt(plan=plan, chunk=plan.chunks[0], bronze_root=tmp_path / "bronze")


def test_collect_historical_evidence_routes_each_kind_to_its_owner() -> None:
    from src.data.collection import HISTORICAL_PROVIDER_ROUTES
    from src.data.schemas import EvidenceKind

    assert HISTORICAL_PROVIDER_ROUTES[EvidenceKind.DAILY_MARKET] == 'krx'
    assert HISTORICAL_PROVIDER_ROUTES[EvidenceKind.SECURITY_MASTER] == 'krx'
    assert HISTORICAL_PROVIDER_ROUTES[EvidenceKind.INVESTOR_FLOW] == 'ls'
    assert HISTORICAL_PROVIDER_ROUTES[EvidenceKind.FINANCIAL_FACTS] == 'opendart'
    assert HISTORICAL_PROVIDER_ROUTES[EvidenceKind.CORPORATE_ACTIONS] == 'opendart_structured_decisions'


def test_collect_opendart_corporate_action_evidence_persists_pages(tmp_path) -> None:
    from src.data.bronze import BronzeStore
    from src.data.collection import collect_opendart_corporate_action_evidence

    class Page:
        endpoint = "fricDecsn.json"
        corp_code = "00123456"
        status = "013"
        records = ()

    class Dart:
        def load_corp_codes(self):
            return {"005930": "00123456"}

        def fetch_corporate_action_decisions(self, **_kwargs):
            return (Page(),)

    receipts = collect_opendart_corporate_action_evidence(
        dart=Dart(), tickers=("KRX:005930",), start=date(2024, 1, 1), end=date(2024, 1, 2),
        bronze=BronzeStore(tmp_path / "bronze"),
    )
    assert len(receipts) == 1


def test_collect_historical_evidence_collects_krx_and_kis_pages(tmp_path) -> None:
    class Krx:
        def fetch_daily_market(self, start, end, *, sessions):
            day = sessions[0]
            row = {
                'BAS_DD': day.isoformat(), 'ISU_SRT_CD': '005930',
                'TDD_OPNPRC': '10', 'TDD_HGPRC': '11', 'TDD_LWPRC': '9',
                'TDD_CLSPRC': '10', 'ACC_TRDVOL': '1', 'ACC_TRDVAL': '10',
                'MKTCAP': '100', 'LIST_SHRS': '10',
            }
            return ({'session': day.isoformat(), 'records': [row]},)

        def fetch_master_lineage(self, start, end, *, sessions):
            return ({'session': sessions[0].isoformat(), 'records': [{'ticker': '005930'}]},)

    class MockLsClient:
        def inquire_investor_trend(self, symbol, start_date, end_date, unit="amount"):
            return ({
                'date': start_date.strftime('%Y%m%d'),
                'tjj0008': '-1', 'tjj0009': '1', 'tjj0018': '0',
            },)

    from src.integrations.ls.investor_flow import LsInvestorFlowCollector

    plan = build_historical_collection_plan(
        sessions=(date(2016, 1, 4),),
        universe=({'symbol': '005930', 'is_common_stock': True},),
        start=date(2016, 1, 4), end=date(2016, 1, 4), artifact_root=tmp_path / 'plans',
    )
    result = collect_historical_evidence(
        plan=plan, krx=Krx(), investor_flow=LsInvestorFlowCollector(('005930',), client=MockLsClient()), investor_flow_provider='ls', dart=object(),
        bronze_root=tmp_path / 'bronze', checkpoint_root=tmp_path / 'checkpoints',
        retrieved_at=datetime(2026, 9, 5, tzinfo=UTC),
        kinds=frozenset({EvidenceKind.DAILY_MARKET, EvidenceKind.SECURITY_MASTER, EvidenceKind.INVESTOR_FLOW}),
    )
    assert set(result) == {EvidenceKind.DAILY_MARKET, EvidenceKind.SECURITY_MASTER, EvidenceKind.INVESTOR_FLOW}


def test_collect_historical_evidence_rejects_invalid_request(tmp_path) -> None:
    import pytest
    from src.data.schemas import EvidenceKind, PITDataError

    plan = build_historical_collection_plan(
        sessions=(date(2016, 1, 4),), universe=({'symbol': '005930', 'is_common_stock': True},),
        start=date(2016, 1, 4), end=date(2016, 1, 4), artifact_root=tmp_path / 'plans',
    )
    with pytest.raises(PITDataError, match='timezone-aware'):
        collect_historical_evidence(plan=plan, krx=object(), investor_flow=None, investor_flow_provider='ls', dart=object(), bronze_root=tmp_path / 'b', checkpoint_root=tmp_path / 'c', retrieved_at=datetime(2026, 1, 1), kinds=frozenset({EvidenceKind.DAILY_MARKET}))
    with pytest.raises(PITDataError, match='at least one'):
        collect_historical_evidence(plan=plan, krx=object(), investor_flow=None, investor_flow_provider='ls', dart=object(), bronze_root=tmp_path / 'b', checkpoint_root=tmp_path / 'c', retrieved_at=datetime(2026, 1, 1, tzinfo=UTC), kinds=frozenset())
    with pytest.raises(PITDataError, match='dedicated'):
        collect_historical_evidence(plan=plan, krx=object(), investor_flow=None, investor_flow_provider='ls', dart=object(), bronze_root=tmp_path / 'b', checkpoint_root=tmp_path / 'c', retrieved_at=datetime(2026, 1, 1, tzinfo=UTC), kinds=frozenset({EvidenceKind.FINANCIAL_FACTS}))
    with pytest.raises(PITDataError, match='investor flow requires'):
        collect_historical_evidence(plan=plan, krx=object(), investor_flow=None, investor_flow_provider='ls', dart=object(), bronze_root=tmp_path / 'b', checkpoint_root=tmp_path / 'c', retrieved_at=datetime(2026, 1, 1, tzinfo=UTC), kinds=frozenset({EvidenceKind.INVESTOR_FLOW}))

    class EmptyKrx:
        def fetch_daily_market(self, *args, **kwargs):
            return ()
        def fetch_master_lineage(self, *args, **kwargs):
            return ()

    with pytest.raises(PITDataError, match='daily market response'):
        collect_historical_evidence(plan=plan, krx=EmptyKrx(), investor_flow=None, investor_flow_provider='ls', dart=object(), bronze_root=tmp_path / 'b', checkpoint_root=tmp_path / 'c', retrieved_at=datetime(2026, 1, 1, tzinfo=UTC), kinds=frozenset({EvidenceKind.DAILY_MARKET}))
    with pytest.raises(PITDataError, match='master lineage response'):
        collect_historical_evidence(plan=plan, krx=EmptyKrx(), investor_flow=None, investor_flow_provider='ls', dart=object(), bronze_root=tmp_path / 'b', checkpoint_root=tmp_path / 'c', retrieved_at=datetime(2026, 1, 1, tzinfo=UTC), kinds=frozenset({EvidenceKind.SECURITY_MASTER}))


def test_collect_daily_market_sessions_persists_only_requested_krx_pages(tmp_path) -> None:
    from datetime import UTC, date, datetime
    from src.data.collection import collect_daily_market_sessions

    requested = (date(2024, 1, 2), date(2024, 1, 4))
    class Krx:
        def __init__(self) -> None:
            self.called = ()
        def fetch_daily_market(self, start, end, *, sessions):
            self.called = tuple(sessions)
            return tuple({'session': day.isoformat(), 'records': [{'BAS_DD': day.strftime('%Y%m%d'), 'ISU_SRT_CD': '005930', 'TDD_OPNPRC': '10', 'TDD_HGPRC': '11', 'TDD_LWPRC': '9', 'TDD_CLSPRC': '10', 'ACC_TRDVOL': '1', 'ACC_TRDVAL': '10', 'MKTCAP': '100', 'LIST_SHRS': '10'}]} for day in sessions)

    krx = Krx()
    artifact = collect_daily_market_sessions(sessions=requested, krx=krx, bronze_root=tmp_path / 'bronze', retrieved_at=datetime(2026, 9, 6, tzinfo=UTC))

    assert krx.called == requested
    assert len(artifact.page_receipts['daily_market']) == 2
    assert artifact.coverage_start == requested[0]
    assert artifact.coverage_end == requested[-1]


def test_collect_daily_market_sessions_rejects_invalid_or_incomplete_pages(tmp_path) -> None:
    import pytest
    from src.data.collection import collect_daily_market_sessions
    from src.data.schemas import PITDataError

    class EmptyKrx:
        def fetch_daily_market(self, *_args, **_kwargs):
            return ()

    with pytest.raises(PITDataError, match="at least one"):
        collect_daily_market_sessions(sessions=(), krx=EmptyKrx(), bronze_root=tmp_path / "bronze", retrieved_at=datetime(2026, 9, 6, tzinfo=UTC))
    with pytest.raises(PITDataError, match="timezone-aware"):
        collect_daily_market_sessions(sessions=(date(2024, 1, 2),), krx=EmptyKrx(), bronze_root=tmp_path / "bronze", retrieved_at=datetime(2026, 9, 6))
    with pytest.raises(PITDataError, match="response is empty"):
        collect_daily_market_sessions(sessions=(date(2024, 1, 2),), krx=EmptyKrx(), bronze_root=tmp_path / "bronze", retrieved_at=datetime(2026, 9, 6, tzinfo=UTC))


def test_daily_market_page_validation_rejects_malformed_facts() -> None:
    import pytest
    from src.data.collection import _daily_market_page_session, _validate_daily_market_page
    from src.data.schemas import PITDataError

    assert _daily_market_page_session({"records": [{"BAS_DD": "20240102"}]}) == date(2024, 1, 2)
    assert _daily_market_page_session({"records": [None, {"session": "2024-01-02"}]}) == date(2024, 1, 2)
    with pytest.raises(PITDataError, match="malformed"):
        _daily_market_page_session({"session": "not-a-date"})
    with pytest.raises(PITDataError, match="missing its trading session"):
        _daily_market_page_session({"records": [{}]})
    with pytest.raises(PITDataError, match="malformed"):
        _daily_market_page_session({"records": [{"BAS_DD": "bad"}]})
    valid = {"records": [{"MKTCAP": "1", "LIST_SHRS": "1", "ISU_SRT_CD": "005930"}]}
    _validate_daily_market_page(valid, session=date(2024, 1, 2))
    for page in ({"records": []}, {"records": [None]}, {"records": [{"LIST_SHRS": "1", "ISU_SRT_CD": "1"}]}, {"records": [{"MKTCAP": "1", "ISU_SRT_CD": "1"}]}, {"records": [{"MKTCAP": "1", "LIST_SHRS": "1"}]}, {"records": [{"MKTCAP": "1", "LIST_SHRS": "1", "ISU_SRT_CD": "1"}, {"MKTCAP": "1", "LIST_SHRS": "1", "ISU_SRT_CD": "1"}]}):
        with pytest.raises(PITDataError):
            _validate_daily_market_page(page, session=date(2024, 1, 2))


def test_collect_daily_market_sessions_rejects_page_assignment_errors(tmp_path) -> None:
    import pytest
    from src.data.collection import collect_daily_market_sessions
    from src.data.schemas import PITDataError

    day = date(2024, 1, 2)
    row = {"BAS_DD": "20240102", "ISU_SRT_CD": "1", "MKTCAP": "1", "LIST_SHRS": "1"}
    class Krx:
        def __init__(self, pages): self.pages = pages
        def fetch_daily_market(self, *_args, **_kwargs): return self.pages

    for pages, match in (([{"session": "2024-01-03", "records": [row]}], "non-requested"), ([{"session": day.isoformat(), "records": [row]}, {"session": day.isoformat(), "records": [row]}], "duplicate pages")):
        with pytest.raises(PITDataError, match=match):
            collect_daily_market_sessions(sessions=(day,), krx=Krx(pages), bronze_root=tmp_path / match, retrieved_at=datetime(2026, 9, 6, tzinfo=UTC))


def test_collect_daily_market_sessions_resumes_after_provider_failure(tmp_path) -> None:
    import pytest
    from src.data.collection import collect_daily_market_sessions
    from src.data.schemas import PITDataError

    days = (date(2024, 1, 2), date(2024, 1, 3))

    def page(day: date) -> dict[str, object]:
        return {'session': day.isoformat(), 'records': [{'BAS_DD': day.strftime('%Y%m%d'), 'ISU_SRT_CD': '005930', 'MKTCAP': '1', 'LIST_SHRS': '1'}]}

    class Interrupted:
        def fetch_daily_market(self, *_args, **_kwargs):
            yield page(days[0])
            raise RuntimeError('network dropped')

    with pytest.raises(PITDataError, match='collection failed'):
        collect_daily_market_sessions(sessions=days, krx=Interrupted(), bronze_root=tmp_path / 'bronze', retrieved_at=datetime(2026, 9, 6, tzinfo=UTC))

    class Retry:
        def __init__(self) -> None:
            self.requested = ()

        def fetch_daily_market(self, _start, _end, *, sessions):
            self.requested = tuple(sessions)
            return iter([page(days[1])])

    retry = Retry()
    artifact = collect_daily_market_sessions(sessions=days, krx=retry, bronze_root=tmp_path / 'bronze', retrieved_at=datetime(2026, 9, 6, tzinfo=UTC))
    assert retry.requested == (days[1],)
    assert len(artifact.page_receipts['daily_market']) == 2


def test_collect_daily_market_sessions_rejects_invalid_requests_and_receipts(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace
    import pytest
    from src.data.collection import collect_daily_market_sessions
    from src.data.schemas import EvidenceKind, PITDataError

    class Krx:
        def fetch_daily_market(self, *_args, **_kwargs):
            return ()

    retrieved_at = datetime(2026, 9, 6, tzinfo=UTC)
    with pytest.raises(PITDataError, match='dates only'):
        collect_daily_market_sessions(sessions=(datetime(2024, 1, 2, tzinfo=UTC),), krx=Krx(), bronze_root=tmp_path / 'bronze', retrieved_at=retrieved_at)
    with pytest.raises(PITDataError, match='duplicates'):
        collect_daily_market_sessions(sessions=(date(2024, 1, 2), date(2024, 1, 2)), krx=Krx(), bronze_root=tmp_path / 'bronze', retrieved_at=retrieved_at)
    receipt = SimpleNamespace(payload_path=tmp_path / 'missing.json', metadata_path=tmp_path / 'receipt.json')
    monkeypatch.setattr('src.data.bronze_aggregation.discover_verified_bronze_receipts', lambda **_kwargs: {EvidenceKind.DAILY_MARKET: (receipt,)})
    with pytest.raises(PITDataError, match='malformed Bronze receipt'):
        collect_daily_market_sessions(sessions=(date(2024, 1, 2),), krx=Krx(), bronze_root=tmp_path / 'bronze', retrieved_at=retrieved_at)


def test_collect_daily_market_sessions_fails_closed_for_unusable_pages(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace
    import pytest
    from src.data.collection import collect_daily_market_sessions
    from src.data.schemas import EvidenceKind, PITDataError

    day = date(2024, 1, 2)
    retrieved_at = datetime(2026, 9, 6, tzinfo=UTC)
    payload = tmp_path / 'payload.json'
    payload.write_text('[]', encoding='utf-8')
    receipt = SimpleNamespace(payload_path=payload, metadata_path=tmp_path / 'receipt.json')
    monkeypatch.setattr('src.data.bronze_aggregation.discover_verified_bronze_receipts', lambda **_kwargs: {EvidenceKind.DAILY_MARKET: (receipt,)})

    class Invalid:
        def fetch_daily_market(self, *_args, **_kwargs):
            return (None,)

    with pytest.raises(PITDataError, match='page is empty'):
        collect_daily_market_sessions(sessions=(day,), krx=Invalid(), bronze_root=tmp_path / 'bronze', retrieved_at=retrieved_at)
    payload.write_text('{}', encoding='utf-8')
    with pytest.raises(PITDataError, match='page is empty'):
        collect_daily_market_sessions(sessions=(day,), krx=Invalid(), bronze_root=tmp_path / 'bronze', retrieved_at=retrieved_at)
    monkeypatch.setattr('src.data.bronze_aggregation.discover_verified_bronze_receipts', lambda **_kwargs: (_ for _ in ()).throw(OSError('bad root')))
    with pytest.raises(PITDataError, match='invalid Bronze root'):
        collect_daily_market_sessions(sessions=(day,), krx=Invalid(), bronze_root=tmp_path / 'bronze', retrieved_at=retrieved_at)

    class Partial:
        def fetch_daily_market(self, *_args, **_kwargs):
            return ({'session': day.isoformat(), 'records': [{'BAS_DD': '20240102', 'ISU_SRT_CD': '1', 'MKTCAP': '1', 'LIST_SHRS': '1'}]},)

    monkeypatch.setattr('src.data.bronze_aggregation.discover_verified_bronze_receipts', lambda **_kwargs: {})
    with pytest.raises(PITDataError, match='missing requested sessions'):
        collect_daily_market_sessions(sessions=(day, date(2024, 1, 3)), krx=Partial(), bronze_root=tmp_path / 'other', retrieved_at=retrieved_at)


def test_kis_flow_reuses_verified_anchor_page_after_interruption(tmp_path) -> None:
    from datetime import UTC, date, datetime
    from src.integrations.kis.investor_flow import KisInvestorFlowCollector

    class Client:
        def __init__(self) -> None: self.calls = 0
        def inquire_investor_trade_by_stock_daily(self, symbol, anchor):
            self.calls += 1
            return ({'stck_bsop_date': anchor.strftime('%Y%m%d'), 'frgn_shnu_tr_pbmn': '1', 'frgn_seln_tr_pbmn': '0', 'frgn_ntby_tr_pbmn': '1', 'orgn_ntby_tr_pbmn': '0', 'prsn_ntby_tr_pbmn': '-1'},)
    client = Client()
    collector = KisInvestorFlowCollector(('005930',), client=client)
    first = tuple(collector.fetch_investor_flow(date(2024, 1, 2), date(2024, 1, 2), bronze_root=tmp_path / 'bronze', retrieved_at=datetime(2026, 9, 6, tzinfo=UTC)))
    second = tuple(collector.fetch_investor_flow(date(2024, 1, 2), date(2024, 1, 2), bronze_root=tmp_path / 'bronze', retrieved_at=datetime(2026, 9, 6, tzinfo=UTC)))
    assert client.calls == 1
    assert first[0]['records'] == second[0]['records']


def test_kis_flow_rejects_unusable_verified_receipts(tmp_path, monkeypatch) -> None:
    import json
    from types import SimpleNamespace
    import pytest
    from src.data.schemas import PITDataError

    collector = KisInvestorFlowCollector(('005930',), client=object())
    payload = tmp_path / 'payload.json'
    receipt = SimpleNamespace(payload_path=payload)

    payload.write_text('{', encoding='utf-8')
    monkeypatch.setattr(
        'src.data.bronze_aggregation.discover_verified_bronze_receipts',
        lambda **_kwargs: {EvidenceKind.INVESTOR_FLOW: (receipt,)},
    )
    with pytest.raises(PITDataError, match='invalid verified KIS Bronze payload'):
        collector._find_verified_anchor_page('005930', date(2024, 1, 2), tmp_path)

    payload.write_text(json.dumps(['not-a-page']), encoding='utf-8')
    with pytest.raises(PITDataError, match='invalid verified KIS Bronze payload'):
        collector._find_verified_anchor_page('005930', date(2024, 1, 2), tmp_path)

    payload.write_text(json.dumps({'symbol': '005930', 'anchor': '2024-01-02', 'records': []}), encoding='utf-8')
    assert collector._find_verified_anchor_page('005930', date(2024, 1, 2), tmp_path) is None


def test_kis_flow_reused_page_advances_by_earliest_session(tmp_path) -> None:
    class Client:
        def __init__(self) -> None:
            self.calls = []

        def inquire_investor_trade_by_stock_daily(self, _symbol, anchor):
            self.calls.append(anchor)
            return ({
                'stck_bsop_date': anchor.strftime('%Y%m%d'),
                'frgn_shnu_tr_pbmn': '1', 'frgn_seln_tr_pbmn': '0',
                'frgn_ntby_tr_pbmn': '1', 'orgn_ntby_tr_pbmn': '0',
                'prsn_ntby_tr_pbmn': '-1',
            },)

    client = Client()
    collector = KisInvestorFlowCollector(('005930',), client=client)
    collector._persist_raw_page(
        '005930', date(2024, 1, 5),
        (
            {'stck_bsop_date': '20240105', 'frgn_shnu_tr_pbmn': '1', 'frgn_seln_tr_pbmn': '0', 'frgn_ntby_tr_pbmn': '1', 'orgn_ntby_tr_pbmn': '0', 'prsn_ntby_tr_pbmn': '-1'},
            {'stck_bsop_date': '20240104', 'frgn_shnu_tr_pbmn': '1', 'frgn_seln_tr_pbmn': '0', 'frgn_ntby_tr_pbmn': '1', 'orgn_ntby_tr_pbmn': '0', 'prsn_ntby_tr_pbmn': '-1'},
        ),
        bronze_root=tmp_path, retrieved_at=datetime(2026, 9, 6, tzinfo=UTC),
    )

    pages = tuple(collector.fetch_investor_flow(date(2024, 1, 2), date(2024, 1, 5), bronze_root=tmp_path))

    assert client.calls == [date(2024, 1, 3), date(2024, 1, 2)]
    assert {row['session'] for page in pages for row in page['records']} == {'2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'}


# test_historical_pipeline_collects_opendart_actions_and_gold_does_not_exclude_by_sentinel
def test_historical_pipeline_routes_corporate_actions_to_opendart() -> None:
    from src.data.collection import HISTORICAL_PROVIDER_ROUTES, EvidenceKind

    assert HISTORICAL_PROVIDER_ROUTES[EvidenceKind.CORPORATE_ACTIONS] == 'opendart_structured_decisions'
    assert 'retained_krx_intervals' not in HISTORICAL_PROVIDER_ROUTES.values()


def test_collect_opendart_actions_persists_direct_mapping_provenance(tmp_path) -> None:
    import json
    from datetime import date
    from src.data.bronze import BronzeStore
    from src.data.collection import collect_opendart_corporate_action_evidence

    class Page:
        endpoint = 'fricDecsn.json'
        corp_code = '00123456'
        status = '013'
        records = ()

    class Dart:
        def load_corp_codes(self):
            return {'005930': '00123456'}

        def fetch_corporate_action_decisions(self, **_kwargs):
            return (Page(),)

    receipt = collect_opendart_corporate_action_evidence(
        dart=Dart(), tickers=('KRX:005930',), start=date(2024, 1, 1), end=date(2024, 1, 2), bronze=BronzeStore(tmp_path / 'bronze'),
    )[0]
    payload = json.loads(receipt.payload_path.read_text(encoding='utf-8'))
    assert payload['requested_instrument_id'] == 'KRX:005930'
    assert payload['instrument_mapping_provenance'] == 'opendart_corp_code_direct'


def test_collect_historical_evidence_propagates_selected_kiwoom_provider(tmp_path, monkeypatch) -> None:
    from datetime import UTC, date, datetime
    from src.data.collection import CollectionArtifact, collect_historical_evidence
    from src.data.collection_plan import build_historical_collection_plan
    from src.data.schemas import EvidenceKind

    day = date(2016, 1, 4)
    plan = build_historical_collection_plan(sessions=(day,), universe=({'symbol': '005930', 'is_common_stock': True},), start=day, end=day, artifact_root=tmp_path / 'plans')
    captured = {}
    def fake_collect(**kwargs):
        captured.update(kwargs)
        return CollectionArtifact(tmp_path / 'bronze', day, day, datetime(2026, 9, 11, tzinfo=UTC), {}, 'b' * 64, tmp_path / 'report.json')
    monkeypatch.setattr('src.data.collection.collect_planned_investor_flow', fake_collect)
    collector = object()
    result = collect_historical_evidence(plan=plan, krx=object(), investor_flow=collector, investor_flow_provider='kiwoom', dart=object(), bronze_root=tmp_path / 'bronze', checkpoint_root=tmp_path / 'checkpoints', retrieved_at=datetime(2026, 9, 11, tzinfo=UTC), kinds=frozenset({EvidenceKind.INVESTOR_FLOW}))
    assert result[EvidenceKind.INVESTOR_FLOW].content_hash == 'b' * 64
    assert captured['provider'] == 'kiwoom'
    assert captured['collector'] is collector
