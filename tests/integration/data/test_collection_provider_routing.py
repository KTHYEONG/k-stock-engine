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
    class Client:
        def inquire_investor_trade_by_stock_daily(self, symbol, anchor):
            return (
                {
                    "stck_bsop_date": anchor.strftime("%Y%m%d"),
                    "frgn_shnu_tr_pbmn": "100",
                    "frgn_seln_tr_pbmn": "40",
                    "frgn_ntby_tr_pbmn": "60",
                    "orgn_ntby_tr_pbmn": "-20",
                    "prsn_ntby_tr_pbmn": "-40",
                },
            )

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
        kis=KisInvestorFlowCollector(("005930",), client=Client()),
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
    assert HISTORICAL_PROVIDER_ROUTES[EvidenceKind.INVESTOR_FLOW] == 'kis'
    assert HISTORICAL_PROVIDER_ROUTES[EvidenceKind.FINANCIAL_FACTS] == 'opendart'
    assert HISTORICAL_PROVIDER_ROUTES[EvidenceKind.CORPORATE_ACTIONS] == 'retained_krx_intervals'


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

    class Client:
        def inquire_investor_trade_by_stock_daily(self, symbol, anchor):
            return ({
                'stck_bsop_date': anchor.strftime('%Y%m%d'),
                'frgn_shnu_tr_pbmn': '1', 'frgn_seln_tr_pbmn': '0',
                'frgn_ntby_tr_pbmn': '1', 'orgn_ntby_tr_pbmn': '0',
                'prsn_ntby_tr_pbmn': '-1',
            },)

    plan = build_historical_collection_plan(
        sessions=(date(2016, 1, 4),),
        universe=({'symbol': '005930', 'is_common_stock': True},),
        start=date(2016, 1, 4), end=date(2016, 1, 4), artifact_root=tmp_path / 'plans',
    )
    result = collect_historical_evidence(
        plan=plan, krx=Krx(), kis=KisInvestorFlowCollector(('005930',), client=Client()), dart=object(),
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
        collect_historical_evidence(plan=plan, krx=object(), kis=None, dart=object(), bronze_root=tmp_path / 'b', checkpoint_root=tmp_path / 'c', retrieved_at=datetime(2026, 1, 1), kinds=frozenset({EvidenceKind.DAILY_MARKET}))
    with pytest.raises(PITDataError, match='at least one'):
        collect_historical_evidence(plan=plan, krx=object(), kis=None, dart=object(), bronze_root=tmp_path / 'b', checkpoint_root=tmp_path / 'c', retrieved_at=datetime(2026, 1, 1, tzinfo=UTC), kinds=frozenset())
    with pytest.raises(PITDataError, match='dedicated'):
        collect_historical_evidence(plan=plan, krx=object(), kis=None, dart=object(), bronze_root=tmp_path / 'b', checkpoint_root=tmp_path / 'c', retrieved_at=datetime(2026, 1, 1, tzinfo=UTC), kinds=frozenset({EvidenceKind.FINANCIAL_FACTS}))
    with pytest.raises(PITDataError, match='KIS collector'):
        collect_historical_evidence(plan=plan, krx=object(), kis=None, dart=object(), bronze_root=tmp_path / 'b', checkpoint_root=tmp_path / 'c', retrieved_at=datetime(2026, 1, 1, tzinfo=UTC), kinds=frozenset({EvidenceKind.INVESTOR_FLOW}))

    class EmptyKrx:
        def fetch_daily_market(self, *args, **kwargs):
            return ()
        def fetch_master_lineage(self, *args, **kwargs):
            return ()

    with pytest.raises(PITDataError, match='daily market response'):
        collect_historical_evidence(plan=plan, krx=EmptyKrx(), kis=None, dart=object(), bronze_root=tmp_path / 'b', checkpoint_root=tmp_path / 'c', retrieved_at=datetime(2026, 1, 1, tzinfo=UTC), kinds=frozenset({EvidenceKind.DAILY_MARKET}))
    with pytest.raises(PITDataError, match='master lineage response'):
        collect_historical_evidence(plan=plan, krx=EmptyKrx(), kis=None, dart=object(), bronze_root=tmp_path / 'b', checkpoint_root=tmp_path / 'c', retrieved_at=datetime(2026, 1, 1, tzinfo=UTC), kinds=frozenset({EvidenceKind.SECURITY_MASTER}))
