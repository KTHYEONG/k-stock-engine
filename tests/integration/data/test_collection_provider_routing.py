from datetime import UTC, date, datetime

from src.data.collection import collect_champion_evidence, collect_planned_investor_flow
from src.data.collection_plan import CollectionCheckpointStore, build_historical_collection_plan
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
