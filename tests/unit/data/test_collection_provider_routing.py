from datetime import UTC, date, datetime
from pathlib import Path

from src.data.collection import collect_planned_investor_flow
from src.data.collection_checkpoint import CollectionCheckpointStore
from src.data.collection_plan import CollectionChunk, HistoricalCollectionPlan
from src.integrations.ls.investor_flow import LsInvestorFlowCollector


def test_collect_planned_investor_flow_with_ls_collector(tmp_path: Path) -> None:
    class MockLsClient:
        def inquire_investor_trend(self, symbol: str, start_date: date, end_date: date, unit: str = "amount"):
            return (
                {
                    "date": "20260306",
                    "tjj0008": "100",
                    "tjj0009": "200",
                    "tjj0018": "-300",
                },
            )

    plan = HistoricalCollectionPlan(
        plan_id="test_plan",
        dataset_name="test_dataset",
        chunks=(
            CollectionChunk(
                chunk_id="005930_2026",
                symbol="005930",
                sessions=(date(2026, 3, 6),),
            ),
        ),
        created_at=datetime.now(UTC),
    )
    collector = LsInvestorFlowCollector(("005930",), client=MockLsClient())
    ckpt = CollectionCheckpointStore(tmp_path / "checkpoints")
    artifact = collect_planned_investor_flow(
        plan=plan,
        collector=collector,
        bronze_root=tmp_path / "bronze",
        retrieved_at=datetime.now(UTC),
        checkpoint_store=ckpt,
    )
    assert artifact.receipt_count >= 1
