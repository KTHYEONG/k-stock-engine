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
        provider="ls",
        collector=collector,
        bronze_root=tmp_path / "bronze",
        retrieved_at=datetime.now(UTC),
        checkpoint_store=ckpt,
    )
    assert artifact.receipt_count >= 1


def test_planned_ls_collection_rejects_over_700_session_chunk_before_fetch(tmp_path) -> None:
    from datetime import UTC, date, datetime, timedelta
    import pytest
    from src.data.collection import collect_planned_investor_flow
    from src.data.collection_checkpoint import CollectionCheckpointStore
    from src.data.collection_plan import HistoricalCollectionPlan, PlanChunk
    from src.data.schemas import PITDataError

    sessions = tuple(date(2016, 1, 1) + timedelta(days=i) for i in range(701))
    plan = HistoricalCollectionPlan('p', sessions[0], sessions[-1], 701, (PlanChunk('p:000020:0000', '000020', sessions),), 'a' * 64)
    class Collector:
        def fetch_investor_flow(self, **_kwargs): raise AssertionError('must not fetch')
    with pytest.raises(PITDataError, match='700'):
        collect_planned_investor_flow(plan=plan, provider='ls', collector=Collector(), bronze_root=tmp_path / 'bronze', retrieved_at=datetime(2026, 9, 11, tzinfo=UTC), checkpoint_store=CollectionCheckpointStore(tmp_path / 'checkpoints'))


def test_planned_collection_rejects_unknown_provider_before_fetch(tmp_path) -> None:
    from datetime import UTC, datetime
    import pytest
    from src.data.collection import collect_planned_investor_flow
    from src.data.collection_checkpoint import CollectionCheckpointStore
    from src.data.collection_plan import HistoricalCollectionPlan
    from src.data.schemas import PITDataError

    plan = HistoricalCollectionPlan(plan_id="p")

    class Collector:
        def fetch_investor_flow(self, **_kwargs):
            raise AssertionError("must not fetch")

    with pytest.raises(PITDataError, match="unsupported investor flow provider"):
        collect_planned_investor_flow(
            plan=plan,
            provider="kis",
            collector=Collector(),
            bronze_root=tmp_path / "bronze",
            retrieved_at=datetime(2026, 9, 11, tzinfo=UTC),
            checkpoint_store=CollectionCheckpointStore(tmp_path / "checkpoints"),
        )


def test_planned_collection_reports_missing_sessions(tmp_path) -> None:
    from datetime import UTC, date, datetime
    import pytest
    from src.data.collection import collect_planned_investor_flow
    from src.data.collection_checkpoint import CollectionCheckpointStore
    from src.data.collection_plan import HistoricalCollectionPlan, PlanChunk
    from src.data.schemas import PITDataError

    day = date(2016, 1, 4)
    plan = HistoricalCollectionPlan(
        "p", day, day, 1, (PlanChunk("p:005930:0000", "005930", (day,)),), "d" * 64
    )

    class EmptyCollector:
        def fetch_investor_flow(self, *args, **kwargs):
            return ({"records": []},)

    with pytest.raises(PITDataError, match="missing requested sessions"):
        collect_planned_investor_flow(
            plan=plan,
            provider="ls",
            collector=EmptyCollector(),
            bronze_root=tmp_path / "bronze",
            retrieved_at=datetime(2026, 9, 11, tzinfo=UTC),
            checkpoint_store=CollectionCheckpointStore(tmp_path / "checkpoints"),
        )
