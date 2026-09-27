"""Flow-target invariants: Silver-only requirement set and range coverage."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

SCOPE_CONFIG = Path("config/research/kr_swing_2019_v1.toml")
D1 = date(2024, 1, 2)


def _cells(rows: list[tuple[date, str]]) -> pl.DataFrame:
    return pl.DataFrame(
        {"session": [day for day, _ in rows], "ticker": [ticker for _, ticker in rows]},
        schema={"session": pl.Date, "ticker": pl.String},
    )


def _range(subject: str, start: date, end: date, status: str):  # type: ignore[no-untyped-def]
    from datetime import UTC, datetime

    from src.data.receipt_catalog import CoverageRange, EvidenceStatus

    return CoverageRange(
        source="ls_investor_flow", subject=subject, start=start, end=end,
        status=EvidenceStatus(status), content_hash=None,
        retrieved_at=datetime(2024, 1, 10, tzinfo=UTC),
    )


def test_only_eligible_tradable_cells(tmp_path: Path) -> None:
    from src.data.dataset_registry import DatasetRegistry
    from src.data.datasets import DatasetIdentity, DatasetLayer, publish_dataset
    from src.data.flow_targets import investor_flow_targets
    from src.data.runtime import load_data_runtime

    runtime = load_data_runtime(scope_config=SCOPE_CONFIG, data_root=tmp_path / "data")
    universe = publish_dataset(
        layer_root=runtime.workspace.silver_root,
        identity=DatasetIdentity(
            kind="ordinary_universe", layer=DatasetLayer.SILVER,
            policy_version="flow-fixture-v1", inputs={}, params={},
        ),
        partitions={"part.parquet": pl.DataFrame(
            {"session": [D1, D1, D1], "ticker": ["000001", "000002", "000003"],
             "eligible": [True, True, False]},
            schema={"session": pl.Date, "ticker": pl.String, "eligible": pl.Boolean},
        )},
    )
    daily = publish_dataset(
        layer_root=runtime.workspace.silver_root,
        identity=DatasetIdentity(
            kind="daily_market", layer=DatasetLayer.SILVER,
            policy_version="flow-fixture-v1", inputs={}, params={},
        ),
        partitions={"part.parquet": pl.DataFrame(
            {"session": [D1, D1, D1], "ticker": ["000001", "000002", "000003"],
             "price_state": ["tradable", "halted", "tradable"]},
            schema={"session": pl.Date, "ticker": pl.String, "price_state": pl.String},
        )},
    )
    registry = DatasetRegistry(runtime.workspace.state_root)
    registry.register("ordinary_universe", universe.dataset_id)
    registry.register("daily_market", daily.dataset_id)

    targets = investor_flow_targets(runtime)

    assert (targets.universe_dataset_id, targets.daily_market_dataset_id) == (universe.dataset_id, daily.dataset_id)
    assert targets.cells.sort(["ticker", "session"]).to_dicts() == [
        {"ticker": "000001", "session": D1}
    ]


def test_ranges_cover_inclusively() -> None:
    from src.data.flow_targets import pending_flow_cells

    days = [date(2024, 1, day) for day in (2, 3, 4, 5, 6)]
    targets = _cells([(day, "000001") for day in days])

    pending = pending_flow_cells(targets, [_range("000001", days[1], days[3], "success")])

    assert pending["session"].to_list() == [days[0], days[4]]


def test_empty_range_also_answers() -> None:
    from src.data.flow_targets import pending_flow_cells

    days = [date(2024, 1, day) for day in (2, 3, 4, 5, 6)]
    targets = _cells([(day, "000001") for day in days])

    pending = pending_flow_cells(targets, [_range("000001", days[0], days[4], "empty")])

    assert pending.height == 0


def test_other_subject_ignored() -> None:
    from src.data.flow_targets import pending_flow_cells

    days = [date(2024, 1, day) for day in (2, 3)]
    targets = _cells([(day, "000001") for day in days])

    pending = pending_flow_cells(targets, [_range("000002", days[0], days[1], "success")])

    assert pending.height == 2


def test_since_keeps_sessions_from_the_cutoff(tmp_path: Path) -> None:
    from src.data.dataset_registry import DatasetRegistry
    from src.data.datasets import DatasetIdentity, DatasetLayer, publish_dataset
    from src.data.flow_targets import investor_flow_targets
    from src.data.runtime import load_data_runtime

    runtime = load_data_runtime(scope_config=SCOPE_CONFIG, data_root=tmp_path / "data")
    days = [date(2024, 1, day) for day in (2, 3, 4)]
    frame = lambda column, value: pl.DataFrame(  # noqa: E731
        {"session": days, "ticker": ["000001"] * 3, column: [value] * 3},
        schema={"session": pl.Date, "ticker": pl.String, column: pl.Boolean if column == "eligible" else pl.String},
    )
    universe = publish_dataset(
        layer_root=runtime.workspace.silver_root,
        identity=DatasetIdentity(kind="ordinary_universe", layer=DatasetLayer.SILVER,
                                 policy_version="flow-fixture-v1", inputs={}, params={}),
        partitions={"part.parquet": frame("eligible", True)},
    )
    daily = publish_dataset(
        layer_root=runtime.workspace.silver_root,
        identity=DatasetIdentity(kind="daily_market", layer=DatasetLayer.SILVER,
                                 policy_version="flow-fixture-v1", inputs={}, params={}),
        partitions={"part.parquet": pl.DataFrame(
            {"session": days, "ticker": ["000001"] * 3, "price_state": ["tradable"] * 3},
            schema={"session": pl.Date, "ticker": pl.String, "price_state": pl.String},
        )},
    )
    registry = DatasetRegistry(runtime.workspace.state_root)
    registry.register("ordinary_universe", universe.dataset_id)
    registry.register("daily_market", daily.dataset_id)

    targets = investor_flow_targets(runtime, since=days[1])

    assert targets.cells["session"].to_list() == days[1:]


def test_missing_registry_dataset_fails_closed(tmp_path: Path) -> None:
    from src.core.pit import PITDataError
    from src.data.flow_targets import investor_flow_targets
    from src.data.runtime import load_data_runtime

    runtime = load_data_runtime(scope_config=SCOPE_CONFIG, data_root=tmp_path / "data")

    with pytest.raises(PITDataError, match="not registered"):
        investor_flow_targets(runtime)


def _publish_pair(runtime, universe_frame, daily_frame):  # type: ignore[no-untyped-def]
    from src.data.dataset_registry import DatasetRegistry
    from src.data.datasets import DatasetIdentity, DatasetLayer, publish_dataset

    universe = publish_dataset(
        layer_root=runtime.workspace.silver_root,
        identity=DatasetIdentity(kind="ordinary_universe", layer=DatasetLayer.SILVER,
                                 policy_version="flow-fixture-v1", inputs={}, params={}),
        partitions={"part.parquet": universe_frame},
    )
    daily = publish_dataset(
        layer_root=runtime.workspace.silver_root,
        identity=DatasetIdentity(kind="daily_market", layer=DatasetLayer.SILVER,
                                 policy_version="flow-fixture-v1", inputs={}, params={}),
        partitions={"part.parquet": daily_frame},
    )
    registry = DatasetRegistry(runtime.workspace.state_root)
    registry.register("ordinary_universe", universe.dataset_id)
    registry.register("daily_market", daily.dataset_id)
    return universe, daily


def test_missing_manifest_fails_closed(tmp_path: Path) -> None:
    import shutil

    import polars as pl

    from src.core.pit import PITDataError
    from src.data.flow_targets import investor_flow_targets
    from src.data.runtime import load_data_runtime

    runtime = load_data_runtime(scope_config=SCOPE_CONFIG, data_root=tmp_path / "data")
    frame = pl.DataFrame(
        {"session": [D1], "ticker": ["000001"], "eligible": [True]},
        schema={"session": pl.Date, "ticker": pl.String, "eligible": pl.Boolean},
    )
    universe, _ = _publish_pair(runtime, frame, frame.rename({"eligible": "price_state"}))
    shutil.rmtree(runtime.workspace.silver_root / universe.dataset_id)

    with pytest.raises(PITDataError, match="invalid ordinary-universe manifest"):
        investor_flow_targets(runtime)


def test_tampered_partition_fails_closed(tmp_path: Path) -> None:
    import polars as pl

    from src.core.pit import PITDataError
    from src.data.flow_targets import investor_flow_targets
    from src.data.runtime import load_data_runtime

    runtime = load_data_runtime(scope_config=SCOPE_CONFIG, data_root=tmp_path / "data")
    frame = pl.DataFrame(
        {"session": [D1], "ticker": ["000001"], "eligible": [True]},
        schema={"session": pl.Date, "ticker": pl.String, "eligible": pl.Boolean},
    )
    universe, _ = _publish_pair(runtime, frame, frame.rename({"eligible": "price_state"}))
    part = runtime.workspace.silver_root / universe.dataset_id / "part.parquet"
    part.write_bytes(part.read_bytes() + b"tampered")

    with pytest.raises(PITDataError, match="invalid ordinary-universe dataset"):
        investor_flow_targets(runtime)


def test_empty_partitions_fail_closed(tmp_path: Path) -> None:
    import polars as pl

    from src.core.pit import PITDataError
    from src.data.dataset_registry import DatasetRegistry
    from src.data.datasets import DatasetIdentity, DatasetLayer, publish_dataset
    from src.data.flow_targets import investor_flow_targets
    from src.data.runtime import load_data_runtime

    runtime = load_data_runtime(scope_config=SCOPE_CONFIG, data_root=tmp_path / "data")
    empty = publish_dataset(
        layer_root=runtime.workspace.silver_root,
        identity=DatasetIdentity(kind="ordinary_universe", layer=DatasetLayer.SILVER,
                                 policy_version="flow-fixture-v1", inputs={}, params={}),
        partitions={},
    )
    full = pl.DataFrame(
        {"session": [D1], "ticker": ["000001"], "price_state": ["tradable"]},
        schema={"session": pl.Date, "ticker": pl.String, "price_state": pl.String},
    )
    daily = publish_dataset(
        layer_root=runtime.workspace.silver_root,
        identity=DatasetIdentity(kind="daily_market", layer=DatasetLayer.SILVER,
                                 policy_version="flow-fixture-v1", inputs={}, params={}),
        partitions={"part.parquet": full},
    )
    registry = DatasetRegistry(runtime.workspace.state_root)
    registry.register("ordinary_universe", empty.dataset_id)
    registry.register("daily_market", daily.dataset_id)

    with pytest.raises(PITDataError, match="invalid ordinary-universe dataset"):
        investor_flow_targets(runtime)


def test_unreadable_silver_inputs_fail_closed(tmp_path: Path) -> None:
    import polars as pl

    from src.core.pit import PITDataError
    from src.data.flow_targets import investor_flow_targets
    from src.data.runtime import load_data_runtime

    runtime = load_data_runtime(scope_config=SCOPE_CONFIG, data_root=tmp_path / "data")
    no_flag = pl.DataFrame(
        {"session": [D1], "ticker": ["000001"]},
        schema={"session": pl.Date, "ticker": pl.String},
    )
    tradable = pl.DataFrame(
        {"session": [D1], "ticker": ["000001"], "price_state": ["tradable"]},
        schema={"session": pl.Date, "ticker": pl.String, "price_state": pl.String},
    )
    _publish_pair(runtime, no_flag, tradable)

    with pytest.raises(PITDataError, match="unreadable"):
        investor_flow_targets(runtime)


def test_empty_targets_stay_empty() -> None:
    from src.data.flow_targets import pending_flow_cells

    empty = pl.DataFrame(
        {"session": [], "ticker": []},
        schema={"session": pl.Date, "ticker": pl.String},
    )

    assert pending_flow_cells(empty, []).height == 0


def test_non_answered_status_is_ignored() -> None:
    from src.data.flow_targets import pending_flow_cells

    targets = _cells([(D1, "000001")])

    pending = pending_flow_cells(targets, [_range("000001", D1, D1, "provider_error")])

    assert pending.height == 1


def test_overlapping_ranges_merge() -> None:
    from datetime import timedelta

    from src.data.flow_targets import pending_flow_cells

    days = [D1 + timedelta(days=index) for index in range(6)]
    targets = _cells([(day, "000001") for day in days])

    pending = pending_flow_cells(
        targets,
        [_range("000001", days[0], days[2], "success"), _range("000001", days[2], days[4], "empty")],
    )

    assert pending["session"].to_list() == [days[5]]
