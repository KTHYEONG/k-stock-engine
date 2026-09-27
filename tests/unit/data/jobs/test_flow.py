"""Range-planned flow-job invariants: calendar windows, KIS convergence, envelopes."""
from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

SCOPE_CONFIG = Path("config/research/kr_swing_2019_v1.toml")
NOW = datetime(2026, 9, 24, 3, 0, tzinfo=UTC)
TICKER = "005930"


def _runtime(tmp_path: Path):  # type: ignore[no-untyped-def]
    from src.data.runtime import load_data_runtime

    return load_data_runtime(scope_config=SCOPE_CONFIG, data_root=tmp_path / "data")


def _provider():  # type: ignore[no-untyped-def]
    from src.config import load_provider_policy, load_runtime_config

    return load_provider_policy(load_runtime_config())


def _sessions(count: int, *, start: date = date(2020, 1, 1)) -> list[date]:
    return [start + timedelta(days=index) for index in range(count)]


def _publish_requirement_sets(runtime, sessions, tickers):  # type: ignore[no-untyped-def]
    """Register ordinary-universe and daily-market Silver covering every cell."""
    from src.data.dataset_registry import DatasetRegistry
    from src.data.datasets import DatasetIdentity, DatasetLayer, publish_dataset

    rows = [(day, ticker) for day in sessions for ticker in tickers]
    universe = publish_dataset(
        layer_root=runtime.workspace.silver_root,
        identity=DatasetIdentity(kind="ordinary_universe", layer=DatasetLayer.SILVER,
                                 policy_version="flow-fixture-v1", inputs={}, params={}),
        partitions={"part.parquet": pl.DataFrame(
            {"session": [day for day, _ in rows], "ticker": [ticker for _, ticker in rows],
             "eligible": [True] * len(rows)},
            schema={"session": pl.Date, "ticker": pl.String, "eligible": pl.Boolean},
        )},
    )
    daily = publish_dataset(
        layer_root=runtime.workspace.silver_root,
        identity=DatasetIdentity(kind="daily_market", layer=DatasetLayer.SILVER,
                                 policy_version="flow-fixture-v1", inputs={}, params={}),
        partitions={"part.parquet": pl.DataFrame(
            {"session": [day for day, _ in rows], "ticker": [ticker for _, ticker in rows],
             "price_state": ["tradable"] * len(rows)},
            schema={"session": pl.Date, "ticker": pl.String, "price_state": pl.String},
        )},
    )
    registry = DatasetRegistry(runtime.workspace.state_root)
    registry.register("ordinary_universe", universe.dataset_id)
    registry.register("daily_market", daily.dataset_id)
    return universe, daily


def _publish_ls_silver(runtime, rows, *, bronze_flow):  # type: ignore[no-untyped-def]
    from src.data.dataset_registry import DatasetRegistry
    from src.data.datasets import DatasetIdentity, DatasetLayer, publish_dataset

    published = publish_dataset(
        layer_root=runtime.workspace.silver_root,
        identity=DatasetIdentity(kind="investor_flow_ls", layer=DatasetLayer.SILVER,
                                 policy_version="flow-fixture-v1",
                                 inputs={"bronze_flow": bronze_flow}, params={}),
        partitions={"part.parquet": pl.DataFrame(
            {"session": [day for day, _ in rows], "ticker": [ticker for _, ticker in rows]},
            schema={"session": pl.Date, "ticker": pl.String},
        )},
    )
    DatasetRegistry(runtime.workspace.state_root).register("investor_flow_ls", published.dataset_id)
    return published


def _ls_row(session: date) -> dict:  # type: ignore[no-untyped-def]
    return {
        "date": session.strftime("%Y%m%d"), "close": 70100, "volume": 12_000_000, "value": 840_000_000_000,
        "tjj0000": 1, "tjj0001": 2, "tjj0002": 3, "tjj0003": 4, "tjj0004": 5, "tjj0005": 6, "tjj0006": -41,
        "tjj0007": 5, "tjj0008": -60, "tjj0009": 10, "tjj0010": 20, "tjj0011": 45,
        "tjj0016": 30, "tjj0017": 50, "tjj0018": -20,
    }


def _kis_row(session: date) -> dict:  # type: ignore[no-untyped-def]
    return {
        "stck_bsop_date": session.strftime("%Y%m%d"),
        "prsn_ntby_qty": "-100", "frgn_ntby_qty": "20", "orgn_ntby_qty": "-10", "etc_ntby_qty": "90",
    }


class _LsStub:
    """Raw t1702 answers for the requested window."""

    def __init__(self, available):  # type: ignore[no-untyped-def]
        self._available = set(available)
        self.calls: list[tuple] = []
        self.health_checks = 0

    def fetch(self, symbol, start, end):  # type: ignore[no-untyped-def]
        from src.integrations.responses import RawResponse

        self.calls.append((symbol, start, end))
        rows = tuple(
            _ls_row(day) for day in self._available
            if start <= day <= end
        )
        return RawResponse(
            query={"symbol": symbol, "start": start.isoformat(), "end": end.isoformat()}, rows=rows,
        )

    def health_check(self) -> None:
        self.health_checks += 1


class _KisStub:
    """Raw FHPTJ04160001 answers covering the last ``kept`` sessions at each anchor."""

    def __init__(self, available, *, kept=None):  # type: ignore[no-untyped-def]
        self._available = set(available)
        self._kept = kept
        self.calls: list[tuple] = []
        self.health_checks = 0

    def fetch(self, symbol, anchor):  # type: ignore[no-untyped-def]
        from src.integrations.responses import RawResponse

        self.calls.append((symbol, anchor))
        covered = sorted(day for day in self._available if day <= anchor)
        if self._kept is not None:
            covered = covered[-self._kept:]
        return RawResponse(
            query={"symbol": symbol, "anchor": anchor.isoformat()},
            rows=tuple(_kis_row(day) for day in covered),
        )

    def health_check(self) -> None:
        self.health_checks += 1


def _ls_ctx(runtime, provider, *, collector=None):  # type: ignore[no-untyped-def]
    from src.data.jobs.flow import build_ls_job_context

    return build_ls_job_context(runtime=runtime, provider=provider, collector=collector, now=lambda: NOW)


def _kis_ctx(runtime, provider, *, collector=None):  # type: ignore[no-untyped-def]
    from src.data.jobs.flow import build_kis_job_context

    return build_kis_job_context(runtime=runtime, provider=provider, collector=collector, now=lambda: NOW)


def _run(spec, ctx, **kwargs):  # type: ignore[no-untyped-def]
    from src.data.jobs.runner import run_job

    emitted: list[dict] = []
    params = {"chunk_size": 10, "max_chunks": None, "dry_run": False, "emit": emitted.append}
    params.update(kwargs)
    return run_job(spec, ctx, **params), emitted


def test_ls_window_spans_calendar_sessions(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    from src.data.evidence_sources import LS_FLOW_SOURCE
    from src.data.jobs.flow import LsInvestorFlowJob
    from src.data.receipt_catalog import CoverageRange, EvidenceStatus, ReceiptCatalog

    sessions = _sessions(800)
    runtime = _runtime(tmp_path)
    provider = _provider()
    _publish_requirement_sets(runtime, sessions, [TICKER])
    catalog = ReceiptCatalog(runtime.workspace.bronze_root / "catalog")
    catalog.publish(
        (),
        ranges=[CoverageRange(
            source=LS_FLOW_SOURCE, subject=TICKER, start=sessions[1], end=sessions[798],
            status=EvidenceStatus.EMPTY, content_hash=None,
            retrieved_at=datetime(2026, 1, 10, tzinfo=UTC),
        )],
    )

    units = LsInvestorFlowJob().pending(_ls_ctx(runtime, provider))

    assert [(unit.payload["symbol"], unit.payload["start"], unit.payload["end"]) for unit in units] == [
        (TICKER, sessions[0].isoformat(), sessions[699].isoformat()),
        (TICKER, sessions[700].isoformat(), sessions[799].isoformat()),
    ]
    assert all(unit.max_requests == 1 for unit in units)


def test_ls_answered_window_not_replanned(tmp_path: Path) -> None:
    from src.data.jobs.flow import LsInvestorFlowJob, build_ls_job_context

    sessions = _sessions(5)
    runtime = _runtime(tmp_path)
    provider = _provider()
    _publish_requirement_sets(runtime, sessions, [TICKER])
    ctx = build_ls_job_context(
        runtime=runtime, provider=provider, collector=_LsStub(sessions), now=lambda: NOW,
    )
    first, _ = _run(LsInvestorFlowJob(), ctx)

    assert first.status == "complete"
    assert first.pending_left == 0

    second, _ = _run(LsInvestorFlowJob(), ctx, dry_run=True)

    assert second.pending_left == 0


def test_kis_skips_ls_valued_cells(tmp_path: Path) -> None:
    from src.data.jobs.flow import KisInvestorFlowJob

    sessions = _sessions(3)
    runtime = _runtime(tmp_path)
    provider = _provider()
    _publish_requirement_sets(runtime, sessions, [TICKER])
    ctx = _kis_ctx(runtime, provider, collector=_KisStub(sessions))
    _publish_ls_silver(
        runtime, [(sessions[0], TICKER), (sessions[1], TICKER)],
        bronze_flow=ctx.catalog.blob_digest(source="ls_investor_flow"),
    )

    units = KisInvestorFlowJob().pending(ctx)

    assert len(units) == 1
    assert units[0].natural_key == TICKER
    assert units[0].max_requests == 1


def test_kis_short_page_converges(tmp_path: Path) -> None:
    from src.data.jobs.flow import KisInvestorFlowJob

    sessions = _sessions(30)
    runtime = _runtime(tmp_path)
    provider = _provider()
    _publish_requirement_sets(runtime, sessions, [TICKER])
    collector = _KisStub(sessions, kept=10)
    ctx = _kis_ctx(runtime, provider, collector=collector)
    _publish_ls_silver(runtime, [], bronze_flow=ctx.catalog.blob_digest(source="ls_investor_flow"))

    first, _ = _run(KisInvestorFlowJob(), ctx)
    assert first.status == "complete"
    assert collector.calls == [(TICKER, sessions[-1])]
    assert ctx.quota_store.remaining_daily_attempts(
        provider="KIS", now=NOW, daily_limit=provider.kis.daily_limit,
    ) == provider.kis.daily_limit - 1

    second, _ = _run(KisInvestorFlowJob(), ctx)
    assert second.status == "complete"
    assert collector.calls[-1] == (TICKER, sessions[-11])
    assert len(collector.calls) == 2


def test_kis_stale_ls_blocks(tmp_path: Path) -> None:
    from src.core.pit import PITDataError
    from src.data.jobs.flow import KisInvestorFlowJob

    sessions = _sessions(3)
    runtime = _runtime(tmp_path)
    provider = _provider()
    _publish_requirement_sets(runtime, sessions, [TICKER])
    collector = _KisStub(sessions)
    ctx = _kis_ctx(runtime, provider, collector=collector)
    _publish_ls_silver(runtime, [], bronze_flow="bronze:" + "0" * 64)

    with pytest.raises(PITDataError, match="investor_flow_ls is stale"):
        KisInvestorFlowJob().pending(ctx)
    assert collector.calls == []


def test_stored_payload_is_a_valid_envelope_without_records(tmp_path: Path) -> None:
    import json

    from src.data.evidence_sources import KIS_FLOW_SOURCE, LS_FLOW_SOURCE, source_contract, validate_envelope
    from src.data.jobs.flow import KisInvestorFlowJob, LsInvestorFlowJob
    from src.data.receipt_catalog import EvidenceStatus

    sessions = _sessions(3)
    runtime = _runtime(tmp_path)
    provider = _provider()
    _publish_requirement_sets(runtime, sessions, [TICKER])
    ls_ctx = _ls_ctx(runtime, provider, collector=_LsStub(sessions))
    report, _ = _run(LsInvestorFlowJob(), ls_ctx)
    assert report.status == "complete"

    for source in (LS_FLOW_SOURCE,):
        for blob in ls_ctx.catalog.blobs(source=source):
            document = json.loads(Path(blob.payload_path).read_bytes())
            assert "records" not in document
            validate_envelope(source_contract(source), Path(blob.payload_path).read_bytes(),
                              status=EvidenceStatus.SUCCESS)

    kis_ctx = _kis_ctx(runtime, provider, collector=_KisStub(sessions))
    _publish_ls_silver(runtime, [], bronze_flow=kis_ctx.catalog.blob_digest(source="ls_investor_flow"))
    kis_report, _ = _run(KisInvestorFlowJob(), kis_ctx)
    assert kis_report.status == "complete"
    for blob in kis_ctx.catalog.blobs(source=KIS_FLOW_SOURCE):
        document = json.loads(Path(blob.payload_path).read_bytes())
        assert "records" not in document
        validate_envelope(source_contract(KIS_FLOW_SOURCE), Path(blob.payload_path).read_bytes(),
                          status=EvidenceStatus.SUCCESS)


def test_runner_reads_provider_policy(tmp_path: Path) -> None:
    from src.data.jobs.flow import build_ls_job_context
    from src.data.jobs.runner import JobUnit
    from src.integrations.errors import ProviderRetryableError

    runtime = _runtime(tmp_path)
    provider = _provider().model_copy(
        update={"ls": _provider().ls.model_copy(update={"circuit_threshold": 2})}
    )
    assert provider.dart.circuit_threshold == 3
    ctx = build_ls_job_context(runtime=runtime, provider=provider, collector=None, now=lambda: NOW)
    assert ctx.runner.circuit_threshold == 2
    assert ctx.runner.avoid_windows_kst == ()

    class _Flaky:
        name = "flaky_flow"
        fail_fast = False

        def pending(self, ctx):  # type: ignore[no-untyped-def]
            return [JobUnit(source="ls_investor_flow", natural_key=f"u{index}",
                            payload={}, max_requests=1) for index in range(2)]

        def fetch(self, ctx, units):  # type: ignore[no-untyped-def]
            raise ProviderRetryableError("LS t1702 throttled")

        def health_check(self, ctx) -> None:  # type: ignore[no-untyped-def]
            return None

    report, _ = _run(_Flaky(), ctx)

    assert (report.status, report.done, report.pending_left) == ("provider_unstable", 0, 2)


def test_flow_job_contexts_use_provider_ledgers(tmp_path: Path) -> None:
    from src.data.jobs.flow import build_kis_job_context, build_ls_job_context

    runtime = _runtime(tmp_path)
    provider = _provider()
    ls_ctx = build_ls_job_context(runtime=runtime, provider=provider)
    kis_ctx = build_kis_job_context(runtime=runtime, provider=provider)

    assert (ls_ctx.runner.quota_provider, ls_ctx.runner.daily_budget) == ("LS", 10000)
    assert ls_ctx.key_env == "LS_APP_KEY"
    assert (kis_ctx.runner.quota_provider, kis_ctx.runner.daily_budget) == ("KIS", 20000)
    assert kis_ctx.key_env == "KIS_APP_KEY"


def test_ls_empty_answer_records_empty_range(tmp_path: Path) -> None:
    from src.data.evidence_sources import LS_FLOW_SOURCE
    from src.data.jobs.flow import LsInvestorFlowJob
    from src.data.receipt_catalog import EvidenceStatus

    sessions = _sessions(2)
    runtime = _runtime(tmp_path)
    provider = _provider()
    _publish_requirement_sets(runtime, sessions, [TICKER])
    ctx = _ls_ctx(runtime, provider, collector=_LsStub(()))

    report, _ = _run(LsInvestorFlowJob(), ctx)

    assert report.status == "complete"
    assert report.done == 1
    ranges = list(ctx.catalog.ranges(source=LS_FLOW_SOURCE))
    assert {item.status for item in ranges} == {EvidenceStatus.EMPTY}
    assert list(ctx.catalog.blobs(source=LS_FLOW_SOURCE)) == []


def test_kis_empty_answer_records_empty_range(tmp_path: Path) -> None:
    from src.data.evidence_sources import KIS_FLOW_SOURCE
    from src.data.jobs.flow import KisInvestorFlowJob
    from src.data.receipt_catalog import EvidenceStatus

    sessions = _sessions(2)
    runtime = _runtime(tmp_path)
    provider = _provider()
    _publish_requirement_sets(runtime, sessions, [TICKER])
    ctx = _kis_ctx(runtime, provider, collector=_KisStub(()))
    _publish_ls_silver(runtime, [], bronze_flow=ctx.catalog.blob_digest(source="ls_investor_flow"))

    report, _ = _run(KisInvestorFlowJob(), ctx)

    assert report.status == "complete"
    ranges = list(ctx.catalog.ranges(source=KIS_FLOW_SOURCE))
    assert [(item.start, item.end, item.status) for item in ranges] == [
        (sessions[0], sessions[1], EvidenceStatus.EMPTY)
    ]
    assert list(ctx.catalog.blobs(source=KIS_FLOW_SOURCE)) == []


def test_kis_undecipherable_row_dates_answer_the_anchor(tmp_path: Path) -> None:
    from src.data.evidence_sources import KIS_FLOW_SOURCE
    from src.data.jobs.flow import KisInvestorFlowJob
    from src.data.receipt_catalog import EvidenceStatus

    sessions = _sessions(2)
    runtime = _runtime(tmp_path)
    provider = _provider()
    _publish_requirement_sets(runtime, sessions, [TICKER])

    class _GarbledKis(_KisStub):
        def fetch(self, symbol, anchor):  # type: ignore[no-untyped-def]
            from src.integrations.responses import RawResponse

            self.calls.append((symbol, anchor))
            return RawResponse(
                query={"symbol": symbol, "anchor": anchor.isoformat()},
                rows=({
                    "stck_bsop_date": "not-a-date",
                    "prsn_ntby_qty": "0", "frgn_ntby_qty": "0",
                    "orgn_ntby_qty": "0", "etc_ntby_qty": "0",
                },),
            )

    ctx = _kis_ctx(runtime, provider, collector=_GarbledKis(sessions))
    _publish_ls_silver(runtime, [], bronze_flow=ctx.catalog.blob_digest(source="ls_investor_flow"))

    report, _ = _run(KisInvestorFlowJob(), ctx)

    assert report.status == "complete"
    ranges = list(ctx.catalog.ranges(source=KIS_FLOW_SOURCE))
    assert [(item.start, item.end, item.status) for item in ranges] == [
        (sessions[1], sessions[1], EvidenceStatus.SUCCESS)
    ]


def test_kis_pending_is_empty_after_completion(tmp_path: Path) -> None:
    from src.data.jobs.flow import KisInvestorFlowJob

    sessions = _sessions(1)
    runtime = _runtime(tmp_path)
    provider = _provider()
    _publish_requirement_sets(runtime, sessions, [TICKER])
    ctx = _kis_ctx(runtime, provider, collector=_KisStub(sessions))
    _publish_ls_silver(runtime, [], bronze_flow=ctx.catalog.blob_digest(source="ls_investor_flow"))

    first, _ = _run(KisInvestorFlowJob(), ctx)
    assert first.status == "complete"

    assert KisInvestorFlowJob().pending(ctx) == ()


def test_kis_invalid_unit_windows_fail_closed(tmp_path: Path) -> None:
    from src.core.pit import PITDataError
    from src.data.jobs.flow import KisInvestorFlowJob
    from src.data.jobs.runner import JobUnit

    sessions = _sessions(1)
    runtime = _runtime(tmp_path)
    provider = _provider()
    _publish_requirement_sets(runtime, sessions, [TICKER])
    ctx = _kis_ctx(runtime, provider, collector=_KisStub(sessions))
    _publish_ls_silver(runtime, [], bronze_flow=ctx.catalog.blob_digest(source="ls_investor_flow"))

    (unit,) = KisInvestorFlowJob().pending(ctx)
    tampered = JobUnit(
        source=unit.source, natural_key=unit.natural_key,
        payload={"symbol": TICKER, "windows": "bogus"}, max_requests=1,
    )

    with pytest.raises(PITDataError, match="unit windows"):
        KisInvestorFlowJob().fetch(ctx, [tampered])


def test_ls_valued_cells_with_missing_manifest_fail_closed(tmp_path: Path) -> None:
    import shutil

    from src.core.pit import PITDataError
    from src.data.jobs.flow import KisInvestorFlowJob

    sessions = _sessions(1)
    runtime = _runtime(tmp_path)
    provider = _provider()
    _publish_requirement_sets(runtime, sessions, [TICKER])
    ctx = _kis_ctx(runtime, provider, collector=_KisStub(sessions))
    published = _publish_ls_silver(
        runtime, [], bronze_flow=ctx.catalog.blob_digest(source="ls_investor_flow")
    )
    shutil.rmtree(runtime.workspace.silver_root / published.dataset_id)

    with pytest.raises(PITDataError, match="invalid investor_flow_ls manifest"):
        KisInvestorFlowJob().pending(ctx)


def test_ls_valued_cells_with_tampered_partition_fail_closed(tmp_path: Path) -> None:
    from src.core.pit import PITDataError
    from src.data.jobs.flow import KisInvestorFlowJob

    sessions = _sessions(1)
    runtime = _runtime(tmp_path)
    provider = _provider()
    _publish_requirement_sets(runtime, sessions, [TICKER])
    ctx = _kis_ctx(runtime, provider, collector=_KisStub(sessions))
    published = _publish_ls_silver(
        runtime, [(sessions[0], TICKER)], bronze_flow=ctx.catalog.blob_digest(source="ls_investor_flow")
    )
    part = runtime.workspace.silver_root / published.dataset_id / "part.parquet"
    part.write_bytes(part.read_bytes() + b"tampered")

    with pytest.raises(PITDataError, match="invalid investor_flow_ls dataset"):
        KisInvestorFlowJob().pending(ctx)


def test_ls_valued_cells_without_ticker_column_fail_closed(tmp_path: Path) -> None:
    import polars as pl

    from src.core.pit import PITDataError
    from src.data.dataset_registry import DatasetRegistry
    from src.data.datasets import DatasetIdentity, DatasetLayer, publish_dataset
    from src.data.jobs.flow import KisInvestorFlowJob

    sessions = _sessions(1)
    runtime = _runtime(tmp_path)
    provider = _provider()
    _publish_requirement_sets(runtime, sessions, [TICKER])
    ctx = _kis_ctx(runtime, provider, collector=_KisStub(sessions))
    published = publish_dataset(
        layer_root=runtime.workspace.silver_root,
        identity=DatasetIdentity(kind="investor_flow_ls", layer=DatasetLayer.SILVER,
                                 policy_version="flow-fixture-v1",
                                 inputs={"bronze_flow": ctx.catalog.blob_digest(source="ls_investor_flow")},
                                 params={}),
        partitions={"part.parquet": pl.DataFrame(
            {"session": sessions},
            schema={"session": pl.Date},
        )},
    )
    DatasetRegistry(runtime.workspace.state_root).register("investor_flow_ls", published.dataset_id)

    with pytest.raises(PITDataError, match="unreadable"):
        KisInvestorFlowJob().pending(ctx)


def test_kis_windows_reject_non_positive_page_size() -> None:
    import polars as pl

    from src.core.pit import PITDataError
    from src.data.jobs.flow import _kis_windows

    pending = pl.DataFrame(
        {"session": [date(2024, 1, 2)], "ticker": [TICKER]},
        schema={"session": pl.Date, "ticker": pl.String},
    )

    with pytest.raises(PITDataError, match="positive integer"):
        _kis_windows(pending=pending, calendar=(date(2024, 1, 2),), rows_per_page=0)


def test_off_calendar_sessions_fail_closed() -> None:
    import polars as pl

    from src.core.pit import PITDataError
    from src.data.jobs.flow import _kis_windows, _ls_units

    pending = pl.DataFrame(
        {"session": [date(2024, 1, 3)], "ticker": [TICKER]},
        schema={"session": pl.Date, "ticker": pl.String},
    )

    with pytest.raises(PITDataError, match="outside the universe calendar"):
        _ls_units(pending=pending, calendar=(date(2024, 1, 2),), limit=700)
    with pytest.raises(PITDataError, match="outside the universe calendar"):
        _kis_windows(pending=pending, calendar=(date(2024, 1, 2),), rows_per_page=30)


def test_empty_calendar_fails_closed(tmp_path: Path) -> None:
    from src.core.pit import PITDataError
    from src.data.dataset_registry import DatasetRegistry
    from src.data.datasets import DatasetIdentity, DatasetLayer, publish_dataset
    from src.data.jobs.flow import _universe_calendar, build_ls_job_context

    runtime = _runtime(tmp_path)
    provider = _provider()
    published = publish_dataset(
        layer_root=runtime.workspace.silver_root,
        identity=DatasetIdentity(kind="ordinary_universe", layer=DatasetLayer.SILVER,
                                 policy_version="flow-fixture-v1", inputs={}, params={}),
        partitions={},
    )
    DatasetRegistry(runtime.workspace.state_root).register("ordinary_universe", published.dataset_id)
    ctx = build_ls_job_context(runtime=runtime, provider=provider, collector=None, now=lambda: NOW)

    with pytest.raises(PITDataError, match="no sessions"):
        _universe_calendar(ctx)


def test_ls_valued_cells_with_empty_partitions_fail_closed(tmp_path: Path) -> None:
    from src.core.pit import PITDataError
    from src.data.dataset_registry import DatasetRegistry
    from src.data.datasets import DatasetIdentity, DatasetLayer, publish_dataset
    from src.data.jobs.flow import KisInvestorFlowJob

    sessions = _sessions(1)
    runtime = _runtime(tmp_path)
    provider = _provider()
    _publish_requirement_sets(runtime, sessions, [TICKER])
    ctx = _kis_ctx(runtime, provider, collector=_KisStub(sessions))
    published = publish_dataset(
        layer_root=runtime.workspace.silver_root,
        identity=DatasetIdentity(kind="investor_flow_ls", layer=DatasetLayer.SILVER,
                                 policy_version="flow-fixture-v1",
                                 inputs={"bronze_flow": ctx.catalog.blob_digest(source="ls_investor_flow")},
                                 params={}),
        partitions={},
    )
    DatasetRegistry(runtime.workspace.state_root).register("investor_flow_ls", published.dataset_id)

    with pytest.raises(PITDataError, match="invalid investor_flow_ls dataset"):
        KisInvestorFlowJob().pending(ctx)
