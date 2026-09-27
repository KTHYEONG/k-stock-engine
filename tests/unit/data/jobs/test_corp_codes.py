"""DART corp-code bridge refresh invariants."""
from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

NOW = datetime(2026, 9, 27, 3, 0, tzinfo=UTC)
SCOPE_CONFIG = Path("config/research/kr_swing_2019_v1.toml")
CORP = "00126380"
TICKER = "005930"


def _runtime(tmp_path: Path):  # type: ignore[no-untyped-def]
    from src.data.runtime import load_data_runtime

    return load_data_runtime(scope_config=SCOPE_CONFIG, data_root=tmp_path / "data")


def _provider():  # type: ignore[no-untyped-def]
    from src.config import load_provider_policy, load_runtime_config

    return load_provider_policy(load_runtime_config())


def _ctx(runtime, provider, *, collector=None, now=None):  # type: ignore[no-untyped-def]
    from src.data.jobs.runner import build_job_context

    return build_job_context(
        runtime=runtime, provider=provider, key_env=None, collector=collector, now=now or (lambda: NOW)
    )


def _universe(runtime, tickers=(TICKER,)) -> None:  # type: ignore[no-untyped-def]
    import polars as pl

    from src.data.dataset_registry import DatasetRegistry
    from src.data.datasets import DatasetIdentity, DatasetLayer, publish_dataset

    published = publish_dataset(
        layer_root=runtime.workspace.silver_root,
        identity=DatasetIdentity(
            kind="ordinary_universe", layer=DatasetLayer.SILVER, policy_version="test-v1", inputs={}, params={}
        ),
        partitions={"part.parquet": pl.DataFrame({"ticker": list(tickers), "eligible": [True] * len(tickers)})},
    )
    DatasetRegistry(runtime.workspace.state_root).register("ordinary_universe", published.dataset_id)


def _persist_bridge(ctx, rows: list[dict], *, as_of: date | None = None) -> None:  # type: ignore[no-untyped-def]
    from src.core.pit import EvidenceKind
    from src.data.receipt_catalog import EvidenceStatus
    from src.data.scoped_ingestion import ScopedRawPayload

    ctx.writer.persist(
        ScopedRawPayload(
            kind=EvidenceKind.SECURITY_MASTER,
            source="dart_corp_codes",
            natural_key="dart_corp_codes",
            as_of=as_of or date(2026, 9, 26),
            fiscal_period=None,
            status=EvidenceStatus.SUCCESS,
            payload=json.dumps(rows, sort_keys=True, ensure_ascii=False).encode("utf-8"),
            retrieved_at=NOW,
            source_label="test:bridge",
        )
    )


class _Collector:
    def __init__(self, records=()):  # type: ignore[no-untyped-def]
        self._records = tuple(records)
        self.calls = 0

    def health_check(self) -> None:
        pass

    def fetch_corp_code_records(self):  # type: ignore[no-untyped-def]
        from src.integrations.dart.client import DartCorpCodeRecord

        self.calls += 1
        return tuple(
            DartCorpCodeRecord(ticker=t, corp_code=c, corp_name=n) for t, c, n in self._records
        )


def test_missing_eligible_ticker_triggers_refresh(tmp_path: Path) -> None:
    from src.data.jobs.corp_codes import DartCorpCodesJob

    runtime = _runtime(tmp_path)
    _universe(runtime, tickers=(TICKER, "000001"))
    provider = _provider()
    ctx = _ctx(runtime, provider)
    _persist_bridge(ctx, [{"ticker": TICKER, "corp_code": CORP, "corp_name": "Test Co"}])

    units = DartCorpCodesJob().pending(ctx)

    assert len(units) == 1


def test_fresh_complete_bridge_is_left_alone(tmp_path: Path) -> None:
    from src.data.jobs.corp_codes import DartCorpCodesJob

    runtime = _runtime(tmp_path)
    _universe(runtime)
    ctx = _ctx(runtime, _provider())
    _persist_bridge(ctx, [{"ticker": TICKER, "corp_code": CORP, "corp_name": "Test Co"}])

    assert tuple(DartCorpCodesJob().pending(ctx)) == ()


def test_delisted_mapping_kept(tmp_path: Path) -> None:
    from src.data.jobs.corp_codes import DartCorpCodesJob
    from src.data.jobs.universe import read_corp_code_bridge

    runtime = _runtime(tmp_path)
    _universe(runtime)
    ctx = _ctx(runtime, _provider())
    _persist_bridge(
        ctx,
        [
            {"ticker": TICKER, "corp_code": CORP, "corp_name": "Old Name"},
            {"ticker": "000001", "corp_code": "00000001", "corp_name": "Delisted Co"},
        ],
    )
    fresh_ctx = _ctx(
        runtime, _provider(), collector=_Collector([(TICKER, CORP, "New Name")]),
        now=lambda: datetime(2026, 9, 27, 4, 0, tzinfo=UTC),
    )

    (payload,) = DartCorpCodesJob().fetch(fresh_ctx, DartCorpCodesJob().pending(ctx)[:1] or _pending(fresh_ctx))

    fresh_ctx.writer.persist(payload)
    mapping, _ = read_corp_code_bridge(fresh_ctx.catalog)

    assert mapping["00000001"] == "000001"
    rows = {row["corp_code"]: row for row in json.loads(payload.payload)}
    assert rows["00000001"]["ticker"] == "000001"
    assert rows[CORP]["corp_name"] == "New Name"


def _pending(ctx):  # type: ignore[no-untyped-def]
    from src.data.jobs.runner import JobUnit

    return (JobUnit(source="dart_corp_codes", natural_key="dart_corp_codes", payload={}, max_requests=1),)


def test_ticker_conflict_fails_closed(tmp_path: Path) -> None:
    from src.core.pit import PITDataError
    from src.data.jobs.corp_codes import DartCorpCodesJob

    runtime = _runtime(tmp_path)
    _universe(runtime)
    ctx = _ctx(runtime, _provider())
    _persist_bridge(ctx, [{"ticker": "000001", "corp_code": "00000001", "corp_name": "Old"}])
    conflict_ctx = _ctx(
        runtime, _provider(), collector=_Collector([("000002", "00000001", "New")]),
    )
    before = list(conflict_ctx.catalog.latest(source="dart_corp_codes", natural_keys={"dart_corp_codes"}).values())

    with pytest.raises(PITDataError, match="remapped"):
        DartCorpCodesJob().fetch(conflict_ctx, _pending(conflict_ctx))

    after = list(conflict_ctx.catalog.latest(source="dart_corp_codes", natural_keys={"dart_corp_codes"}).values())
    assert [entry.content_hash for entry in before] == [entry.content_hash for entry in after]


def test_reader_uses_catalog_only(tmp_path: Path) -> None:
    import hashlib

    from src.data.jobs.universe import read_corp_code_bridge

    runtime = _runtime(tmp_path)
    _universe(runtime)
    ctx = _ctx(runtime, _provider())
    _persist_bridge(ctx, [{"ticker": TICKER, "corp_code": CORP, "corp_name": "Catalog Co"}])
    stray = runtime.workspace.bronze_root / "dart_corp_codes" / ("f" * 64) / "payload.json"
    stray.parent.mkdir(parents=True, exist_ok=True)
    stray.write_bytes(
        json.dumps([{"ticker": "999999", "corp_code": "99999999", "corp_name": "Stray"}]).encode("utf-8")
    )
    _ = hashlib.sha256(b"x").hexdigest()

    mapping, content_hash = read_corp_code_bridge(ctx.catalog)

    assert mapping == {CORP: TICKER}
    assert len(content_hash) == 64


def test_missing_bridge_triggers_refresh(tmp_path: Path) -> None:
    from src.data.jobs.corp_codes import DartCorpCodesJob

    runtime = _runtime(tmp_path)
    _universe(runtime)
    ctx = _ctx(runtime, _provider())

    assert len(DartCorpCodesJob().pending(ctx)) == 1


def test_stale_bridge_triggers_refresh(tmp_path: Path) -> None:
    from src.data.jobs.corp_codes import DartCorpCodesJob

    runtime = _runtime(tmp_path)
    _universe(runtime)
    ctx = _ctx(runtime, _provider())
    _persist_bridge(
        ctx, [{"ticker": TICKER, "corp_code": CORP, "corp_name": "Test"}],
        as_of=date(2026, 7, 1),
    )

    assert len(DartCorpCodesJob().pending(ctx)) == 1


def test_fetch_without_collector_fails_closed(tmp_path: Path) -> None:
    import pytest

    from src.core.pit import PITDataError
    from src.data.jobs.corp_codes import DartCorpCodesJob

    runtime = _runtime(tmp_path)
    _universe(runtime)
    ctx = _ctx(runtime, _provider(), collector=None)

    with pytest.raises(PITDataError, match="not configured"):
        DartCorpCodesJob().fetch(ctx, _pending(ctx))


def test_health_check_delegates(tmp_path: Path) -> None:
    from src.data.jobs.corp_codes import DartCorpCodesJob

    runtime = _runtime(tmp_path)
    _universe(runtime)
    ctx = _ctx(runtime, _provider(), collector=_Collector([]))

    DartCorpCodesJob().health_check(ctx)


def test_pending_with_naive_now_and_missing_as_of(tmp_path: Path) -> None:
    from datetime import datetime as _datetime

    from src.core.pit import EvidenceKind
    from src.data.jobs.corp_codes import DartCorpCodesJob
    from src.data.receipt_catalog import EvidenceStatus
    from src.data.scoped_ingestion import ScopedRawPayload

    runtime = _runtime(tmp_path)
    _universe(runtime)
    naive_ctx = _ctx(runtime, _provider(), now=lambda: _datetime(2026, 9, 27, 3, 0))
    assert len(DartCorpCodesJob().pending(naive_ctx)) == 1

    ctx = _ctx(runtime, _provider())
    ctx.writer.persist(
        ScopedRawPayload(
            kind=EvidenceKind.SECURITY_MASTER,
            source="dart_corp_codes",
            natural_key="dart_corp_codes",
            as_of=None,
            fiscal_period=None,
            status=EvidenceStatus.SUCCESS,
            payload=json.dumps(
                [{"ticker": TICKER, "corp_code": CORP, "corp_name": "Test"}]
            ).encode("utf-8"),
            retrieved_at=datetime(2026, 9, 26, 3, 0, tzinfo=UTC),
            source_label="test:bridge-no-asof",
        )
    )
    assert tuple(DartCorpCodesJob().pending(ctx)) == ()


def test_stale_bridge_without_asof_triggers_refresh(tmp_path: Path) -> None:
    from src.core.pit import EvidenceKind
    from src.data.jobs.corp_codes import DartCorpCodesJob
    from src.data.receipt_catalog import EvidenceStatus
    from src.data.scoped_ingestion import ScopedRawPayload

    runtime = _runtime(tmp_path)
    _universe(runtime)
    ctx = _ctx(runtime, _provider())
    ctx.writer.persist(
        ScopedRawPayload(
            kind=EvidenceKind.SECURITY_MASTER,
            source="dart_corp_codes",
            natural_key="dart_corp_codes",
            as_of=None,
            fiscal_period=None,
            status=EvidenceStatus.SUCCESS,
            payload=json.dumps(
                [{"ticker": TICKER, "corp_code": CORP, "corp_name": "Test"}]
            ).encode("utf-8"),
            retrieved_at=datetime(2026, 7, 1, 3, 0, tzinfo=UTC),
            source_label="test:stale-no-asof",
        )
    )

    assert len(DartCorpCodesJob().pending(ctx)) == 1


def test_fetch_uses_legacy_loader_and_empty_old_bridge(tmp_path: Path) -> None:
    from datetime import datetime as _datetime

    from src.data.jobs.corp_codes import DartCorpCodesJob
    from src.data.jobs.universe import read_corp_code_bridge
    from src.integrations.dart.client import DartCorpCodeRecord

    runtime = _runtime(tmp_path)
    _universe(runtime)

    class _LegacyLoader:
        def health_check(self) -> None:
            pass

        def load_corp_code_records(self):  # type: ignore[no-untyped-def]
            return (DartCorpCodeRecord(ticker=TICKER, corp_code=CORP, corp_name="Legacy"),)

    ctx = _ctx(
        runtime, _provider(), collector=_LegacyLoader(),
        now=lambda: _datetime(2026, 9, 27, 4, 0, tzinfo=UTC),
    )

    (payload,) = DartCorpCodesJob().fetch(ctx, _pending(ctx))
    ctx.writer.persist(payload)
    mapping, _ = read_corp_code_bridge(ctx.catalog)

    assert mapping == {CORP: TICKER}


def test_fetch_empty_records_fails_closed(tmp_path: Path) -> None:
    import pytest

    from src.core.pit import PITDataError
    from src.data.jobs.corp_codes import DartCorpCodesJob

    runtime = _runtime(tmp_path)
    _universe(runtime)
    ctx = _ctx(runtime, _provider(), collector=_Collector([]))

    with pytest.raises(PITDataError, match="no listed tickers"):
        DartCorpCodesJob().fetch(ctx, _pending(ctx))


def test_fetch_without_loader_fails_closed(tmp_path: Path) -> None:
    import pytest

    from src.core.pit import PITDataError
    from src.data.jobs.corp_codes import DartCorpCodesJob

    class _NoLoader:
        def health_check(self) -> None:
            pass

    runtime = _runtime(tmp_path)
    _universe(runtime)
    ctx = _ctx(runtime, _provider(), collector=_NoLoader())

    with pytest.raises(PITDataError, match="not configured"):
        DartCorpCodesJob().fetch(ctx, _pending(ctx))


def test_fetch_skips_blank_records(tmp_path: Path) -> None:
    from datetime import datetime as _datetime

    from src.data.jobs.corp_codes import DartCorpCodesJob
    from src.data.jobs.universe import read_corp_code_bridge
    from src.integrations.dart.client import DartCorpCodeRecord

    runtime = _runtime(tmp_path)
    _universe(runtime)

    class _BlankMixed:
        def health_check(self) -> None:
            pass

        def fetch_corp_code_records(self):  # type: ignore[no-untyped-def]
            return (
                DartCorpCodeRecord(ticker="", corp_code="", corp_name="Blank"),
                DartCorpCodeRecord(ticker=TICKER, corp_code=CORP, corp_name="Kept"),
            )

    ctx = _ctx(
        runtime, _provider(), collector=_BlankMixed(),
        now=lambda: _datetime(2026, 9, 27, 4, 0, tzinfo=UTC),
    )

    (payload,) = DartCorpCodesJob().fetch(ctx, _pending(ctx))
    ctx.writer.persist(payload)

    assert read_corp_code_bridge(ctx.catalog)[0] == {CORP: TICKER}
