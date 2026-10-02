"""Silver cash-series materialization tests."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime, time
from pathlib import Path

import polars as pl
import pytest

from src.core.pit import PITDataError
from src.core.time import KRX_TZ
from src.data.cash_series_silver import (
    POLICY_VERSION,
    CashSeriesConfig,
    load_cash_series_config,
    materialize_cash_series_silver,
)
from src.data.receipt_catalog import EvidenceStatus, ReceiptCatalog, ReceiptIndexEntry
from tests.fixtures import seed_receipts

DAY1 = date(2026, 3, 4)
DAY2 = date(2026, 3, 5)
DAY3 = date(2026, 3, 6)


def _config(**overrides: object) -> CashSeriesConfig:
    values: dict[str, object] = {
        "collection_start": DAY1,
        "ticker": "153130",
        "available_time": time(18, 0),
    }
    values.update(overrides)
    return CashSeriesConfig.model_validate(values)


def _universe(silver_root: Path, sessions: tuple[date, ...]) -> Path:
    from src.data.datasets import DatasetIdentity, DatasetLayer, publish_dataset

    partitions = {
        f"session={day.isoformat()}/part.parquet": pl.DataFrame(
            {"session": [day], "instrument_id": ["KRX:005930"], "ticker": ["005930"], "eligible": [True]}
        )
        for day in sessions
    }
    return publish_dataset(
        layer_root=Path(silver_root),
        identity=DatasetIdentity(
            kind="ordinary_universe",
            layer=DatasetLayer.SILVER,
            policy_version="krx-ordinary-equity-v1",
            inputs={},
            params={"calendar": ",".join(day.isoformat() for day in sessions)},
        ),
        partitions=partitions,
    ).path


def _etf_record(session: date, **overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "_endpoint": "etf",
        "ISU_CD": "153130",
        "ISU_SRT_CD": "153130",
        "BAS_DD": session.strftime("%Y%m%d"),
        "TDD_CLSPRC": "106000",
    }
    record.update(overrides)
    return record


def _publish(
    catalog_root: Path,
    payload_dir: Path,
    session: date,
    records: object,
    *,
    status: EvidenceStatus = EvidenceStatus.SUCCESS,
    as_of: date | None = None,
    raw_bytes: bytes | None = None,
) -> str:
    payload_dir.mkdir(parents=True, exist_ok=True)
    if raw_bytes is None:
        raw = json.dumps({"session": session.isoformat(), "records": records}, sort_keys=True).encode("utf-8")
    else:
        raw = raw_bytes
    digest = hashlib.sha256(raw).hexdigest()
    payload_path = payload_dir / f"{session.isoformat()}.json"
    payload_path.write_bytes(raw)
    seed_receipts(
        ReceiptCatalog(catalog_root),
        [
            ReceiptIndexEntry(
                source="krx_cash_series",
                natural_key=session.isoformat(),
                as_of=session if as_of is None else as_of,
                fiscal_period=None,
                status=status,
                content_hash=digest,
                retrieved_at=datetime(2026, 3, 7, tzinfo=UTC),
                payload_path=payload_path,
            )
        ],
    )
    return digest


def _setup(tmp_path: Path, sessions: tuple[date, ...]) -> tuple[Path, Path]:
    catalog_root = tmp_path / "catalog"
    silver_root = tmp_path / "silver"
    _universe(silver_root, sessions)
    return catalog_root, silver_root


def _pages(tmp_path: Path) -> Path:
    return tmp_path / "pages"


def _frame(dataset_path: Path, session: date) -> pl.DataFrame:
    return pl.read_parquet(dataset_path / f"session={session.isoformat()}" / "part.parquet")


def test_silver_one_row_per_session(tmp_path: Path) -> None:
    catalog_root, silver_root = _setup(tmp_path, (DAY1, DAY2, DAY3))
    pages = _pages(tmp_path)
    digest1 = _publish(catalog_root, pages, DAY1, [_etf_record(DAY1)])
    _publish(catalog_root, pages, DAY2, [_etf_record(DAY2)])
    digest3 = _publish(catalog_root, pages, DAY3, [_etf_record(DAY3)])
    result = materialize_cash_series_silver(
        catalog=ReceiptCatalog(catalog_root),
        universe_root=silver_root,
        silver_root=silver_root,
        config=_config(),
    )
    assert result.sessions == 3
    assert result.dataset_id.startswith("cash_series_")
    assert _frame(result.dataset_path, DAY1)["cash_close"].to_list() == [106000]
    assert _frame(result.dataset_path, DAY3)["cash_close"].to_list() == [106000]
    assert _frame(result.dataset_path, DAY1)["source_hash"].to_list() == [digest1]
    assert _frame(result.dataset_path, DAY3)["source_hash"].to_list() == [digest3]
    assert _frame(result.dataset_path, DAY1)["available_at"].to_list() == [
        datetime(2026, 3, 4, 18, 0, tzinfo=KRX_TZ)
    ]
    assert _frame(result.dataset_path, DAY1)["policy_version"].to_list() == [POLICY_VERSION]


def test_missing_ticker_fails_closed(tmp_path: Path) -> None:
    catalog_root, silver_root = _setup(tmp_path, (DAY1, DAY2, DAY3))
    pages = _pages(tmp_path)
    _publish(catalog_root, pages, DAY1, [_etf_record(DAY1)])
    _publish(catalog_root, pages, DAY2, [{"_endpoint": "etf", "ISU_CD": "999999", "BAS_DD": "20260305", "TDD_CLSPRC": "1"}])
    _publish(catalog_root, pages, DAY3, [_etf_record(DAY3)])
    with pytest.raises(PITDataError, match="2026-03-05"):
        materialize_cash_series_silver(
            catalog=ReceiptCatalog(catalog_root),
            universe_root=silver_root,
            silver_root=silver_root,
            config=_config(),
        )
    assert list(Path(silver_root).glob("cash_series_*")) == []


def test_missing_receipt_fails_closed(tmp_path: Path) -> None:
    catalog_root, silver_root = _setup(tmp_path, (DAY1, DAY2, DAY3))
    pages = _pages(tmp_path)
    _publish(catalog_root, pages, DAY1, [_etf_record(DAY1)])
    _publish(catalog_root, pages, DAY3, [_etf_record(DAY3)])
    with pytest.raises(PITDataError, match="2026-03-05"):
        materialize_cash_series_silver(
            catalog=ReceiptCatalog(catalog_root),
            universe_root=silver_root,
            silver_root=silver_root,
            config=_config(),
        )
    assert list(Path(silver_root).glob("cash_series_*")) == []


def test_config_validation(tmp_path: Path) -> None:
    good = tmp_path / "good.toml"
    good.write_text(
        'collection_start = 2017-01-02\nticker = "153130"\navailable_time = "18:00:00"\n',
        encoding="utf-8",
    )
    assert load_cash_series_config(good).collection_start == date(2017, 1, 2)
    assert load_cash_series_config(Path("config/data/cash_series.toml")).ticker == "153130"
    bad_ticker = tmp_path / "bad.toml"
    bad_ticker.write_text(
        'collection_start = 2017-01-02\nticker = "15313"\navailable_time = "18:00:00"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cash-series config"):
        load_cash_series_config(bad_ticker)
    unknown = tmp_path / "unknown.toml"
    unknown.write_text(good.read_text(encoding="utf-8") + 'extra_key = 1\n', encoding="utf-8")
    with pytest.raises(ValueError, match="cash-series config"):
        load_cash_series_config(unknown)
    missing = tmp_path / "missing.toml"
    missing.write_text('collection_start = 2017-01-02\navailable_time = "18:00:00"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="cash-series config"):
        load_cash_series_config(missing)
    broken = tmp_path / "broken.toml"
    broken.write_text("collection_start = [\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cash-series config"):
        load_cash_series_config(broken)
    with pytest.raises(ValueError, match="cash-series config"):
        load_cash_series_config(tmp_path / "absent.toml")
    with pytest.raises(ValueError, match="6-digit"):
        CashSeriesConfig.model_validate(
            {"collection_start": DAY1, "ticker": "15313", "available_time": time(18, 0)}
        )


def test_duplicated_etf_row_fails_closed(tmp_path: Path) -> None:
    catalog_root, silver_root = _setup(tmp_path, (DAY1, DAY2, DAY3))
    pages = _pages(tmp_path)
    _publish(catalog_root, pages, DAY1, [_etf_record(DAY1)])
    _publish(catalog_root, pages, DAY2, [_etf_record(DAY2)])
    _publish(catalog_root, pages, DAY3, [_etf_record(DAY3), _etf_record(DAY3)])
    with pytest.raises(PITDataError, match="duplicated"):
        materialize_cash_series_silver(
            catalog=ReceiptCatalog(catalog_root),
            universe_root=silver_root,
            silver_root=silver_root,
            config=_config(),
        )


def test_hash_mismatch_and_bad_payloads_fail_closed(tmp_path: Path) -> None:
    catalog_root, silver_root = _setup(tmp_path, (DAY1, DAY2, DAY3))
    pages = _pages(tmp_path)
    _publish(catalog_root, pages, DAY1, [_etf_record(DAY1)])
    _publish(catalog_root, pages, DAY2, [_etf_record(DAY2)])
    _publish(catalog_root, pages, DAY3, [_etf_record(DAY3)])
    (pages / f"{DAY2.isoformat()}.json").write_bytes(b"tampered")
    with pytest.raises(PITDataError):
        materialize_cash_series_silver(
            catalog=ReceiptCatalog(catalog_root),
            universe_root=silver_root,
            silver_root=silver_root,
            config=_config(),
        )


def test_page_date_and_close_validation(tmp_path: Path) -> None:
    catalog_root, silver_root = _setup(tmp_path, (DAY1,))
    pages = _pages(tmp_path)
    _publish(catalog_root, pages, DAY1, [_etf_record(DAY1, BAS_DD="20260305")])
    with pytest.raises(PITDataError):
        materialize_cash_series_silver(
            catalog=ReceiptCatalog(catalog_root),
            universe_root=silver_root,
            silver_root=silver_root,
            config=_config(),
        )


@pytest.mark.parametrize("close", [None, "", True, 10.5, "abc", "1.5", "0", -5])
def test_malformed_close_fails_closed(tmp_path: Path, close: object) -> None:
    catalog_root, silver_root = _setup(tmp_path, (DAY1,))
    pages = _pages(tmp_path)
    _publish(catalog_root, pages, DAY1, [_etf_record(DAY1, TDD_CLSPRC=close)])
    with pytest.raises(PITDataError):
        materialize_cash_series_silver(
            catalog=ReceiptCatalog(catalog_root),
            universe_root=silver_root,
            silver_root=silver_root,
            config=_config(),
        )


@pytest.mark.parametrize("close", [106000, 106000.0, "106,000", "106000.0"])
def test_numeric_close_forms_accepted(tmp_path: Path, close: object) -> None:
    catalog_root, silver_root = _setup(tmp_path, (DAY1,))
    pages = _pages(tmp_path)
    _publish(catalog_root, pages, DAY1, [_etf_record(DAY1, TDD_CLSPRC=close)])
    result = materialize_cash_series_silver(
        catalog=ReceiptCatalog(catalog_root),
        universe_root=silver_root,
        silver_root=silver_root,
        config=_config(),
    )
    assert _frame(result.dataset_path, DAY1)["cash_close"].to_list() == [106000]


def test_no_sessions_and_catalog_conflict_fail_closed(tmp_path: Path) -> None:
    catalog_root, silver_root = _setup(tmp_path, (DAY1,))
    with pytest.raises(PITDataError):
        materialize_cash_series_silver(
            catalog=ReceiptCatalog(catalog_root),
            universe_root=silver_root,
            silver_root=silver_root,
            config=_config(collection_start=date(2027, 1, 4)),
        )
    pages = _pages(tmp_path)
    _publish(catalog_root, pages, DAY1, [_etf_record(DAY1)], as_of=DAY2)
    with pytest.raises(PITDataError):
        materialize_cash_series_silver(
            catalog=ReceiptCatalog(catalog_root),
            universe_root=silver_root,
            silver_root=silver_root,
            config=_config(),
        )


def test_universe_autoresolve_requires_exactly_one(tmp_path: Path) -> None:
    empty_root = tmp_path / "empty-silver"
    empty_root.mkdir(parents=True, exist_ok=True)
    with pytest.raises(PITDataError):
        materialize_cash_series_silver(
            catalog=ReceiptCatalog(tmp_path / "catalog"),
            universe_root=empty_root,
            silver_root=tmp_path / "out",
            config=_config(),
        )


def test_malformed_payload_documents_fail_closed(tmp_path: Path) -> None:
    for index, raw_bytes in enumerate((b"not-json", b"[1, 2]", b'{"session": "2026-03-04"}')):
        catalog_root = tmp_path / f"catalog-{index}"
        silver_root = tmp_path / f"silver-{index}"
        _universe(silver_root, (DAY1,))
        pages = tmp_path / f"pages-{hashlib.sha256(raw_bytes).hexdigest()[:8]}"
        _publish(catalog_root, pages, DAY1, [_etf_record(DAY1)], raw_bytes=raw_bytes)
        with pytest.raises(PITDataError):
            materialize_cash_series_silver(
                catalog=ReceiptCatalog(catalog_root),
                universe_root=silver_root,
                silver_root=silver_root,
                config=_config(),
            )


def test_unreadable_payload_fails_closed(tmp_path: Path) -> None:
    catalog_root, silver_root = _setup(tmp_path, (DAY1,))
    pages = _pages(tmp_path)
    _publish(catalog_root, pages, DAY1, [_etf_record(DAY1)])
    (pages / f"{DAY1.isoformat()}.json").unlink()
    with pytest.raises(PITDataError):
        materialize_cash_series_silver(
            catalog=ReceiptCatalog(catalog_root),
            universe_root=silver_root,
            silver_root=silver_root,
            config=_config(),
        )


def test_materialize_is_idempotent(tmp_path: Path) -> None:
    catalog_root, silver_root = _setup(tmp_path, (DAY1, DAY2))
    pages = _pages(tmp_path)
    _publish(catalog_root, pages, DAY1, [_etf_record(DAY1)])
    _publish(catalog_root, pages, DAY2, [_etf_record(DAY2)])
    first = materialize_cash_series_silver(
        catalog=ReceiptCatalog(catalog_root),
        universe_root=silver_root,
        silver_root=silver_root,
        config=_config(),
    )
    second = materialize_cash_series_silver(
        catalog=ReceiptCatalog(catalog_root),
        universe_root=silver_root,
        silver_root=silver_root,
        config=_config(),
    )
    assert first.dataset_id == second.dataset_id
    _ = POLICY_VERSION


# --- Client-level guards exercising fetch_etf_rows branches ---


class _FakeResponse:
    def __init__(self, payload=None, *, json_error: bool = False) -> None:
        self._payload = payload
        self._json_error = json_error

    def json(self):  # type: ignore[no-untyped-def]
        if self._json_error:
            raise ValueError("not json")
        return self._payload


class _FakeTransport:
    def __init__(self, handler):  # type: ignore[no-untyped-def]
        self._handler = handler
        self.calls: list[tuple] = []

    def get(self, endpoint, params, *, headers=None, classify=None):  # type: ignore[no-untyped-def]
        self.calls.append((endpoint, dict(params), dict(headers or {})))
        response = self._handler(endpoint, dict(params))
        if classify is not None:
            classify(response)
        return response


def _client(handler, **kwargs):  # type: ignore[no-untyped-def]
    from src.integrations.krx.client import KrxApiClient

    return KrxApiClient("test-key", transport=_FakeTransport(handler), **kwargs)


SESSION = date(2026, 1, 5)


def _etf_handler(rows):  # type: ignore[no-untyped-def]
    def _handle(endpoint, params):  # type: ignore[no-untyped-def]
        assert endpoint == "etp/etf_bydd_trd"
        return _FakeResponse({"OutBlock_1": [dict(row) for row in rows]})

    return _handle


def _cash_row(**overrides):  # type: ignore[no-untyped-def]
    row = {"ISU_CD": "153130", "BAS_DD": "20260105", "TDD_CLSPRC": "106000"}
    row.update(overrides)
    return row


def test_etf_rows_filtered_and_tagged() -> None:
    rows = [_cash_row(), {"ISU_CD": "999999", "BAS_DD": "20260105", "TDD_CLSPRC": "1"}]
    client = _client(_etf_handler(rows))
    records = client.fetch_etf_rows(SESSION, tickers=("153130",))
    assert len(records) == 1
    assert records[0]["_endpoint"] == "etf"
    assert records[0]["ISU_CD"] == "153130"
    assert [call[0] for call in client._transport.calls] == ["etp/etf_bydd_trd"]


def test_etf_rows_empty_and_absent_ticker() -> None:
    empty = _client(_etf_handler([]))
    assert empty.fetch_etf_rows(SESSION, tickers=("153130",)) == []
    absent = _client(_etf_handler([{"ISU_CD": "999999", "BAS_DD": "20260105", "TDD_CLSPRC": "1"}]))
    assert absent.fetch_etf_rows(SESSION, tickers=("153130",)) == []


def test_etf_duplicate_rejected() -> None:
    from src.integrations.errors import ProviderTerminalError

    dup = _client(_etf_handler([_cash_row(), _cash_row()]))
    with pytest.raises(ProviderTerminalError):
        dup.fetch_etf_rows(SESSION, tickers=("153130",))


def test_etf_invalid_arguments() -> None:
    client = _client(_etf_handler([_cash_row()]))
    with pytest.raises(ValueError, match="as_of"):
        client.fetch_etf_rows("2026-01-05", tickers=("153130",))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tickers"):
        client.fetch_etf_rows(SESSION, tickers=("  ",))


def test_etf_holiday_passthrough() -> None:
    from src.integrations.krx.client import KrxHolidayError

    payload = {"OutBlock_1": [], "RESULT": {"MESSAGE": "휴장일로 거래가 없습니다"}}
    client = _client(lambda endpoint, params: _FakeResponse(payload))
    with pytest.raises(KrxHolidayError):
        client.fetch_etf_rows(SESSION, tickers=("153130",))


@pytest.mark.parametrize("bas_dd", ["2026-01-05", "20261301", "", None, "20260106"])
def test_etf_bad_dates_rejected(bas_dd: object) -> None:
    from src.integrations.errors import ProviderTerminalError

    client = _client(_etf_handler([_cash_row(BAS_DD=bas_dd)]))
    with pytest.raises(ProviderTerminalError):
        client.fetch_etf_rows(SESSION, tickers=("153130",))


@pytest.mark.parametrize("close", [None, "", True, "abc", "1.5", "0", 0, -3, 10.5, "10.5"])
def test_etf_bad_closes_rejected(close: object) -> None:
    from src.integrations.errors import ProviderTerminalError

    client = _client(_etf_handler([_cash_row(TDD_CLSPRC=close)]))
    with pytest.raises(ProviderTerminalError):
        client.fetch_etf_rows(SESSION, tickers=("153130",))


@pytest.mark.parametrize("close", [106000, 106000.0, "106,000", "106000.0"])
def test_etf_numeric_closes_accepted(close: object) -> None:
    client = _client(_etf_handler([_cash_row(TDD_CLSPRC=close)]))
    assert client.fetch_etf_rows(SESSION, tickers=("153130",))[0]["_endpoint"] == "etf"


# --- Job-level guards exercising KrxCashSeriesJob branches ---


def _krx_runtime(tmp_path: Path):  # type: ignore[no-untyped-def]
    from src.data.runtime import load_data_runtime

    return load_data_runtime(scope_config=Path("config/research/kr_swing_2019_v1.toml"), data_root=tmp_path / "data")


def _krx_provider():  # type: ignore[no-untyped-def]
    from src.config import load_provider_policy, load_runtime_config

    return load_provider_policy(load_runtime_config())


class _CashCollector:
    def __init__(self, pages=None):  # type: ignore[no-untyped-def]
        self._pages = {day: [dict(row) for row in rows] for day, rows in dict(pages or {}).items()}
        self.fetch_calls: list = []

    def fetch_etf_rows(self, session, *, tickers):  # type: ignore[no-untyped-def]
        self.fetch_calls.append((session, tuple(tickers)))
        return [dict(row) for row in self._pages.get(session, ())]

    def health_check(self) -> None:
        return None


def _krx_ctx(runtime, provider, *, collector=None, now=None):  # type: ignore[no-untyped-def]
    from src.data.jobs.krx import build_krx_job_context

    return build_krx_job_context(runtime=runtime, provider=provider, collector=collector, now=now)


@pytest.mark.parametrize("bas_dd", ["2026-03-04", "20261301", "", None])
def test_invalid_bas_dd_fails_closed(tmp_path: Path, bas_dd: object) -> None:
    tag = hashlib.sha256(str(bas_dd).encode()).hexdigest()[:8]
    catalog_root = tmp_path / f"catalog-bad-{tag}"
    silver_root = tmp_path / f"silver-bad-{tag}"
    _universe(silver_root, (DAY1,))
    pages = tmp_path / f"pages-bad-{tag}"
    _publish(catalog_root, pages, DAY1, [_etf_record(DAY1, BAS_DD=bas_dd)])
    with pytest.raises(PITDataError):
        materialize_cash_series_silver(
            catalog=ReceiptCatalog(catalog_root),
            universe_root=silver_root,
            silver_root=silver_root,
            config=_config(),
        )


def test_job_pending_empty_before_collection_start(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import src.data.jobs.krx as krx_jobs
    from datetime import UTC as _UTC
    from datetime import datetime as _datetime

    sessions = (date(2026, 1, 2), date(2026, 1, 5), date(2026, 1, 6))
    monkeypatch.setattr(
        krx_jobs, "_xkrx_sessions", lambda start, end: tuple(d for d in sessions if start <= d <= end)
    )
    runtime = _krx_runtime(tmp_path)
    provider = _krx_provider()
    early = _datetime(2017, 1, 1, 9, 1, tzinfo=_UTC)
    assert krx_jobs.KrxCashSeriesJob().pending(_krx_ctx(runtime, provider, now=lambda: early)) == ()
    ctx = _krx_ctx(
        runtime, provider, collector=_CashCollector(), now=lambda: _datetime(2026, 1, 6, 9, 1, tzinfo=_UTC)
    )
    units = krx_jobs.KrxCashSeriesJob().pending(ctx)
    assert [u.natural_key for u in units] == [d.isoformat() for d in sessions]
    ctx.writer.persist_many(
        (
            krx_jobs.krx_cash_series_scoped_payload(
                records=[{"ISU_CD": "153130"}],
                session=sessions[0],
                retrieved_at=_datetime(2026, 1, 6, 9, 1, tzinfo=_UTC),
            ),
        )
    )
    remaining = krx_jobs.KrxCashSeriesJob().pending(ctx)
    assert [u.natural_key for u in remaining] == [d.isoformat() for d in sessions[1:]]
    assert krx_jobs.KrxCashSeriesJob().source == "krx_cash_series"
    assert krx_jobs.KrxCashSeriesJob().name == "krx_cash_series"
    assert krx_jobs.resolve_krx_job("krx_cash_series").source == "krx_cash_series"
    krx_jobs.KrxCashSeriesJob().health_check(ctx)
    assert krx_jobs.krx_cash_series_scoped_payload(
        records=[], session=sessions[0], retrieved_at=_datetime(2026, 1, 6, 9, 1, tzinfo=_UTC)
    ).source_label == f"krx:cash-series:{sessions[0].isoformat()}"


def test_job_fetch_success_and_empty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import src.data.jobs.krx as krx_jobs
    from datetime import UTC as _UTC
    from datetime import datetime as _datetime

    from src.data.jobs.runner import run_job
    from src.data.receipt_catalog import EvidenceStatus

    sessions = (date(2026, 1, 2), date(2026, 1, 5))
    monkeypatch.setattr(
        krx_jobs, "_xkrx_sessions", lambda start, end: tuple(d for d in sessions if start <= d <= end)
    )
    runtime = _krx_runtime(tmp_path)
    provider = _krx_provider()
    now = lambda: _datetime(2026, 1, 6, 9, 1, tzinfo=_UTC)  # noqa: E731
    pages = {
        day: [{"_endpoint": "etf", "ISU_CD": "153130", "BAS_DD": day.strftime("%Y%m%d"), "TDD_CLSPRC": "1"}]
        for day in sessions
    }
    ctx = _krx_ctx(runtime, provider, collector=_CashCollector(pages=pages), now=now)
    report = run_job(
        krx_jobs.KrxCashSeriesJob(), ctx, chunk_size=5, max_chunks=None, dry_run=False, emit=lambda p: None
    )
    assert report.done == 2
    entries = ctx.catalog.latest(source="krx_cash_series", natural_keys={d.isoformat() for d in sessions})
    assert {k: v.status for k, v in entries.items()} == {d.isoformat(): EvidenceStatus.SUCCESS for d in sessions}
    runtime2 = _krx_runtime(tmp_path / "second")
    ctx2 = _krx_ctx(runtime2, provider, collector=_CashCollector(), now=now)
    report2 = run_job(
        krx_jobs.KrxCashSeriesJob(), ctx2, chunk_size=5, max_chunks=None, dry_run=False, emit=lambda p: None
    )
    assert report2.done == 2
    entries2 = ctx2.catalog.latest(source="krx_cash_series", natural_keys={d.isoformat() for d in sessions})
    assert {k: v.status for k, v in entries2.items()} == {d.isoformat(): EvidenceStatus.EMPTY for d in sessions}


def test_cash_source_contract() -> None:
    from src.data.evidence_sources import KRX_CASH_SERIES_SOURCE, source_contract

    assert KRX_CASH_SERIES_SOURCE == "krx_cash_series"
    assert source_contract(KRX_CASH_SERIES_SOURCE).source == KRX_CASH_SERIES_SOURCE


def test_build_and_collect_commands_registered() -> None:
    from src.data.cli import _ensure_registered
    from src.data.cli.registry import commands

    _ensure_registered()
    names = {command.name for command in commands()}
    assert "build-cash-series-silver" in names
    assert "collect-krx-cash-series" in names


def test_build_cash_series_silver_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import hashlib as _hashlib
    import json as _json

    from src.data.cli import main
    from src.data.receipt_catalog import EvidenceStatus as _Status
    from src.data.receipt_catalog import ReceiptCatalog as _Catalog
    from src.data.receipt_catalog import ReceiptIndexEntry as _Entry
    from src.data.runtime import load_data_runtime as _load_runtime
    from tests.fixtures import seed_receipts as _seed

    scope_config = Path("config/research/kr_swing_2019_v1.toml")
    runtime = _load_runtime(scope_config=scope_config, data_root=tmp_path / "data")
    from src.data.cli.build import _run_ordinary as _ordinary  # noqa: F401
    from src.data.datasets import DatasetIdentity as _Identity
    from src.data.datasets import DatasetLayer as _Layer
    from src.data.datasets import publish_dataset as _publish_ds

    _publish_ds(
        layer_root=runtime.workspace.silver_root,
        identity=_Identity(
            kind="ordinary_universe",
            layer=_Layer.SILVER,
            policy_version="krx-ordinary-equity-v1",
            inputs={},
            params={"calendar": "2026-03-04"},
        ),
        partitions={
            "session=2026-03-04/part.parquet": pl.DataFrame(
                {"session": [date(2026, 3, 4)], "instrument_id": ["KRX:005930"], "eligible": [True]}
            )
        },
    )
    from src.data.dataset_registry import DatasetRegistry as _Registry

    universe_id = _Registry(runtime.workspace.state_root).current("ordinary_universe")
    if universe_id is None:
        candidates = sorted((runtime.workspace.silver_root).glob("ordinary_universe_*"))
        universe_id = candidates[0].name
        _Registry(runtime.workspace.state_root).register("ordinary_universe", universe_id)
    records = [{"_endpoint": "etf", "ISU_CD": "153130", "BAS_DD": "20260304", "TDD_CLSPRC": "106000"}]
    raw = _json.dumps({"session": "2026-03-04", "records": records}, sort_keys=True).encode("utf-8")
    page_path = runtime.workspace.bronze_root / "krx-cash-page.json"
    page_path.parent.mkdir(parents=True, exist_ok=True)
    page_path.write_bytes(raw)
    _seed(
        _Catalog(runtime.workspace.bronze_root / "catalog"),
        [
            _Entry(
                source="krx_cash_series",
                natural_key="2026-03-04",
                as_of=date(2026, 3, 4),
                fiscal_period=None,
                status=_Status.SUCCESS,
                content_hash=_hashlib.sha256(raw).hexdigest(),
                retrieved_at=datetime(2026, 3, 5, tzinfo=UTC),
                payload_path=page_path,
            )
        ],
    )
    assert main(["build-cash-series-silver", "--scope-config", str(scope_config), "--data-root", str(tmp_path / "data")]) == 0
    emitted = _json.loads(capsys.readouterr().out)
    assert emitted["dataset_id"].startswith("cash_series_")
    assert emitted["sessions"] == 1
