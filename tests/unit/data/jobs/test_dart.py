"""DART job invariants: page shapes, windows, latest-wins, 014 empties, universe."""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

NOW = datetime(2026, 9, 24, 3, 0, tzinfo=UTC)  # 12:00 KST, last completed 2026-09-23
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


def _bridge(bronze_root: Path, rows=None) -> None:  # type: ignore[no-untyped-def]
    rows = rows if rows is not None else [{"ticker": TICKER, "corp_code": CORP, "corp_name": "Test Co"}]
    raw = json.dumps(rows, sort_keys=True, ensure_ascii=False).encode("utf-8")
    target = bronze_root / "dart_corp_codes" / hashlib.sha256(raw).hexdigest() / "payload.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)


def _universe(runtime) -> None:  # type: ignore[no-untyped-def]
    import polars as pl

    from src.data.dataset_registry import DatasetRegistry
    from src.data.datasets import DatasetIdentity, DatasetLayer, publish_dataset

    published = publish_dataset(
        layer_root=runtime.workspace.silver_root,
        identity=DatasetIdentity(
            kind="ordinary_universe", layer=DatasetLayer.SILVER, policy_version="test-v1", inputs={}, params={}
        ),
        partitions={"part.parquet": pl.DataFrame({"ticker": [TICKER], "eligible": [True]})},
    )
    DatasetRegistry(runtime.workspace.state_root).register("ordinary_universe", published.dataset_id)


def _write_disclosure_page(bronze_root: Path, payload: dict) -> None:  # type: ignore[no-untyped-def]
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    target = bronze_root / "disclosures" / hashlib.sha256(raw).hexdigest() / "payload.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)


def _per_corp_record(rcept_no: str, rcept_dt: str, report_nm: str) -> dict:
    return {"corp_code": CORP, "corp_name": "Test Co", "rcept_no": rcept_no, "rcept_dt": rcept_dt, "report_nm": report_nm, "rm": ""}


def _per_corp_page(records: list, *, start: str = "2015-01-01", end: str = "2019-06-30") -> dict:
    return {"records": records, "start": start, "end": end, "corp_code": CORP}


class _Collector:
    """Scripted stand-in for the DART collector surface the jobs use."""

    def __init__(self, *, windows=None, archives=None, fact_pages=None, healthy=True):  # type: ignore[no-untyped-def]
        self._windows = windows or {}
        self._archives = archives or {}
        self._fact_pages = fact_pages
        self._healthy = healthy
        self.aborted = False
        self.list_calls: list[tuple] = []
        self.archive_calls: list[str] = []

    def health_check(self) -> None:
        if not self._healthy:
            raise RuntimeError("connection reset")

    def list_disclosures(self, start, end, *, detail_type=None):  # type: ignore[no-untyped-def]
        self.list_calls.append((start, end, detail_type))
        outcome = self._windows.get((start.isoformat(), end.isoformat(), detail_type), [])
        if isinstance(outcome, Exception):
            raise outcome
        return [dict(item) for item in outcome]

    def fetch_document_archive(self, rcept_no: str) -> bytes:
        self.archive_calls.append(rcept_no)
        outcome = self._archives.get(rcept_no, b"PK\x03\x04fake-zip")
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def fetch_financial_fact_sources(self, identities):  # type: ignore[no-untyped-def]
        if isinstance(self._fact_pages, Exception):
            raise self._fact_pages
        return iter([dict(page) for page in (self._fact_pages or [])])


def _standard_fact_page(identity: dict, *, status: str = "000", records=None) -> dict:  # type: ignore[no-untyped-def]
    return {
        "source_kind": "opendart_standard",
        "status": status,
        "identity": dict(identity),
        "records": records if records is not None else [{"fact": "sales"}],
        "mapping_version": "test-v1",
        "diagnostics": (),
        "raw_document_hash": None,
        "raw_provenance": {},
        **identity,
    }


# --- universe ---------------------------------------------------------------


def test_eligible_tickers_comes_from_registry_universe(tmp_path: Path) -> None:
    from src.data.jobs.universe import eligible_tickers

    runtime = _runtime(tmp_path)
    _universe(runtime)
    ctx = _ctx(runtime, _provider())

    assert eligible_tickers(ctx) == frozenset({TICKER})


def test_eligible_tickers_fails_closed(tmp_path: Path) -> None:
    import polars as pl

    from src.core.pit import PITDataError
    from src.data.jobs.universe import eligible_tickers

    runtime = _runtime(tmp_path)
    ctx = _ctx(runtime, _provider())
    with pytest.raises(PITDataError, match="not registered"):
        eligible_tickers(ctx)

    _universe(runtime)
    dataset_id = {"ordinary_universe": None}
    from src.data.dataset_registry import DatasetRegistry

    dataset_id["ordinary_universe"] = DatasetRegistry(runtime.workspace.state_root).require("ordinary_universe")
    dataset_dir = runtime.workspace.silver_root / str(dataset_id["ordinary_universe"])
    for path in dataset_dir.rglob("*.parquet"):
        path.unlink()
    with pytest.raises(PITDataError, match="no published partitions"):
        eligible_tickers(ctx)

    (dataset_dir / "junk.parquet").write_bytes(b"not a parquet file")
    with pytest.raises(PITDataError, match="unreadable"):
        eligible_tickers(ctx)
    (dataset_dir / "junk.parquet").unlink()

    pl.DataFrame({"ticker": [TICKER], "eligible": [False]}).write_parquet(dataset_dir / "flat.parquet")
    with pytest.raises(PITDataError, match="no eligible tickers"):
        eligible_tickers(ctx)

    import shutil

    shutil.rmtree(dataset_dir)
    with pytest.raises(PITDataError, match="dataset is missing"):
        eligible_tickers(ctx)


def test_corp_code_bridge_and_index(tmp_path: Path) -> None:
    from src.core.pit import PITDataError
    from src.data.jobs.universe import corp_code_bridge, index_corp_codes
    from src.integrations.dart.client import DartCorpCodeRecord

    runtime = _runtime(tmp_path)
    _bridge(runtime.workspace.bronze_root)
    ctx = _ctx(runtime, _provider())

    assert dict(corp_code_bridge(ctx)) == {CORP: TICKER}

    indexed = index_corp_codes(
        [
            DartCorpCodeRecord(ticker=TICKER, corp_code=CORP, corp_name="Test"),
            DartCorpCodeRecord(ticker="abcdef", corp_code="00000001", corp_name="Unlisted"),
            DartCorpCodeRecord(ticker=TICKER, corp_code=CORP, corp_name="Repeat"),
            DartCorpCodeRecord(ticker="005930", corp_code="", corp_name="NoCode"),
        ]
    )
    assert indexed == {TICKER: CORP}
    with pytest.raises(PITDataError, match="multiple corp codes"):
        index_corp_codes(
            [
                DartCorpCodeRecord(ticker=TICKER, corp_code=CORP, corp_name="A"),
                DartCorpCodeRecord(ticker=TICKER, corp_code="00000002", corp_name="B"),
            ]
        )


# --- window helpers ----------------------------------------------------------


def test_disclosure_windows_tile_three_month_blocks() -> None:
    from src.data.jobs.dart import disclosure_windows

    assert disclosure_windows(date(2016, 1, 1), date(2016, 7, 15)) == [
        (date(2016, 1, 1), date(2016, 3, 31)),
        (date(2016, 4, 1), date(2016, 6, 30)),
        (date(2016, 7, 1), date(2016, 7, 15)),
    ]
    assert disclosure_windows(date(2016, 7, 15), date(2016, 1, 1)) == []


def test_required_periods_follow_scope_floor_and_publication() -> None:
    from src.data.jobs.dart import last_completed_kst_day, required_periods

    assert last_completed_kst_day(NOW) == date(2026, 9, 23)
    assert required_periods(fiscal_start="2016Q1", today=date(2016, 8, 1)) == ("2016Q1",)
    assert required_periods(fiscal_start="2016Q1", today=date(2016, 2, 1)) == ()
    assert required_periods(fiscal_start="2016Q1", today=date(2016, 11, 20)) == ("2016Q1", "2016Q2", "2016Q3")


def test_resolve_dart_job_registry() -> None:
    from src.core.pit import PITDataError
    from src.data.jobs.dart import DART_JOBS, resolve_dart_job

    assert set(DART_JOBS) == {"dart_disclosures", "dart_facts", "dividend_decisions"}
    assert resolve_dart_job("dart_facts").name == "dart_facts"
    with pytest.raises(PITDataError, match="unknown DART job"):
        resolve_dart_job("nope")


# --- disclosures job ----------------------------------------------------------


def _disclosure_fixtures(runtime) -> None:  # type: ignore[no-untyped-def]
    _bridge(runtime.workspace.bronze_root)
    _universe(runtime)


def test_disclosures_skip_per_corp_covered_windows(tmp_path: Path) -> None:
    from src.data.jobs.dart import DartDisclosuresJob

    runtime = _runtime(tmp_path)
    _disclosure_fixtures(runtime)
    _write_disclosure_page(
        runtime.workspace.bronze_root,
        _per_corp_page([_per_corp_record("20160330001234", "20160330", "사업보고서 (2015.12)")]),
    )
    ctx = _ctx(runtime, _provider())

    units = DartDisclosuresJob().pending(ctx)

    assert units
    assert {unit.payload["detail_type"] for unit in units} == {"A", "I001"}
    assert min(unit.payload["window_start"] for unit in units) >= "2019-07-01"
    assert all(unit.max_requests > 0 for unit in units)


def test_open_window_stays_pending_after_run(tmp_path: Path) -> None:
    from src.data.jobs.dart import DartDisclosuresJob
    from src.data.jobs.runner import run_job

    runtime = _runtime(tmp_path)
    _disclosure_fixtures(runtime)
    collector = _Collector()
    ctx = _ctx(runtime, _provider(), collector=collector)
    spec = DartDisclosuresJob()

    before = spec.pending(ctx)
    open_keys = {unit.natural_key for unit in before if unit.payload["window_end"] >= "2026-09-24"}
    assert open_keys

    emitted: list[dict] = []
    report = run_job(spec, ctx, chunk_size=500, max_chunks=None, dry_run=False, emit=emitted.append)

    assert report.status == "complete"
    assert collector.list_calls
    after = {unit.natural_key for unit in spec.pending(ctx)}
    assert open_keys <= after
    assert [line["phase"] for line in emitted] == ["plan", "chunk", "done"]


def test_disclosure_window_records_round_trip(tmp_path: Path) -> None:
    from src.data.jobs.dart import DartDisclosuresJob

    runtime = _runtime(tmp_path)
    _disclosure_fixtures(runtime)
    rows = [
        {"rcept_no": "20160330001234", "rcept_dt": "20160330", "corp_code": CORP, "report_nm": "사업보고서 (2015.12)", "rm": ""},
    ]
    collector = _Collector(windows={("2016-01-01", "2016-03-31", "A"): rows})
    ctx = _ctx(runtime, _provider(), collector=collector)
    unit = next(
        unit
        for unit in DartDisclosuresJob().pending(ctx)
        if unit.natural_key == "A:2016-01-01..2016-03-31"
    )

    (payload,) = DartDisclosuresJob().fetch(ctx, [unit])

    assert payload.status.value == "success"
    assert json.loads(payload.payload)["records"] == rows


def test_disclosures_quota_block_stops_run(tmp_path: Path) -> None:
    from src.data.jobs.dart import DartDisclosuresJob
    from src.data.jobs.runner import run_job
    from src.integrations.errors import ProviderQuotaExhaustedError

    runtime = _runtime(tmp_path)
    _disclosure_fixtures(runtime)
    ctx = _ctx(
        runtime, _provider(), collector=_Collector(windows={}), now=lambda: NOW,
    )
    spec = DartDisclosuresJob()
    units = [unit for unit in spec.pending(ctx) if unit.payload["window_start"] >= "2019-07-01"][:2]
    assert len(units) == 2

    class _Blocked(_Collector):
        def list_disclosures(self, start, end, *, detail_type=None):  # type: ignore[no-untyped-def]
            raise ProviderQuotaExhaustedError("blocked")

    ctx_blocked = _ctx(runtime, _provider(), collector=_Blocked())
    report = run_job(spec, ctx_blocked, chunk_size=500, max_chunks=None, dry_run=False, emit=lambda _p: None)

    assert report.status == "quota_blocked"
    assert report.done == 0


def test_per_corp_coverage_ignores_malformed_pages(tmp_path: Path) -> None:
    from src.data.jobs.dart import _per_corp_coverage, _window_covered

    bronze = tmp_path / "bronze"
    disclosures = bronze / "disclosures"
    (disclosures / "bad-json").mkdir(parents=True)
    (disclosures / "bad-json" / "payload.json").write_text("{broken", encoding="utf-8")
    (disclosures / "non-dict").mkdir()
    (disclosures / "non-dict" / "payload.json").write_text("[1]", encoding="utf-8")
    _write_disclosure_page(bronze, {"records": [], "start": "not-a-date", "end": "2019-06-30", "corp_code": CORP})
    _write_disclosure_page(bronze, {"records": [], "start": "2019-06-30", "end": "2019-01-01", "corp_code": CORP})
    _write_disclosure_page(bronze, {"records": [], "start": "2016-01-01", "end": "2016-03-31"})

    coverage = _per_corp_coverage(bronze)

    assert coverage == {}
    assert _window_covered(coverage, frozenset({CORP}), date(2016, 1, 1), date(2016, 3, 31)) is False
    _write_disclosure_page(
        bronze, _per_corp_page([], start="2015-01-01", end="2019-06-30"),
    )
    coverage = _per_corp_coverage(bronze)
    assert _window_covered(coverage, frozenset({CORP}), date(2016, 1, 1), date(2016, 3, 31)) is True
    assert _window_covered(coverage, frozenset({CORP}), date(2019, 7, 1), date(2019, 9, 30)) is False


def test_legacy_dividend_cache_imports_once_then_deletes(tmp_path: Path) -> None:
    from src.data.jobs.dart import DartDisclosuresJob, import_legacy_dividend_cache

    runtime = _runtime(tmp_path)
    _disclosure_fixtures(runtime)
    cache = runtime.workspace.state_root / "dividend_decision_lists.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"rcept_no": "20190814001234", "rcept_dt": "20190814", "corp_code": CORP, "report_nm": "현금배당결정", "rm": ""}]
    cache.write_text(json.dumps({"2019-07-01..2019-09-30": rows}), encoding="utf-8")
    ctx = _ctx(runtime, _provider())

    assert import_legacy_dividend_cache(ctx) == 1
    assert not cache.exists()

    units = DartDisclosuresJob().pending(ctx)
    assert "I001:2019-07-01..2019-09-30" not in {unit.natural_key for unit in units}
    assert import_legacy_dividend_cache(ctx) == 0


def test_legacy_dividend_cache_rejects_malformed_cache(tmp_path: Path) -> None:
    from src.core.pit import PITDataError
    from src.data.jobs.dart import import_legacy_dividend_cache

    runtime = _runtime(tmp_path)
    _disclosure_fixtures(runtime)
    cache = runtime.workspace.state_root / "dividend_decision_lists.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("not-json", encoding="utf-8")
    with pytest.raises(PITDataError, match="unreadable"):
        import_legacy_dividend_cache(_ctx(runtime, _provider()))
    cache.write_text(json.dumps(["not", "a", "mapping"]), encoding="utf-8")
    with pytest.raises(PITDataError, match="must hold a mapping"):
        import_legacy_dividend_cache(_ctx(runtime, _provider()))
    cache.write_text(json.dumps({"bogus-window": []}), encoding="utf-8")
    with pytest.raises(PITDataError, match="invalid window"):
        import_legacy_dividend_cache(_ctx(runtime, _provider()))


# --- facts job -----------------------------------------------------------------


def _facts_fixtures(runtime, records: list) -> None:  # type: ignore[no-untyped-def]
    _bridge(runtime.workspace.bronze_root)
    _universe(runtime)
    _write_disclosure_page(runtime.workspace.bronze_root, _per_corp_page(records))


def test_facts_read_both_disclosure_page_shapes(tmp_path: Path) -> None:
    from src.data.jobs.dart import DartFactsJob

    runtime = _runtime(tmp_path)
    _bridge(runtime.workspace.bronze_root)
    _universe(runtime)
    _write_disclosure_page(
        runtime.workspace.bronze_root,
        _per_corp_page([_per_corp_record("20170330001234", "20170330", "사업보고서 (2016.12)")]),
    )
    _write_disclosure_page(
        runtime.workspace.bronze_root,
        {
            "detail_type": "A",
            "start": "2016-04-01",
            "end": "2016-06-30",
            "records": [_per_corp_record("20160516001235", "20160516", "분기보고서 (2016.03)")],
        },
    )
    ctx = _ctx(runtime, _provider())

    units = DartFactsJob().pending(ctx)

    assert {unit.natural_key for unit in units} == {f"{CORP}:2016:11011", f"{CORP}:2016:11013"}
    assert all(unit.max_requests == 3 for unit in units)


def test_facts_latest_filing_wins(tmp_path: Path) -> None:
    from src.data.jobs.dart import DartFactsJob

    runtime = _runtime(tmp_path)
    _facts_fixtures(
        runtime,
        [
            _per_corp_record("20160516001235", "20160516", "분기보고서 (2016.03)"),
            _per_corp_record("20160520001236", "20160520", "[기재정정]분기보고서 (2016.03)"),
        ],
    )
    ctx = _ctx(runtime, _provider())

    (unit,) = DartFactsJob().pending(ctx)

    assert unit.natural_key == f"{CORP}:2016:11013"
    assert unit.payload["filing_id"] == "20160520001236"


def test_facts_unavailable_answers_are_retried_success_is_not(tmp_path: Path) -> None:
    from src.data.jobs.dart import DartFactsJob
    from src.data.receipt_catalog import EvidenceStatus
    from src.core.pit import EvidenceKind
    from src.data.scoped_ingestion import ScopedRawPayload

    runtime = _runtime(tmp_path)
    _facts_fixtures(runtime, [_per_corp_record("20160516001235", "20160516", "분기보고서 (2016.03)")])
    ctx = _ctx(runtime, _provider())
    (unit,) = DartFactsJob().pending(ctx)
    assert unit.natural_key == f"{CORP}:2016:11013"
    ctx.writer.persist(
        ScopedRawPayload(
            kind=EvidenceKind.FINANCIAL_FACTS,
            source="financial_facts",
            natural_key=unit.natural_key,
            as_of=date(2016, 5, 16),
            fiscal_period="2016Q1",
            status=EvidenceStatus.PROVIDER_UNAVAILABLE,
            payload=b"{}",
            retrieved_at=NOW,
            source_label="test:unavailable",
        )
    )

    assert [item.natural_key for item in DartFactsJob().pending(ctx)] == [unit.natural_key]

    ctx.writer.persist(
        ScopedRawPayload(
            kind=EvidenceKind.FINANCIAL_FACTS,
            source="financial_facts",
            natural_key=unit.natural_key,
            as_of=date(2016, 5, 16),
            fiscal_period="2016Q1",
            status=EvidenceStatus.SUCCESS,
            payload=json.dumps({"records": [{"fact": "sales"}]}).encode("utf-8"),
            retrieved_at=datetime(2026, 9, 24, 4, 0, tzinfo=UTC),
            source_label="test:success",
        )
    )

    assert DartFactsJob().pending(ctx) == []


def test_facts_fetch_converts_pages_and_surfaces_quota_and_transport(tmp_path: Path) -> None:
    from src.data.jobs.dart import DartFactsJob
    from src.integrations.dart.client import ProviderQuotaExhaustedError, ProviderRetryableError

    runtime = _runtime(tmp_path)
    _facts_fixtures(runtime, [_per_corp_record("20160516001235", "20160516", "분기보고서 (2016.03)")])
    identity = {
        "corp_code": CORP, "filing_id": "20160516001235", "rcept_no": "20160516001235",
        "biz_year": "2016", "reprt_code": "11013", "fs_div": "CFS",
        "published_at": "2016-05-16", "ticker": TICKER,
    }
    archive = b"PK\x03\x04legacy-zip-bytes"
    collector = _Collector(
        fact_pages=[
            _standard_fact_page(identity),
            {**_standard_fact_page(identity, status="013", records=[]), "source_kind": "legacy_document",
             "raw_archive": archive, "diagnostics": ("legacy_fallback",)},
        ]
    )
    ctx = _ctx(runtime, _provider(), collector=collector)
    (unit,) = DartFactsJob().pending(ctx)

    payloads = DartFactsJob().fetch(ctx, [unit])

    assert len(payloads) == 2
    assert payloads[0].natural_key == f"{CORP}:2016:11013"
    assert payloads[0].fiscal_period == "2016Q1"
    assert payloads[0].status.value == "success"

    blocked_ctx = _ctx(
        runtime, _provider(),
        collector=_Collector(fact_pages=[{"source_kind": "blocked", "status": "020", "identity": identity}]),
    )
    with pytest.raises(ProviderQuotaExhaustedError):
        DartFactsJob().fetch(blocked_ctx, [unit])

    transport_ctx = _ctx(
        runtime, _provider(),
        collector=_Collector(
            fact_pages=[{"source_kind": "unavailable", "status": "900", "identity": identity,
                         "diagnostics": ("DART status 900: throttled",)}]
        ),
    )
    with pytest.raises(ProviderRetryableError):
        DartFactsJob().fetch(transport_ctx, [unit])

    empty_ctx = _ctx(
        runtime, _provider(),
        collector=_Collector(
            fact_pages=[{"source_kind": "unavailable", "status": "013", "identity": identity,
                         "diagnostics": ("empty_archive",), **identity}]
        ),
    )
    (unavailable,) = DartFactsJob().fetch(empty_ctx, [unit])
    assert unavailable.status.value == "provider_unavailable"


def test_transport_failure_page_classifier() -> None:
    from src.integrations.dart.xbrl import is_transport_failure_page

    assert is_transport_failure_page({"source_kind": "unavailable", "diagnostics": ("transport failed",)}) is True
    assert is_transport_failure_page({"source_kind": "unavailable", "diagnostics": ("dart_error: nope",)}) is False
    assert is_transport_failure_page({"source_kind": "opendart_standard", "diagnostics": ()}) is False


def test_xbrl_facts_falls_back_to_private_validated_seam() -> None:
    from src.integrations.dart.xbrl import DartXbrlCollector

    seen: list[tuple] = []

    class _LegacyClient:
        def _request_validated(self, endpoint, params):  # type: ignore[no-untyped-def]
            seen.append((endpoint, params))
            return {"status": "000", "list": [{"account_id": "x", "account_nm": "y"}]}

    collector = DartXbrlCollector(api_key="k", client=_LegacyClient(), min_interval=0.2, max_workers=1)
    identity = {
        "corp_code": CORP, "filing_id": "20160516001235", "biz_year": "2016",
        "reprt_code": "11013", "fs_div": "CFS",
    }

    pages = list(collector.fetch_xbrl_facts((identity,)))

    assert len(pages) == 1
    assert seen[0][0] == "fnlttSinglAcntAll.json"


def test_fact_source_archive_failure_is_terminal() -> None:
    from src.core.pit import PITDataError
    from src.integrations.dart.xbrl import DartXbrlCollector

    def _empty(endpoint, params):  # type: ignore[no-untyped-def]
        return {"status": "013"}

    def _boom(endpoint, params):  # type: ignore[no-untyped-def]
        raise RuntimeError("socket gone")

    collector = DartXbrlCollector(
        api_key="k", request_json=_empty, request_bytes=_boom, min_interval=0.2, max_workers=1
    )
    identity = {
        "corp_code": CORP, "filing_id": "20160516001235", "rcept_no": "20160516001235",
        "biz_year": "2016", "reprt_code": "11013", "fs_div": "CFS", "published_at": "2016-05-16",
    }

    with pytest.raises(PITDataError, match="document archive failed"):
        list(collector.fetch_financial_fact_sources((identity,)))


# --- dividend job -----------------------------------------------------------------


def _decision_record(rcept_no: str, rcept_dt: str) -> dict:
    return _per_corp_record(rcept_no, rcept_dt, "현금배당결정")


def test_dividend_reads_both_page_shapes(tmp_path: Path) -> None:
    from src.data.jobs.dart import DividendDecisionsJob

    runtime = _runtime(tmp_path)
    _bridge(runtime.workspace.bronze_root)
    _universe(runtime)
    _write_disclosure_page(
        runtime.workspace.bronze_root, _per_corp_page([_decision_record("20170330876543", "20170330")])
    )
    _write_disclosure_page(
        runtime.workspace.bronze_root,
        {
            "detail_type": "I001",
            "start": "2019-07-01",
            "end": "2019-09-30",
            "records": [_decision_record("20190814001234", "20190814")],
        },
    )
    ctx = _ctx(runtime, _provider())

    units = DividendDecisionsJob().pending(ctx)

    assert [unit.natural_key for unit in units] == ["20170330876543", "20190814001234"]
    assert all(unit.max_requests == 1 for unit in units)


def test_job_health_checks_delegate_to_collector(tmp_path: Path) -> None:
    from src.data.jobs.dart import DartDisclosuresJob, DartFactsJob, DividendDecisionsJob

    runtime = _runtime(tmp_path)
    _bridge(runtime.workspace.bronze_root)
    _universe(runtime)
    collector = _Collector(healthy=True)
    ctx = _ctx(runtime, _provider(), collector=collector)

    for spec in (DartDisclosuresJob(), DartFactsJob(), DividendDecisionsJob()):
        spec.health_check(ctx)


def test_dividend_pending_ignores_malformed_records(tmp_path: Path) -> None:
    from src.core.pit import PITDataError
    from src.data.jobs.dart import DividendDecisionsJob

    runtime = _runtime(tmp_path)
    _bridge(runtime.workspace.bronze_root)
    _universe(runtime)
    bronze = runtime.workspace.bronze_root
    _write_disclosure_page(
        bronze,
        {
            "records": [
                "not-a-record",
                {"corp_code": "99999999", "rcept_no": "20170330876544", "rcept_dt": "20170330", "report_nm": "현금배당결정"},
                {"corp_code": CORP, "rcept_no": "20170330876545", "rcept_dt": "baddate1", "report_nm": "현금배당결정"},
                {"corp_code": CORP, "rcept_no": "20170330876546", "rcept_dt": "20300101", "report_nm": "현금배당결정"},
                {"corp_code": CORP, "rcept_no": "20170330876547", "rcept_dt": "20170330", "report_nm": "사업보고서 (2016.12)"},
                {"corp_code": CORP, "rcept_no": "", "rcept_dt": "20170330", "report_nm": "현금배당결정"},
                _decision_record("20170330876543", "20170330"),
            ],
            "start": "2015-01-01",
            "end": "2019-06-30",
            "corp_code": CORP,
        },
    )
    _write_disclosure_page(bronze, {"detail_type": "I001", "start": "2019-07-01", "end": "2019-09-30", "records": []})
    _write_disclosure_page(bronze, [1, 2, 3])
    _write_disclosure_page(bronze, {"records": {"rcept_no": "20170330876548"}})
    ctx = _ctx(runtime, _provider())

    units = DividendDecisionsJob().pending(ctx)

    assert [unit.natural_key for unit in units] == ["20170330876543"]

    unreadable = bronze / "disclosures" / ("e" * 64)
    (unreadable / "payload.json").parent.mkdir(parents=True, exist_ok=True)
    (unreadable / "payload.json").mkdir()
    with pytest.raises(PITDataError, match="unreadable"):
        DividendDecisionsJob().pending(ctx)


def test_dividend_014_body_is_empty_and_not_retried(tmp_path: Path) -> None:
    from src.data.jobs.dart import DividendDecisionsJob
    from src.data.jobs.runner import run_job

    runtime = _runtime(tmp_path)
    _bridge(runtime.workspace.bronze_root)
    _universe(runtime)
    _write_disclosure_page(
        runtime.workspace.bronze_root, _per_corp_page([_decision_record("20170330876543", "20170330")])
    )
    absent = b'<?xml version="1.0"?><response><status>014</status><message>absent</message></response>'
    collector = _Collector(archives={"20170330876543": absent})
    ctx = _ctx(runtime, _provider(), collector=collector)
    spec = DividendDecisionsJob()
    (unit,) = spec.pending(ctx)

    (payload,) = spec.fetch(ctx, [unit])

    assert payload.status.value == "empty"
    envelope = json.loads(payload.payload)
    assert envelope["document_status"] == "unavailable"

    report = run_job(spec, ctx, chunk_size=10, max_chunks=None, dry_run=False, emit=lambda _p: None)
    assert (report.status, report.done, report.pending_left) == ("complete", 1, 0)
    assert spec.pending(ctx) == []
    answered = ctx.catalog.latest(source="opendart:dividend_decision", natural_keys={"20170330876543"})
    assert answered["20170330876543"].status.value == "empty"


def test_dividend_zip_envelope_matches_registered_format(tmp_path: Path) -> None:
    import base64

    from src.data.dividend_events import _iter_decision_envelopes
    from src.data.jobs.dart import DividendDecisionsJob

    runtime = _runtime(tmp_path)
    _bridge(runtime.workspace.bronze_root)
    _universe(runtime)
    _write_disclosure_page(
        runtime.workspace.bronze_root, _per_corp_page([_decision_record("20170330876543", "20170330")])
    )
    archive = b"PK\x03\x04dividend-zip-bytes"
    ctx = _ctx(runtime, _provider(), collector=_Collector(archives={"20170330876543": archive}))
    spec = DividendDecisionsJob()
    (unit,) = spec.pending(ctx)

    (payload,) = spec.fetch(ctx, [unit])

    assert payload.status.value == "success"
    envelope = json.loads(payload.payload)
    assert set(envelope) == {"rcept_no", "corp_code", "received_on", "report_nm", "archive_b64", "archive_sha256"}
    assert base64.b64decode(envelope["archive_b64"]) == archive
    ctx.writer.persist(payload)
    assert len(_iter_decision_envelopes(runtime.workspace.bronze_root)) == 1
