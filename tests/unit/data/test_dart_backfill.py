import dataclasses
import json
from datetime import date

import pytest


def test_backfill_plan_requires_exact_ticker_to_corp_code_bridge() -> None:
    from datetime import UTC, date, datetime
    import polars as pl
    from src.data.dart_backfill import build_dart_historical_backfill_plan
    from src.integrations.dart.client import DartCorpCodeRecord

    master = pl.DataFrame({"instrument_id": ["KRX:005930", "KRX:000001"], "ticker": ["005930", "000001"], "share_class": ["common", "common"], "valid_from": [datetime(2010, 1, 1, tzinfo=UTC)] * 2, "valid_to": [None, None], "available_at": [datetime(2010, 1, 1, tzinfo=UTC)] * 2})
    records = (DartCorpCodeRecord(ticker="005930", corp_code="00126380", corp_name="A"),)

    plan = build_dart_historical_backfill_plan(security_master=master, corp_code_records=records, validation_start=date(2016, 1, 4), validation_end=date(2016, 12, 29), corp_code_receipt_hash="a" * 64)

    assert plan.required_periods == ("2014Q3", "2014Q4", "2015Q1", "2015Q2", "2015Q3")
    assert dict(plan.ticker_by_corp_code) == {"00126380": "005930"}
    assert plan.unresolved_tickers == ("000001",)


def test_backfill_batch_writes_deterministic_plan_artifact_before_fact_collection(tmp_path, monkeypatch) -> None:
    from datetime import UTC, date, datetime
    import json
    import polars as pl
    from src.data.dart_backfill import DartHistoricalBackfillRequest, run_dart_historical_backfill_batch
    from src.integrations.dart.client import DartCorpCodeRecord

    class Collector:
        def fetch_corp_code_records(self):
            return (DartCorpCodeRecord(ticker="005930", corp_code="00126380", corp_name="A"),)
    master = pl.DataFrame({"instrument_id": ["KRX:005930"], "ticker": ["005930"], "share_class": ["common"], "valid_from": [datetime(2010, 1, 1, tzinfo=UTC)], "valid_to": [None], "available_at": [datetime(2010, 1, 1, tzinfo=UTC)]})
    monkeypatch.setattr("src.data.dart_backfill._load_security_master", lambda _root, **_kwargs: master)
    monkeypatch.setattr("src.data.dart_backfill._persist_corp_code_receipt", lambda **_kwargs: "c" * 64)
    monkeypatch.setattr("src.data.dart_backfill.collect_dart_disclosures", lambda **_kwargs: None)
    monkeypatch.setattr("src.data.dart_backfill.DartXbrlCollector.filing_identities_from_bronze", lambda *_args, **_kwargs: ())

    request = DartHistoricalBackfillRequest(bronze_root=tmp_path / "bronze", artifact_root=tmp_path / "artifacts", silver_root=tmp_path / "silver", validation_start=date(2016, 1, 4), validation_end=date(2016, 12, 29), retrieved_at=datetime(2016, 1, 4, tzinfo=UTC), offset=0, limit=1)
    plan = run_dart_historical_backfill_batch(request=request, dart=Collector())

    payload = json.loads((tmp_path / "artifacts" / "dart_backfill" / f"{plan.plan_id}.json").read_text())
    assert payload["ticker_by_corp_code"] == {"00126380": "005930"}
    assert payload["required_periods"] == ["2014Q3", "2014Q4", "2015Q1", "2015Q2", "2015Q3"]


def test_dedupe_endpoint_identities_keeps_latest_correction() -> None:
    from src.data.dart_backfill import _dedupe_endpoint_identities

    identities = (
        {"corp_code": "001", "biz_year": "2015", "reprt_code": "11013", "fs_div": "CFS", "filing_id": "F1", "published_at": "2015-05-15"},
        {"corp_code": "001", "biz_year": "2015", "reprt_code": "11013", "fs_div": "CFS", "filing_id": "F2", "published_at": "2015-06-01"},
        {"corp_code": "001", "biz_year": "2015", "reprt_code": "11012", "fs_div": "CFS", "filing_id": "F3", "published_at": "2015-08-15"},
    )

    result = _dedupe_endpoint_identities(identities)

    assert [item["filing_id"] for item in result] == ["F3", "F2"]


def test_missing_facts_batch_selects_only_uncovered_retained_identity(tmp_path, monkeypatch) -> None:
    from datetime import UTC, datetime
    import json
    from types import SimpleNamespace

    from src.data.dart_backfill import DartMissingFactsRequest, run_dart_missing_facts_batch

    artifact = tmp_path / "backfill.json"
    artifact.write_text(json.dumps({"ticker_by_corp_code": {"00126380": "005930"}, "required_periods": ["2015Q1"], "validation_start": "2016-01-04"}))
    bronze = tmp_path / "bronze"
    (bronze / "financial_facts" / "a").mkdir(parents=True)
    (bronze / "financial_facts" / "a" / "payload.json").write_text(json.dumps({"corp_code": "00126380", "biz_year": "2015", "reprt_code": "11011", "status": "000", "records": [{"x": 1}]}))
    identities = (
        {"corp_code": "00126380", "ticker": "005930", "fiscal_period": "2015Q1", "biz_year": "2015", "reprt_code": "11013", "fs_div": "CFS", "filing_id": "F1", "published_at": "2015-05-15"},
        {"corp_code": "00126380", "ticker": "005930", "fiscal_period": "2015Q1", "biz_year": "2015", "reprt_code": "11011", "fs_div": "CFS", "filing_id": "F2", "published_at": "2016-03-30"},
    )
    monkeypatch.setattr("src.data.dart_backfill.DartXbrlCollector.filing_identities_from_bronze", lambda *_args, **_kwargs: identities)
    captured = {}
    def collect(**kwargs):
        captured["identities"] = kwargs["identities"]
        return SimpleNamespace(content_hash="h")

    monkeypatch.setattr("src.data.dart_backfill.collect_dart_financial_facts", collect)
    plan = run_dart_missing_facts_batch(request=DartMissingFactsRequest(bronze, tmp_path / "artifacts", artifact, datetime(2026, 9, 21, tzinfo=UTC), 0, 1), dart=object())
    assert plan.candidate_count == 1
    assert captured["identities"][0]["filing_id"] == "F1"


def test_build_single_account_request_plan_batches_under_quota() -> None:
    from src.data.dart_backfill import build_single_account_request_plan

    # Given: 2 distinct corp codes (one duplicated) across 2 fiscal years.
    batches = build_single_account_request_plan(
        corp_codes=('00126380', '00413046', '00126380'),
        first_fiscal_year=2016,
        last_fiscal_year=2017,
        daily_call_budget=10,
    )

    # Then: 2 corps x 2 years x 4 reports x 2 bases = 32 calls in batches of 10.
    flat = [request for batch in batches for request in batch]
    assert len(flat) == 32
    assert all(len(batch) <= 10 for batch in batches)
    assert [len(batch) for batch in batches] == [10, 10, 10, 2]
    assert flat[0].corp_code == '00126380'
    assert flat[0].biz_year == 2016
    assert flat[0].reprt_code == '11013'
    assert flat[0].fs_div == 'CFS'
    assert flat[1].fs_div == 'OFS'
    assert {request.fs_div for request in flat} == {'CFS', 'OFS'}

    # And: the plan is deterministic for identical inputs.
    assert build_single_account_request_plan(
        corp_codes=('00413046', '00126380'),
        first_fiscal_year=2016,
        last_fiscal_year=2017,
        daily_call_budget=10,
    ) == batches


def test_build_single_account_request_plan_validates_inputs() -> None:
    import pytest

    from src.data.dart_backfill import (
        SingleAccountBackfillRequest,
        build_single_account_request_plan,
    )

    # When/Then: every malformed input fails closed.
    with pytest.raises(ValueError, match='corp_codes'):
        build_single_account_request_plan(
            corp_codes=(), first_fiscal_year=2016, last_fiscal_year=2016
        )
    with pytest.raises(ValueError, match='corp_code'):
        build_single_account_request_plan(
            corp_codes=('123',), first_fiscal_year=2016, last_fiscal_year=2016
        )
    with pytest.raises(ValueError, match='fiscal_year'):
        build_single_account_request_plan(
            corp_codes=('00126380',), first_fiscal_year=2018, last_fiscal_year=2016
        )
    with pytest.raises(ValueError, match='daily_call_budget'):
        build_single_account_request_plan(
            corp_codes=('00126380',),
            first_fiscal_year=2016,
            last_fiscal_year=2016,
            daily_call_budget=0,
        )

    # And: CFS-only planning halves the call count.
    only_cfs = build_single_account_request_plan(
        corp_codes=('00126380',),
        first_fiscal_year=2016,
        last_fiscal_year=2016,
        include_separate_fallback=False,
    )
    flat = [request for batch in only_cfs for request in batch]
    assert len(flat) == 4
    assert {request.fs_div for request in flat} == {'CFS'}

    # And: the request itself rejects an invalid accounting basis.
    with pytest.raises(ValueError, match='fs_div'):
        SingleAccountBackfillRequest(
            corp_code='00126380', biz_year=2016, reprt_code='11011', fs_div='XXX'
        )


def test_single_account_backfill_request_rejects_each_malformed_field() -> None:
    import pytest

    from src.data.dart_backfill import SingleAccountBackfillRequest

    # Given: one valid request shape to mutate a single field at a time.
    valid = {"corp_code": "00126380", "biz_year": 2016, "reprt_code": "11011", "fs_div": "CFS"}

    # When/Then: every identity field fails closed on its own.
    with pytest.raises(ValueError, match="corp_code"):
        SingleAccountBackfillRequest(**{**valid, "corp_code": "126380"})
    with pytest.raises(ValueError, match="fiscal_year"):
        SingleAccountBackfillRequest(**{**valid, "biz_year": 1999})
    with pytest.raises(ValueError, match="reprt_code"):
        SingleAccountBackfillRequest(**{**valid, "reprt_code": "11015"})
    with pytest.raises(ValueError, match="fs_div"):
        SingleAccountBackfillRequest(**{**valid, "fs_div": "cfs"})

    # And: the valid shape constructs.
    assert SingleAccountBackfillRequest(**valid).fs_div == "CFS"


def test_load_security_master_selects_latest_dataset_only(tmp_path) -> None:
    from datetime import UTC, datetime

    from src.data.dart_backfill import _load_security_master
    from src.data.schemas import SilverTable
    from src.data.silver import SilverStore, complete_minimal_fixture

    # Given: two certified security_master publishes at different decision times
    # (distinct content, so distinct dataset directories under the table root).
    store = SilverStore(tmp_path / "silver")
    first_time = datetime(2024, 1, 3, tzinfo=UTC)
    tables_1, _, report_1 = complete_minimal_fixture(decision_time=first_time)
    store.materialize_all(tables_1, report=report_1, decision_time=first_time)

    second_time = datetime(2024, 1, 10, tzinfo=UTC)
    tables_2, _, report_2 = complete_minimal_fixture(decision_time=second_time)
    store.materialize_all(tables_2, report=report_2, decision_time=second_time)

    dataset_dirs = list((tmp_path / "silver" / "security_master").iterdir())
    assert len(dataset_dirs) == 2, "fixture must produce two distinct dataset versions"

    # When
    read_time = datetime(2024, 2, 1, tzinfo=UTC)
    master = _load_security_master(tmp_path / "silver", decision_time=read_time)

    # Then: exactly one version's rows are returned, never both concatenated.
    assert master.height == tables_2[SilverTable.SECURITY_MASTER].height
    assert master.height == 1


def test_backfill_batch_uses_wide_fixed_disclosure_fetch_range_and_narrow_identity_range(tmp_path, monkeypatch) -> None:
    from datetime import UTC, date, datetime
    import polars as pl
    from src.data.dart_backfill import DartHistoricalBackfillRequest, run_dart_historical_backfill_batch
    from src.data.research_period import OPENDART_FIRST_FISCAL_YEAR
    from src.integrations.dart.client import DartCorpCodeRecord

    class Collector:
        def fetch_corp_code_records(self):
            return (DartCorpCodeRecord(ticker="005930", corp_code="00126380", corp_name="A"),)

    master = pl.DataFrame({
        "instrument_id": ["KRX:005930"], "ticker": ["005930"], "share_class": ["common"],
        "valid_from": [datetime(2010, 1, 1, tzinfo=UTC)], "valid_to": [None],
        "available_at": [datetime(2010, 1, 1, tzinfo=UTC)],
    })
    monkeypatch.setattr("src.data.dart_backfill._load_security_master", lambda _root, **_kwargs: master)
    monkeypatch.setattr("src.data.dart_backfill._persist_corp_code_receipt", lambda **_kwargs: "c" * 64)

    captured_disclosure: dict[str, object] = {}

    def fake_collect_dart_disclosures(**kwargs):
        captured_disclosure["start"] = kwargs["start"]
        captured_disclosure["end"] = kwargs["end"]
        return None

    monkeypatch.setattr("src.data.dart_backfill.collect_dart_disclosures", fake_collect_dart_disclosures)

    captured_identity: dict[str, object] = {}

    def fake_filing_identities_from_bronze(*_args, **kwargs):
        captured_identity["start"] = kwargs["start"]
        captured_identity["end"] = kwargs["end"]
        return ()

    monkeypatch.setattr(
        "src.data.dart_backfill.DartXbrlCollector.filing_identities_from_bronze",
        fake_filing_identities_from_bronze,
    )

    validation_start = date(2017, 4, 4)
    retrieved_at = datetime(2017, 4, 4, 9, 0, tzinfo=UTC)
    request = DartHistoricalBackfillRequest(
        bronze_root=tmp_path / "bronze", artifact_root=tmp_path / "artifacts", silver_root=tmp_path / "silver",
        validation_start=validation_start, validation_end=validation_start, retrieved_at=retrieved_at,
        offset=0, limit=1,
    )
    run_dart_historical_backfill_batch(request=request, dart=Collector())

    # Then: disclosure FETCH range is the wide fixed floor..retrieved_at.date() (cache-friendly).
    assert captured_disclosure["start"] == date(OPENDART_FIRST_FISCAL_YEAR - 1, 1, 1)
    assert captured_disclosure["end"] == retrieved_at.date()
    # And: identity EXTRACTION keeps the narrow, PIT-bounded per-window range unchanged.
    assert captured_identity["start"] == date(validation_start.year - 2, 1, 1)
    assert captured_identity["end"] == validation_start


def _scoped_runtime(tmp_path):  # type: ignore[no-untyped-def]
    from pathlib import Path as _Path

    from src.data.runtime import load_data_runtime

    return load_data_runtime(scope_config=_Path("config/research/kr_swing_2019_v1.toml"), data_root=tmp_path / "data")


def _scoped_catalog(runtime):  # type: ignore[no-untyped-def]
    from src.data.receipt_catalog import ReceiptCatalog

    return ReceiptCatalog(runtime.workspace.bronze_root / "catalog")


def _filing(corp="00126380", filing="F1", biz="2019", reprt="11013", published="2019-05-15"):
    return {
        "corp_code": corp, "filing_id": filing, "biz_year": biz, "reprt_code": reprt,
        "published_at": published, "ticker": "005930",
    }


def test_scoped_batch_selects_catalog_gap(tmp_path) -> None:
    from src.data.dart_backfill import build_scoped_dart_fact_batch

    runtime = _scoped_runtime(tmp_path)
    batch = build_scoped_dart_fact_batch(
        runtime=runtime, catalog=_scoped_catalog(runtime),
        filing_identities=[_filing()], offset=0, limit=20,
    )

    assert batch.scope_hash == runtime.scope.content_hash
    assert batch.plan_id.startswith("dart-facts-")
    assert [dict(item)["filing_id"] for item in batch.identities] == ["F1"]
    assert batch.missing_without_filing == ()
    assert batch.estimated_request_ceiling == 3


def test_scoped_batch_excludes_pre_2019_filing(tmp_path) -> None:
    from src.data.dart_backfill import build_scoped_dart_fact_batch

    runtime = _scoped_runtime(tmp_path)
    batch = build_scoped_dart_fact_batch(
        runtime=runtime, catalog=_scoped_catalog(runtime),
        filing_identities=[_filing(filing="F0", biz="2018", reprt="11011", published="2019-03-30")],
        offset=0, limit=20,
    )

    assert batch.identities == ()
    assert batch.missing_without_filing == ()
    assert batch.estimated_request_ceiling == 0


def test_scoped_batch_reports_missing_without_discovery(tmp_path, monkeypatch) -> None:
    from src.data.dart_backfill import build_scoped_dart_fact_batch
    from src.integrations.dart import xbrl as dart_xbrl

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("disclosure-list discovery must not run")

    monkeypatch.setattr(dart_xbrl.DartXbrlCollector, "filing_identities_from_bronze", _forbidden)
    runtime = _scoped_runtime(tmp_path)
    incomplete = {"corp_code": "00126380", "biz_year": "2019", "reprt_code": "11013", "ticker": "005930"}
    batch = build_scoped_dart_fact_batch(
        runtime=runtime, catalog=_scoped_catalog(runtime),
        filing_identities=[incomplete, _filing()], offset=0, limit=20,
    )

    assert [item.natural_key for item in batch.missing_without_filing] == ["00126380:2019:11013"]
    assert batch.missing_without_filing[0].required is True
    assert [dict(item)["filing_id"] for item in batch.identities] == ["F1"]


def test_scoped_batch_quota_ceiling_limits_identities(tmp_path) -> None:
    from src.data.dart_backfill import build_scoped_dart_fact_batch
    from src.data.research_scope import CollectionBudget
    from src.data.runtime import DataRuntime
    from src.data.workspace import build_workspace

    runtime = _scoped_runtime(tmp_path)
    capped_scope = runtime.scope.model_copy(
        update={"collection": CollectionBudget(dart_daily_budget=1200, dart_batch_identities=500)}
    )
    capped = DataRuntime(scope=capped_scope, workspace=build_workspace(data_root=tmp_path / "data", scope=capped_scope))
    identities = [
        _filing(corp=f"{index:08d}", filing=f"F{index}", published="2019-05-15") for index in range(500)
    ]
    batch = build_scoped_dart_fact_batch(
        runtime=capped, catalog=_scoped_catalog(capped),
        filing_identities=identities, offset=0, limit=500,
    )

    assert len(batch.identities) == 266
    assert batch.estimated_request_ceiling == 798

    tiny_scope = runtime.scope.model_copy(
        update={"collection": CollectionBudget(dart_daily_budget=16000, dart_batch_identities=2)}
    )
    tiny = DataRuntime(scope=tiny_scope, workspace=build_workspace(data_root=tmp_path / "data", scope=tiny_scope))
    batch = build_scoped_dart_fact_batch(
        runtime=tiny, catalog=_scoped_catalog(tiny),
        filing_identities=identities[:10], offset=0, limit=10,
    )
    assert len(batch.identities) == 2


def test_scoped_batch_reserves_provider_wide_quota_before_selecting(tmp_path) -> None:
    from datetime import UTC, datetime

    from src.data.dart_backfill import build_scoped_dart_fact_batch
    from src.data.research_scope import CollectionBudget
    from src.data.runtime import DataRuntime
    from src.data.workspace import build_workspace
    from src.integrations.quota import ProviderQuotaStateStore

    runtime = _scoped_runtime(tmp_path)
    scope = runtime.scope.model_copy(
        update={"collection": CollectionBudget(dart_daily_budget=100, dart_batch_identities=50, dart_daily_reserve=10)}
    )
    capped = DataRuntime(scope=scope, workspace=build_workspace(data_root=tmp_path / "data", scope=scope))
    store = ProviderQuotaStateStore(capped.workspace.state_root / "quota")
    moment = datetime(2026, 9, 21, 14, tzinfo=UTC)
    for index in range(82):
        store.record_attempt(provider="OpenDART", endpoint="list.json" if index == 0 else "fnlttSinglAcntAll.json", now=moment)

    batch = build_scoped_dart_fact_batch(
        runtime=capped,
        catalog=_scoped_catalog(capped),
        filing_identities=[_filing(corp=f"{index:08d}", filing=f"F{index}") for index in range(10)],
        offset=0,
        limit=10,
        quota_store=store,
        now=moment,
    )

    assert batch.available_request_headroom == 8
    assert len(batch.identities) == 2


def test_scoped_collector_uses_collection_policy(tmp_path, monkeypatch) -> None:
    import src.data.dart_backfill as backfill
    from src.data.dart_backfill import build_scoped_dart_collector

    captured: dict[str, object] = {}

    class _Collector:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(backfill, "DartXbrlCollector", _Collector)
    runtime = _scoped_runtime(tmp_path)
    build_scoped_dart_collector(runtime=runtime)

    assert captured["daily_request_limit"] == runtime.scope.collection.dart_daily_budget
    assert captured["min_interval"] == runtime.scope.collection.dart_request_min_interval_seconds
    assert captured["max_workers"] == runtime.scope.collection.dart_max_workers


def test_scoped_batch_catalog_success_suppresses_retry(tmp_path) -> None:
    import hashlib
    from datetime import UTC, datetime

    from src.data.dart_backfill import build_scoped_dart_fact_batch
    from src.data.receipt_catalog import EvidenceStatus, ReceiptIndexEntry

    runtime = _scoped_runtime(tmp_path)
    catalog = _scoped_catalog(runtime)
    body = b'{"records": [{"fact": 1}]}'
    payload_path = tmp_path / "fact.json"
    payload_path.write_bytes(body)
    catalog.publish(
        (
            ReceiptIndexEntry(
                source="financial_facts", natural_key="00126380:2019:11013",
                as_of=date(2019, 5, 16), fiscal_period="2019Q1", status=EvidenceStatus.SUCCESS,
                content_hash=hashlib.sha256(body).hexdigest(),
                retrieved_at=datetime(2019, 5, 17, tzinfo=UTC), payload_path=payload_path,
            ),
        )
    )
    batch = build_scoped_dart_fact_batch(
        runtime=runtime, catalog=catalog, filing_identities=[_filing()], offset=0, limit=20
    )

    assert batch.identities == ()


def test_scoped_batch_dedupes_to_latest_correction(tmp_path) -> None:
    from src.data.dart_backfill import build_scoped_dart_fact_batch

    runtime = _scoped_runtime(tmp_path)
    batch = build_scoped_dart_fact_batch(
        runtime=runtime, catalog=_scoped_catalog(runtime),
        filing_identities=[
            _filing(filing="F2", published="2019-06-01"),
            _filing(filing="F1", published="2019-05-15"),
        ],
        offset=0, limit=20,
    )

    assert [dict(item)["filing_id"] for item in batch.identities] == ["F2"]
    same_inputs = [
        _filing(filing="F2", published="2019-06-01"),
        _filing(filing="F1", published="2019-05-15"),
    ]
    first = build_scoped_dart_fact_batch(
        runtime=runtime, catalog=_scoped_catalog(runtime),
        filing_identities=same_inputs, offset=0, limit=20,
    )
    second = build_scoped_dart_fact_batch(
        runtime=runtime, catalog=_scoped_catalog(runtime),
        filing_identities=same_inputs, offset=0, limit=20,
    )
    assert first.plan_id == second.plan_id


def test_scoped_batch_rejects_invalid_inputs(tmp_path) -> None:
    from src.data.dart_backfill import build_scoped_dart_fact_batch
    from src.data.schemas import PITDataError

    runtime = _scoped_runtime(tmp_path)
    catalog = _scoped_catalog(runtime)
    with pytest.raises(PITDataError, match="offset"):
        build_scoped_dart_fact_batch(runtime=runtime, catalog=catalog, filing_identities=[], offset=-1, limit=1)
    with pytest.raises(PITDataError, match="offset"):
        build_scoped_dart_fact_batch(runtime=runtime, catalog=catalog, filing_identities=[], offset=0, limit=0)
    with pytest.raises(PITDataError, match="corp code"):
        build_scoped_dart_fact_batch(
            runtime=runtime, catalog=catalog, filing_identities=[{"biz_year": "2019"}], offset=0, limit=1
        )
    with pytest.raises(PITDataError, match="fiscal period"):
        build_scoped_dart_fact_batch(
            runtime=runtime, catalog=catalog,
            filing_identities=[{"corp_code": "00126380", "biz_year": "2019", "reprt_code": "11999"}],
            offset=0, limit=1,
        )
    with pytest.raises(PITDataError, match="fiscal period"):
        build_scoped_dart_fact_batch(
            runtime=runtime, catalog=catalog,
            filing_identities=[{"corp_code": "c", "biz_year": "b", "reprt_code": "r", "fiscal_period": "bogus"}],
            offset=0, limit=1,
        )
    with pytest.raises(ValueError, match="not-a-date"):
        build_scoped_dart_fact_batch(
            runtime=runtime, catalog=catalog,
            filing_identities=[{"corp_code": "00126380", "biz_year": "2019", "reprt_code": "11013", "as_of": "not-a-date"}],
            offset=0, limit=1,
        )


def test_facts_scoped_commands_plan_batch(tmp_path, capsys, monkeypatch) -> None:
    import json

    from src.data.cli import main

    filings = [_filing(), {"corp_code": "00126380", "biz_year": "2019", "reprt_code": "11012", "ticker": "005930"}]
    filings_path = tmp_path / "filings.json"
    filings_path.write_text(json.dumps(filings), encoding="utf-8")
    base = ["--scope-config", "config/research/kr_swing_2019_v1.toml", "--data-root", str(tmp_path / "data"),
            "--filings", str(filings_path), "--offset", "0", "--limit", "20"]

    assert main(["collect-dart-facts-scoped", *base]) == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["selected"] == 1
    assert out["missing_without_filing"] == 1

    assert main(["collect-missing-dart-facts-scoped", *base]) == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["missing_without_filing"] == 1
    assert out["selected"] == 1

    import src.data.cli as cli
    import src.data.dart_backfill as backfill

    collector = object()
    monkeypatch.setattr(backfill, "build_scoped_dart_collector", lambda **_kwargs: collector)
    monkeypatch.setattr(
        cli,
        "collect_dart_financial_facts",
        lambda **kwargs: type("Result", (), {"content_hash": "fact-hash"})(),
    )
    assert main(["collect-dart-facts-scoped", *base, "--execute"]) == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["content_hash"] == "fact-hash"


def test_legacy_missing_fact_batch_rejects_invalid_inputs_and_empty_candidates(tmp_path, monkeypatch) -> None:
    from datetime import UTC, datetime

    import src.data.dart_backfill as backfill
    from src.data.dart_backfill import DartMissingFactsRequest
    from src.data.schemas import PITDataError

    request = DartMissingFactsRequest(
        bronze_root=tmp_path / "bronze",
        artifact_root=tmp_path / "artifacts",
        backfill_artifact=tmp_path / "missing.json",
        retrieved_at=datetime(2020, 1, 1),
        offset=0,
        limit=1,
    )
    with pytest.raises(PITDataError, match="timezone-aware"):
        backfill.run_dart_missing_facts_batch(request=request, dart=object())

    aware = dataclasses.replace(request, retrieved_at=datetime(2020, 1, 1, tzinfo=UTC), offset=-1)
    with pytest.raises(PITDataError, match="offset"):
        backfill.run_dart_missing_facts_batch(request=aware, dart=object())
    request.backfill_artifact.write_text("{broken", encoding="utf-8")
    with pytest.raises(PITDataError, match="invalid DART backfill artifact"):
        backfill.run_dart_missing_facts_batch(
            request=dataclasses.replace(aware, offset=0), dart=object()
        )

    request.backfill_artifact.write_text(
        json.dumps(
            {
                "ticker_by_corp_code": {"00126380": "005930"},
                "required_periods": ["2019Q1"],
                "validation_start": "2020-01-01",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        backfill.DartXbrlCollector,
        "filing_identities_from_bronze",
        lambda *_args, **_kwargs: (),
    )
    with pytest.raises(PITDataError, match="batch is empty"):
        backfill.run_dart_missing_facts_batch(
            request=dataclasses.replace(aware, offset=0), dart=object()
        )
    request.backfill_artifact.write_text(
        json.dumps(
            {"ticker_by_corp_code": {}, "required_periods": [], "validation_start": "2020-01-01"}
        ),
        encoding="utf-8",
    )
    with pytest.raises(PITDataError, match="invalid DART backfill artifact"):
        backfill.run_dart_missing_facts_batch(
            request=dataclasses.replace(aware, offset=0), dart=object()
        )


def test_legacy_successful_fact_endpoint_reader_filters_failures(tmp_path) -> None:
    import src.data.dart_backfill as backfill
    from src.data.schemas import PITDataError

    root = tmp_path / "bronze" / "financial_facts"

    def write_payload(name: str, body: object) -> None:
        path = root / name
        path.mkdir(parents=True)
        (path / "payload.json").write_text(json.dumps(body), encoding="utf-8")

    write_payload(
        "success",
        {"corp_code": "00126380", "biz_year": "2019", "reprt_code": "11013", "records": [1]},
    )
    write_payload("unusable", {"status": "013", "records": [1]})
    write_payload("not-a-mapping", [1])
    assert backfill._successful_fact_endpoints(tmp_path / "bronze") == {
        ("00126380", "2019", "11013", "CFS")
    }

    (root / "broken").mkdir()
    (root / "broken" / "payload.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(PITDataError, match="unreadable"):
        backfill._successful_fact_endpoints(tmp_path / "bronze")


def test_scoped_identity_accepts_explicit_valid_fiscal_period() -> None:
    import src.data.dart_backfill as backfill

    assert backfill._identity_fiscal_period({"fiscal_period": "2019Q2"}) == "2019Q2"
