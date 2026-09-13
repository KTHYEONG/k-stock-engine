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
