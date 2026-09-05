def test_backfill_plan_requires_exact_ticker_to_corp_code_bridge() -> None:
    from datetime import UTC, date, datetime
    import polars as pl
    from src.data.dart_backfill import build_dart_historical_backfill_plan
    from src.integrations.dart.client import DartCorpCodeRecord

    master = pl.DataFrame({"instrument_id": ["KRX:005930", "KRX:000001"], "ticker": ["005930", "000001"], "share_class": ["common", "common"], "valid_from": [datetime(2010, 1, 1, tzinfo=UTC)] * 2, "valid_to": [None, None], "available_at": [datetime(2010, 1, 1, tzinfo=UTC)] * 2})
    records = (DartCorpCodeRecord(ticker="005930", corp_code="00126380", corp_name="A"),)

    plan = build_dart_historical_backfill_plan(security_master=master, corp_code_records=records, validation_start=date(2016, 1, 4), validation_end=date(2016, 12, 29), corp_code_receipt_hash="a" * 64)

    assert plan.required_periods == ("2014Q4", "2015Q1", "2015Q2", "2015Q3")
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
    monkeypatch.setattr("src.data.dart_backfill._load_security_master", lambda _root: master)
    monkeypatch.setattr("src.data.dart_backfill._persist_corp_code_receipt", lambda **_kwargs: "c" * 64)
    monkeypatch.setattr("src.data.dart_backfill.collect_dart_disclosures", lambda **_kwargs: None)
    monkeypatch.setattr("src.data.dart_backfill.DartXbrlCollector.filing_identities_from_bronze", lambda *_args, **_kwargs: ())

    request = DartHistoricalBackfillRequest(bronze_root=tmp_path / "bronze", artifact_root=tmp_path / "artifacts", silver_root=tmp_path / "silver", validation_start=date(2016, 1, 4), validation_end=date(2016, 12, 29), retrieved_at=datetime(2016, 1, 4, tzinfo=UTC), offset=0, limit=1)
    plan = run_dart_historical_backfill_batch(request=request, dart=Collector())

    payload = json.loads((tmp_path / "artifacts" / "dart_backfill" / f"{plan.plan_id}.json").read_text())
    assert payload["ticker_by_corp_code"] == {"00126380": "005930"}
    assert payload["required_periods"] == ["2014Q4", "2015Q1", "2015Q2", "2015Q3"]


def test_dedupe_endpoint_identities_keeps_latest_correction() -> None:
    from src.data.dart_backfill import _dedupe_endpoint_identities

    identities = (
        {"corp_code": "001", "biz_year": "2015", "reprt_code": "11013", "fs_div": "CFS", "filing_id": "F1", "published_at": "2015-05-15"},
        {"corp_code": "001", "biz_year": "2015", "reprt_code": "11013", "fs_div": "CFS", "filing_id": "F2", "published_at": "2015-06-01"},
        {"corp_code": "001", "biz_year": "2015", "reprt_code": "11012", "fs_div": "CFS", "filing_id": "F3", "published_at": "2015-08-15"},
    )

    result = _dedupe_endpoint_identities(identities)

    assert [item["filing_id"] for item in result] == ["F3", "F2"]
