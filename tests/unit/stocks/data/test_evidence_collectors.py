"""Tests for KRX and OpenDART evidence collectors."""
from __future__ import annotations

import fcntl
import json
from collections.abc import Callable
from datetime import UTC, date, datetime

import pytest
import requests

from src.stocks.data.evidence import load_krx_calendar_snapshot
from src.stocks.data.evidence_collectors import (
    DartRetryPolicy,
    EvidenceCollectionError,
    KRXEvidenceCollector,
    OpenDartEvidenceCollector,
)


def _dart_record(page: int, receipt_date: str = "20240102") -> dict[str, str]:
    return {
        "rcept_no": f"2024010200000{page}",
        "rcept_dt": receipt_date,
        "corp_code": "00126380",
        "corp_name": "테스트",
        "report_nm": "현금배당결정",
        "rm": "",
    }


def _dart_collector(request: Callable[[str, dict[str, str]], dict]) -> OpenDartEvidenceCollector:
    return OpenDartEvidenceCollector(
        api_key="fixture-key",
        raw_request_json=request,
        generated_time=datetime(2026, 1, 1, tzinfo=UTC),
        sleep=lambda _: None,
        monotonic=lambda: 0.0,
    )


def _dart_noop_policy() -> DartRetryPolicy:
    return DartRetryPolicy(
        max_attempts=2,
        initial_backoff_seconds=0.0,
        max_backoff_seconds=0.0,
        min_request_interval_seconds=0.0,
    )


def _session_request(session_dates: set[str]) -> Callable[[str, dict[str, str]], dict]:
    def request(endpoint: str, params: dict[str, str]) -> dict:
        if not endpoint.endswith("_bydd_trd"):
            return {"OutBlock_1": []}
        return (
            {"OutBlock_1": [{"ISU_CD": "005930"}]}
            if params["basDd"] in session_dates
            else {"OutBlock_1": []}
        )

    return request


def test_krx_master_classification_uses_explicit_security_type() -> None:
    def request(endpoint: str, _params: dict[str, str]) -> dict:
        if endpoint.endswith("stk_isu_base_info"):
            return {
                "OutBlock_1": [
                    {
                        "ISU_SRT_CD": "005930",
                        "ISU_CD": "KR7005930003",
                        "ISU_ABBRV": "삼성전자",
                        "KIND_STKCERT_TP_NM": "보통주",
                        "LIST_DD": "19750611",
                    },
                    {
                        "ISU_SRT_CD": "0001A0",
                        "ISU_CD": "KR70001A0000",
                        "ISU_ABBRV": "특수상품",
                        "KIND_STKCERT_TP_NM": "신주인수권증서",
                        "LIST_DD": "20260305",
                    },
                ]
            }
        return {"OutBlock_1": []}

    collector = KRXEvidenceCollector(
        request_json=request,
        generated_time=datetime(2026, 3, 10, tzinfo=UTC),
    )
    snapshot = collector.collect_master_snapshot(date(2026, 3, 10))

    records = {record.source_identifier: record for record in snapshot.records}
    assert records["005930"].is_common_stock is True
    assert records["0001A0"].is_common_stock is False
    assert records["0001A0"].asset_type == "krx:신주인수권증서"


def test_krx_calendar_uses_only_dates_with_market_records() -> None:
    def request(endpoint: str, params: dict[str, str]) -> dict:
        if endpoint.endswith("_bydd_trd"):
            return {"OutBlock_1": [{"ISU_CD": "005930"}]} if params["basDd"] == "20240102" else {"OutBlock_1": []}
        return {"OutBlock_1": []}

    collector = KRXEvidenceCollector(request_json=request, generated_time=datetime(2026, 1, 1, tzinfo=UTC))
    calendar = collector.collect_session_calendar(date(2024, 1, 1), date(2024, 1, 3))
    assert calendar.sessions == (date(2024, 1, 2),)

def test_resumable_collection_skips_valid_completed_month(tmp_path) -> None:
    session_dates = {"20240102", "20240201"}
    collector = KRXEvidenceCollector(
        request_json=_session_request(session_dates),
        generated_time=datetime(2026, 1, 1, tzinfo=UTC),
    )
    output_dir = tmp_path / "parts"
    collector.collect_calendar_partitions(output_dir, date(2024, 1, 1), date(2024, 2, 29))

    assert (output_dir / "months" / "2024-01.json").is_file()
    assert (output_dir / "months" / "2024-02.json").is_file()
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["months"]["2024-01"]["status"] == "complete"
    assert manifest["months"]["2024-02"]["status"] == "complete"

    def fail_if_called(endpoint: str, params: dict[str, str]) -> dict:
        raise AssertionError("no KRX request should be made on a rerun")

    rerun = KRXEvidenceCollector(
        request_json=fail_if_called,
        generated_time=datetime(2026, 1, 1, tzinfo=UTC),
    )
    rerun.collect_calendar_partitions(output_dir, date(2024, 1, 1), date(2024, 2, 29))


def test_resumable_collection_retries_only_failed_month(tmp_path) -> None:
    def request(endpoint: str, params: dict[str, str]) -> dict:
        if not endpoint.endswith("_bydd_trd"):
            return {"OutBlock_1": []}
        if params["basDd"].startswith("2024-02"):
            raise EvidenceCollectionError("KRX transport failure")
        return (
            {"OutBlock_1": [{"ISU_CD": "005930"}]}
            if params["basDd"] == "20240102"
            else {"OutBlock_1": []}
        )

    collector = KRXEvidenceCollector(
        request_json=request,
        generated_time=datetime(2026, 1, 1, tzinfo=UTC),
    )
    output_dir = tmp_path / "parts"
    with pytest.raises(EvidenceCollectionError):
        collector.collect_calendar_partitions(output_dir, date(2024, 1, 1), date(2024, 2, 29))

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["months"]["2024-01"]["status"] == "complete"
    assert manifest["months"]["2024-02"]["status"] == "incomplete"
    assert not (tmp_path / "calendar.json").exists()

    jan_calls: list[str] = []

    def retry_request(endpoint: str, params: dict[str, str]) -> dict:
        if not endpoint.endswith("_bydd_trd"):
            return {"OutBlock_1": []}
        if params["basDd"].startswith("2024-01"):
            jan_calls.append(params["basDd"])
            raise AssertionError("completed month must not be re-requested")
        return (
            {"OutBlock_1": [{"ISU_CD": "005930"}]}
            if params["basDd"] == "20240201"
            else {"OutBlock_1": []}
        )

    rerun = KRXEvidenceCollector(
        request_json=retry_request,
        generated_time=datetime(2026, 1, 1, tzinfo=UTC),
    )
    rerun.collect_calendar_partitions(output_dir, date(2024, 1, 1), date(2024, 2, 29))
    assert jan_calls == []
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["months"]["2024-01"]["status"] == "complete"
    assert manifest["months"]["2024-02"]["status"] == "complete"


def test_calendar_merge_rejects_missing_or_overlapping_month(tmp_path) -> None:
    collector = KRXEvidenceCollector(
        request_json=_session_request({"20240102", "20240201"}),
        generated_time=datetime(2026, 1, 1, tzinfo=UTC),
    )
    output_dir = tmp_path / "parts"
    final = tmp_path / "calendar.json"
    collector.collect_calendar_partitions(output_dir, date(2024, 1, 1), date(2024, 2, 29))

    manifest_path = output_dir / "manifest.json"
    valid_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    def corrupt_manifest(patch: Callable[[dict], None]) -> None:
        manifest = json.loads(json.dumps(valid_manifest))
        patch(manifest)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    corrupt_manifest(lambda manifest: manifest["months"].pop("2024-02"))
    with pytest.raises(EvidenceCollectionError, match="2024-02"):
        collector.merge_calendar_partitions(output_dir, date(2024, 1, 1), date(2024, 2, 29), final)
    assert not final.exists()

    corrupt_manifest(lambda manifest: manifest["months"]["2024-02"].update({"sha256": "0" * 64}))
    with pytest.raises(EvidenceCollectionError, match="2024-02"):
        collector.merge_calendar_partitions(output_dir, date(2024, 1, 1), date(2024, 2, 29), final)
    assert not final.exists()

    month_file = output_dir / "months" / "2024-01.json"
    original = month_file.read_text(encoding="utf-8")
    month_file.write_text(original.replace("2024-01-02", "2024-01-03"), encoding="utf-8")
    with pytest.raises(EvidenceCollectionError, match="2024-01"):
        collector.merge_calendar_partitions(output_dir, date(2024, 1, 1), date(2024, 2, 29), final)
    assert not final.exists()

    month_file.write_text(original, encoding="utf-8")
    manifest_path.write_text(json.dumps(valid_manifest), encoding="utf-8")
    collector.merge_calendar_partitions(output_dir, date(2024, 1, 1), date(2024, 2, 29), final)
    assert final.is_file()


def test_calendar_merge_produces_loaded_final_artifact(tmp_path) -> None:
    collector = KRXEvidenceCollector(
        request_json=_session_request({"20240102", "20240201"}),
        generated_time=datetime(2026, 1, 1, tzinfo=UTC),
    )
    output_dir = tmp_path / "parts"
    final = tmp_path / "calendar.json"
    collector.collect_calendar_partitions(output_dir, date(2024, 1, 1), date(2024, 2, 29))
    collector.merge_calendar_partitions(output_dir, date(2024, 1, 1), date(2024, 2, 29), final)

    calendar = load_krx_calendar_snapshot(final)
    assert calendar.sessions == (date(2024, 1, 2), date(2024, 2, 1))
    assert calendar.version == "krx-calendar-2024-01-01-2024-02-29"
    assert list(calendar.sessions) == sorted(calendar.sessions)
    assert len(set(calendar.sessions)) == len(calendar.sessions)
    assert all(date(2024, 1, 1) <= day <= date(2024, 2, 29) for day in calendar.sessions)


def test_resumable_collection_rejects_incompatible_manifest(tmp_path) -> None:
    output_dir = tmp_path / "parts"
    output_dir.mkdir()
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "krx-calendar-manifest-1",
                "requested_start": "2025-01-01",
                "requested_end": "2025-12-31",
                "partition": "month",
                "months": {},
            }
        ),
        encoding="utf-8",
    )
    collector = KRXEvidenceCollector(
        request_json=_session_request({"20240102"}),
        generated_time=datetime(2026, 1, 1, tzinfo=UTC),
    )
    with pytest.raises(ValueError, match="different range"):
        collector.collect_calendar_partitions(output_dir, date(2024, 1, 1), date(2024, 2, 29))


def test_dart_disclosures_preserve_receipt_identity_and_candidates() -> None:
    def request(endpoint: str, params: dict[str, str]) -> dict:
        assert endpoint == "list.json"
        if params["page_no"] == "1":
            return {
                "status": "000",
                "total_page": 1,
                "list": [
                    {
                        "rcept_no": "20240102000001",
                        "rcept_dt": "20240102",
                        "corp_code": "00126380",
                        "corp_name": "테스트",
                        "report_nm": "현금배당결정",
                        "rm": "",
                    }
                ],
            }
        return {"status": "000", "total_page": 1, "list": []}

    collector = OpenDartEvidenceCollector(
        api_key="fixture-key",
        request_json=request,
        generated_time=datetime(2026, 1, 1, tzinfo=UTC),
    )
    records = collector.collect_disclosures(date(2024, 1, 1), date(2024, 1, 3))
    candidates = collector.collect_corporate_action_candidates(date(2024, 1, 1), date(2024, 1, 3))
    assert records[0]["rcept_no"] == "20240102000001"
    assert records[0]["rcept_dt"] == "20240102"
    assert candidates[0]["verification_status"] == "candidate_only"


def test_dart_page_checkpoint_resume_continues_from_verified_page(tmp_path) -> None:
    # scenario: SCENARIO_DART_PAGE_CHECKPOINT_RESUME
    policy = _dart_noop_policy()
    first_calls: list[int] = []

    def failing(endpoint: str, params: dict[str, str]) -> dict:
        assert endpoint == "list.json"
        page = int(params["page_no"])
        first_calls.append(page)
        if page == 1:
            return {"status": "000", "total_page": 2, "list": [_dart_record(1)]}
        raise requests.ConnectionError("transient network failure")

    output_dir = tmp_path / "parts"
    with pytest.raises(EvidenceCollectionError, match="2024-01"):
        _dart_collector(failing).collect_disclosure_partitions(
            output_dir, date(2024, 1, 1), date(2024, 1, 31), retry_policy=policy
        )
    assert (output_dir / "pages" / "2024-01" / "00001.json").is_file()
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["months"]["2024-01"]["status"] == "incomplete"
    assert manifest["months"]["2024-01"]["attempt_count"] == 2
    assert manifest["months"]["2024-01"]["next_page"] == 2

    resumed_calls: list[int] = []

    def resumed(endpoint: str, params: dict[str, str]) -> dict:
        page = int(params["page_no"])
        resumed_calls.append(page)
        assert page == 2
        return {"status": "000", "total_page": 2, "list": [_dart_record(2)]}

    _dart_collector(resumed).collect_disclosure_partitions(
        output_dir, date(2024, 1, 1), date(2024, 1, 31), retry_policy=policy
    )
    assert resumed_calls == [2]
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["months"]["2024-01"]["status"] == "complete"
    assert manifest["months"]["2024-01"]["record_count"] == 2
    assert not list((output_dir / "pages" / "2024-01").glob("*.json"))


def test_dart_retry_classification_backs_off_only_transient_errors(tmp_path) -> None:
    # scenario: SCENARIO_DART_BACKOFF_CLASSIFICATION
    sleeps: list[float] = []
    attempts: list[int] = []
    policy = DartRetryPolicy(
        max_attempts=3,
        initial_backoff_seconds=1.0,
        max_backoff_seconds=30.0,
        min_request_interval_seconds=0.0,
    )

    def request(endpoint: str, params: dict[str, str]) -> dict:
        attempts.append(int(params["page_no"]))
        return {"status": "800", "message": "try again later"}

    collector = OpenDartEvidenceCollector(
        api_key="fixture-key",
        raw_request_json=request,
        generated_time=datetime(2026, 1, 1, tzinfo=UTC),
        sleep=sleeps.append,
        monotonic=lambda: 0.0,
    )
    output_dir = tmp_path / "parts"
    with pytest.raises(EvidenceCollectionError, match="2024-01"):
        collector.collect_disclosure_partitions(
            output_dir, date(2024, 1, 1), date(2024, 1, 31), retry_policy=policy
        )
    assert attempts == [1, 1, 1]
    assert sleeps == [1.0, 2.0]
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["months"]["2024-01"]["status"] == "incomplete"
    assert manifest["months"]["2024-01"]["attempt_count"] == 3
    assert not (output_dir / "months" / "2024-01.json").exists()


def test_dart_quota_020_blocks_without_retry_or_artifact(tmp_path) -> None:
    # scenario: SCENARIO_DART_BACKOFF_CLASSIFICATION
    sleeps: list[float] = []
    attempts: list[int] = []

    def request(endpoint: str, params: dict[str, str]) -> dict:
        attempts.append(int(params["page_no"]))
        return {"status": "020", "message": "quota exhausted"}

    collector = OpenDartEvidenceCollector(
        api_key="fixture-key",
        raw_request_json=request,
        generated_time=datetime(2026, 1, 1, tzinfo=UTC),
        sleep=sleeps.append,
        monotonic=lambda: 0.0,
    )
    output_dir = tmp_path / "parts"
    with pytest.raises(EvidenceCollectionError, match="blocked"):
        collector.collect_disclosure_partitions(
            output_dir, date(2024, 1, 1), date(2024, 1, 31), retry_policy=_dart_noop_policy()
        )
    assert attempts == [1]
    assert sleeps == []
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["months"]["2024-01"]["status"] == "blocked"
    assert not (output_dir / "months" / "2024-01.json").exists()


def test_dart_terminal_status_stops_without_complete_artifact(tmp_path) -> None:
    # scenario: SCENARIO_DART_BACKOFF_CLASSIFICATION
    attempts: list[int] = []

    def request(endpoint: str, params: dict[str, str]) -> dict:
        attempts.append(int(params["page_no"]))
        return {"status": "010", "message": "invalid api key"}

    output_dir = tmp_path / "parts"
    with pytest.raises(EvidenceCollectionError, match="2024-01"):
        _dart_collector(request).collect_disclosure_partitions(
            output_dir, date(2024, 1, 1), date(2024, 1, 31), retry_policy=_dart_noop_policy()
        )
    assert attempts == [1]
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["months"]["2024-01"]["status"] == "incomplete"
    assert not (output_dir / "months" / "2024-01.json").exists()


def test_dart_status_013_produces_complete_empty_month(tmp_path) -> None:
    # scenario: SCENARIO_DART_EMPTY_MONTH_COMPLETE
    calls: list[str] = []

    def request(endpoint: str, params: dict[str, str]) -> dict:
        calls.append(params["page_no"])
        return {"status": "013", "message": "no data"}

    collector = _dart_collector(request)
    output_dir = tmp_path / "parts"
    collector.collect_disclosure_partitions(output_dir, date(2024, 1, 1), date(2024, 1, 31))
    assert calls == ["1"]
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["months"]["2024-01"]["status"] == "complete"
    assert manifest["months"]["2024-01"]["record_count"] == 0
    month = json.loads((output_dir / "months" / "2024-01.json").read_text(encoding="utf-8"))
    assert month["records"] == []
    assert month["version"] == "dart-disclosures-month-2024-01"

    def fail_if_called(endpoint: str, params: dict[str, str]) -> dict:
        raise AssertionError("completed empty month must not be re-requested")

    _dart_collector(fail_if_called).collect_disclosure_partitions(
        output_dir, date(2024, 1, 1), date(2024, 1, 31)
    )


def test_dart_partitions_recollect_tampered_complete_month(tmp_path) -> None:
    output_dir = tmp_path / "parts"

    def request(endpoint: str, params: dict[str, str]) -> dict:
        return {"status": "000", "total_page": 1, "list": [_dart_record(1)]}

    _dart_collector(request).collect_disclosure_partitions(
        output_dir, date(2024, 1, 1), date(2024, 1, 31)
    )
    month_path = output_dir / "months" / "2024-01.json"
    month_path.write_text(
        month_path.read_text(encoding="utf-8").replace("20240102000001", "20240102999999"),
        encoding="utf-8",
    )
    _dart_collector(request).collect_disclosure_partitions(
        output_dir, date(2024, 1, 1), date(2024, 1, 31)
    )
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["months"]["2024-01"]["status"] == "complete"
    payload = json.loads(month_path.read_text(encoding="utf-8"))
    assert payload["records"][0]["rcept_no"] == "20240102000001"


def test_dart_partitions_reject_range_or_policy_mismatch(tmp_path) -> None:
    output_dir = tmp_path / "parts"

    def request(endpoint: str, params: dict[str, str]) -> dict:
        return {"status": "000", "total_page": 1, "list": [_dart_record(1)]}

    collector = _dart_collector(request)
    collector.collect_disclosure_partitions(output_dir, date(2024, 1, 1), date(2024, 1, 31))
    with pytest.raises(ValueError, match="different range"):
        collector.collect_disclosure_partitions(output_dir, date(2025, 1, 1), date(2025, 1, 31))
    with pytest.raises(ValueError, match="page_count"):
        collector.collect_disclosure_partitions(
            output_dir, date(2024, 1, 1), date(2024, 1, 31), page_count=50
        )


def test_dart_partitions_reject_concurrent_lock(tmp_path) -> None:
    output_dir = tmp_path / "parts"
    output_dir.mkdir()
    lock = output_dir / "collector.lock"
    calls: list[str] = []

    def request(endpoint: str, params: dict[str, str]) -> dict:
        calls.append(params["page_no"])
        return {"status": "000", "total_page": 1, "list": []}

    with lock.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(EvidenceCollectionError, match="lock"):
            _dart_collector(request).collect_disclosure_partitions(
                output_dir, date(2024, 1, 1), date(2024, 1, 31)
            )
        assert calls == []

    _dart_collector(request).collect_disclosure_partitions(
        output_dir, date(2024, 1, 1), date(2024, 1, 31)
    )
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["months"]["2024-01"]["status"] == "complete"


def test_dart_merge_produces_sorted_unique_final(tmp_path) -> None:
    # scenario: SCENARIO_DART_MERGE_FAILS_ON_GAP_OR_TAMPER
    def request(endpoint: str, params: dict[str, str]) -> dict:
        bgn_de = params["bgn_de"]
        if bgn_de.startswith("202401"):
            return {"status": "000", "total_page": 1, "list": [_dart_record(1)]}
        return {
            "status": "000",
            "total_page": 1,
            "list": [_dart_record(2, receipt_date="20240201")],
        }

    collector = _dart_collector(request)
    output_dir = tmp_path / "parts"
    final = tmp_path / "disclosures.json"
    collector.collect_disclosure_partitions(output_dir, date(2024, 1, 1), date(2024, 2, 29))
    collector.merge_disclosure_partitions(output_dir, date(2024, 1, 1), date(2024, 2, 29), final)

    payload = json.loads(final.read_text(encoding="utf-8"))
    assert payload["version"] == "dart-disclosures-2024-01-01-2024-02-29"
    assert payload["record_count"] == 2
    assert payload["records"] == [_dart_record(1), _dart_record(2, receipt_date="20240201")]


def test_dart_merge_fails_on_missing_or_incomplete_month(tmp_path) -> None:
    # scenario: SCENARIO_DART_MERGE_FAILS_ON_GAP_OR_TAMPER
    def request(endpoint: str, params: dict[str, str]) -> dict:
        if params["bgn_de"].startswith("202401"):
            return {"status": "000", "total_page": 1, "list": [_dart_record(1)]}
        return {"status": "000", "total_page": 1, "list": [_dart_record(2, receipt_date="20240201")]}

    collector = _dart_collector(request)
    output_dir = tmp_path / "parts"
    final = tmp_path / "disclosures.json"
    collector.collect_disclosure_partitions(output_dir, date(2024, 1, 1), date(2024, 2, 29))

    (output_dir / "months" / "2024-02.json").unlink()
    with pytest.raises(EvidenceCollectionError, match="2024-02"):
        collector.merge_disclosure_partitions(output_dir, date(2024, 1, 1), date(2024, 2, 29), final)
    assert not final.exists()


def test_dart_merge_rejects_digest_tamper(tmp_path) -> None:
    # scenario: SCENARIO_DART_MERGE_FAILS_ON_GAP_OR_TAMPER
    def request(endpoint: str, params: dict[str, str]) -> dict:
        if params["bgn_de"].startswith("202401"):
            return {"status": "000", "total_page": 1, "list": [_dart_record(1)]}
        return {"status": "000", "total_page": 1, "list": [_dart_record(2, receipt_date="20240201")]}

    collector = _dart_collector(request)
    output_dir = tmp_path / "parts"
    final = tmp_path / "disclosures.json"
    collector.collect_disclosure_partitions(output_dir, date(2024, 1, 1), date(2024, 2, 29))

    month_path = output_dir / "months" / "2024-02.json"
    month_path.write_text(
        month_path.read_text(encoding="utf-8").replace("20240102000002", "20240102999999"),
        encoding="utf-8",
    )
    with pytest.raises(EvidenceCollectionError, match="2024-02"):
        collector.merge_disclosure_partitions(output_dir, date(2024, 1, 1), date(2024, 2, 29), final)
    assert not final.exists()


def test_dart_merge_never_overwrites_different_output(tmp_path) -> None:
    # scenario: SCENARIO_DART_MERGE_FAILS_ON_GAP_OR_TAMPER
    def request(endpoint: str, params: dict[str, str]) -> dict:
        return {"status": "000", "total_page": 1, "list": [_dart_record(1)]}

    collector = _dart_collector(request)
    output_dir = tmp_path / "parts"
    final = tmp_path / "disclosures.json"
    collector.collect_disclosure_partitions(output_dir, date(2024, 1, 1), date(2024, 1, 31))
    collector.merge_disclosure_partitions(output_dir, date(2024, 1, 1), date(2024, 1, 31), final)
    original = final.read_bytes()

    final.write_text("stale conflicting content", encoding="utf-8")
    with pytest.raises(EvidenceCollectionError, match="different content"):
        collector.merge_disclosure_partitions(output_dir, date(2024, 1, 1), date(2024, 1, 31), final)
    assert final.read_bytes() == b"stale conflicting content"

    final.write_bytes(original)
    collector.merge_disclosure_partitions(output_dir, date(2024, 1, 1), date(2024, 1, 31), final)
    assert final.read_bytes() == original
