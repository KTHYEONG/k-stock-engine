

from datetime import datetime

from src.core.time import KRX_TZ
from src.data.collection import collect_dart_lifecycle_evidence
from src.data.lifecycle import LifecycleCandidate, LifecycleEvidence, LifecycleResolutionKind

class FakeCollector:
    def collect(self, candidate):
        return LifecycleEvidence(candidate, LifecycleResolutionKind.UNRESOLVED, 'unresolved', 'missing_terms', None, None, None, None, None, None, 'opendart', None, None, None)

def test_collect_dart_lifecycle_evidence_writes_one_auditable_bronze_envelope_per_candidate(tmp_path):
    from src.data.bronze import BronzeStore
    session = datetime(2016, 1, 21, 9, tzinfo=KRX_TZ)
    candidate = LifecycleCandidate('KRX:003945', '003945', datetime(2016, 1, 20, 9, tzinfo=KRX_TZ), session, ('master-hash',))
    receipts = collect_dart_lifecycle_evidence(candidates=(candidate,), collector=FakeCollector(), bronze=BronzeStore(tmp_path), retrieved_at=session)
    assert len(receipts) == 1
    assert receipts[0].kind.value == 'lifecycle_events'
    assert receipts[0].content_hash


def test_corp_code_disclosure_is_cached_true_when_existing_payload_fully_contains_range(tmp_path) -> None:
    import json
    from datetime import date
    from src.data.collection import _corp_code_disclosure_is_cached

    # Given: an existing Bronze disclosure payload covering a WIDE range for corp A.
    receipt_dir = tmp_path / "disclosures" / "abc123"
    receipt_dir.mkdir(parents=True)
    (receipt_dir / "payload.json").write_text(
        json.dumps({"corp_code": "00126380", "start": "2015-01-01", "end": "2020-01-01", "records": []}),
        encoding="utf-8",
    )

    # When: checking a NARROWER sub-range for the same corp_code.
    result = _corp_code_disclosure_is_cached(
        tmp_path, corp_code="00126380", start=date(2016, 1, 1), end=date(2017, 1, 1)
    )

    # Then
    assert result is True


def test_corp_code_disclosure_is_cached_false_when_partial_overlap_only(tmp_path) -> None:
    import json
    from datetime import date
    from src.data.collection import _corp_code_disclosure_is_cached

    # Given: existing payload ends BEFORE the requested range ends (partial overlap only).
    receipt_dir = tmp_path / "disclosures" / "abc123"
    receipt_dir.mkdir(parents=True)
    (receipt_dir / "payload.json").write_text(
        json.dumps({"corp_code": "00126380", "start": "2014-01-01", "end": "2016-06-01", "records": []}),
        encoding="utf-8",
    )

    # When
    result = _corp_code_disclosure_is_cached(
        tmp_path, corp_code="00126380", start=date(2015, 1, 1), end=date(2017, 1, 1)
    )

    # Then: not fully contained -> must report not-cached (fetch must still happen).
    assert result is False


def test_corp_code_disclosure_is_cached_false_when_no_bronze_directory(tmp_path) -> None:
    from datetime import date
    from src.data.collection import _corp_code_disclosure_is_cached

    # Given: tmp_path has no 'disclosures' subdirectory at all (fresh Bronze root).

    # When
    result = _corp_code_disclosure_is_cached(
        tmp_path, corp_code="00126380", start=date(2016, 1, 1), end=date(2017, 1, 1)
    )

    # Then
    assert result is False


def test_corp_code_disclosure_is_cached_fails_open_on_unreadable_payload(tmp_path) -> None:
    from datetime import date
    from src.data.collection import _corp_code_disclosure_is_cached

    # Given: a payload.json that is not valid JSON.
    receipt_dir = tmp_path / "disclosures" / "corrupt1"
    receipt_dir.mkdir(parents=True)
    (receipt_dir / "payload.json").write_text("{not valid json", encoding="utf-8")

    # When/Then: must not raise, and must fail open (treat as not-cached).
    result = _corp_code_disclosure_is_cached(
        tmp_path, corp_code="00126380", start=date(2016, 1, 1), end=date(2017, 1, 1)
    )
    assert result is False


def test_collect_dart_disclosures_skips_already_cached_corp_codes(tmp_path) -> None:
    import json
    from datetime import UTC, date, datetime
    from src.data.collection import collect_dart_disclosures
    from src.data.schemas import EvidenceKind

    bronze_root = tmp_path / "bronze"
    receipt_dir = bronze_root / "disclosures" / "cached1"
    receipt_dir.mkdir(parents=True)
    receipt_dir.joinpath("payload.json").write_text(
        json.dumps({"corp_code": "00126380", "start": "2014-01-01", "end": "2026-01-01", "records": [{"rcept_no": "1", "rcept_dt": "20160102", "corp_code": "00126380"}]}),
        encoding="utf-8",
    )

    calls: list[object] = []

    class NeverCalledDart:
        def fetch_disclosures(self, start, end, *, corp_codes=None):
            calls.append(corp_codes)
            raise AssertionError("fetch_disclosures must not be called when every corp_code is cached")

    artifact = collect_dart_disclosures(
        dart=NeverCalledDart(),
        start=date(2016, 1, 1),
        end=date(2017, 1, 1),
        bronze_root=bronze_root,
        retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
        corp_codes=("00126380",),
    )

    # Then: no live fetch happened, yet a valid empty-receipts artifact was returned.
    assert calls == []
    assert EvidenceKind.DISCLOSURES not in artifact.receipts
    assert artifact.page_receipts is not None
    assert artifact.page_receipts[EvidenceKind.DISCLOSURES.value] == ()
    assert artifact.report_path.exists()


def test_collect_dart_disclosures_fetches_only_uncached_corp_codes(tmp_path) -> None:
    import json
    from datetime import UTC, date, datetime
    from src.data.collection import collect_dart_disclosures

    bronze_root = tmp_path / "bronze"
    receipt_dir = bronze_root / "disclosures" / "cached1"
    receipt_dir.mkdir(parents=True)
    receipt_dir.joinpath("payload.json").write_text(
        json.dumps({"corp_code": "AAA", "start": "2014-01-01", "end": "2026-01-01", "records": []}),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    class RecordingDart:
        def fetch_disclosures(self, start, end, *, corp_codes=None):
            captured["corp_codes"] = corp_codes
            return [{"records": [], "start": start.isoformat(), "end": end.isoformat(), "corp_code": c} for c in corp_codes]

    collect_dart_disclosures(
        dart=RecordingDart(),
        start=date(2016, 1, 1),
        end=date(2017, 1, 1),
        bronze_root=bronze_root,
        retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
        corp_codes=("AAA", "BBB"),
    )

    # Then: only the uncached corp_code was actually fetched.
    assert captured["corp_codes"] == ("BBB",)


def test_collect_dart_disclosures_corp_codes_none_branch_unchanged(tmp_path) -> None:
    from datetime import UTC, date, datetime
    from src.data.collection import collect_dart_disclosures
    from src.data.schemas import EvidenceKind

    calls: list[tuple[object, object]] = []

    class WholeMarketDart:
        def fetch_disclosures(self, start, end):
            calls.append((start, end))
            return [{"records": [{"rcept_no": "1", "rcept_dt": "20160102", "corp_code": "X"}], "start": start.isoformat(), "end": end.isoformat()}]

    artifact = collect_dart_disclosures(
        dart=WholeMarketDart(),
        start=date(2016, 1, 1),
        end=date(2016, 1, 31),
        bronze_root=tmp_path / "bronze",
        retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    # Then: fetched exactly once, whole-market call untouched by the corp_codes caching path.
    assert calls == [(date(2016, 1, 1), date(2016, 1, 31))]
    assert EvidenceKind.DISCLOSURES in artifact.receipts
