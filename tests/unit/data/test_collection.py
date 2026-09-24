

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


def test_collect_dart_disclosures_filtered_fetch_never_widens(tmp_path) -> None:
    from datetime import UTC, date, datetime

    import pytest

    from src.data.collection import collect_dart_disclosures

    calls: list[object] = []

    class CorpCodeOnlyDart:
        def fetch_disclosures(self, start, end, *, corp_codes):
            calls.append(corp_codes)
            raise TypeError("fetch_disclosures() missing support for corp_codes")

    with pytest.raises(TypeError, match="corp_codes"):
        collect_dart_disclosures(
            dart=CorpCodeOnlyDart(),
            start=date(2016, 1, 1),
            end=date(2016, 1, 31),
            bronze_root=tmp_path / "bronze",
            retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
            corp_codes=("00126380",),
        )

    assert calls == [("00126380",)]


def _action_plan(symbols=(("005930", "00126380"),), start=None, end=None, endpoints=None, reports=None, tmp=None):
    from datetime import date as _date
    from src.data.collection_plan import HistoricalCollectionPlan, PlanChunk, build_corporate_action_collection_plan

    start = start or _date(2019, 1, 1)
    end = end or _date(2025, 12, 31)
    chunks = tuple(PlanChunk(f"p:{s}:0000", s, (start,)) for s, _ in symbols)
    plan = HistoricalCollectionPlan("hist-1", start, end, 1, chunks, "h" * 64)
    mapping = dict(symbols)
    return build_corporate_action_collection_plan(
        historical_plan=plan,
        ticker_to_corp_code=mapping,
        action_endpoints=endpoints or ("fricDecsn.json", "crDecsn.json", "piicDecsn.json", "cmpDvDecsn.json", "cmpMgDecsn.json"),
        dividend_endpoint="alotMatter.json",
        dividend_report_codes=reports or ("11011", "11012", "11013", "11014"),
        artifact_root=tmp,
    )


class _ActionDart:
    def __init__(self, mapping, status="000", records=(), wrong=None):
        self.mapping = dict(mapping)
        self.status = status
        self.records = list(records)
        self.wrong = wrong
        self.calls = {"structured": 0, "dividend": 0}

    def load_corp_codes(self):
        return dict(self.mapping)

    def fetch_corporate_action_decisions(self, *, corp_codes, start, end):
        from types import SimpleNamespace

        self.calls["structured"] += 1
        pages = []
        for code in corp_codes:
            for endpoint in ("fricDecsn.json", "crDecsn.json", "piicDecsn.json", "cmpDvDecsn.json", "cmpMgDecsn.json"):
                ep = endpoint
                cc = code
                if self.wrong == "structured-endpoint":
                    ep = "other.json"
                if self.wrong == "structured-corp":
                    cc = "99999999"
                pages.append(SimpleNamespace(endpoint=ep, corp_code=cc, status=self.status, records=tuple(self.records)))
        return tuple(pages)

    def fetch_dividend_disclosures(self, *, corp_codes, bsns_years):
        from types import SimpleNamespace

        self.calls["dividend"] += 1
        pages = []
        for code in corp_codes:
            for year in bsns_years:
                for report in ("11011", "11012", "11013", "11014"):
                    cc, yy, rc = code, year, report
                    if self.wrong == "dividend-year":
                        yy = "1999"
                    if self.wrong == "dividend-report":
                        rc = "99999"
                    pages.append(SimpleNamespace(corp_code=cc, bsns_year=yy, reprt_code=rc, status=self.status, records=tuple(self.records)))
        return tuple(pages)


def test_build_corporate_action_plan_expands_deterministic_requests(tmp_path):
    plan = _action_plan(symbols=(("005930", "00126380"), ("000660", "00164742")), tmp=tmp_path / "plans")
    assert len(plan.requests) == 2 * (5 + 28)
    keys = [(r.requested_instrument_id, r.endpoint, r.bsns_year or "", r.reprt_code or "") for r in plan.requests]
    assert keys == sorted(keys)
    per_symbol = {}
    for r in plan.requests:
        per_symbol.setdefault(r.requested_instrument_id, {"structured": 0, "dividend": 0})
        per_symbol[r.requested_instrument_id]["structured" if r.bsns_year is None else "dividend"] += 1
    assert per_symbol == {"000660": {"structured": 5, "dividend": 28}, "005930": {"structured": 5, "dividend": 28}}
    assert (tmp_path / "plans" / f"{plan.plan_id}.json").exists()


def test_build_corporate_action_plan_covers_inclusive_fiscal_years(tmp_path):
    from datetime import date

    plan = _action_plan(
        symbols=(("005930", "00126380"),),
        start=date(2019, 6, 1),
        end=date(2020, 2, 1),
        tmp=tmp_path / "plans",
    )
    dividend = [r for r in plan.requests if r.bsns_year is not None]
    assert {(r.bsns_year, r.reprt_code) for r in dividend} == {
        (year, code) for year in ("2019", "2020") for code in ("11011", "11012", "11013", "11014")
    }


def test_build_corporate_action_plan_preserves_unmapped_symbols(tmp_path):
    from datetime import date
    from src.data.collection_plan import HistoricalCollectionPlan, PlanChunk, build_corporate_action_collection_plan

    day = date(2020, 1, 1)
    hist = HistoricalCollectionPlan(
        "hist-u", day, date(2020, 12, 31), 1,
        (PlanChunk("hist-u:005930:0000", "005930", (day,)), PlanChunk("hist-u:999999:0000", "999999", (day,))),
        "u" * 64,
    )
    plan = build_corporate_action_collection_plan(
        historical_plan=hist,
        ticker_to_corp_code={"005930": "00126380"},
        action_endpoints=("fricDecsn.json",),
        dividend_endpoint="alotMatter.json",
        dividend_report_codes=("11011",),
        artifact_root=tmp_path / "plans",
    )
    assert plan.unresolved_instruments == ("999999",)
    assert all(r.requested_instrument_id == "005930" for r in plan.requests)


def test_build_corporate_action_plan_supports_prefixed_identifiers(tmp_path):
    from datetime import date
    from src.data.collection_plan import HistoricalCollectionPlan, PlanChunk, build_corporate_action_collection_plan

    day = date(2020, 1, 1)
    hist = HistoricalCollectionPlan(
        "hist-p", day, date(2020, 12, 31), 1, (PlanChunk("hist-p:x:0000", "KRX:005930", (day,)),), "p" * 64
    )
    plan = build_corporate_action_collection_plan(
        historical_plan=hist,
        ticker_to_corp_code={"005930": "00126380"},
        action_endpoints=("fricDecsn.json",),
        dividend_endpoint="alotMatter.json",
        dividend_report_codes=("11011",),
        artifact_root=tmp_path / "plans",
    )
    assert plan.unresolved_instruments == ()
    assert all(r.requested_instrument_id == "KRX:005930" for r in plan.requests)
    assert all(r.corp_code == "00126380" for r in plan.requests)


def test_build_corporate_action_plan_rejects_conflicting_and_malformed_mappings(tmp_path):
    import pytest
    from datetime import date
    from src.data.collection_plan import HistoricalCollectionPlan, PlanChunk, build_corporate_action_collection_plan
    from src.data.schemas import PITDataError

    day = date(2020, 1, 1)
    hist = HistoricalCollectionPlan(
        "hist-c", day, date(2020, 12, 31), 1, (PlanChunk("hist-c:x:0000", "KRX:005930", (day,)),), "c" * 64
    )
    base = {
        "historical_plan": hist,
        "action_endpoints": ("fricDecsn.json",),
        "dividend_endpoint": "alotMatter.json",
        "dividend_report_codes": ("11011",),
        "artifact_root": tmp_path / "plans",
    }
    with pytest.raises(PITDataError):
        build_corporate_action_collection_plan(ticker_to_corp_code={"KRX:005930": "00126380", "005930": "00999999"}, **base)
    with pytest.raises(PITDataError):
        build_corporate_action_collection_plan(ticker_to_corp_code={"005930": "  "}, **base)
    with pytest.raises(PITDataError):
        build_corporate_action_collection_plan(ticker_to_corp_code={"005930": "ABC"}, **base)
    with pytest.raises(PITDataError):
        build_corporate_action_collection_plan(ticker_to_corp_code={"005930": "00126380"}, action_endpoints=("a.json", "a.json"), **{k: v for k, v in base.items() if k != "action_endpoints"})
    with pytest.raises(PITDataError):
        build_corporate_action_collection_plan(ticker_to_corp_code={"005930": "00126380"}, dividend_endpoint=" ", **{k: v for k, v in base.items() if k != "dividend_endpoint"})
    with pytest.raises(PITDataError):
        build_corporate_action_collection_plan(ticker_to_corp_code={"005930": "00126380"}, dividend_report_codes=(), **{k: v for k, v in base.items() if k != "dividend_report_codes"})


def test_build_corporate_action_plan_hash_is_order_stable(tmp_path):
    first = _action_plan(symbols=(("005930", "00126380"), ("000660", "00164742")), tmp=tmp_path / "p1")
    second = _action_plan(symbols=(("000660", "00164742"), ("005930", "00126380")), tmp=tmp_path / "p2")
    assert [r.request_id for r in first.requests] == [r.request_id for r in second.requests]
    assert (first.plan_id, first.content_hash) == (second.plan_id, second.content_hash)


def test_collect_planned_structured_page_identity_mismatch_raises_without_checkpoint(tmp_path):
    from datetime import UTC, datetime
    from src.data.collection import collect_planned_corporate_actions
    from src.data.collection_plan import CollectionCheckpointStore
    from src.data.schemas import PITDataError
    import pytest

    plan = _action_plan(
        symbols=(("005930", "00126380"),),
        start=__import__("datetime").date(2020, 1, 1),
        end=__import__("datetime").date(2020, 12, 31),
        endpoints=("fricDecsn.json",),
        reports=("11011",),
        tmp=tmp_path / "plans",
    )
    dart = _ActionDart({"005930": "00126380"}, wrong="structured-endpoint")
    store = CollectionCheckpointStore(tmp_path / "ckpt")
    with pytest.raises(PITDataError, match="identity mismatch"):
        collect_planned_corporate_actions(
            plan=plan, dart=dart, bronze_root=tmp_path / "bronze",
            retrieved_at=datetime(2026, 1, 1, tzinfo=UTC), checkpoint_store=store,
        )
    structured = next(r for r in plan.requests if r.bsns_year is None)
    assert not store._chunk_path(plan.plan_id, structured.request_id).exists()


def test_collect_planned_dividend_page_identity_mismatch_raises_without_checkpoint(tmp_path):
    from datetime import UTC, date, datetime
    from src.data.collection import collect_planned_corporate_actions
    from src.data.collection_plan import CollectionCheckpointStore
    from src.data.schemas import PITDataError
    import pytest

    plan = _action_plan(
        symbols=(("005930", "00126380"),), start=date(2020, 1, 1), end=date(2020, 12, 31),
        endpoints=("fricDecsn.json",), reports=("11011",), tmp=tmp_path / "plans",
    )
    dart = _ActionDart({"005930": "00126380"}, wrong="dividend-report")
    checkpoint_store = CollectionCheckpointStore(tmp_path / "ckpt")
    with pytest.raises(PITDataError, match="identity mismatch"):
        collect_planned_corporate_actions(
            plan=plan, dart=dart, bronze_root=tmp_path / "bronze",
            retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
            checkpoint_store=checkpoint_store,
        )
    dividend = next(r for r in plan.requests if r.bsns_year is not None)
    assert not checkpoint_store._chunk_path(plan.plan_id, dividend.request_id).exists()


def test_collect_planned_empty_disclosure_persists_auditable_receipt(tmp_path):
    import json
    from datetime import UTC, date, datetime
    from src.data.collection import collect_planned_corporate_actions
    from src.data.collection_plan import CollectionCheckpointStore
    from src.data.schemas import EvidenceKind

    plan = _action_plan(
        symbols=(("005930", "00126380"),), start=date(2020, 1, 1), end=date(2020, 12, 31),
        endpoints=("fricDecsn.json",), reports=("11011",), tmp=tmp_path / "plans",
    )
    dart = _ActionDart({"005930": "00126380"}, status="013")
    artifact = collect_planned_corporate_actions(
        plan=plan, dart=dart, bronze_root=tmp_path / "bronze",
        retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
        checkpoint_store=CollectionCheckpointStore(tmp_path / "ckpt"),
    )
    receipts = artifact.page_receipts[EvidenceKind.CORPORATE_ACTIONS.value]
    assert len(receipts) == len(plan.requests)
    payload = json.loads(receipts[0].payload_path.read_bytes())
    assert payload["status"] == "013"
    assert payload["records"] == []
    assert payload["request_id"] == plan.requests[0].request_id
    ledger = json.loads(artifact.report_path.read_text(encoding="utf-8"))
    assert ledger["empty"] == len(plan.requests)
    assert ledger["successful"] == 0
    assert ledger["planned"] == len(plan.requests)
    assert ledger["pending"] == 0


def test_collect_planned_successful_disclosure_retains_provenance(tmp_path):
    import json
    from datetime import UTC, date, datetime
    from src.data.collection import collect_planned_corporate_actions
    from src.data.collection_plan import CollectionCheckpointStore
    from src.data.schemas import EvidenceKind

    records = ({"rcept_no": "20200101000001", "corp_code": "00126380"},)
    plan = _action_plan(
        symbols=(("005930", "00126380"),), start=date(2020, 1, 1), end=date(2020, 12, 31),
        endpoints=("fricDecsn.json",), reports=("11011",), tmp=tmp_path / "plans",
    )
    dart = _ActionDart({"005930": "00126380"}, status="000", records=records)
    artifact = collect_planned_corporate_actions(
        plan=plan, dart=dart, bronze_root=tmp_path / "bronze",
        retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
        checkpoint_store=CollectionCheckpointStore(tmp_path / "ckpt"),
    )
    payload = json.loads(artifact.page_receipts[EvidenceKind.CORPORATE_ACTIONS.value][0].payload_path.read_bytes())
    assert payload["records"] == [dict(records[0])]
    assert payload["instrument_mapping_provenance"] == "opendart_corp_code_direct"
    assert payload["plan_id"] == plan.plan_id
    assert payload["corp_code"] == "00126380"
    assert artifact.content_hash
    assert len(artifact.page_receipts[EvidenceKind.CORPORATE_ACTIONS.value]) == len(plan.requests)


def test_collect_planned_resumption_skips_verified_requests(tmp_path):
    from datetime import UTC, date, datetime
    from src.data.collection import collect_planned_corporate_actions
    from src.data.collection_plan import CollectionCheckpointStore

    plan = _action_plan(
        symbols=(("005930", "00126380"),), start=date(2020, 1, 1), end=date(2020, 12, 31),
        endpoints=("fricDecsn.json",), reports=("11011",), tmp=tmp_path / "plans",
    )
    bronze = tmp_path / "bronze"
    store = CollectionCheckpointStore(tmp_path / "ckpt")
    first_dart = _ActionDart({"005930": "00126380"}, status="000")
    first = collect_planned_corporate_actions(
        plan=plan, dart=first_dart, bronze_root=bronze,
        retrieved_at=datetime(2026, 1, 1, tzinfo=UTC), checkpoint_store=store,
    )
    second_dart = _ActionDart({"005930": "00126380"}, status="000")
    second = collect_planned_corporate_actions(
        plan=plan, dart=second_dart, bronze_root=bronze,
        retrieved_at=datetime(2026, 1, 1, tzinfo=UTC), checkpoint_store=store,
    )
    assert second_dart.calls == {"structured": 0, "dividend": 0}
    assert second.previously_completed_chunks == len(plan.requests)
    assert second.content_hash == first.content_hash


def test_collect_planned_corporate_actions_reuses_provider_batches_per_company_and_year(tmp_path):
    from datetime import UTC, datetime
    from src.data.collection import collect_planned_corporate_actions
    from src.data.collection_plan import CollectionCheckpointStore

    plan = _action_plan(tmp=tmp_path / "plans")
    dart = _ActionDart({"005930": "00126380"})
    collect_planned_corporate_actions(
        plan=plan,
        dart=dart,
        bronze_root=tmp_path / "bronze",
        retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
        checkpoint_store=CollectionCheckpointStore(tmp_path / "ckpt"),
    )
    assert dart.calls == {"structured": 1, "dividend": 7}


def test_collect_planned_receipt_revalidation_recollects_on_missing_or_mismatch(tmp_path):
    import shutil
    from datetime import UTC, date, datetime
    from src.data.collection import _verified_action_receipts, collect_planned_corporate_actions
    from src.data.collection_plan import CollectionCheckpointStore

    plan = _action_plan(
        symbols=(("005930", "00126380"),), start=date(2020, 1, 1), end=date(2020, 12, 31),
        endpoints=("fricDecsn.json",), reports=("11011",), tmp=tmp_path / "plans",
    )
    bronze = tmp_path / "bronze"
    store = CollectionCheckpointStore(tmp_path / "ckpt")
    collect_planned_corporate_actions(
        plan=plan, dart=_ActionDart({"005930": "00126380"}), bronze_root=bronze,
        retrieved_at=datetime(2026, 1, 1, tzinfo=UTC), checkpoint_store=store,
    )
    shutil.rmtree(bronze / "corporate_actions")
    redart = _ActionDart({"005930": "00126380"})
    rerun = collect_planned_corporate_actions(
        plan=plan, dart=redart, bronze_root=bronze,
        retrieved_at=datetime(2026, 1, 1, tzinfo=UTC), checkpoint_store=store,
    )
    assert redart.calls["structured"] + redart.calls["dividend"] > 0
    assert rerun.previously_completed_chunks == 0

    import json as _reval_json

    stored = _reval_json.loads(store._chunk_path(plan.plan_id, plan.requests[0].request_id).read_text(encoding="utf-8"))
    tampered = bronze / "corporate_actions" / str(stored["receipt_hashes"][0]) / "payload.json"
    tampered.write_bytes(b'{"request_id":"tampered"}')
    assert _verified_action_receipts(
        checkpoint_store=store, plan=plan,
        request_id=plan.requests[0].request_id, bronze_root=bronze,
    ) is None


def test_collect_planned_rejects_invalid_inputs_before_provider_call(tmp_path):
    from datetime import UTC, date, datetime
    from src.data.collection import collect_planned_corporate_actions
    from src.data.collection_plan import CollectionCheckpointStore, CorporateActionCollectionPlan
    from src.data.schemas import PITDataError
    import pytest

    class _ExplodingDart:
        def fetch_corporate_action_decisions(self, **_: object):
            raise AssertionError("must not be called")

        def fetch_dividend_disclosures(self, **_: object):
            raise AssertionError("must not be called")

    plan = _action_plan(
        symbols=(("005930", "00126380"),), start=date(2020, 1, 1), end=date(2020, 12, 31),
        endpoints=("fricDecsn.json",), reports=("11011",), tmp=tmp_path / "plans",
    )
    with pytest.raises(PITDataError):
        collect_planned_corporate_actions(
            plan=plan, dart=_ExplodingDart(), bronze_root=tmp_path / "bronze",
            retrieved_at=datetime(2026, 1, 1), checkpoint_store=CollectionCheckpointStore(tmp_path / "ckpt"),
        )
    with pytest.raises(PITDataError):
        collect_planned_corporate_actions(
            plan=CorporateActionCollectionPlan(plan.plan_id, plan.coverage_start, plan.coverage_end, plan.input_receipt_digest, (), (), "x"),
            dart=_ExplodingDart(), bronze_root=tmp_path / "bronze",
            retrieved_at=datetime(2026, 1, 1, tzinfo=UTC), checkpoint_store=CollectionCheckpointStore(tmp_path / "ckpt"),
        )


def test_action_page_helpers_support_dict_pages_and_reject_invalid_records(tmp_path):
    from datetime import UTC, date, datetime
    from src.data.collection import _action_page_field, _action_page_records, collect_planned_corporate_actions
    from src.data.collection_plan import CollectionCheckpointStore
    from src.data.schemas import PITDataError
    import pytest

    assert _action_page_field({"endpoint": "fricDecsn.json"}, "endpoint") == "fricDecsn.json"
    assert _action_page_field({"endpoint": None}, "endpoint") == ""
    assert _action_page_records({}) == []
    with pytest.raises(PITDataError, match="invalid records"):
        _action_page_records({"records": "bad"})
    with pytest.raises(PITDataError, match="invalid records"):
        _action_page_records({"records": [42]})

    plan = _action_plan(
        symbols=(("005930", "00126380"),), start=date(2020, 1, 1), end=date(2020, 12, 31),
        endpoints=("fricDecsn.json",), reports=("11011",), tmp=tmp_path / "plans",
    )

    class _DictDart:
        def fetch_corporate_action_decisions(self, *, corp_codes, start, end):
            assert corp_codes == ("00126380",)
            return ({"endpoint": "fricDecsn.json", "corp_code": "00126380", "status": "000", "records": []},)

        def fetch_dividend_disclosures(self, *, corp_codes, bsns_years):
            assert corp_codes == ("00126380",)
            return ({"corp_code": "00126380", "bsns_year": "2020", "reprt_code": "11011", "status": "013", "records": []},)

    artifact = collect_planned_corporate_actions(
        plan=plan, dart=_DictDart(), bronze_root=tmp_path / "bronze",
        retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
        checkpoint_store=CollectionCheckpointStore(tmp_path / "ckpt"),
    )
    assert artifact.planned_chunks == len(plan.requests)

    class _BadRecordsDart:
        def fetch_corporate_action_decisions(self, *, corp_codes, start, end):
            return ({"endpoint": "fricDecsn.json", "corp_code": "00126380", "status": "000", "records": [7]},)

        def fetch_dividend_disclosures(self, *, corp_codes, bsns_years):
            return ({"corp_code": "00126380", "bsns_year": "2020", "reprt_code": "11011", "status": "013", "records": []},)

    with pytest.raises(PITDataError, match="invalid records"):
        collect_planned_corporate_actions(
            plan=plan, dart=_BadRecordsDart(), bronze_root=tmp_path / "bronze-bad",
            retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
            checkpoint_store=CollectionCheckpointStore(tmp_path / "ckpt-bad"),
        )


def test_action_checkpoint_mismatch_is_not_counted_complete(tmp_path):
    import json
    from src.data.collection import _verified_action_receipts
    from src.data.collection_plan import CollectionCheckpointStore

    plan = _action_plan(symbols=(("005930", "00126380"),), tmp=tmp_path / "plans")
    store = CollectionCheckpointStore(tmp_path / "ckpt")
    request_id = plan.requests[0].request_id
    store.mark_complete(plan_id=plan.plan_id, chunk_id=request_id, receipt_digest="x" * 64, plan_digest="other-digest", receipt_hashes=("y" * 64,))
    assert _verified_action_receipts(checkpoint_store=store, plan=plan, request_id=request_id, bronze_root=tmp_path / "bronze") is None
    store.mark_complete(plan_id=plan.plan_id, chunk_id=request_id, receipt_digest="x" * 64, plan_digest=plan.content_hash, receipt_hashes=("", " "))
    assert _verified_action_receipts(checkpoint_store=store, plan=plan, request_id=request_id, bronze_root=tmp_path / "bronze") is None
    path = store._chunk_path(plan.plan_id, request_id)
    path.write_text(json.dumps({"plan_digest": plan.content_hash, "receipt_hashes": []}), encoding="utf-8")
    assert _verified_action_receipts(checkpoint_store=store, plan=plan, request_id=request_id, bronze_root=tmp_path / "bronze") is None


def test_build_corporate_action_plan_rejects_malformed_inputs(tmp_path):
    from datetime import date
    from pathlib import Path

    import pytest
    from src.data.collection_plan import HistoricalCollectionPlan, PlanChunk, build_corporate_action_collection_plan
    from src.data.schemas import PITDataError

    day = date(2020, 1, 1)
    good_chunks = (PlanChunk("h:005930:0000", "005930", (day,)),)
    base = {
        "ticker_to_corp_code": {"005930": "00126380"},
        "action_endpoints": ("fricDecsn.json",),
        "dividend_endpoint": "alotMatter.json",
        "dividend_report_codes": ("11011",),
        "artifact_root": tmp_path / "plans",
    }
    inverted = HistoricalCollectionPlan("inv", date(2020, 12, 31), day, 1, good_chunks, "i" * 64)
    with pytest.raises(PITDataError, match="coverage_start"):
        build_corporate_action_collection_plan(historical_plan=inverted, **base)
    hist = HistoricalCollectionPlan("h", day, date(2020, 12, 31), 1, good_chunks, "h" * 64)
    with pytest.raises(PITDataError, match="mapping"):
        build_corporate_action_collection_plan(historical_plan=hist, **{**base, "ticker_to_corp_code": [("005930", "00126380")]})
    with pytest.raises(PITDataError, match="blank instrument key"):
        build_corporate_action_collection_plan(historical_plan=hist, **{**base, "ticker_to_corp_code": {"  ": "00126380"}})
    empty = HistoricalCollectionPlan("e", day, date(2020, 12, 31), 1, (), "e" * 64)
    with pytest.raises(PITDataError, match="no eligible instruments"):
        build_corporate_action_collection_plan(historical_plan=empty, **base)
    undigested = HistoricalCollectionPlan("", day, date(2020, 12, 31), 1, good_chunks, "")
    with pytest.raises(PITDataError, match="digest is missing"):
        build_corporate_action_collection_plan(historical_plan=undigested, **base)
    with pytest.raises(PITDataError, match="receipt write failed"):
        build_corporate_action_collection_plan(historical_plan=hist, **{**base, "artifact_root": Path("/proc/stock-engine-action-test")})


def test_collect_planned_rejects_bad_plan_identity_and_statuses(tmp_path):
    from datetime import UTC, date, datetime

    import pytest
    from src.data.collection import collect_planned_corporate_actions
    from src.data.collection_plan import (
        CollectionCheckpointStore,
        CorporateActionCollectionPlan,
        CorporateActionRequest,
    )
    from src.data.schemas import PITDataError

    day = date(2020, 1, 1)
    one = CorporateActionRequest("ca-1", "005930", "00126380", "fricDecsn.json", None, None)
    good = CorporateActionCollectionPlan("cap-g", day, date(2020, 12, 31), "h" * 64, (one,), (), "d" * 64)
    inverted = CorporateActionCollectionPlan("cap-i", date(2020, 12, 31), day, "h" * 64, (one,), (), "d" * 64)
    dup = CorporateActionCollectionPlan("cap-d", day, date(2020, 12, 31), "h" * 64, (one, one), (), "d" * 64)

    class _NeverCalled:
        def fetch_corporate_action_decisions(self, **_: object):
            raise AssertionError("must not be called")

        def fetch_dividend_disclosures(self, **_: object):
            raise AssertionError("must not be called")

    for bad in (inverted, dup):
        with pytest.raises(PITDataError):
            collect_planned_corporate_actions(
                plan=bad, dart=_NeverCalled(), bronze_root=tmp_path / "bronze",
                retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
                checkpoint_store=CollectionCheckpointStore(tmp_path / "ckpt"),
            )

    broken_dividend = CorporateActionRequest("ca-9", "005930", "00126380", "alotMatter.json", "2020", None)
    broken_plan = CorporateActionCollectionPlan("cap-b", day, date(2020, 12, 31), "h" * 64, (broken_dividend,), (), "d" * 64)
    with pytest.raises(PITDataError, match="fiscal identity"):
        collect_planned_corporate_actions(
            plan=broken_plan, dart=_NeverCalled(), bronze_root=tmp_path / "bronze",
            retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
            checkpoint_store=CollectionCheckpointStore(tmp_path / "ckpt"),
        )

    class _BadStatusDart:
        def __init__(self, status, records):
            self.status = status
            self.records = records

        def fetch_corporate_action_decisions(self, *, corp_codes, start, end):
            return ({"endpoint": "fricDecsn.json", "corp_code": "00126380", "status": self.status, "records": self.records},)

        def fetch_dividend_disclosures(self, **_: object):
            raise AssertionError("unexpected call")

    with pytest.raises(PITDataError, match="unexpected OpenDART status"):
        collect_planned_corporate_actions(
            plan=good, dart=_BadStatusDart("999", []), bronze_root=tmp_path / "bronze-s",
            retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
            checkpoint_store=CollectionCheckpointStore(tmp_path / "ckpt-s"),
        )
    with pytest.raises(PITDataError, match="carries records"):
        collect_planned_corporate_actions(
            plan=good, dart=_BadStatusDart("013", [{"r": 1}]), bronze_root=tmp_path / "bronze-e",
            retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
            checkpoint_store=CollectionCheckpointStore(tmp_path / "ckpt-e"),
        )

    class _NoneDart:
        def fetch_corporate_action_decisions(self, *, corp_codes, start, end):
            return None

        def fetch_dividend_disclosures(self, **_: object):
            raise AssertionError("unexpected call")

    with pytest.raises(PITDataError, match="identity mismatch"):
        collect_planned_corporate_actions(
            plan=good, dart=_NoneDart(), bronze_root=tmp_path / "bronze-n",
            retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
            checkpoint_store=CollectionCheckpointStore(tmp_path / "ckpt-n"),
        )


def test_collect_planned_report_reconciles_to_immutable_plan(tmp_path):
    from datetime import UTC, date, datetime

    import pytest
    from src.data.collection import collect_planned_corporate_actions
    from src.data.collection_plan import CollectionCheckpointStore, CorporateActionRequest
    from src.data.schemas import PITDataError
    from types import SimpleNamespace

    day = date(2020, 1, 1)
    one = CorporateActionRequest("ca-1", "005930", "00126380", "fricDecsn.json", None, None)

    class _LyingRequests:
        def __iter__(self):
            return iter((one,))

        def __len__(self):
            return 2

    lying = SimpleNamespace(
        plan_id="cap-lie", coverage_start=day, coverage_end=date(2020, 12, 31),
        content_hash="d" * 64, requests=_LyingRequests(), unresolved_instruments=(),
    )

    class _OkDart:
        def fetch_corporate_action_decisions(self, *, corp_codes, start, end):
            return ({"endpoint": "fricDecsn.json", "corp_code": "00126380", "status": "000", "records": []},)

        def fetch_dividend_disclosures(self, **_: object):
            raise AssertionError("unexpected call")

    with pytest.raises(PITDataError, match="does not reconcile"):
        collect_planned_corporate_actions(
            plan=lying, dart=_OkDart(), bronze_root=tmp_path / "bronze",
            retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
            checkpoint_store=CollectionCheckpointStore(tmp_path / "ckpt"),
        )


def test_collect_planned_swapped_and_tampered_receipts_fail_revalidation(tmp_path):
    import json
    from datetime import UTC, date, datetime
    from src.data.collection import _verified_action_receipts, collect_planned_corporate_actions
    from src.data.collection_plan import CollectionCheckpointStore
    from src.data.schemas import PITDataError
    import pytest

    plan = _action_plan(
        symbols=(("005930", "00126380"),), start=date(2020, 1, 1), end=date(2020, 12, 31),
        endpoints=("fricDecsn.json",), reports=("11011",), tmp=tmp_path / "plans",
    )
    bronze = tmp_path / "bronze"
    store = CollectionCheckpointStore(tmp_path / "ckpt")
    collect_planned_corporate_actions(
        plan=plan, dart=_ActionDart({"005930": "00126380"}), bronze_root=bronze,
        retrieved_at=datetime(2026, 1, 1, tzinfo=UTC), checkpoint_store=store,
    )
    first, second = plan.requests[0], plan.requests[1]
    first_path = store._chunk_path(plan.plan_id, first.request_id)
    second_path = store._chunk_path(plan.plan_id, second.request_id)
    first_stored = json.loads(first_path.read_text(encoding="utf-8"))
    second_stored = json.loads(second_path.read_text(encoding="utf-8"))
    first_path.write_text(json.dumps({**first_stored, "receipt_hashes": second_stored["receipt_hashes"]}), encoding="utf-8")
    assert _verified_action_receipts(checkpoint_store=store, plan=plan, request_id=first.request_id, bronze_root=bronze) is None

    meta_path = bronze / "corporate_actions" / str(second_stored["receipt_hashes"][0]) / "receipt.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta_path.write_text(json.dumps({**meta, "retrieved_at": "not-a-date"}), encoding="utf-8")
    with pytest.raises(PITDataError, match="unreadable"):
        _verified_action_receipts(checkpoint_store=store, plan=plan, request_id=second.request_id, bronze_root=bronze)


def test_collect_planned_resumed_empty_pages_count_as_empty(tmp_path):
    from datetime import UTC, date, datetime
    from src.data.collection import collect_planned_corporate_actions
    from src.data.collection_plan import CollectionCheckpointStore
    from src.data.schemas import EvidenceKind

    plan = _action_plan(
        symbols=(("005930", "00126380"),), start=date(2020, 1, 1), end=date(2020, 12, 31),
        endpoints=("fricDecsn.json",), reports=("11011",), tmp=tmp_path / "plans",
    )
    bronze = tmp_path / "bronze"
    checkpoint_store = CollectionCheckpointStore(tmp_path / "ckpt")
    collect_planned_corporate_actions(
        plan=plan, dart=_ActionDart({"005930": "00126380"}, status="013"), bronze_root=bronze,
        retrieved_at=datetime(2026, 1, 1, tzinfo=UTC), checkpoint_store=checkpoint_store,
    )
    quiet = _ActionDart({"005930": "00126380"}, status="013")
    rerun = collect_planned_corporate_actions(
        plan=plan, dart=quiet, bronze_root=bronze,
        retrieved_at=datetime(2026, 1, 1, tzinfo=UTC), checkpoint_store=checkpoint_store,
    )
    assert quiet.calls == {"structured": 0, "dividend": 0}
    assert rerun.previously_completed_chunks == len(plan.requests)
    assert len(rerun.page_receipts[EvidenceKind.CORPORATE_ACTIONS.value]) == len(plan.requests)


def test_collect_planned_unresolved_blocks_completion_with_ledger(tmp_path):
    import json
    from datetime import UTC, date, datetime

    import pytest
    from src.data.collection import collect_planned_corporate_actions
    from src.data.collection_plan import HistoricalCollectionPlan, PlanChunk, build_corporate_action_collection_plan, CollectionCheckpointStore
    from src.data.schemas import PITDataError

    day = date(2020, 1, 1)
    hist = HistoricalCollectionPlan(
        "hist-n", day, date(2020, 12, 31), 1,
        (PlanChunk("hist-n:a:0000", "005930", (day,)), PlanChunk("hist-n:b:0000", "999999", (day,))),
        "n" * 64,
    )
    plan = build_corporate_action_collection_plan(
        historical_plan=hist,
        ticker_to_corp_code={"005930": "00126380"},
        action_endpoints=("fricDecsn.json",),
        dividend_endpoint="alotMatter.json",
        dividend_report_codes=("11011",),
        artifact_root=tmp_path / "plans",
    )
    with pytest.raises(PITDataError, match="999999"):
        collect_planned_corporate_actions(
            plan=plan, dart=_ActionDart({"005930": "00126380"}), bronze_root=tmp_path / "bronze",
            retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
            checkpoint_store=CollectionCheckpointStore(tmp_path / "ckpt"),
        )
    ledgers = list((tmp_path / "bronze").parent.glob("artifacts/collections/*-corporate-actions.json"))
    assert len(ledgers) == 1
    assert json.loads(ledgers[0].read_text(encoding="utf-8"))["unresolved_instrument_count"] == 1


def test_action_revalidation_rejects_foreign_plan_receipts(tmp_path):
    import json
    from datetime import UTC, datetime
    from src.data.bronze import BronzeStore
    from src.data.collection import _verified_action_receipts
    from src.data.collection_plan import CollectionCheckpointStore
    from src.data.schemas import EvidenceKind

    plan = _action_plan(
        symbols=(("005930", "00126380"),),
        start=__import__("datetime").date(2020, 1, 1),
        end=__import__("datetime").date(2020, 12, 31),
        endpoints=("fricDecsn.json",),
        reports=("11011",),
        tmp=tmp_path / "plans",
    )
    bronze = tmp_path / "bronze"
    store = BronzeStore(bronze)
    checkpoint_store = CollectionCheckpointStore(tmp_path / "ckpt")
    stamp = datetime(2026, 1, 1, tzinfo=UTC)
    for index, request in enumerate(plan.requests):
        foreign = {
            "request_id": request.request_id,
            "requested_instrument_id": request.requested_instrument_id,
            "corp_code": request.corp_code,
            "endpoint": request.endpoint,
            "bsns_year": request.bsns_year,
            "reprt_code": request.reprt_code,
            "coverage_start": plan.coverage_start.isoformat(),
            "coverage_end": plan.coverage_end.isoformat(),
            "status": "000",
            "records": [],
            "plan_id": "cap-other" if index == 0 else plan.plan_id,
            "plan_hash": plan.content_hash if index == 0 else "0" * 64,
            "instrument_mapping_provenance": "opendart_corp_code_direct",
        }
        receipt = store.import_bytes(
            json.dumps(foreign, sort_keys=True, ensure_ascii=False).encode("utf-8"),
            kind=EvidenceKind.CORPORATE_ACTIONS, retrieved_at=stamp, source_label=f"foreign:{index}",
        )
        checkpoint_store.mark_complete(
            plan_id=plan.plan_id, chunk_id=request.request_id, receipt_digest=receipt.content_hash,
            plan_digest=plan.content_hash, receipt_hashes=(receipt.content_hash,),
        )
        assert _verified_action_receipts(
            checkpoint_store=checkpoint_store, plan=plan, request_id=request.request_id, bronze_root=bronze,
        ) is None


def test_collect_historical_evidence_corporate_action_route(tmp_path):
    import json
    from datetime import UTC, date, datetime
    from types import SimpleNamespace
    from src.data.collection import collect_historical_evidence
    from src.data.collection_plan import HistoricalCollectionPlan, PlanChunk
    from src.data.schemas import EvidenceKind

    day = date(2020, 1, 1)
    plan = HistoricalCollectionPlan(
        "hist-w", day, date(2020, 12, 31), 1, (PlanChunk("hist-w:005930:0000", "005930", (day,)),), "w" * 64
    )

    class _WireDart:
        def load_corp_codes(self):
            return {"005930": "00126380"}

        def fetch_corporate_action_decisions(self, *, corp_codes, start, end):
            assert start == day
            assert end == date(2020, 12, 31)
            return tuple(
                SimpleNamespace(endpoint=endpoint, corp_code="00126380", status="013", records=())
                for endpoint in ("fricDecsn.json", "crDecsn.json", "piicDecsn.json", "cmpDvDecsn.json", "cmpMgDecsn.json")
            )

        def fetch_dividend_disclosures(self, *, corp_codes, bsns_years):
            assert bsns_years == ("2020",)
            return tuple(
                SimpleNamespace(corp_code="00126380", bsns_year="2020", reprt_code=code, status="000", records=())
                for code in ("11011", "11012", "11013", "11014")
            )

    results = collect_historical_evidence(
        plan=plan, krx=object(), investor_flow=None, investor_flow_provider="ls", dart=_WireDart(),
        bronze_root=tmp_path / "bronze", checkpoint_root=tmp_path / "checkpoints",
        retrieved_at=datetime(2026, 1, 1, tzinfo=UTC), kinds=frozenset({EvidenceKind.CORPORATE_ACTIONS}),
    )
    artifact = results[EvidenceKind.CORPORATE_ACTIONS]
    assert artifact.planned_chunks == 5 + 4
    ledger = json.loads(artifact.report_path.read_text(encoding="utf-8"))
    assert ledger["planned"] == 9
    assert ledger["pending"] == 0


def _flow_backfill_plan(count: int, *, plan_id: str = "backfill-plan", digest: str = "b" * 64) -> object:
    from datetime import date

    from src.data.collection_plan import HistoricalCollectionPlan, PlanChunk

    day = date(2026, 3, 6)
    chunks = tuple(PlanChunk(f"{plan_id}:{index:04d}", "005930", (day,)) for index in range(count))
    return HistoricalCollectionPlan(plan_id, day, day, 1, chunks, digest)


class _BackfillSuccessCollector:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...] | None] = []
        self.closed = 0

    def fetch_investor_flow(self, start: object, end: object, **kwargs: object) -> tuple[dict[str, object], ...]:
        symbols = kwargs.get("symbols")
        assert isinstance(symbols, tuple)
        self.calls.append(symbols)
        return (
            {
                "records": [
                    {
                        "ticker": "005930",
                        "session": "20260306",
                        "unit": "shares",
                        "individual_net_shares": "-3",
                        "foreign_net_shares": "1",
                        "institution_net_shares": "2",
                        "other_net_shares": "0",
                    }
                ]
            },
        )

    def close(self) -> None:
        self.closed += 1


def test_backfill_iterator_reuses_one_collector_for_all_batches(tmp_path, monkeypatch) -> None:
    from datetime import UTC, datetime

    import src.data.collection as collection_module
    from src.data.collection import iter_planned_investor_flow_backfill
    from src.data.collection_plan import CollectionCheckpointStore

    plan = _flow_backfill_plan(6)
    collector = _BackfillSuccessCollector()
    seen: list[int] = []
    real = collection_module.collect_planned_investor_flow

    def recording(**kwargs: object) -> object:
        seen.append(id(kwargs["collector"]))
        return real(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(collection_module, "collect_planned_investor_flow", recording)
    progress = list(
        iter_planned_investor_flow_backfill(
            plan=plan,  # type: ignore[arg-type]
            provider="ls",
            collector=collector,
            bronze_root=tmp_path / "bronze",
            retrieved_at_factory=lambda: datetime(2026, 9, 20, tzinfo=UTC),
            checkpoint_store=CollectionCheckpointStore(tmp_path / "ckpt"),
            chunk_batch_size=2,
        )
    )
    assert len(progress) == 3
    assert seen == [id(collector)] * 3
    assert len(collector.calls) == 6


def test_backfill_iterator_yields_contiguous_stable_slices(tmp_path, monkeypatch) -> None:
    from datetime import UTC, datetime

    import src.data.collection as collection_module
    from src.data.collection import iter_planned_investor_flow_backfill
    from src.data.collection_plan import CollectionCheckpointStore

    plan = _flow_backfill_plan(5)
    captured: list[tuple[int, tuple[str, ...]]] = []
    real = collection_module.collect_planned_investor_flow

    def recording(**kwargs: object) -> object:
        sub = kwargs["plan"]
        captured.append((len(captured), tuple(chunk.chunk_id for chunk in sub.chunks)))
        return real(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(collection_module, "collect_planned_investor_flow", recording)
    progress = list(
        iter_planned_investor_flow_backfill(
            plan=plan,  # type: ignore[arg-type]
            provider="ls",
            collector=_BackfillSuccessCollector(),
            bronze_root=tmp_path / "bronze",
            retrieved_at_factory=lambda: datetime(2026, 9, 20, tzinfo=UTC),
            checkpoint_store=CollectionCheckpointStore(tmp_path / "ckpt"),
            chunk_batch_size=2,
        )
    )
    assert [item.chunk_offset for item in progress] == [0, 2, 4]
    assert [chunk for _, ids in captured for chunk in ids] == [chunk.chunk_id for chunk in plan.chunks]  # type: ignore[union-attr]


def test_backfill_iterator_preserves_plan_identity_for_checkpoints(tmp_path) -> None:
    import json
    from datetime import UTC, datetime

    from src.data.collection import iter_planned_investor_flow_backfill
    from src.data.collection_plan import CollectionCheckpointStore

    plan = _flow_backfill_plan(2, plan_id="custom-plan", digest="a" * 64)
    store = CollectionCheckpointStore(tmp_path / "ckpt")
    list(
        iter_planned_investor_flow_backfill(
            plan=plan,  # type: ignore[arg-type]
            provider="ls",
            collector=_BackfillSuccessCollector(),
            bronze_root=tmp_path / "bronze",
            retrieved_at_factory=lambda: datetime(2026, 9, 20, tzinfo=UTC),
            checkpoint_store=store,
            chunk_batch_size=2,
        )
    )
    for chunk in plan.chunks:  # type: ignore[union-attr]
        stored = json.loads(store._chunk_path("custom-plan", chunk.chunk_id).read_text(encoding="utf-8"))
        assert stored["plan_digest"] == "a" * 64


def test_backfill_iterator_skips_verified_checkpoint_chunk(tmp_path) -> None:
    from datetime import UTC, datetime

    from src.data.collection import collect_planned_investor_flow, iter_planned_investor_flow_backfill
    from src.data.collection_plan import CollectionCheckpointStore

    plan = _flow_backfill_plan(2)
    bronze = tmp_path / "bronze"
    store = CollectionCheckpointStore(tmp_path / "ckpt")
    first_chunk = plan.chunks[0]  # type: ignore[union-attr]
    from src.data.collection_plan import HistoricalCollectionPlan

    single = HistoricalCollectionPlan(
        plan.plan_id, plan.coverage_start, plan.coverage_end, 1, (first_chunk,), plan.content_hash  # type: ignore[union-attr]
    )
    collect_planned_investor_flow(
        plan=single,
        provider="ls",
        collector=_BackfillSuccessCollector(),
        bronze_root=bronze,
        retrieved_at=datetime(2026, 9, 20, tzinfo=UTC),
        checkpoint_store=store,
    )
    collector = _BackfillSuccessCollector()
    progress = list(
        iter_planned_investor_flow_backfill(
            plan=plan,  # type: ignore[arg-type]
            provider="ls",
            collector=collector,
            bronze_root=bronze,
            retrieved_at_factory=lambda: datetime(2026, 9, 20, tzinfo=UTC),
            checkpoint_store=store,
            chunk_batch_size=2,
        )
    )
    assert len(collector.calls) == 1
    assert progress[0].artifact.previously_completed_chunks == 1


def test_backfill_iterator_records_provider_error_once_and_continues(tmp_path) -> None:
    import json
    from datetime import UTC, date, datetime

    from src.data.collection import iter_planned_investor_flow_backfill
    from src.data.collection_plan import CollectionCheckpointStore, HistoricalCollectionPlan, PlanChunk
    from src.data.schemas import PITDataError

    day = date(2026, 3, 6)
    plan = HistoricalCollectionPlan(
        "err-plan",
        day,
        day,
        1,
        (PlanChunk("err-plan:0000", "000001", (day,)), PlanChunk("err-plan:0001", "005930", (day,))),
        "e" * 64,
    )

    class Flaky:
        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []

        def fetch_investor_flow(self, start: object, end: object, **kwargs: object) -> tuple[dict[str, object], ...]:
            symbols = kwargs["symbols"]
            assert isinstance(symbols, tuple)
            self.calls.append(symbols)
            if symbols == ("000001",):
                raise PITDataError("boom")
            return (
                {
                    "records": [
                        {
                            "ticker": "005930",
                            "session": "20260306",
                            "unit": "shares",
                            "individual_net_shares": "-3",
                            "foreign_net_shares": "1",
                            "institution_net_shares": "2",
                            "other_net_shares": "0",
                        }
                    ]
                },
            )

    collector = Flaky()
    progress = list(
        iter_planned_investor_flow_backfill(
            plan=plan,
            provider="ls",
            collector=collector,
            bronze_root=tmp_path / "bronze",
            retrieved_at_factory=lambda: datetime(2026, 9, 20, tzinfo=UTC),
            checkpoint_store=CollectionCheckpointStore(tmp_path / "ckpt"),
            chunk_batch_size=2,
        )
    )
    assert collector.calls.count(("000001",)) == 1
    assert progress[0].artifact.provider_error_chunks == 1
    assert progress[0].artifact.completed_chunks == 1
    for payload_path in (tmp_path / "bronze" / "investor_flow").glob("*/payload.json"):
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        records = payload.get("records") or []
        for record in records:
            assert not (
                record.get("ticker") == "000001"
                and record.get("foreign_net_value") == 0
                and record.get("institution_net_value") == 0
            )


def test_backfill_iterator_bounds_failure_accounting_per_batch(tmp_path) -> None:
    from datetime import UTC, datetime

    from src.data.collection import iter_planned_investor_flow_backfill
    from src.data.collection_plan import CollectionCheckpointStore

    class AlwaysFail:
        def fetch_investor_flow(self, *_args: object, **_kwargs: object) -> tuple[dict[str, object], ...]:
            raise TimeoutError("timed out")

    progress = list(
        iter_planned_investor_flow_backfill(
            plan=_flow_backfill_plan(6),  # type: ignore[arg-type]
            provider="ls",
            collector=AlwaysFail(),
            bronze_root=tmp_path / "bronze",
            retrieved_at_factory=lambda: datetime(2026, 9, 20, tzinfo=UTC),
            checkpoint_store=CollectionCheckpointStore(tmp_path / "ckpt"),
            chunk_batch_size=2,
        )
    )
    assert len(progress) == 3
    for item in progress:
        assert item.artifact.planned_chunks == 2
        assert item.artifact.provider_error_chunks == 2
        assert len(item.artifact.page_receipts["investor_flow"]) == 2


def test_backfill_iterator_rejects_uncertified_provider_and_factory(tmp_path) -> None:
    from datetime import UTC, datetime

    import pytest

    from src.data.collection import iter_planned_investor_flow_backfill
    from src.data.collection_plan import CollectionCheckpointStore
    from src.data.schemas import PITDataError

    for bad_provider in ("kis", "toss", "krx"):
        collector = _BackfillSuccessCollector()
        with pytest.raises(PITDataError):
            iter_planned_investor_flow_backfill(
                plan=_flow_backfill_plan(1),  # type: ignore[arg-type]
                provider=bad_provider,
                collector=collector,
                bronze_root=tmp_path / "bronze",
                retrieved_at_factory=lambda: datetime(2026, 9, 20, tzinfo=UTC),
                checkpoint_store=CollectionCheckpointStore(tmp_path / "ckpt"),
                chunk_batch_size=1,
            )
        assert collector.calls == []

    with pytest.raises(PITDataError):
        iter_planned_investor_flow_backfill(
            plan=_flow_backfill_plan(1),  # type: ignore[arg-type]
            provider="ls",
            collector=_BackfillSuccessCollector(),
            bronze_root=tmp_path / "bronze",
            retrieved_at_factory="not-callable",  # type: ignore[arg-type]
            checkpoint_store=CollectionCheckpointStore(tmp_path / "ckpt"),
            chunk_batch_size=1,
        )


def test_backfill_iterator_rejects_invalid_batch_size_without_request(tmp_path) -> None:
    import pytest

    from src.data.collection import iter_planned_investor_flow_backfill
    from src.data.collection_plan import CollectionCheckpointStore
    from src.data.schemas import PITDataError
    from datetime import UTC, datetime

    for bad in (0, -1, True, False):
        collector = _BackfillSuccessCollector()

        with pytest.raises(PITDataError):
            iter_planned_investor_flow_backfill(
                plan=_flow_backfill_plan(1),  # type: ignore[arg-type]
                provider="ls",
                collector=collector,
                bronze_root=tmp_path / "bronze",
                retrieved_at_factory=lambda: datetime(2026, 9, 20, tzinfo=UTC),
                checkpoint_store=CollectionCheckpointStore(tmp_path / "ckpt"),
                chunk_batch_size=bad,
            )
        assert collector.calls == []


def test_backfill_iterator_rejects_naive_retrieval_timestamp(tmp_path) -> None:
    from datetime import datetime

    import pytest

    from src.data.collection import iter_planned_investor_flow_backfill
    from src.data.collection_plan import CollectionCheckpointStore
    from src.data.schemas import PITDataError

    with pytest.raises(PITDataError):
        list(
            iter_planned_investor_flow_backfill(
                plan=_flow_backfill_plan(1),  # type: ignore[arg-type]
                provider="ls",
                collector=_BackfillSuccessCollector(),
                bronze_root=tmp_path / "bronze",
                retrieved_at_factory=lambda: datetime(2026, 9, 20),
                checkpoint_store=CollectionCheckpointStore(tmp_path / "ckpt"),
                chunk_batch_size=1,
            )
        )
    assert not list((tmp_path / "bronze" / "investor_flow").glob("*/payload.json"))


def test_backfill_iterator_closes_collector_on_exhaustion_and_exception(tmp_path) -> None:
    from datetime import UTC, datetime

    import pytest

    from src.data.collection import iter_planned_investor_flow_backfill
    from src.data.collection_plan import CollectionCheckpointStore

    done = _BackfillSuccessCollector()
    list(
        iter_planned_investor_flow_backfill(
            plan=_flow_backfill_plan(2),  # type: ignore[arg-type]
            provider="ls",
            collector=done,
            bronze_root=tmp_path / "bronze-ok",
            retrieved_at_factory=lambda: datetime(2026, 9, 20, tzinfo=UTC),
            checkpoint_store=CollectionCheckpointStore(tmp_path / "ckpt-ok"),
            chunk_batch_size=2,
        )
    )
    assert done.closed == 1

    class Boom:
        def __init__(self) -> None:
            self.closed = 0

        def fetch_investor_flow(self, *_args: object, **_kwargs: object) -> tuple[dict[str, object], ...]:
            raise RuntimeError("boom")

        def close(self) -> None:
            self.closed += 1

    failing = Boom()
    with pytest.raises(RuntimeError, match="boom"):
        list(
            iter_planned_investor_flow_backfill(
                plan=_flow_backfill_plan(1),  # type: ignore[arg-type]
                provider="ls",
                collector=failing,
                bronze_root=tmp_path / "bronze-boom",
                retrieved_at_factory=lambda: datetime(2026, 9, 20, tzinfo=UTC),
                checkpoint_store=CollectionCheckpointStore(tmp_path / "ckpt-boom"),
                chunk_batch_size=1,
                allow_source_unavailable=False,
            )
        )
    assert failing.closed == 1

    class NoClose:
        def fetch_investor_flow(self, *_args: object, **_kwargs: object) -> tuple[dict[str, object], ...]:
            return (
                {
                    "records": [
                        {
                            "ticker": "005930",
                            "session": "20260306",
                            "unit": "shares",
                            "individual_net_shares": "-3",
                            "foreign_net_shares": "1",
                            "institution_net_shares": "2",
                            "other_net_shares": "0",
                        }
                    ]
                },
            )

    progress = list(
        iter_planned_investor_flow_backfill(
            plan=_flow_backfill_plan(1),  # type: ignore[arg-type]
            provider="ls",
            collector=NoClose(),
            bronze_root=tmp_path / "bronze-plain",
            retrieved_at_factory=lambda: datetime(2026, 9, 20, tzinfo=UTC),
            checkpoint_store=CollectionCheckpointStore(tmp_path / "ckpt-plain"),
            chunk_batch_size=1,
        )
    )
    assert len(progress) == 1


def test_ls_chunk_validates_share_records_and_checkpoints(tmp_path) -> None:
    from datetime import UTC, date, datetime

    from src.data.collection import _validate_flow_pages, collect_planned_investor_flow
    from src.data.collection_plan import CollectionCheckpointStore, HistoricalCollectionPlan, PlanChunk

    day = date(2026, 3, 6)
    plan = HistoricalCollectionPlan("ls-share-plan", day, day, 1, (PlanChunk("ls-share-plan:0000", "005930", (day,)),), "f" * 64)
    record = {
        "ticker": "005930",
        "session": "20260306",
        "unit": "shares",
        "_source_provider": "LS",
        "individual_net_shares": 927,
        "foreign_net_shares": -828,
        "institution_net_shares": -228,
        "other_net_shares": 129,
    }

    class ShareCollector:
        def fetch_investor_flow(self, start: object, end: object, **kwargs: object) -> tuple[dict[str, object], ...]:
            return ({"provider": "ls", "records": [dict(record)]},)

    store = CollectionCheckpointStore(tmp_path / "ckpt")
    artifact = collect_planned_investor_flow(
        plan=plan,
        provider="ls",
        collector=ShareCollector(),
        bronze_root=tmp_path / "bronze",
        retrieved_at=datetime(2026, 9, 20, tzinfo=UTC),
        checkpoint_store=store,
    )
    assert artifact.completed_chunks == 1
    assert store.has_verified_receipt(plan=plan, chunk=plan.chunks[0], bronze_root=tmp_path / "bronze")
    canonical = _validate_flow_pages(
        chunk_symbol="005930", norm_provider="ls", pages=({"provider": "ls", "records": [dict(record)]},)
    )
    assert canonical["2026-03-06"] == {
        "individual_net_shares": 927.0,
        "foreign_net_shares": -828.0,
        "institution_net_shares": -228.0,
        "other_net_shares": 129.0,
    }


def test_ls_legacy_value_records_rejected_without_checkpoint(tmp_path) -> None:
    from datetime import UTC, date, datetime

    import pytest

    from src.data.collection import collect_planned_investor_flow
    from src.data.collection_plan import CollectionCheckpointStore, HistoricalCollectionPlan, PlanChunk
    from src.data.schemas import PITDataError

    day = date(2026, 3, 6)
    plan = HistoricalCollectionPlan("ls-legacy-plan", day, day, 1, (PlanChunk("ls-legacy-plan:0000", "005930", (day,)),), "e" * 64)

    class LegacyCollector:
        def fetch_investor_flow(self, start: object, end: object, **kwargs: object) -> tuple[dict[str, object], ...]:
            return ({"provider": "ls", "records": [{"ticker": "005930", "session": "20260306", "foreign_net_value": "1"}]},)

    store = CollectionCheckpointStore(tmp_path / "ckpt")
    with pytest.raises(PITDataError, match="unit contract"):
        collect_planned_investor_flow(
            plan=plan,
            provider="ls",
            collector=LegacyCollector(),
            bronze_root=tmp_path / "bronze",
            retrieved_at=datetime(2026, 9, 20, tzinfo=UTC),
            checkpoint_store=store,
        )
    assert not store._chunk_path("ls-legacy-plan", "ls-legacy-plan:0000").exists()


def test_non_ls_provider_keeps_krw_value_contract(tmp_path) -> None:
    from datetime import UTC, date, datetime

    from src.data.collection import _validate_flow_pages, collect_planned_investor_flow
    from src.data.collection_plan import CollectionCheckpointStore, HistoricalCollectionPlan, PlanChunk

    day = date(2026, 3, 6)
    plan = HistoricalCollectionPlan("kw-plan", day, day, 1, (PlanChunk("kw-plan:0000", "005930", (day,)),), "d" * 64)
    record = {
        "ticker": "005930",
        "session": "20260306",
        "foreign_net_value": "1",
        "institution_net_value": "2",
        "retail_net_value": "-3",
    }

    class KiwoomCollector:
        def fetch_investor_flow(self, start: object, end: object, **kwargs: object) -> tuple[dict[str, object], ...]:
            return ({"provider": "kiwoom", "records": [dict(record)]},)

    artifact = collect_planned_investor_flow(
        plan=plan,
        provider="kiwoom",
        collector=KiwoomCollector(),
        bronze_root=tmp_path / "bronze",
        retrieved_at=datetime(2026, 9, 20, tzinfo=UTC),
        checkpoint_store=CollectionCheckpointStore(tmp_path / "ckpt"),
    )
    assert artifact.completed_chunks == 1
    canonical = _validate_flow_pages(
        chunk_symbol="005930", norm_provider="kiwoom", pages=({"provider": "kiwoom", "records": [dict(record)]},)
    )
    assert canonical["2026-03-06"] == {"foreign_net_value": 1.0, "institution_net_value": 2.0, "retail_net_value": -3.0}


def test_ls_share_field_shapes_rejected_under_unit_contract() -> None:
    import pytest

    from src.data.collection import _validate_flow_pages
    from src.data.schemas import PITDataError

    def base_record(**overrides):
        record = {
            "ticker": "005930",
            "session": "20260306",
            "unit": "shares",
            "individual_net_shares": 927,
            "foreign_net_shares": -828,
            "institution_net_shares": -228,
            "other_net_shares": 129,
        }
        record.update(overrides)
        return record

    with pytest.raises(PITDataError, match="unit contract"):
        _validate_flow_pages(chunk_symbol="005930", norm_provider="ls", pages=({"records": [base_record(foreign_net_shares=None)]},))
    with pytest.raises(PITDataError, match="unit contract"):
        _validate_flow_pages(chunk_symbol="005930", norm_provider="ls", pages=({"records": [base_record(foreign_net_shares=True)]},))
    with pytest.raises(PITDataError, match="unit contract"):
        _validate_flow_pages(chunk_symbol="005930", norm_provider="ls", pages=({"records": [base_record(foreign_net_shares="abc")]},))
    with pytest.raises(PITDataError, match="unit contract"):
        _validate_flow_pages(chunk_symbol="005930", norm_provider="ls", pages=({"records": [base_record(foreign_net_shares="1.5")]},))
    with pytest.raises(PITDataError, match="unit contract"):
        _validate_flow_pages(
            chunk_symbol="005930",
            norm_provider="ls",
            pages=({"records": [dict(base_record(), foreign_net_value=1.0)]},),
        )


def test_collect_dart_financial_facts_persists_blocked_pages_and_counts_them(tmp_path) -> None:
    import json
    from datetime import UTC, datetime

    from src.data.collection import collect_dart_financial_facts
    from src.data.schemas import EvidenceKind

    def _page(source_kind: str, filing_id: str, status: str = "000") -> dict[str, object]:
        return {
            "source_kind": source_kind,
            "status": status,
            "identity": {"corp_code": "00126380", "biz_year": "2020", "reprt_code": "11011"},
            "records": [] if source_kind != "opendart_standard" else [{"fact": "sales"}],
            "filing_id": filing_id,
            "corp_code": "00126380",
            "biz_year": "2020",
            "reprt_code": "11011",
        }

    pages = (
        _page("opendart_standard", "F1"),
        _page("opendart_standard", "F2"),
        {**_page("blocked", "F3", status="020"), "diagnostics": ("dart_quota_exhausted",)},
        _page("unavailable", "F4", status="013"),
    )

    class _FakeDart:
        def fetch_financial_fact_sources(self, identities: object) -> tuple[dict[str, object], ...]:
            assert identities
            return pages

    identities = tuple(
        {"corp_code": "00126380", "filing_id": f"F{i}", "biz_year": "2020", "reprt_code": "11011"}
        for i in range(1, 5)
    )
    artifact = collect_dart_financial_facts(
        dart=_FakeDart(),
        identities=identities,  # type: ignore[arg-type]
        bronze_root=tmp_path / "bronze",
        retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    # Then: never aborts, all 4 pages persisted, blocked counted once.
    assert artifact.page_receipts is not None
    assert len(artifact.page_receipts[EvidenceKind.FINANCIAL_FACTS.value]) == 4
    report = json.loads(artifact.report_path.read_text(encoding="utf-8"))
    assert report["blocked"] == 1
    assert report["standardized"] == 2
    assert report["unavailable"] == 1
