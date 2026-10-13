"""KIND relay bundle invariants: canonical encoding, strict decoding, freshness."""
from __future__ import annotations

import base64
import copy
import hashlib
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from tests.fixtures.kind_html import kind_row_html, kind_search_html

FETCHED_AT = datetime(2026, 9, 24, 3, 0, tzinfo=UTC)
NOW = FETCHED_AT + timedelta(hours=1)


def _provider():  # type: ignore[no-untyped-def]
    from src.config import load_provider_policy, load_runtime_config

    return load_provider_policy(load_runtime_config())


def _bundle():  # type: ignore[no-untyped-def]
    from src.data.jobs.kind import collect_notice_document, collect_search_window
    from src.data.kind_relay import RelayBundle
    from src.integrations.krx.kind import parse_kind_search_page

    rows_html = [
        kind_row_html("20240101000001", "2024-01-02 15:30", "상장폐지", "005930", "유가증권시장본부"),
        kind_row_html("20240101000002", "2024-01-02 15:31", "상장폐지", "005930", "유가증권시장본부"),
    ]

    class _Collector:
        def search_page(self, keyword, start, end, *, page_index):  # type: ignore[no-untyped-def]
            return kind_search_html(rows_html, 2)

        def fetch_document(self, acptno):  # type: ignore[no-untyped-def]
            from src.integrations.krx.kind import KindNoticeDocument

            return KindNoticeDocument(acptno=acptno, doc_no="1", template="a.htm", html="<html>x</html>")

    collector = _Collector()
    search = collect_search_window(
        collector, keyword="상장폐지", start=date(2019, 7, 1), end=date(2019, 9, 30),
        today=date(2019, 10, 1), retrieved_at=FETCHED_AT,
    )
    (row,) = [row for row in parse_kind_search_page(kind_search_html(rows_html, 2)).rows
              if row.acptno == "20240101000001"]
    document = collect_notice_document(collector, row, retrieved_at=FETCHED_AT)
    fetch_date_kst = (FETCHED_AT + timedelta(hours=9)).date()
    return RelayBundle(
        fetched_at=FETCHED_AT, fetch_date_kst=fetch_date_kst,
        producer={"run_id": "r1"}, items=(search, document),
    )


def test_encode_decode_roundtrip_is_exact() -> None:
    from src.data.kind_relay import decode_relay_bundle, encode_relay_bundle

    bundle = _bundle()
    raw = encode_relay_bundle(bundle)
    decoded = decode_relay_bundle(raw)

    assert decoded.items == tuple(sorted(bundle.items, key=lambda item: (item.source, item.natural_key)))
    assert encode_relay_bundle(decoded) == raw


def test_decode_rejects_tampered_payload() -> None:
    import pytest

    from src.core.pit import PITDataError
    from src.data.kind_relay import decode_relay_bundle, encode_relay_bundle

    document = json.loads(encode_relay_bundle(_bundle()))
    payload = bytearray(base64.b64decode(document["items"][0]["payload_b64"]))
    payload[0] ^= 1
    document["items"][0]["payload_b64"] = base64.b64encode(payload).decode("ascii")
    document["items_sha256"] = hashlib.sha256(
        json.dumps(document["items"], sort_keys=True, ensure_ascii=False,
                   separators=(",", ":")).encode("utf-8")).hexdigest()
    with pytest.raises(PITDataError, match="payload_sha256"):
        decode_relay_bundle(json.dumps(document).encode("utf-8"))


def test_decode_rejects_unknown_keys_schema_and_duplicates() -> None:
    import pytest

    from src.core.pit import PITDataError
    from src.data.kind_relay import decode_relay_bundle, encode_relay_bundle

    raw = json.loads(encode_relay_bundle(_bundle()))

    extra = copy.deepcopy(raw)
    extra["extra"] = 1
    with pytest.raises(PITDataError):
        decode_relay_bundle(json.dumps(extra).encode("utf-8"))

    bad_schema = copy.deepcopy(raw)
    bad_schema["schema"] = "kind-relay-v2"
    with pytest.raises(PITDataError):
        decode_relay_bundle(json.dumps(bad_schema).encode("utf-8"))

    dup = copy.deepcopy(raw)
    dup["items"] = [dup["items"][0], copy.deepcopy(dup["items"][0])]
    dup["items_sha256"] = hashlib.sha256(
        json.dumps(dup["items"], sort_keys=True, ensure_ascii=False,
                   separators=(",", ":")).encode("utf-8")).hexdigest()
    for item in dup["items"]:
        item["payload_sha256"] = hashlib.sha256(base64.b64decode(item["payload_b64"])).hexdigest()
    with pytest.raises(PITDataError):
        decode_relay_bundle(json.dumps(dup).encode("utf-8"))


def test_decode_rejects_inconsistent_fetch_date() -> None:
    import pytest

    from src.core.pit import PITDataError
    from src.data.kind_relay import decode_relay_bundle, encode_relay_bundle

    raw = copy.deepcopy(json.loads(encode_relay_bundle(_bundle())))
    raw["fetch_date_kst"] = "2000-01-01"
    with pytest.raises(PITDataError):
        decode_relay_bundle(json.dumps(raw).encode("utf-8"))


def test_decode_rejects_oversize_input() -> None:
    import pytest

    from src.core.pit import PITDataError
    from src.data.kind_relay import RELAY_MAX_BYTES, decode_relay_bundle

    with pytest.raises(PITDataError):
        decode_relay_bundle(b"x" * (RELAY_MAX_BYTES + 1))


def test_freshness_boundaries() -> None:
    import dataclasses

    import pytest

    from src.core.pit import PITDataError
    from src.data.kind_relay import check_relay_freshness

    policy = _provider().kind
    now = NOW
    base = _bundle()

    for offset, accepted in (
        (timedelta(seconds=299), True),
        (timedelta(seconds=301), False),
        (-timedelta(hours=36), True),
        (-timedelta(hours=36, seconds=1), False),
    ):
        shifted = dataclasses.replace(base, fetched_at=now + offset,
                                      fetch_date_kst=((now + offset) + timedelta(hours=9)).date())
        if accepted:
            check_relay_freshness(shifted, now=now, policy=policy, last_accepted_at=None)
        else:
            with pytest.raises(PITDataError):
                check_relay_freshness(shifted, now=now, policy=policy, last_accepted_at=None)


def test_freshness_rejects_regression_but_allows_equal() -> None:
    import dataclasses

    import pytest

    from src.core.pit import PITDataError
    from src.data.kind_relay import check_relay_freshness

    policy = _provider().kind
    base = _bundle()
    watermark = base.fetched_at
    earlier = dataclasses.replace(
        base, fetched_at=watermark - timedelta(seconds=1),
        fetch_date_kst=((watermark - timedelta(seconds=1)) + timedelta(hours=9)).date())
    with pytest.raises(PITDataError):
        check_relay_freshness(earlier, now=NOW, policy=policy, last_accepted_at=watermark)
    check_relay_freshness(base, now=NOW, policy=policy, last_accepted_at=watermark)


def test_transport_selector_is_strict() -> None:
    import pytest

    from src.core.pit import PITDataError
    from src.data.kind_relay import kind_transport

    assert kind_transport({}) == "direct"
    assert kind_transport({"KSE_KIND_TRANSPORT": ""}) == "direct"
    assert kind_transport({"KSE_KIND_TRANSPORT": "relay"}) == "relay"
    assert kind_transport({"KSE_KIND_TRANSPORT": "direct"}) == "direct"
    with pytest.raises(PITDataError):
        kind_transport({"KSE_KIND_TRANSPORT": "Relay"})
    with pytest.raises(PITDataError):
        kind_transport({"KSE_KIND_TRANSPORT": "x"})


def test_decode_rejects_malformed_envelopes() -> None:
    import copy

    import pytest

    from src.core.pit import PITDataError
    from src.data.kind_relay import RELAY_MAX_ITEMS, decode_relay_bundle, encode_relay_bundle

    good = json.loads(encode_relay_bundle(_bundle()))

    with pytest.raises(PITDataError):
        decode_relay_bundle(b"\xff\xfe")
    with pytest.raises(PITDataError):
        decode_relay_bundle(b"{broken")
    with pytest.raises(PITDataError):
        decode_relay_bundle(json.dumps([1, 2]).encode("utf-8"))

    for mutated in (
        lambda doc: doc.update({"fetched_at": 123}),
        lambda doc: doc.update({"items_sha256": 123}),
        lambda doc: doc.update({"fetched_at": "bad-date"}),
        lambda doc: doc.update({"fetched_at": "2026-09-24T03:00:00"}),
        lambda doc: doc.update({"fetched_at": "2026-09-24T12:00:00+09:00"}),
        lambda doc: doc.update({"fetch_date_kst": "bad-date"}),
        lambda doc: doc.update({"producer": []}),
        lambda doc: doc.update({"producer": {"run_id": 1}}),
        lambda doc: doc.update({"items": {}}),
    ):
        doc = copy.deepcopy(good)
        mutated(doc)
        with pytest.raises(PITDataError):
            decode_relay_bundle(json.dumps(doc).encode("utf-8"))

    doc = copy.deepcopy(good)
    doc["items"] = [{"kind": "x"}]
    with pytest.raises(PITDataError):
        decode_relay_bundle(json.dumps(doc).encode("utf-8"))

    oversized = copy.deepcopy(good)
    oversized["items"] = oversized["items"] * (RELAY_MAX_ITEMS + 1)
    with pytest.raises(PITDataError):
        decode_relay_bundle(json.dumps(oversized).encode("utf-8"))


def test_decode_rejects_malformed_items() -> None:
    import base64
    import copy
    import hashlib

    import pytest

    from src.core.pit import PITDataError
    from src.data.kind_relay import decode_relay_bundle, encode_relay_bundle

    def _decode_with(item_mutation):  # type: ignore[no-untyped-def]
        doc = copy.deepcopy(json.loads(encode_relay_bundle(_bundle())))
        item_mutation(doc["items"][0])
        doc["items_sha256"] = hashlib.sha256(
            json.dumps(doc["items"], sort_keys=True, ensure_ascii=False,
                       separators=(",", ":")).encode("utf-8")).hexdigest()
        decode_relay_bundle(json.dumps(doc).encode("utf-8"))

    with pytest.raises(PITDataError):
        _decode_with(lambda item: item.update({"source": "dart_disclosures"}))
    with pytest.raises(PITDataError):
        _decode_with(lambda item: item.update({"kind": "nope"}))
    with pytest.raises(PITDataError):
        _decode_with(lambda item: item.update({"status": "nope"}))
    with pytest.raises(PITDataError):
        _decode_with(lambda item: item.update({"as_of": "bad-date"}))
    with pytest.raises(PITDataError):
        _decode_with(lambda item: item.update({"as_of": 123}))
    with pytest.raises(PITDataError):
        _decode_with(lambda item: item.update({"fiscal_period": 123}))
    with pytest.raises(PITDataError):
        _decode_with(lambda item: item.update({"source_label": ""}))
    with pytest.raises(PITDataError):
        _decode_with(lambda item: item.update({"retrieved_at": "bad-date"}))
    with pytest.raises(PITDataError):
        _decode_with(lambda item: item.update({"retrieved_at": "2026-09-24T03:00:00"}))
    with pytest.raises(PITDataError):
        _decode_with(lambda item: item.update({"payload_b64": "!!!"}))
    with pytest.raises(PITDataError):
        _decode_with(lambda item: item.update({"payload_b64": 123}))
    with pytest.raises(PITDataError):
        _decode_with(lambda item: item.update({"payload_sha256": "0" * 64}))
    with pytest.raises(PITDataError):
        _decode_with(lambda item: item.pop("payload_sha256"))

    doc = copy.deepcopy(json.loads(encode_relay_bundle(_bundle())))
    raw_payload = base64.b64decode(doc["items"][0]["payload_b64"])
    assert hashlib.sha256(raw_payload).hexdigest() == doc["items"][0]["payload_sha256"]


def test_roundtrip_supports_null_as_of() -> None:
    from src.core.pit import EvidenceKind
    from src.data.evidence_sources import KIND_NOTICE_SEARCH_SOURCE
    from src.data.kind_relay import RelayBundle, decode_relay_bundle, encode_relay_bundle
    from src.data.receipt_catalog import EvidenceStatus
    from src.data.scoped_ingestion import ScopedRawPayload

    payload = ScopedRawPayload(
        kind=EvidenceKind.DISCLOSURES, source=KIND_NOTICE_SEARCH_SOURCE, natural_key="k",
        as_of=None, fiscal_period=None, status=EvidenceStatus.EMPTY,
        payload=b"{}", retrieved_at=FETCHED_AT, source_label="test",
    )
    bundle = RelayBundle(fetched_at=FETCHED_AT,
                         fetch_date_kst=(FETCHED_AT + timedelta(hours=9)).date(),
                         producer={}, items=(payload,))
    assert decode_relay_bundle(encode_relay_bundle(bundle)).items[0].as_of is None


def test_module_is_import_light() -> None:
    import src.data.kind_relay as relay

    source = Path(relay.__file__).read_text(encoding="utf-8")
    assert "import polars" not in source
    assert "import requests" not in source


def test_decode_rejects_duplicate_json_keys_and_nonfinite_values() -> None:
    import pytest

    from src.core.pit import PITDataError
    from src.data.kind_relay import decode_relay_bundle, encode_relay_bundle

    raw = encode_relay_bundle(_bundle())
    duplicate = raw.replace(b'"schema":', b'"schema":"kind-relay-v1","schema":')
    with pytest.raises(PITDataError, match="duplicate JSON key"):
        decode_relay_bundle(duplicate)
    nested_duplicate = raw.replace(b'"run_id":"r1"', b'"run_id":"r0","run_id":"r1"')
    with pytest.raises(PITDataError, match="duplicate JSON key"):
        decode_relay_bundle(nested_duplicate)
    for value in (float("nan"), float("inf")):
        document = json.loads(raw)
        document["items"][0]["as_of"] = value
        with pytest.raises(PITDataError):
            decode_relay_bundle(json.dumps(document).encode("utf-8"))
    document = json.loads(raw)
    document["fetched_at"] = "9999-12-31T23:00:00+00:00"
    with pytest.raises(PITDataError):
        decode_relay_bundle(json.dumps(document).encode("utf-8"))
    with pytest.raises(PITDataError):
        decode_relay_bundle(b"[" * 2000 + b"]" * 2000)


def test_naive_observation_times_fail_closed() -> None:
    from dataclasses import replace

    import pytest

    from src.core.pit import PITDataError
    from src.data.kind_relay import check_relay_freshness, encode_relay_bundle

    bundle = _bundle()
    naive = FETCHED_AT.replace(tzinfo=None)
    with pytest.raises(PITDataError):
        encode_relay_bundle(replace(bundle, fetched_at=naive))
    for fetched_at, now, watermark in (
        (naive, NOW, None), (FETCHED_AT, naive, None), (FETCHED_AT, NOW, naive),
    ):
        with pytest.raises(PITDataError):
            check_relay_freshness(replace(bundle, fetched_at=fetched_at), now=now,
                                  policy=_provider().kind, last_accepted_at=watermark)
