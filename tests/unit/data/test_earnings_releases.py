"""Earnings-release Silver invariants: versions, availability, quarantine and benchmark."""
from __future__ import annotations

import base64
import io
import json
import zipfile
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

from src.core.time import KRX_TZ, SessionCalendar
from src.data.datasets import DatasetIdentity, DatasetLayer, load_manifest, publish_dataset, read_dataset
from src.data.earnings_releases import benchmark_earnings_releases, materialize_earnings_releases

CORP = "00126380"
TICKER = "005930"
OTHER_CORP = "00126381"


def _zip(rows: list[list[str]]) -> bytes:
    body = "<table>" + "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
    ) + "</table>"
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("document.xml", body)
    return output.getvalue()


def _preliminary_rows(*, caption: str = "1. 연결실적내용", unit: str = "단위 : 백만원, %") -> list[list[str]]:
    return [
        [caption, unit],
        ["구분", "당기실적", "전기실적", "전기대비증감율(%)", "전년동기실적", "전년동기대비증감율(%)"],
        ["('18.4Q)", "('18.3Q)", "('17.4Q)"],
        ["매출액", "당해실적", "321,223", "376,466", "-14.7", "87,836", "265.7"],
        ["누계실적", "1,365,439", "1,044,216", "-", "87,836", "1,454.5"],
    ]


def _calendar(*days: date) -> SessionCalendar:
    return SessionCalendar(tuple(datetime(day.year, day.month, day.day, 9, tzinfo=KRX_TZ) for day in days))


def _catalog(bronze: Path):  # type: ignore[no-untyped-def]
    from src.data.receipt_catalog import ReceiptCatalog

    return ReceiptCatalog(bronze / "catalog")


def _seed_bridge(bronze: Path, rows: list[dict[str, str]] | None = None, *, retrieved_at=None) -> None:  # type: ignore[no-untyped-def]
    from datetime import date as _date
    from datetime import datetime as _datetime
    from datetime import UTC as _UTC

    from src.core.pit import EvidenceKind as _Kind
    from src.data.bronze import BronzeStore as _Store
    from src.data.receipt_catalog import BlobEntry as _Blob, EvidenceStatus as _Status
    from src.data.receipt_catalog import ReceiptCatalog as _Catalog, ReceiptIndexEntry as _Entry

    rows = rows if rows is not None else [{"ticker": TICKER, "corp_code": CORP, "corp_name": "Test Co"}]
    raw = json.dumps(rows, sort_keys=True, ensure_ascii=False).encode("utf-8")
    store = _Store(bronze)
    moment = retrieved_at if retrieved_at is not None else _datetime(2024, 1, 2, tzinfo=_UTC)
    receipt = store.import_bytes(
        raw, kind=_Kind.SECURITY_MASTER, retrieved_at=moment,
        source_label="test:bridge",
    )
    catalog = _Catalog(bronze / "catalog")
    catalog.publish(
        [_Entry(source="dart_corp_codes", natural_key="dart_corp_codes", as_of=_date(2024, 1, 2),
                fiscal_period=None, status=_Status.SUCCESS, content_hash=receipt.content_hash,
                retrieved_at=receipt.retrieved_at, payload_path=receipt.payload_path)],
        blobs=[_Blob(content_hash=receipt.content_hash, kind=_Kind.SECURITY_MASTER, source="dart_corp_codes",
                     usable=True, unusable_reason=None, retrieved_at=receipt.retrieved_at,
                     payload_path=receipt.payload_path)],
    )


def _seed_release(
    bronze: Path, *, rcept_no: str, corp_code: str = CORP, received_on: str,
    report_nm: str = "연결재무제표기준영업(잠정)실적(공정공시)", archive: bytes | None = None,
    retrieved_at=None,  # type: ignore[no-untyped-def]
) -> None:
    from datetime import date as _date
    from datetime import datetime as _datetime
    from datetime import UTC as _UTC

    from src.core.pit import EvidenceKind as _Kind
    from src.data.bronze import BronzeStore as _Store
    from src.data.evidence_sources import EARNINGS_RELEASE_SOURCE
    from src.data.receipt_catalog import BlobEntry as _Blob, EvidenceStatus as _Status
    from src.data.receipt_catalog import ReceiptCatalog as _Catalog, ReceiptIndexEntry as _Entry

    import hashlib

    body = _zip(_preliminary_rows()) if archive is None else archive
    envelope = {
        "rcept_no": rcept_no,
        "corp_code": corp_code,
        "received_on": received_on,
        "report_nm": report_nm,
        "archive_b64": base64.b64encode(body).decode("ascii"),
        "archive_sha256": hashlib.sha256(body).hexdigest(),
    }
    raw = json.dumps(envelope, sort_keys=True, ensure_ascii=False).encode("utf-8")
    store = _Store(bronze)
    moment = retrieved_at if retrieved_at is not None else _datetime(2024, 1, 2, tzinfo=_UTC)
    receipt = store.import_bytes(
        raw, kind=_Kind.DISCLOSURES, retrieved_at=moment,
        source_label="test:release",
    )
    catalog = _Catalog(bronze / "catalog")
    catalog.publish(
        [_Entry(source=EARNINGS_RELEASE_SOURCE, natural_key=rcept_no, as_of=_date.fromisoformat(received_on),
                fiscal_period=None, status=_Status.SUCCESS, content_hash=receipt.content_hash,
                retrieved_at=receipt.retrieved_at, payload_path=receipt.payload_path)],
        blobs=[_Blob(content_hash=receipt.content_hash, kind=_Kind.DISCLOSURES, source=EARNINGS_RELEASE_SOURCE,
                     usable=True, unusable_reason=None, retrieved_at=receipt.retrieved_at,
                     payload_path=receipt.payload_path)],
    )


def _materialize(bronze: Path, silver: Path, calendar: SessionCalendar) -> Path:
    from src.config.providers import EarningsReleasePolicy

    return materialize_earnings_releases(
        catalog=_catalog(bronze), silver_root=silver, calendar=calendar,
        policy=EarningsReleasePolicy(),
    )


def test_versions_keep_their_own_availability(tmp_path: Path) -> None:
    bronze, silver = tmp_path / "bronze", tmp_path / "silver"
    _seed_bridge(bronze)
    _seed_release(bronze, rcept_no="20190131000001", received_on="2019-01-31")
    _seed_release(
        bronze, rcept_no="20190302000002", received_on="2019-03-02",
        report_nm="[기재정정]연결재무제표기준영업(잠정)실적(공정공시)",
    )
    calendar = _calendar(date(2019, 2, 1), date(2019, 3, 4), date(2019, 3, 5))

    path = _materialize(bronze, silver, calendar)
    frame = read_dataset(path).collect().sort("rcept_no")

    assert set(frame["rcept_no"].to_list()) == {"20190131000001", "20190302000002"}
    assert set(frame.filter(pl.col("rcept_no") == "20190131000001")["available_at"].to_list()) == {
        datetime(2019, 2, 1, 9, tzinfo=KRX_TZ)
    }
    assert set(frame.filter(pl.col("rcept_no") == "20190302000002")["available_at"].to_list()) == {
        datetime(2019, 3, 4, 9, tzinfo=KRX_TZ)
    }
    assert frame.filter(pl.col("rcept_no") == "20190131000001").height == 2


def test_availability_is_the_next_session(tmp_path: Path) -> None:
    bronze, silver = tmp_path / "bronze", tmp_path / "silver"
    _seed_bridge(bronze)
    rows = [row[:] for row in _preliminary_rows()]
    rows[2] = ["(2023.4Q)", "(2023.3Q)", "(2022.4Q)"]
    _seed_release(bronze, rcept_no="20240105000001", received_on="2024-01-05", archive=_zip(rows))
    calendar = _calendar(date(2024, 1, 4), date(2024, 1, 9))

    path = _materialize(bronze, silver, calendar)
    frame = read_dataset(path).collect()

    assert frame.height == 2
    assert set(frame["available_at"].to_list()) == {datetime(2024, 1, 9, 9, tzinfo=KRX_TZ)}


def test_quarantine_records_reasons(tmp_path: Path) -> None:
    bronze, silver = tmp_path / "bronze", tmp_path / "silver"
    _seed_bridge(bronze)
    diff_rows = [["정정항목", "정정전", "정정후"], ["매출액(당해실적)", "240,204", "245,695"]]
    _seed_release(
        bronze, rcept_no="20230209000001", received_on="2023-02-09", archive=_zip(diff_rows),
        report_nm="[기재정정]연결재무제표기준영업(잠정)실적(공정공시)",
    )
    _seed_release(
        bronze, rcept_no="20190131000002", received_on="2019-01-31",
        archive=_zip(_preliminary_rows(unit="단위 : 달러")),
    )
    calendar = _calendar(date(2019, 2, 1), date(2023, 2, 10))

    path = _materialize(bronze, silver, calendar)

    assert read_dataset(path).collect().height == 0
    quarantine = json.loads((path / "quarantine.json").read_text(encoding="utf-8"))
    assert {entry["reason"] for entry in quarantine} == {"no_result_table", "unknown_unit"}
    manifest = load_manifest(path)
    assert manifest.details["quarantined_filings"] == 2


def test_unmapped_corp_counted_not_guessed(tmp_path: Path) -> None:
    bronze, silver = tmp_path / "bronze", tmp_path / "silver"
    _seed_bridge(bronze)
    _seed_release(bronze, rcept_no="20190131000001", corp_code="00999999", received_on="2019-01-31")
    calendar = _calendar(date(2019, 2, 1))

    path = _materialize(bronze, silver, calendar)

    assert read_dataset(path).collect().height == 0
    assert load_manifest(path).details["unmapped_filings"] == 1


def test_deterministic_republication(tmp_path: Path) -> None:
    import hashlib

    bronze, silver = tmp_path / "bronze", tmp_path / "silver"
    _seed_bridge(bronze)
    _seed_release(bronze, rcept_no="20190131000001", received_on="2019-01-31")
    calendar = _calendar(date(2019, 2, 1))

    first = _materialize(bronze, silver, calendar)
    second = _materialize(bronze, silver, calendar)

    assert first == second
    assert (first / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()
    digest = hashlib.sha256((first / "part-00000.parquet").read_bytes()).hexdigest()
    assert digest == hashlib.sha256((second / "part-00000.parquet").read_bytes()).hexdigest()


def _publish_facts(silver: Path, values: dict[str, float]) -> Path:
    rows = [
        {
            "company_id": ticker,
            "ticker": ticker,
            "fiscal_period": "2018Q4",
            "filing_id": f"F-{ticker}",
            "fact": "sales",
            "published_at": datetime(2019, 3, 29, tzinfo=UTC),
            "available_at": datetime(2019, 4, 1, tzinfo=UTC),
            "value": value,
            "unit": "KRW",
            "consolidated": True,
            "restatement_id": "",
        }
        for ticker, value in values.items()
    ]
    published = publish_dataset(
        layer_root=silver,
        identity=DatasetIdentity(
            kind="financial_facts", layer=DatasetLayer.SILVER, policy_version="test-v1",
            inputs={}, params={},
        ),
        partitions={"part-00000.parquet": pl.DataFrame(rows)},
    )
    return published.path


def test_benchmark_agreement_statistics(tmp_path: Path) -> None:
    bronze, silver = tmp_path / "bronze", tmp_path / "silver"
    tickers = [f"{index:06d}" for index in range(10)]
    corps = [f"{int(CORP) + index:08d}" for index in range(10)]
    _seed_bridge(
        bronze,
        rows=[{"ticker": ticker, "corp_code": corp, "corp_name": "T"}
              for ticker, corp in zip(tickers, corps, strict=True)],
    )
    for position, corp in enumerate(corps):
        _seed_release(
            bronze, rcept_no=f"20190131{position:06d}", corp_code=corp,
            received_on="2019-01-31",
        )
    calendar = _calendar(date(2019, 2, 1))
    releases_path = _materialize(bronze, silver, calendar)
    facts_values = dict.fromkeys(tickers, 1365439 * 1e6)
    facts_values[tickers[-1]] = 1365439 * 1e6 * 1.2
    facts_path = _publish_facts(silver, facts_values)

    (metric_report,) = [entry for entry in benchmark_earnings_releases(
        releases_path=releases_path, facts_path=facts_path) if entry.metric == "sales"]

    assert metric_report.matched == 10
    assert metric_report.within_5pct == 0.9


def test_tampered_and_unreadable_envelopes_fail_closed(tmp_path: Path) -> None:
    import pytest

    from src.core.pit import PITDataError
    from src.data.receipt_catalog import ReceiptCatalog

    bronze, silver = tmp_path / "bronze", tmp_path / "silver"
    _seed_bridge(bronze)
    _seed_release(bronze, rcept_no="20190131000001", received_on="2019-01-31")
    catalog = ReceiptCatalog(bronze / "catalog")
    entry = catalog.latest(source="opendart:earnings_release", natural_keys={"20190131000001"})[
        "20190131000001"
    ]
    Path(entry.payload_path).write_bytes(b"corrupted")
    with pytest.raises(PITDataError, match="hash mismatch"):
        _materialize(bronze, silver, _calendar(date(2019, 2, 1)))

    Path(entry.payload_path).unlink()
    with pytest.raises(PITDataError, match="unreadable"):
        _materialize(bronze, silver, _calendar(date(2019, 2, 1)))


def test_skipped_envelope_shapes_yield_no_rows(tmp_path: Path) -> None:
    from datetime import date as _date
    from datetime import datetime as _datetime
    from datetime import UTC as _UTC

    from src.core.pit import EvidenceKind as _Kind
    from src.data.bronze import BronzeStore as _Store
    from src.data.evidence_sources import EARNINGS_RELEASE_SOURCE
    from src.data.receipt_catalog import BlobEntry as _Blob, EvidenceStatus as _Status
    from src.data.receipt_catalog import ReceiptCatalog as _Catalog, ReceiptIndexEntry as _Entry

    bronze, silver = tmp_path / "bronze", tmp_path / "silver"
    _seed_bridge(bronze)
    bodies = [
        b"{broken",
        json.dumps([1, 2, 3]).encode(),
        json.dumps({"rcept_no": "1"}).encode(),
        json.dumps({"rcept_no": "2", "archive_b64": "!!!"}).encode(),
        b"PK\x03\x04raw-zip-bytes",
    ]
    absent = (
        b'<?xml version="1.0"?><response><status>014</status><message>x</message></response>'
    )
    import base64 as _b64

    bodies.append(
        json.dumps({"rcept_no": "3", "archive_b64": _b64.b64encode(absent).decode()}).encode()
    )
    store = _Store(bronze)
    catalog = _Catalog(bronze / "catalog")
    for position, raw in enumerate(bodies):
        receipt = store.import_bytes(
            raw, kind=_Kind.DISCLOSURES, retrieved_at=_datetime(2024, 1, 2, tzinfo=_UTC),
            source_label="test:skipped",
        )
        catalog.publish(
            [_Entry(source=EARNINGS_RELEASE_SOURCE, natural_key=f"skip-{position}",
                    as_of=_date(2024, 1, 2), fiscal_period=None, status=_Status.SUCCESS,
                    content_hash=receipt.content_hash, retrieved_at=receipt.retrieved_at,
                    payload_path=receipt.payload_path)],
            blobs=[_Blob(content_hash=receipt.content_hash, kind=_Kind.DISCLOSURES,
                         source=EARNINGS_RELEASE_SOURCE, usable=True, unusable_reason=None,
                         retrieved_at=receipt.retrieved_at, payload_path=receipt.payload_path)],
        )
    path = _materialize(bronze, silver, _calendar(date(2024, 1, 3)))

    assert read_dataset(path).collect().height == 0
    assert load_manifest(path).details["filings"] == 0


def test_malformed_envelope_fields_fail_closed(tmp_path: Path) -> None:
    import pytest

    from src.core.pit import PITDataError
    from src.data.earnings_releases import _release_from_envelope

    with pytest.raises(PITDataError, match="invalid earnings-release Bronze envelope"):
        _release_from_envelope(
            {"rcept_no": "20190131000001", "corp_code": CORP, "received_on": "bogus"},
            max_filing_lag_days=135,
        )


def test_invalid_bridge_and_empty_calendar_fail_closed(tmp_path: Path) -> None:
    import pytest

    from src.core.pit import PITDataError

    bronze, silver = tmp_path / "bronze", tmp_path / "silver"
    _seed_bridge(bronze, rows=[{"ticker": TICKER}])
    with pytest.raises(PITDataError, match="invalid dart corp-code bridge"):
        _materialize(bronze, silver, _calendar(date(2019, 2, 1)))

    bronze2 = tmp_path / "bronze2"
    _seed_bridge(bronze2)
    with pytest.raises(PITDataError, match="at least one certified session"):
        _materialize(bronze2, silver, SessionCalendar(()))


def test_no_session_after_receipt_fails_closed(tmp_path: Path) -> None:
    import pytest

    from src.core.pit import PITDataError

    bronze, silver = tmp_path / "bronze", tmp_path / "silver"
    _seed_bridge(bronze)
    _seed_release(bronze, rcept_no="20190131000001", received_on="2019-01-31")

    with pytest.raises(PITDataError, match="no certified session open after"):
        _materialize(bronze, silver, _calendar(date(2019, 1, 2)))


def test_duplicate_receipts_collapse_to_one(tmp_path: Path) -> None:
    bronze, silver = tmp_path / "bronze", tmp_path / "silver"
    _seed_bridge(bronze)
    _seed_release(bronze, rcept_no="20190131000001", received_on="2019-01-31")
    rows = [row[:] for row in _preliminary_rows()]
    rows[3] = ["매출액", "당해실적", "999,999", "376,466", "-14.7", "87,836", "265.7"]
    _seed_release(
        bronze, rcept_no="20190131000001", received_on="2019-02-15", archive=_zip(rows),
        retrieved_at=datetime(2024, 1, 3, tzinfo=UTC),
    )

    path = _materialize(bronze, silver, _calendar(date(2019, 2, 1), date(2019, 2, 18)))

    assert load_manifest(path).details["filings"] == 1


def test_benchmark_skips_and_zero_facts(tmp_path: Path) -> None:
    bronze, silver = tmp_path / "bronze", tmp_path / "silver"
    _seed_bridge(
        bronze,
        rows=[
            {"ticker": "000001", "corp_code": CORP, "corp_name": "A"},
            {"ticker": "000002", "corp_code": OTHER_CORP, "corp_name": "B"},
        ],
    )
    quarter_rows = [
        ["1. 연결실적내용", "단위 : 백만원, %"],
        ["구분", "당기실적", "전기실적", "전기대비증감율(%)", "전년동기실적", "전년동기대비증감율(%)"],
        ["(2023.1Q)", "(2022.4Q)", "(2022.1Q)"],
        ["매출액", "당해실적", "100", "90", "11.1", "80", "25.0"],
        ["영업이익", "당해실적", "-", "-", "-", "-", "-"],
        ["누계실적", "-", "-", "-", "-", "-"],
    ]
    _seed_release(
        bronze, rcept_no="20230420000001", corp_code=CORP, received_on="2023-04-20",
        archive=_zip(quarter_rows),
    )
    separate_rows = [row[:] for row in quarter_rows]
    separate_rows[0] = ["1. 실적내용", "단위 : 백만원, %"]
    _seed_release(
        bronze, rcept_no="20230420000002", corp_code=OTHER_CORP, received_on="2023-04-20",
        report_nm="영업(잠정)실적(공정공시)", archive=_zip(separate_rows),
    )
    calendar = _calendar(date(2023, 4, 21))
    releases_path = _materialize(bronze, silver, calendar)
    facts_rows = [
        {
            "company_id": "000001", "ticker": "000001", "fiscal_period": "2023Q1", "filing_id": "F1",
            "fact": "sales", "published_at": datetime(2023, 5, 15, tzinfo=UTC),
            "available_at": datetime(2023, 5, 16, tzinfo=UTC), "value": 100 * 1e6,
            "unit": "KRW", "consolidated": True, "restatement_id": "",
        },
        {
            "company_id": "000001", "ticker": "000001", "fiscal_period": "2023Q1", "filing_id": "F0",
            "fact": "sales", "published_at": datetime(2023, 5, 15, tzinfo=UTC),
            "available_at": datetime(2023, 5, 16, tzinfo=UTC), "value": 999.0,
            "unit": "KRW", "consolidated": False, "restatement_id": "",
        },
        {
            "company_id": "000001", "ticker": "000001", "fiscal_period": "2023Q1", "filing_id": "F2",
            "fact": "sales", "published_at": datetime(2023, 5, 15, tzinfo=UTC),
            "available_at": datetime(2023, 5, 16, tzinfo=UTC), "value": None,
            "unit": "KRW", "consolidated": True, "restatement_id": "",
        },
    ]
    facts_path = publish_dataset(
        layer_root=silver,
        identity=DatasetIdentity(
            kind="financial_facts", layer=DatasetLayer.SILVER, policy_version="test-v1",
            inputs={}, params={},
        ),
        partitions={"part-00000.parquet": pl.DataFrame(facts_rows, strict=False)},
    ).path

    (report,) = [
        entry for entry in benchmark_earnings_releases(
            releases_path=releases_path, facts_path=facts_path)
        if entry.metric == "sales"
    ]

    assert report.matched == 1
    assert report.within_5pct == 1.0


def test_benchmark_rejects_malformed_facts(tmp_path: Path) -> None:
    import pytest

    from src.core.pit import PITDataError

    bronze, silver = tmp_path / "bronze", tmp_path / "silver"
    _seed_bridge(bronze)
    _seed_release(bronze, rcept_no="20190131000001", received_on="2019-01-31")
    releases_path = _materialize(bronze, silver, _calendar(date(2019, 2, 1)))
    facts_path = publish_dataset(
        layer_root=silver,
        identity=DatasetIdentity(
            kind="financial_facts", layer=DatasetLayer.SILVER, policy_version="test-v1",
            inputs={}, params={},
        ),
        partitions={
            "part-00000.parquet": pl.DataFrame(
                [{
                    "company_id": "12345", "ticker": 12345, "fiscal_period": "2018Q4", "filing_id": "F",
                    "fact": "sales", "published_at": datetime(2019, 3, 29, tzinfo=UTC),
                    "available_at": datetime(2019, 4, 1, tzinfo=UTC), "value": 1.0,
                    "unit": "KRW", "consolidated": True, "restatement_id": "",
                }],
                strict=False,
            )
        },
    ).path

    with pytest.raises(PITDataError, match="invalid financial fact"):
        benchmark_earnings_releases(releases_path=releases_path, facts_path=facts_path)


def test_benchmark_skip_shapes_and_zero_facts(tmp_path: Path) -> None:
    silver = tmp_path / "silver"
    releases = [
        ("KRX:000001", "2023Q1", "sales", "quarter", 100.0, "consolidated"),
        ("KRX:000009", "2023Q1", "sales", "quarter", 50.0, "consolidated"),
        ("KRX:000001", "2023Q1", "gross_profit", "quarter", 10.0, "consolidated"),
        ("KRX:000001", "2023Q1", "sales", "quarter", None, "consolidated"),
        ("KRX:000001", "FY23", "sales", "quarter", 10.0, "consolidated"),
        ("KRX:000001", "2023Q1", "sales", "cumulative", 10.0, "consolidated"),
        ("KRX:000001", "2023Q4", "sales", "quarter", 10.0, "consolidated"),
        ("KRX:000001", "2023Q1", "operating_profit", "quarter", 0.0, "consolidated"),
        ("KRX:000002", "2023Q4", "sales", "annual", 50.0, "consolidated"),
        ("KRX:000001", "2023Q1", "sales", "weekly", 5.0, "consolidated"),
        ("KRX:000001", "2023Q1", "sales", "quarter", 10.0, "separate"),
    ]
    releases_path = publish_dataset(
        layer_root=silver,
        identity=DatasetIdentity(
            kind="earnings_releases", layer=DatasetLayer.SILVER, policy_version="test-v1",
            inputs={}, params={},
        ),
        partitions={
            "part-00000.parquet": pl.DataFrame(
                {
                    "instrument_id": [row[0] for row in releases],
                    "fiscal_period": [row[1] for row in releases],
                    "metric": [row[2] for row in releases],
                    "span": [row[3] for row in releases],
                    "value_krw": [row[4] for row in releases],
                    "basis": [row[5] for row in releases],
                },
                strict=False,
            )
        },
    ).path
    facts_path = publish_dataset(
        layer_root=silver,
        identity=DatasetIdentity(
            kind="financial_facts", layer=DatasetLayer.SILVER, policy_version="test-v1",
            inputs={}, params={},
        ),
        partitions={
            "part-00000.parquet": pl.DataFrame(
                {
                    "company_id": ["000001", "000001", "000002"],
                    "ticker": ["000001", "000001", "000002"],
                    "fiscal_period": ["2023Q1", "2023Q1", "2023Q4"],
                    "fact": ["sales", "operating_profit", "sales"],
                    "value": [100.0, 0.0, 0.0],
                    "consolidated": [True, True, True],
                }
            )
        },
    ).path

    by_metric = {
        entry.metric: entry
        for entry in benchmark_earnings_releases(releases_path=releases_path, facts_path=facts_path)
    }

    assert (by_metric["sales"].matched, by_metric["sales"].within_5pct) == (2, 0.5)
    assert by_metric["sales"].median_abs_rel_error == 0.0
    assert (by_metric["operating_profit"].matched, by_metric["operating_profit"].within_1pct) == (1, 1.0)


def test_benchmark_rejects_malformed_releases(tmp_path: Path) -> None:
    import pytest

    from src.core.pit import PITDataError

    silver = tmp_path / "silver"
    releases_path = publish_dataset(
        layer_root=silver,
        identity=DatasetIdentity(
            kind="earnings_releases", layer=DatasetLayer.SILVER, policy_version="test-v1",
            inputs={}, params={},
        ),
        partitions={
            "part-00000.parquet": pl.DataFrame(
                {
                    "instrument_id": ["KRX:000001"],
                    "fiscal_period": ["2023Q1"],
                    "metric": ["sales"],
                    "span": ["quarter"],
                    "value_krw": ["not-a-number"],
                    "basis": ["consolidated"],
                },
                strict=False,
            )
        },
    ).path
    facts_path = publish_dataset(
        layer_root=silver,
        identity=DatasetIdentity(
            kind="financial_facts", layer=DatasetLayer.SILVER, policy_version="test-v1",
            inputs={}, params={},
        ),
        partitions={
            "part-00000.parquet": pl.DataFrame(
                {
                    "company_id": ["000001"],
                    "ticker": ["000001"],
                    "fiscal_period": ["2023Q1"],
                    "fact": ["sales"],
                    "value": [100.0],
                    "consolidated": [True],
                }
            )
        },
    ).path

    with pytest.raises(PITDataError, match="invalid earnings releases for benchmark"):
        benchmark_earnings_releases(releases_path=releases_path, facts_path=facts_path)
