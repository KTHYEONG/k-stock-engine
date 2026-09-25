"""Dividend event silver build tests (offline Bronze fixtures only)."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import zipfile
from datetime import date, datetime
from pathlib import Path

from src.core.time import KRX_TZ, SessionCalendar


def _archive(record: str, pay: str, dps: int) -> bytes:
    html = (
        '<?xml version="1.0" encoding="utf-8"?><document><table>'
        f"<tr><td>주당 배당금(원)</td><td>보통주 {dps}원</td></tr>"
        f"<tr><td>배당기준일</td><td>{record}</td></tr>"
        f"<tr><td>배당금지급 예정일자</td><td>{pay}</td></tr>"
        "</table></document>"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("doc.xml", html.encode("utf-8"))
    return buf.getvalue()


def _write_bridge(bronze_root: Path) -> None:
    payload = [{"ticker": "005930", "corp_code": "00126380", "corp_name": "삼성전자"}]
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    target = bronze_root / "dart_corp_codes" / digest
    target.mkdir(parents=True, exist_ok=True)
    (target / "payload.json").write_bytes(raw)


def _write_envelope(
    bronze_root: Path, *, rcept_no: str, received_on: str, record: str, pay: str, dps: int, corp_code: str = "00126380"
) -> None:
    archive = _archive(record, pay, dps)
    envelope = {
        "rcept_no": rcept_no,
        "corp_code": corp_code,
        "received_on": received_on,
        "report_nm": "현금ㆍ현물배당결정",
        "archive_b64": base64.b64encode(archive).decode("ascii"),
        "archive_sha256": hashlib.sha256(archive).hexdigest(),
    }
    raw = json.dumps(envelope, ensure_ascii=False, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    target = bronze_root / "corporate_actions" / digest
    target.mkdir(parents=True, exist_ok=True)
    (target / "payload.json").write_bytes(raw)


def _calendar(sessions: list[date]) -> SessionCalendar:
    return SessionCalendar(
        sessions=tuple(datetime(s.year, s.month, s.day, 9, 0, tzinfo=KRX_TZ) for s in sessions)
    )


def _materialize(bronze_root: Path, calendar: SessionCalendar) -> tuple[Path, dict]:
    import polars as pl

    from src.data.dividend_events import materialize_dividend_events

    silver_root = bronze_root.parent / "silver"
    universe_root = bronze_root.parent / "universe"
    universe_root.mkdir(parents=True, exist_ok=True)
    out = materialize_dividend_events(
        bronze_root=bronze_root, universe_root=universe_root, silver_root=silver_root, calendar=calendar
    )
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    frame = pl.read_parquet(out / "part-00000.parquet")
    return out, {"manifest": manifest, "frame": frame}


def test_materialize_dividend_events_resolves_ex_session_before_record_session(tmp_path: Path) -> None:
    bronze_root = tmp_path / "bronze"
    _write_bridge(bronze_root)
    _write_envelope(
        bronze_root,
        rcept_no="20200102001234",
        received_on="2020-01-02",
        record="2019-12-31",
        pay="2020-04-20",
        dps=300,
    )
    calendar = _calendar([date(2019, 12, 26), date(2019, 12, 27), date(2019, 12, 30), date(2020, 1, 2), date(2020, 4, 20)])

    _, result = _materialize(bronze_root, calendar)
    frame = result["frame"]

    assert frame.height == 1
    assert str(frame["ex_session"][0]) == "2019-12-27"


def test_materialize_dividend_events_latest_correction_wins(tmp_path: Path) -> None:
    bronze_root = tmp_path / "bronze"
    _write_bridge(bronze_root)
    _write_envelope(
        bronze_root,
        rcept_no="20200102001234",
        received_on="2020-01-02",
        record="2019-12-31",
        pay="2020-04-20",
        dps=300,
    )
    _write_envelope(
        bronze_root,
        rcept_no="20200110001234",
        received_on="2020-01-10",
        record="2019-12-31",
        pay="2020-04-20",
        dps=350,
    )
    calendar = _calendar([date(2019, 12, 26), date(2019, 12, 27), date(2019, 12, 30), date(2020, 1, 13), date(2020, 4, 20)])

    _, result = _materialize(bronze_root, calendar)
    frame = result["frame"]

    assert frame.height == 1
    assert int(frame["dps_krw"][0]) == 350
    available_at = frame["available_at"][0]
    assert available_at.date().isoformat() > "2020-01-10"


def test_materialize_dividend_events_undecided_pay_date_excluded_and_counted(tmp_path: Path) -> None:
    bronze_root = tmp_path / "bronze"
    _write_bridge(bronze_root)
    _write_envelope(
        bronze_root,
        rcept_no="20200102001234",
        received_on="2020-01-02",
        record="2019-12-31",
        pay="-",
        dps=300,
    )
    calendar = _calendar([date(2019, 12, 26), date(2019, 12, 27), date(2019, 12, 30), date(2020, 1, 2)])

    _, result = _materialize(bronze_root, calendar)

    assert result["frame"].height == 0
    assert result["manifest"]["unpaid_date_rows"] == 1


def test_materialize_dividend_events_counts_unmapped_corp_and_skips_junk_pages(tmp_path: Path) -> None:
    bronze_root = tmp_path / "bronze"
    _write_bridge(bronze_root)
    _write_envelope(
        bronze_root,
        rcept_no="20200102009999",
        received_on="2020-01-02",
        record="2019-12-31",
        pay="2020-04-20",
        dps=300,
        corp_code="00999999",
    )
    junk_dir = bronze_root / "corporate_actions" / ("f" * 64)
    junk_dir.mkdir(parents=True, exist_ok=True)
    (junk_dir / "payload.json").write_bytes(b"PK\x03\x04raw-zip-bytes")
    bad_dir = bronze_root / "corporate_actions" / ("e" * 64)
    bad_dir.mkdir(parents=True, exist_ok=True)
    (bad_dir / "payload.json").write_bytes(b"{not json")
    list_dir = bronze_root / "corporate_actions" / ("d" * 64)
    list_dir.mkdir(parents=True, exist_ok=True)
    (list_dir / "payload.json").write_bytes(b"[1, 2]")
    thin_dir = bronze_root / "corporate_actions" / ("c" * 64)
    thin_dir.mkdir(parents=True, exist_ok=True)
    (thin_dir / "payload.json").write_bytes(json.dumps({"rcept_no": "x"}).encode("utf-8"))
    calendar = _calendar([date(2019, 12, 26), date(2019, 12, 27), date(2019, 12, 30), date(2020, 1, 2), date(2020, 4, 20)])

    _, result = _materialize(bronze_root, calendar)

    assert result["frame"].height == 0
    assert result["manifest"]["unmapped_rows"] == 1


def test_materialize_dividend_events_rejects_broken_bridge(tmp_path: Path) -> None:
    import pytest

    from src.data.schemas import PITDataError
    from src.data.dividend_events import materialize_dividend_events

    calendar = _calendar([date(2019, 12, 26), date(2019, 12, 27)])
    universe_root = tmp_path / "universe"
    universe_root.mkdir(parents=True, exist_ok=True)
    silver_root = tmp_path / "silver"

    with pytest.raises(PITDataError, match="bridge is missing"):
        materialize_dividend_events(
            bronze_root=tmp_path / "bronze",
            universe_root=universe_root,
            silver_root=silver_root,
            calendar=calendar,
        )

    def _bridge_with(payload: object) -> Path:
        root = tmp_path / f"bronze_{abs(hash(json.dumps(payload, sort_keys=True, default=str))) % 10_000_000}"
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        target = root / "dart_corp_codes" / hashlib.sha256(raw).hexdigest()
        target.mkdir(parents=True, exist_ok=True)
        (target / "payload.json").write_bytes(raw)
        return root

    with pytest.raises(PITDataError, match="invalid"):
        materialize_dividend_events(
            bronze_root=_bridge_with({"oops": True}),
            universe_root=universe_root,
            silver_root=silver_root,
            calendar=calendar,
        )
    with pytest.raises(PITDataError, match="multiple tickers"):
        materialize_dividend_events(
            bronze_root=_bridge_with([
                {"ticker": "005930", "corp_code": "00126380"},
                {"ticker": "000660", "corp_code": "00126380"},
            ]),
            universe_root=universe_root,
            silver_root=silver_root,
            calendar=calendar,
        )
    with pytest.raises(PITDataError, match="empty"):
        materialize_dividend_events(
            bronze_root=_bridge_with(["junk", {"ticker": "", "corp_code": ""}]),
            universe_root=universe_root,
            silver_root=silver_root,
            calendar=calendar,
        )


def test_materialize_dividend_events_rejects_bad_envelope_and_session_math(tmp_path: Path) -> None:
    import pytest

    from src.data.schemas import PITDataError
    from src.data.dividend_events import materialize_dividend_events

    def _run(bronze_root: Path, calendar: SessionCalendar) -> Path:
        universe_root = bronze_root.parent / "universe"
        universe_root.mkdir(parents=True, exist_ok=True)
        return materialize_dividend_events(
            bronze_root=bronze_root,
            universe_root=universe_root,
            silver_root=bronze_root.parent / "silver",
            calendar=calendar,
        )

    corrupt = tmp_path / "corrupt" / "bronze"
    _write_bridge(corrupt)
    raw = json.dumps({
        "rcept_no": "20200102001234",
        "corp_code": "00126380",
        "received_on": "not-a-date",
        "archive_b64": "!!!",
    }).encode("utf-8")
    target = corrupt / "corporate_actions" / hashlib.sha256(raw).hexdigest()
    target.mkdir(parents=True, exist_ok=True)
    (target / "payload.json").write_bytes(raw)
    with pytest.raises(PITDataError, match="envelope"):
        _run(corrupt, _calendar([date(2019, 12, 26), date(2019, 12, 27)]))

    early = tmp_path / "early" / "bronze"
    _write_bridge(early)
    _write_envelope(
        early, rcept_no="20200102001234", received_on="2020-01-02",
        record="2019-12-31", pay="2020-04-20", dps=300,
    )
    with pytest.raises(PITDataError, match="on or before record"):
        _run(early, _calendar([date(2020, 1, 5), date(2020, 1, 6)]))
    with pytest.raises(PITDataError, match="no prior certified session"):
        _run(early, _calendar([date(2019, 12, 31), date(2020, 1, 6), date(2020, 4, 20)]))
    early_pay = tmp_path / "early_pay" / "bronze"
    _write_bridge(early_pay)
    _write_envelope(
        early_pay, rcept_no="20200102001234", received_on="2020-01-02",
        record="2019-12-31", pay="2019-01-05", dps=300,
    )
    with pytest.raises(PITDataError, match="on or before pay"):
        _run(early_pay, _calendar([date(2019, 12, 26), date(2019, 12, 27), date(2019, 12, 30), date(2020, 1, 2)]))
    with pytest.raises(PITDataError, match="open after receipt"):
        _run(early, _calendar([date(2019, 12, 26), date(2019, 12, 27), date(2019, 12, 30), date(2020, 1, 2), date(2020, 4, 20)][:-1]))
    with pytest.raises(PITDataError, match="at least two"):
        _run(early, _calendar([date(2019, 12, 30)]))


def test_materialize_dividend_events_is_idempotent_and_detects_conflicts(tmp_path: Path) -> None:
    import pytest

    from src.data.schemas import PITDataError
    from src.data.dividend_events import materialize_dividend_events

    bronze_root = tmp_path / "bronze"
    _write_bridge(bronze_root)
    _write_envelope(
        bronze_root, rcept_no="20200102001234", received_on="2020-01-02",
        record="2019-12-31", pay="2020-04-20", dps=300,
    )
    calendar = _calendar([date(2019, 12, 26), date(2019, 12, 27), date(2019, 12, 30), date(2020, 1, 2), date(2020, 4, 20)])
    universe_root = tmp_path / "universe"
    universe_root.mkdir(parents=True, exist_ok=True)
    silver_root = tmp_path / "silver"
    kwargs = {"bronze_root": bronze_root, "universe_root": universe_root, "silver_root": silver_root, "calendar": calendar}

    first = materialize_dividend_events(**kwargs)
    assert materialize_dividend_events(**kwargs) == first

    (first / "manifest.json").write_text('{"dataset_id": "tampered"}', encoding="utf-8")
    with pytest.raises(PITDataError, match="differs"):
        materialize_dividend_events(**kwargs)

    (first / "manifest.json").unlink()
    (first / "manifest.json").mkdir()
    with pytest.raises(PITDataError, match="unreadable"):
        materialize_dividend_events(**kwargs)
