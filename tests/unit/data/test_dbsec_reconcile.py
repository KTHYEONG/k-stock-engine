"""Shadow reconciliation invariants: DB EOD evidence against KRX daily-market Silver."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from src.core.pit import PITDataError
from src.data.dbsec_reconcile import (
    FieldMismatch,
    ReconcileVerdict,
    SessionReconciliation,
    fallback_readiness,
    load_dbsec_session_rows,
    read_reconciliation_history,
    reconcile_session,
    write_reconciliation,
)
from src.data.receipt_catalog import EvidenceStatus

DAY = date(2026, 10, 6)

_KRX_SCHEMA = {
    "ticker": pl.String,
    "market": pl.String,
    "open": pl.Int64,
    "high": pl.Int64,
    "low": pl.Int64,
    "close": pl.Int64,
    "volume": pl.Int64,
    "price_state": pl.String,
}

_DB_SCHEMA = {
    "ticker": pl.String,
    "db_open": pl.Int64,
    "db_high": pl.Int64,
    "db_low": pl.Int64,
    "db_last": pl.Int64,
    "db_volume": pl.Int64,
    "db_close": pl.Int64,
    "db_closing_bar_volume": pl.Int64,
}


def _krx_row(ticker: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "ticker": ticker,
        "market": "KOSPI",
        "open": 100,
        "high": 110,
        "low": 90,
        "close": 105,
        "volume": 1000,
        "price_state": "tradable",
    }
    row.update(overrides)
    return row


def _krx(rows: list[dict[str, object]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=_KRX_SCHEMA)
    return pl.DataFrame(rows, schema=_KRX_SCHEMA)


def _db_row(ticker: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "ticker": ticker,
        "db_open": 100,
        "db_high": 110,
        "db_low": 90,
        "db_last": 105,
        "db_volume": 1000,
        "db_close": 105,
        "db_closing_bar_volume": 10,
    }
    row.update(overrides)
    return row


def _db(rows: list[dict[str, object]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=_DB_SCHEMA)
    return pl.DataFrame(rows, schema=_DB_SCHEMA)


def _reconcile(krx: pl.DataFrame, db: pl.DataFrame, **overrides: object) -> SessionReconciliation:
    params: dict[str, object] = {
        "session": DAY,
        "krx": krx,
        "db": db,
        "min_match_rate": 0.999,
        "min_coverage": 0.995,
        "sample_limit": 20,
    }
    params.update(overrides)
    return reconcile_session(**params)  # type: ignore[arg-type]


def _report(session: date, verdict: ReconcileVerdict) -> SessionReconciliation:
    return SessionReconciliation(
        session=session,
        verdict=verdict,
        krx_tradable=1,
        db_daily_rows=1,
        db_closing_bars=1,
        coverage=1.0,
        close_match_rate=1.0,
        volume_match_rate=1.0,
        open_match_rate=1.0,
        high_match_rate=1.0,
        low_match_rate=1.0,
        missing_closing_bar=0,
        samples=(),
    )


def test_reconcile_passes_on_exact_close_and_volume() -> None:
    tickers = ["005930", "000660", "035420"]
    report = _reconcile(_krx([_krx_row(ticker) for ticker in tickers]), _db([_db_row(ticker) for ticker in tickers]))

    assert report.verdict is ReconcileVerdict.PASS
    assert (report.close_match_rate, report.volume_match_rate, report.open_match_rate) == (1.0, 1.0, 1.0)
    assert report.coverage == 1.0
    assert report.samples == ()


def test_reconcile_never_uses_daily_last_as_close() -> None:
    krx = _krx([_krx_row("005930")])
    db = _db([_db_row("005930", db_last=105, db_close=999, db_closing_bar_volume=11)])

    report = _reconcile(krx, db)

    assert report.close_match_rate == 0.0
    assert report.verdict is ReconcileVerdict.FAIL


def test_reconcile_counts_missing_closing_bar_without_substitution() -> None:
    krx = _krx([_krx_row("005930"), _krx_row("000660")])
    db = _db([_db_row("005930"), _db_row("000660", db_close=None, db_closing_bar_volume=None)])

    report = _reconcile(krx, db)

    assert report.missing_closing_bar == 1
    assert report.close_match_rate == 1.0
    assert (report.db_daily_rows, report.db_closing_bars) == (2, 1)
    (sample,) = [item for item in report.samples if item.ticker == "000660"]
    assert (sample.field, sample.krx, sample.db) == ("close", 105, None)


def test_reconcile_ignores_high_low_for_verdict() -> None:
    krx = _krx([_krx_row("005930")])
    db = _db([_db_row("005930", db_high=1, db_low=2)])

    report = _reconcile(krx, db)

    assert report.verdict is ReconcileVerdict.PASS
    assert report.high_match_rate == 0.0


def test_reconcile_excludes_non_tradable_krx_rows() -> None:
    krx = _krx([_krx_row("005930"), _krx_row("000660", price_state="zero_volume", volume=0)])
    db = _db([_db_row("005930")])

    report = _reconcile(krx, db)

    assert report.krx_tradable == 1
    assert report.verdict is ReconcileVerdict.PASS


def test_reconcile_fails_on_low_coverage() -> None:
    tickers = [f"{index:06d}" for index in range(20)]
    krx = _krx([_krx_row(ticker) for ticker in tickers])
    db = _db([_db_row(ticker) for ticker in tickers[:18]])

    report = _reconcile(krx, db)

    assert report.coverage == pytest.approx(0.9)
    assert report.verdict is ReconcileVerdict.FAIL


def test_reconcile_no_krx_rows_is_no_data() -> None:
    report = _reconcile(_krx([]), _db([_db_row("005930")]))

    assert report.verdict is ReconcileVerdict.NO_DATA
    assert report.krx_tradable == 0


def test_reconcile_open_rate_skips_zero_krx_open() -> None:
    krx = _krx([_krx_row("005930", open=0)])
    db = _db([_db_row("005930", db_open=538)])

    report = _reconcile(krx, db)

    assert report.open_match_rate == 0.0
    assert report.verdict is ReconcileVerdict.PASS
    assert all(item.field != "open" for item in report.samples)


def test_reconcile_samples_bounded_and_ordered() -> None:
    krx_rows = [_krx_row(f"c{index:02d}") for index in range(5)]
    krx_rows += [_krx_row(f"v{index:02d}") for index in range(50)]
    krx_rows += [_krx_row(f"o{index:02d}") for index in range(10)]
    db_rows = [_db_row(f"c{index:02d}", db_close=1) for index in range(5)]
    db_rows += [_db_row(f"v{index:02d}", db_volume=1) for index in range(50)]
    db_rows += [_db_row(f"o{index:02d}", db_open=1) for index in range(10)]

    report = _reconcile(_krx(krx_rows), _db(db_rows), sample_limit=20)

    assert len(report.samples) == 20
    fields = [item.field for item in report.samples]
    assert fields == ["close"] * 5 + ["volume"] * 15
    assert [item.ticker for item in report.samples[:5]] == sorted(f"c{index:02d}" for index in range(5))
    assert [item.ticker for item in report.samples[5:]] == sorted(f"v{index:02d}" for index in range(15))


class _FakeCatalog:
    def __init__(self, entries: dict[tuple[str, str], SimpleNamespace]) -> None:
        self._entries = entries

    def latest(self, *, source: str, natural_keys: object) -> dict[str, SimpleNamespace]:
        return {
            key: self._entries[(source, key)]
            for key in list(natural_keys)  # type: ignore[union-attr]
            if (source, key) in self._entries
        }


def _payload_dir(tmp_path: Path) -> Path:
    bronze = tmp_path / "bronze"
    (bronze / "catalog").mkdir(parents=True, exist_ok=True)
    return bronze


def _store(bronze: Path, name: str, rows: list[dict[str, object]] | None, *, status: EvidenceStatus) -> SimpleNamespace:
    path = bronze / "dbsec" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"rows": rows if rows is not None else []}), encoding="utf-8")
    return SimpleNamespace(status=status, payload_path=path)


def _daily_row(day: date, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "Date": day.strftime("%Y%m%d"),
        "Oprc": "100",
        "Hprc": "110",
        "Lprc": "90",
        "Prpr": "105",
        "AcmlVol": "1000",
    }
    row.update(overrides)
    return row


def _closing_row(day: date, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "Date": day.strftime("%Y%m%d"),
        "Hour": "153000",
        "Oprc": "100",
        "Hprc": "110",
        "Lprc": "90",
        "Prpr": "105",
        "CntgVol": "10",
    }
    row.update(overrides)
    return row


def test_load_rows_keeps_nulls_for_empty_receipts(tmp_path: Path) -> None:
    bronze = _payload_dir(tmp_path)
    entries = {
        ("dbsec_daily_chart", f"{DAY.isoformat()}:005930"): _store(
            bronze, "daily", [_daily_row(DAY)], status=EvidenceStatus.SUCCESS
        ),
        ("dbsec_closing_bar", f"{DAY.isoformat()}:005930"): _store(
            bronze, "closing", [], status=EvidenceStatus.EMPTY
        ),
    }

    frame = load_dbsec_session_rows(_FakeCatalog(entries), DAY, ("005930",))  # type: ignore[arg-type]

    assert frame["db_close"].to_list() == [None]
    assert frame["db_volume"].to_list() == [1000]


def test_load_rows_parses_native_numeric_types_strictly(tmp_path: Path) -> None:
    bronze = _payload_dir(tmp_path)
    entries = {
        ("dbsec_daily_chart", f"{DAY.isoformat()}:005930"): _store(
            bronze,
            "daily-int",
            [_daily_row(DAY, Oprc=100, Hprc=110, Lprc=90, Prpr=105, AcmlVol="1,000")],
            status=EvidenceStatus.SUCCESS,
        ),
        ("dbsec_closing_bar", f"{DAY.isoformat()}:005930"): _store(
            bronze, "closing-float", [_closing_row(DAY, Prpr=105.0, CntgVol=10.0)], status=EvidenceStatus.SUCCESS
        ),
        ("dbsec_daily_chart", f"{DAY.isoformat()}:000660"): _store(
            bronze,
            "daily-null",
            [_daily_row(DAY, Oprc=None, Hprc="", Lprc="90", Prpr="105", AcmlVol="1000")],
            status=EvidenceStatus.SUCCESS,
        ),
        ("dbsec_closing_bar", f"{DAY.isoformat()}:000660"): _store(
            bronze, "closing-null", [_closing_row(DAY)], status=EvidenceStatus.SUCCESS
        ),
    }

    frame = load_dbsec_session_rows(_FakeCatalog(entries), DAY, ("005930", "000660"))  # type: ignore[arg-type]
    by_ticker = {row["ticker"]: row for row in frame.to_dicts()}

    assert by_ticker["005930"]["db_open"] == 100
    assert by_ticker["005930"]["db_volume"] == 1000
    assert by_ticker["005930"]["db_close"] == 105
    assert by_ticker["000660"]["db_open"] is None
    assert by_ticker["000660"]["db_high"] is None


def test_load_rows_rejects_non_plain_numeric_values(tmp_path: Path) -> None:
    bronze = _payload_dir(tmp_path)
    bad_daily = [
        ("bool", {"Oprc": True}),
        ("float", {"AcmlVol": 1000.5}),
        ("text", {"Hprc": "abc"}),
        ("structured", {"Lprc": ["90"]}),
    ]
    for name, override in bad_daily:
        entries = {
            ("dbsec_daily_chart", f"{DAY.isoformat()}:005930"): _store(
                bronze, f"daily-{name}", [_daily_row(DAY, **override)], status=EvidenceStatus.SUCCESS
            ),
            ("dbsec_closing_bar", f"{DAY.isoformat()}:005930"): _store(
                bronze, f"closing-{name}", [_closing_row(DAY)], status=EvidenceStatus.SUCCESS
            ),
        }
        with pytest.raises(PITDataError):
            load_dbsec_session_rows(_FakeCatalog(entries), DAY, ("005930",))  # type: ignore[arg-type]


def test_load_rows_rejects_malformed_envelopes(tmp_path: Path) -> None:
    bronze = _payload_dir(tmp_path)
    key = f"{DAY.isoformat()}:005930"
    bad_payloads = [
        ("not-json", b"{oops"),
        ("array", b"[1, 2]"),
        ("no-rows", b'{"query": {}}'),
        ("rows-str", b'{"rows": "nope"}'),
        ("rows-empty", b'{"rows": []}'),
        ("row-scalar", b'{"rows": [42]}'),
    ]
    for name, payload in bad_payloads:
        path = bronze / "dbsec" / f"bad-{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        entries = {
            ("dbsec_daily_chart", key): SimpleNamespace(status=EvidenceStatus.SUCCESS, payload_path=path),
            ("dbsec_closing_bar", key): _store(bronze, f"closing-{name}", [_closing_row(DAY)], status=EvidenceStatus.SUCCESS),
        }
        with pytest.raises(PITDataError):
            load_dbsec_session_rows(_FakeCatalog(entries), DAY, ("005930",))  # type: ignore[arg-type]

    outside = tmp_path / "elsewhere" / "daily.json"
    outside.parent.mkdir(parents=True, exist_ok=True)
    outside.write_text('{"rows": []}', encoding="utf-8")
    entries = {
        ("dbsec_daily_chart", key): SimpleNamespace(status=EvidenceStatus.SUCCESS, payload_path=outside),
        ("dbsec_closing_bar", key): _store(bronze, "closing-outside", [_closing_row(DAY)], status=EvidenceStatus.SUCCESS),
    }
    with pytest.raises(PITDataError):
        load_dbsec_session_rows(_FakeCatalog(entries), DAY, ("005930",))  # type: ignore[arg-type]

    incomplete_daily = _daily_row(DAY)
    del incomplete_daily["AcmlVol"]
    incomplete_closing = _closing_row(DAY)
    del incomplete_closing["CntgVol"]
    entries = {
        ("dbsec_daily_chart", key): _store(bronze, "daily-short", [incomplete_daily], status=EvidenceStatus.SUCCESS),
        ("dbsec_closing_bar", key): _store(bronze, "closing-ok", [_closing_row(DAY)], status=EvidenceStatus.SUCCESS),
    }
    with pytest.raises(PITDataError, match="AcmlVol"):
        load_dbsec_session_rows(_FakeCatalog(entries), DAY, ("005930",))  # type: ignore[arg-type]
    entries = {
        ("dbsec_daily_chart", key): _store(bronze, "daily-ok", [_daily_row(DAY)], status=EvidenceStatus.SUCCESS),
        ("dbsec_closing_bar", key): _store(bronze, "closing-short", [incomplete_closing], status=EvidenceStatus.SUCCESS),
    }
    with pytest.raises(PITDataError, match="CntgVol"):
        load_dbsec_session_rows(_FakeCatalog(entries), DAY, ("005930",))  # type: ignore[arg-type]


def test_load_rows_rejects_non_integer_strings(tmp_path: Path) -> None:
    bronze = _payload_dir(tmp_path)
    entries = {
        ("dbsec_daily_chart", f"{DAY.isoformat()}:005930"): _store(
            bronze, "daily", [_daily_row(DAY, AcmlVol="1.5")], status=EvidenceStatus.SUCCESS
        ),
        ("dbsec_closing_bar", f"{DAY.isoformat()}:005930"): _store(
            bronze, "closing", [_closing_row(DAY)], status=EvidenceStatus.SUCCESS
        ),
    }

    with pytest.raises(PITDataError) as excinfo:
        load_dbsec_session_rows(_FakeCatalog(entries), DAY, ("005930",))  # type: ignore[arg-type]

    assert "1.5" not in str(excinfo.value)
    assert "005930" in str(excinfo.value)


@pytest.mark.parametrize("text", ["Infinity", "NaN", "1e400"])
def test_load_rows_rejects_non_finite_strings(tmp_path: Path, text: str) -> None:
    bronze = _payload_dir(tmp_path)
    entries = {
        ("dbsec_daily_chart", f"{DAY.isoformat()}:005930"): _store(
            bronze, "daily", [_daily_row(DAY, AcmlVol=text)], status=EvidenceStatus.SUCCESS
        ),
    }

    with pytest.raises(PITDataError, match="AcmlVol"):
        load_dbsec_session_rows(_FakeCatalog(entries), DAY, ("005930",))  # type: ignore[arg-type]


def test_reconcile_tolerates_null_and_float_cells() -> None:
    krx = pl.DataFrame(
        [
            {"ticker": "005930", "market": "KOSPI", "open": 100, "high": "110",
             "low": True, "close": 105.0, "volume": 1000, "price_state": "tradable"},
            {"ticker": "000660", "market": "KOSPI", "open": 100, "high": "110",
             "low": False, "close": None, "volume": None, "price_state": "tradable"},
        ],
        schema={
            "ticker": pl.String, "market": pl.String, "open": pl.Int64, "high": pl.String,
            "low": pl.Boolean, "close": pl.Float64, "volume": pl.Float64, "price_state": pl.String,
        },
    )
    db = pl.DataFrame(
        [_db_row("005930"), _db_row("000660"), {**_db_row("005930"), "ticker": None}],
        schema=_DB_SCHEMA,
    )

    report = _reconcile(krx, db)

    assert report.verdict is ReconcileVerdict.PASS
    assert report.high_match_rate == 0.0
    assert report.low_match_rate == 0.0
    assert report.missing_closing_bar == 0


def _sessions(count: int) -> list[date]:
    from src.core.krx_calendar import xkrx_session_calendar

    calendar = xkrx_session_calendar(start=date(2026, 9, 1), end=date(2026, 10, 9))
    ordered = sorted({instant.date() for instant in calendar.sessions})
    assert len(ordered) >= count
    return ordered[-count:]


def test_readiness_requires_consecutive_pass_sessions() -> None:
    window = _sessions(21)
    gapped = [day for index, day in enumerate(window) if index != 10]

    assert fallback_readiness([_report(day, ReconcileVerdict.PASS) for day in gapped], ready_sessions=20) is False
    assert fallback_readiness([_report(day, ReconcileVerdict.PASS) for day in window[:20]], ready_sessions=20) is True


def test_readiness_skips_no_data_sessions() -> None:
    window = _sessions(21)
    history = [
        _report(day, ReconcileVerdict.NO_DATA if day == window[10] else ReconcileVerdict.PASS) for day in window
    ]

    assert fallback_readiness(history, ready_sessions=20) is True


def test_readiness_rejects_fail_short_and_invalid_windows() -> None:
    window = _sessions(20)

    assert fallback_readiness([_report(day, ReconcileVerdict.PASS) for day in window[:5]], ready_sessions=20) is False
    failed = [_report(day, ReconcileVerdict.FAIL if day == window[3] else ReconcileVerdict.PASS) for day in window]
    assert fallback_readiness(failed, ready_sessions=20) is False
    with pytest.raises(ValueError, match="ready_sessions"):
        fallback_readiness(failed, ready_sessions=0)


def test_readiness_fails_closed_outside_calendar_range() -> None:
    history = [_report(date(1980, 1, 5), ReconcileVerdict.PASS), _report(date(1980, 1, 6), ReconcileVerdict.PASS)]

    assert fallback_readiness(history, ready_sessions=2) is False


def test_reconciliation_write_is_idempotent(tmp_path: Path) -> None:
    report = SessionReconciliation(
        session=DAY,
        verdict=ReconcileVerdict.FAIL,
        krx_tradable=2,
        db_daily_rows=2,
        db_closing_bars=1,
        coverage=1.0,
        close_match_rate=0.5,
        volume_match_rate=1.0,
        open_match_rate=1.0,
        high_match_rate=1.0,
        low_match_rate=0.0,
        missing_closing_bar=1,
        samples=(FieldMismatch(ticker="000660", field="close", krx=105, db=None),),
    )
    root = tmp_path / "dbsec_reconcile"

    first = write_reconciliation(report, root=root)
    second = write_reconciliation(report, root=root)

    assert first == second
    assert [path.name for path in root.glob("*.json")] == [f"{DAY.isoformat()}.json"]
    assert first.read_bytes() == second.read_bytes()
    (loaded,) = read_reconciliation_history(root)
    assert loaded == report


def test_read_history_rejects_unreadable_files(tmp_path: Path) -> None:
    root = tmp_path / "dbsec_reconcile"
    root.mkdir(parents=True, exist_ok=True)
    (root / "2026-10-06.json").write_text("{broken", encoding="utf-8")

    with pytest.raises(PITDataError):
        read_reconciliation_history(root)


def test_read_history_rejects_wrongly_typed_fields(tmp_path: Path) -> None:
    root = tmp_path / "dbsec_reconcile"
    root.mkdir(parents=True, exist_ok=True)
    (root / "2026-10-06.json").write_text(
        json.dumps({
            "session": "2026-10-06", "verdict": "pass", "krx_tradable": 1, "db_daily_rows": 1,
            "db_closing_bars": 1, "coverage": 1.0, "close_match_rate": 1.0, "volume_match_rate": 1.0,
            "open_match_rate": 1.0, "high_match_rate": 1.0, "low_match_rate": 1.0,
            "missing_closing_bar": 0,
            "samples": [{"ticker": "005930", "field": "close", "krx": True, "db": None}],
        }),
        encoding="utf-8",
    )

    with pytest.raises(PITDataError):
        read_reconciliation_history(root)


def test_read_history_rejects_non_mapping_reports(tmp_path: Path) -> None:
    root = tmp_path / "dbsec_reconcile"
    root.mkdir(parents=True, exist_ok=True)
    (root / "2026-10-06.json").write_text("[1, 2]", encoding="utf-8")

    with pytest.raises(PITDataError):
        read_reconciliation_history(root)


def test_read_history_rejects_non_list_samples(tmp_path: Path) -> None:
    root = tmp_path / "dbsec_reconcile"
    root.mkdir(parents=True, exist_ok=True)
    (root / "2026-10-06.json").write_text(
        json.dumps({
            "session": "2026-10-06", "verdict": "pass", "krx_tradable": 1, "db_daily_rows": 1,
            "db_closing_bars": 1, "coverage": 1.0, "close_match_rate": 1.0, "volume_match_rate": 1.0,
            "open_match_rate": 1.0, "high_match_rate": 1.0, "low_match_rate": 1.0,
            "missing_closing_bar": 0, "samples": {"ticker": "005930"},
        }),
        encoding="utf-8",
    )

    with pytest.raises(PITDataError):
        read_reconciliation_history(root)


def test_read_history_skips_non_file_entries(tmp_path: Path) -> None:
    root = tmp_path / "dbsec_reconcile"
    root.mkdir(parents=True, exist_ok=True)
    (root / "2026-10-07.json").mkdir()
    target = root / "2026-10-06.json"
    target.write_text(
        json.dumps({
            "session": "2026-10-06", "verdict": "pass", "krx_tradable": 1, "db_daily_rows": 1,
            "db_closing_bars": 1, "coverage": 1.0, "close_match_rate": 1.0, "volume_match_rate": 1.0,
            "open_match_rate": 1.0, "high_match_rate": 1.0, "low_match_rate": 1.0,
            "missing_closing_bar": 0, "samples": [],
        }),
        encoding="utf-8",
    )
    (root / "2026-10-08.json").symlink_to(target)

    (loaded,) = read_reconciliation_history(root)

    assert loaded.session == date(2026, 10, 6)


def test_read_history_reports_unscannable_directories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(self: object, pattern: str) -> object:
        raise OSError("disk gone")

    monkeypatch.setattr(Path, "glob", _boom)

    with pytest.raises(PITDataError, match="history"):
        read_reconciliation_history(tmp_path)
