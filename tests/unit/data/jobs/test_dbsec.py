"""DB Securities fallback job invariants: finality gating, master-gated bars, raw envelopes."""
from __future__ import annotations

import json
from datetime import UTC, date, datetime, time
from pathlib import Path

import pytest

SCOPE_CONFIG = Path("config/research/kr_swing_2019_v1.toml")
SESSIONS = (
    date(2026, 9, 28),
    date(2026, 9, 29),
    date(2026, 9, 30),
    date(2026, 10, 1),
    date(2026, 10, 2),
    date(2026, 10, 5),
    date(2026, 10, 6),
)
TICKERS = ("005930", "000020")


def _runtime(tmp_path: Path, *, evidence_start: date | None = None):  # type: ignore[no-untyped-def]
    import dataclasses

    from src.data.runtime import load_data_runtime

    runtime = load_data_runtime(scope_config=SCOPE_CONFIG, data_root=tmp_path / "data")
    if evidence_start is None:
        return runtime
    return dataclasses.replace(runtime, scope=runtime.scope.model_copy(update={"evidence_start": evidence_start}))


def _provider():  # type: ignore[no-untyped-def]
    from src.config import load_provider_policy, load_runtime_config

    return load_provider_policy(load_runtime_config())


def _provider_with_lookback(lookback: int):  # type: ignore[no-untyped-def]
    provider = _provider()
    return provider.model_copy(update={"dbsec": provider.dbsec.model_copy(update={"lookback_sessions": lookback})})


def _stub_calendar(start: date, end: date) -> tuple[date, ...]:
    return tuple(day for day in SESSIONS if start <= day <= end)


class _DbsecCollector:
    """Scripted stand-in for the DbsecClient surface the jobs use."""

    def __init__(self, *, master=(), daily=None, minute=None):  # type: ignore[no-untyped-def]
        self._master = [dict(row) for row in master]
        self._daily = {(ticker, day): [dict(row) for row in rows] for (ticker, day), rows in dict(daily or {}).items()}
        self._minute = {
            (ticker, day): [dict(row) for row in rows] for (ticker, day), rows in dict(minute or {}).items()
        }
        self.calls: list[tuple] = []
        self.health_checks = 0

    def fetch_ticker_master(self, market_div):  # type: ignore[no-untyped-def]
        self.calls.append(("master", market_div))
        return tuple(self._master)

    def fetch_daily_chart(self, ticker, start, end, *, adjusted):  # type: ignore[no-untyped-def]
        self.calls.append(("daily", ticker, start, end, adjusted))
        return tuple(self._daily.get((ticker, start), ()))

    def fetch_minute_chart(self, ticker, anchor, *, bar_count):  # type: ignore[no-untyped-def]
        self.calls.append(("minute", ticker, anchor, bar_count))
        return tuple(self._minute.get((ticker, anchor), ()))

    def health_check(self) -> None:
        self.health_checks += 1


def _master_row(ticker: str, market: str = "거래소"):  # type: ignore[no-untyped-def]
    return {
        "Iscd": ticker,
        "StndIscd": f"KR7{ticker}003",
        "KorIsnm": f"name-{ticker}",
        "MrktClsCode": "STK",
        "MrktClsName": market,
    }


def _daily_row(day: date, **overrides: object):  # type: ignore[no-untyped-def]
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


def _bar_row(day: date, hour: str, **overrides: object):  # type: ignore[no-untyped-def]
    row: dict[str, object] = {
        "Date": day.strftime("%Y%m%d"),
        "Hour": hour,
        "Oprc": "100",
        "Hprc": "110",
        "Lprc": "90",
        "Prpr": "105",
        "CntgVol": "10",
    }
    row.update(overrides)
    return row


def _ctx(runtime, provider, *, collector=None, now=None):  # type: ignore[no-untyped-def]
    from src.data.jobs.dbsec import build_dbsec_job_context

    return build_dbsec_job_context(runtime=runtime, provider=provider, collector=collector, now=now)


def _run(spec, ctx, **kwargs):  # type: ignore[no-untyped-def]
    from src.data.jobs.runner import run_job

    emitted: list[dict] = []
    params = {"chunk_size": 50, "max_chunks": None, "dry_run": False, "emit": emitted.append}
    params.update(kwargs)
    return run_job(spec, ctx, **params), emitted


def _patch_calendar(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.data.jobs.dbsec as dbsec_jobs

    monkeypatch.setattr(dbsec_jobs, "_xkrx_sessions", _stub_calendar)


def _persist_master(ctx, job, day, collector):  # type: ignore[no-untyped-def]
    from src.data.jobs.runner import JobUnit

    units = [
        JobUnit(
            source=job.source, natural_key=day.isoformat(),
            payload={"session": day.isoformat()}, max_requests=1,
        )
    ]
    payloads = job.fetch(ctx, units)
    assert collector.calls[-1] == ("master", "J")
    ctx.writer.persist_many(payloads)
    return payloads


def _envelope_rows(payload) -> list:  # type: ignore[no-untyped-def]
    return json.loads(payload.payload)["rows"]


def test_final_sessions_exclude_today_before_cutoff(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_calendar(monkeypatch)
    from src.data.jobs.dbsec import dbsec_final_sessions

    before = dbsec_final_sessions(
        now=datetime(2026, 10, 2, 10, 59, tzinfo=UTC),  # 19:59 KST
        final_time_kst=time(20, 10), lookback=5, calendar=_stub_calendar,
    )
    at_cutoff = dbsec_final_sessions(
        now=datetime(2026, 10, 2, 11, 10, tzinfo=UTC),  # 20:10 KST
        final_time_kst=time(20, 10), lookback=5, calendar=_stub_calendar,
    )

    assert before[-1] == date(2026, 10, 1)
    assert date(2026, 10, 2) not in before
    assert at_cutoff[-1] == date(2026, 10, 2)


def test_final_sessions_respect_lookback_and_evidence_start(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_calendar(monkeypatch)
    from src.data.jobs.dbsec import DbsecDailyChartJob

    def evening() -> datetime:
        return datetime(2026, 10, 2, 12, 0, tzinfo=UTC)  # 21:00 KST
    collector = _DbsecCollector(master=[_master_row(ticker) for ticker in TICKERS])
    ctx = _ctx(_runtime(tmp_path), _provider_with_lookback(2), collector=collector, now=evening)
    from src.data.jobs.dbsec import DbsecTickerMasterJob

    for day in SESSIONS[:5]:
        _persist_master(ctx, DbsecTickerMasterJob(), day, collector)

    units = DbsecDailyChartJob().pending(ctx)
    sessions = sorted({unit.payload["session"] for unit in units})

    assert sessions == ["2026-10-01", "2026-10-02"]
    assert len(units) <= 5 * len(TICKERS)

    late_ctx = _ctx(
        _runtime(tmp_path, evidence_start=date(2026, 10, 2)), _provider(), collector=collector, now=evening
    )
    late_units = DbsecDailyChartJob().pending(late_ctx)
    assert {unit.payload["session"] for unit in late_units} <= {"2026-10-02"}


def test_master_planned_only_for_newest_final_session(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_calendar(monkeypatch)
    from src.data.jobs.dbsec import DbsecTickerMasterJob

    def now() -> datetime:
        return datetime(2026, 9, 30, 12, 0, tzinfo=UTC)  # 21:00 KST
    ctx = _ctx(_runtime(tmp_path), _provider(), collector=_DbsecCollector(), now=now)

    units = DbsecTickerMasterJob().pending(ctx)

    assert [unit.natural_key for unit in units] == ["2026-09-30"]
    assert units[0].max_requests == 1


def test_master_not_planned_after_validity_window(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_calendar(monkeypatch)
    from src.data.jobs.dbsec import DbsecTickerMasterJob

    def now() -> datetime:
        return datetime(2026, 10, 1, 23, 1, tzinfo=UTC)  # 10-02 08:01 KST
    ctx = _ctx(_runtime(tmp_path), _provider(), collector=_DbsecCollector(), now=now)

    assert DbsecTickerMasterJob().pending(ctx) == ()


def test_master_roundtrip_and_context_uses_dbsec_ledger(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_calendar(monkeypatch)
    import src.data.jobs.dbsec as dbsec_jobs

    def now() -> datetime:
        return datetime(2026, 9, 30, 12, 0, tzinfo=UTC)  # 21:00 KST
    collector = _DbsecCollector(master=[_master_row("005930")])
    ctx = _ctx(_runtime(tmp_path), _provider(), collector=collector, now=now)

    assert ctx.runner.quota_provider == "DBSEC"
    assert ctx.key_env == "DB_APP_KEY"

    report, _ = _run(dbsec_jobs.DbsecTickerMasterJob(), ctx)

    assert report.status == "complete"
    assert report.done == 1
    assert collector.health_checks == 1
    second, _ = _run(dbsec_jobs.DbsecTickerMasterJob(), ctx)
    assert second.done == 0


def test_bar_units_require_same_session_master(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_calendar(monkeypatch)
    import src.data.jobs.dbsec as dbsec_jobs

    def evening() -> datetime:
        return datetime(2026, 10, 2, 12, 0, tzinfo=UTC)  # 21:00 KST
    collector = _DbsecCollector(master=[_master_row(ticker) for ticker in TICKERS])
    ctx = _ctx(_runtime(tmp_path), _provider_with_lookback(2), collector=collector, now=evening)
    _persist_master(ctx, dbsec_jobs.DbsecTickerMasterJob(), date(2026, 10, 2), collector)

    units = dbsec_jobs.DbsecDailyChartJob().pending(ctx)

    assert units
    assert {unit.payload["session"] for unit in units} == {"2026-10-02"}


def test_bar_units_skip_answered_keys(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_calendar(monkeypatch)
    import src.data.jobs.dbsec as dbsec_jobs

    def evening() -> datetime:
        return datetime(2026, 10, 2, 12, 0, tzinfo=UTC)  # 21:00 KST
    collector = _DbsecCollector(master=[_master_row(ticker) for ticker in TICKERS])
    ctx = _ctx(_runtime(tmp_path), _provider_with_lookback(1), collector=collector, now=evening)
    _persist_master(ctx, dbsec_jobs.DbsecTickerMasterJob(), date(2026, 10, 2), collector)

    report, _ = _run(dbsec_jobs.DbsecDailyChartJob(), ctx)

    assert report.status == "complete"
    assert dbsec_jobs.DbsecDailyChartJob().pending(ctx) == ()
    entries = ctx.catalog.latest(
        source=dbsec_jobs.DBSEC_DAILY_CHART_SOURCE, natural_keys={"2026-10-02:000020"}
    )
    from src.data.receipt_catalog import EvidenceStatus

    assert entries["2026-10-02:000020"].status is EvidenceStatus.EMPTY


def test_master_tickers_keep_equities_only() -> None:
    import src.data.jobs.dbsec as dbsec_jobs
    from src.data.evidence_sources import DBSEC_TICKER_MASTER_SOURCE, raw_rows_envelope, source_contract

    rows = [
        _master_row("005930", "거래소"),
        _master_row("000020", "코스닥"),
        _master_row("000020", "코스닥"),
        _master_row("123456", "ETF"),
        _master_row("234567", "ETN"),
        _master_row("34567", "거래소"),
        _master_row("AB12", "코스닥"),
    ]
    payload = raw_rows_envelope(
        source_contract(DBSEC_TICKER_MASTER_SOURCE),
        query={"market_div": "J", "session": "2026-10-02"},
        rows=rows,
    )

    assert dbsec_jobs.dbsec_master_tickers(payload, market_names=("거래소", "코스닥")) == ("000020", "005930")


def test_closing_bar_stores_only_1530_row(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_calendar(monkeypatch)
    import src.data.jobs.dbsec as dbsec_jobs
    from src.data.jobs.runner import JobUnit

    day = date(2026, 10, 2)
    hours = [f"{index:06d}" for index in range(400)]
    hours[200] = "153000"
    collector = _DbsecCollector(minute={("005930", day): [_bar_row(day, hour) for hour in hours]})
    ctx = _ctx(_runtime(tmp_path), _provider(), collector=collector)
    unit = JobUnit(
        source=dbsec_jobs.DBSEC_CLOSING_BAR_SOURCE, natural_key=f"{day.isoformat()}:005930",
        payload={"session": day.isoformat(), "ticker": "005930"}, max_requests=1,
    )

    (payload,) = dbsec_jobs.DbsecClosingBarJob().fetch(ctx, [unit])

    assert _envelope_rows(payload) == [_bar_row(day, "153000")]
    assert json.loads(payload.payload)["query"]["row_filter"] == f"Date={day.strftime('%Y%m%d')},Hour=153000"


def test_closing_bar_missing_is_empty_not_substituted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_calendar(monkeypatch)
    import src.data.jobs.dbsec as dbsec_jobs
    from src.data.jobs.runner import JobUnit
    from src.data.receipt_catalog import EvidenceStatus

    day = date(2026, 10, 2)
    collector = _DbsecCollector(
        minute={("005930", day): [_bar_row(day, "145900"), _bar_row(day, "200000")]}
    )
    ctx = _ctx(_runtime(tmp_path), _provider(), collector=collector)
    unit = JobUnit(
        source=dbsec_jobs.DBSEC_CLOSING_BAR_SOURCE, natural_key=f"{day.isoformat()}:005930",
        payload={"session": day.isoformat(), "ticker": "005930"}, max_requests=1,
    )

    (payload,) = dbsec_jobs.DbsecClosingBarJob().fetch(ctx, [unit])

    assert payload.status is EvidenceStatus.EMPTY
    assert _envelope_rows(payload) == []


def test_closing_bar_duplicate_1530_fails_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_calendar(monkeypatch)
    import src.data.jobs.dbsec as dbsec_jobs
    from src.core.pit import PITDataError
    from src.data.jobs.runner import JobUnit

    day = date(2026, 10, 2)
    collector = _DbsecCollector(
        minute={("005930", day): [_bar_row(day, "153000"), _bar_row(day, "153000")]}
    )
    ctx = _ctx(_runtime(tmp_path), _provider(), collector=collector)
    unit = JobUnit(
        source=dbsec_jobs.DBSEC_CLOSING_BAR_SOURCE, natural_key=f"{day.isoformat()}:005930",
        payload={"session": day.isoformat(), "ticker": "005930"}, max_requests=1,
    )

    with pytest.raises(PITDataError, match="1530"):
        dbsec_jobs.DbsecClosingBarJob().fetch(ctx, [unit])


def test_daily_chart_keeps_only_session_row(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_calendar(monkeypatch)
    import src.data.jobs.dbsec as dbsec_jobs
    from src.data.jobs.runner import JobUnit

    day = date(2026, 10, 2)
    prev = date(2026, 10, 1)
    collector = _DbsecCollector(daily={("005930", day): [_daily_row(prev), _daily_row(day)]})
    ctx = _ctx(_runtime(tmp_path), _provider(), collector=collector)
    unit = JobUnit(
        source=dbsec_jobs.DBSEC_DAILY_CHART_SOURCE, natural_key=f"{day.isoformat()}:005930",
        payload={"session": day.isoformat(), "ticker": "005930"}, max_requests=1,
    )

    (payload,) = dbsec_jobs.DbsecDailyChartJob().fetch(ctx, [unit])

    assert _envelope_rows(payload) == [_daily_row(day)]
    assert collector.calls[-1] == ("daily", "005930", day, day, False)


def test_payloads_pass_writer_contract(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_calendar(monkeypatch)
    import src.data.jobs.dbsec as dbsec_jobs
    from src.data.receipt_catalog import EvidenceStatus

    day = date(2026, 10, 2)
    collector = _DbsecCollector(
        master=[_master_row(ticker) for ticker in TICKERS],
        daily={("005930", day): [_daily_row(day)]},
    )
    ctx = _ctx(
        _runtime(tmp_path), _provider_with_lookback(1), collector=collector,
        now=lambda: datetime(2026, 10, 2, 12, 0, tzinfo=UTC),
    )
    _persist_master(ctx, dbsec_jobs.DbsecTickerMasterJob(), day, collector)
    units = dbsec_jobs.DbsecDailyChartJob().pending(ctx)
    assert len(units) == 2

    payloads = dbsec_jobs.DbsecDailyChartJob().fetch(ctx, units)
    ctx.writer.persist_many(tuple(payloads))

    entries = ctx.catalog.latest(
        source=dbsec_jobs.DBSEC_DAILY_CHART_SOURCE,
        natural_keys={unit.natural_key for unit in units},
    )
    assert set(entries) == {unit.natural_key for unit in units}
    assert entries[f"{day.isoformat()}:005930"].status is EvidenceStatus.SUCCESS
    assert entries[f"{day.isoformat()}:000020"].status is EvidenceStatus.EMPTY


def test_jobs_registry_order_is_master_first() -> None:
    import src.data.jobs.dbsec as dbsec_jobs
    from src.core.pit import PITDataError

    assert list(dbsec_jobs.DBSEC_JOBS) == ["dbsec_ticker_master", "dbsec_daily_chart", "dbsec_closing_bar"]
    assert isinstance(
        dbsec_jobs.resolve_dbsec_job("dbsec_closing_bar"), dbsec_jobs.DbsecClosingBarJob
    )
    with pytest.raises(PITDataError, match="unknown DB job"):
        dbsec_jobs.resolve_dbsec_job("dbsec_nope")


def test_invalid_final_inputs_fail_closed() -> None:
    import src.data.jobs.dbsec as dbsec_jobs
    from src.core.pit import PITDataError

    with pytest.raises(PITDataError, match="HH:MM"):
        dbsec_jobs._parse_hhmm("25:00")
    with pytest.raises(ValueError, match="lookback"):
        dbsec_jobs.dbsec_final_sessions(
            now=datetime(2026, 10, 2, 12, 0, tzinfo=UTC),
            final_time_kst=time(20, 10), lookback=0, calendar=_stub_calendar,
        )


def test_next_session_skips_empty_windows_and_fails_closed() -> None:
    import src.data.jobs.dbsec as dbsec_jobs
    from src.core.pit import PITDataError

    calls: list[tuple[date, date]] = []

    def _flaky(start: date, end: date) -> tuple[date, ...]:
        calls.append((start, end))
        if len(calls) == 1:
            raise ValueError("no XKRX session in range")
        return (date(2026, 10, 5),)

    assert dbsec_jobs._next_session(date(2026, 10, 2), calendar=_flaky) == date(2026, 10, 5)
    with pytest.raises(PITDataError, match="no XKRX session after"):
        dbsec_jobs._next_session(date(2026, 10, 2), calendar=lambda start, end: ())


def test_unreadable_master_payload_fails_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_calendar(monkeypatch)
    import src.data.jobs.dbsec as dbsec_jobs
    from src.core.pit import PITDataError
    from src.data.receipt_catalog import EvidenceStatus, ReceiptIndexEntry

    entry = ReceiptIndexEntry(
        source=dbsec_jobs.DBSEC_TICKER_MASTER_SOURCE, natural_key="2026-10-02",
        as_of=date(2026, 10, 2), fiscal_period=None, status=EvidenceStatus.SUCCESS,
        content_hash="0" * 64, retrieved_at=datetime(2026, 10, 2, 12, 0, tzinfo=UTC),
        payload_path=tmp_path / "ghost.json",
    )
    with pytest.raises(PITDataError, match="missing"):
        dbsec_jobs._master_envelope_bytes(_ctx(_runtime(tmp_path), _provider()), entry)

    monkeypatch.setattr(dbsec_jobs, "read_payload", lambda path: b"tampered")
    with pytest.raises(PITDataError, match="hash mismatch"):
        dbsec_jobs._master_envelope_bytes(_ctx(_runtime(tmp_path), _provider()), entry)


def test_no_final_session_plans_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_calendar(monkeypatch)
    import src.data.jobs.dbsec as dbsec_jobs

    def now() -> datetime:
        return datetime(2026, 9, 28, 1, 0, tzinfo=UTC)  # 10:00 KST, before any finalization

    ctx = _ctx(_runtime(tmp_path), _provider(), collector=_DbsecCollector(), now=now)

    assert dbsec_jobs.DbsecTickerMasterJob().pending(ctx) == ()
    assert dbsec_jobs.DbsecDailyChartJob().pending(ctx) == ()


def test_closing_bar_roundtrip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_calendar(monkeypatch)
    import src.data.jobs.dbsec as dbsec_jobs
    from src.data.receipt_catalog import EvidenceStatus

    day = date(2026, 10, 2)

    def evening() -> datetime:
        return datetime(2026, 10, 2, 12, 0, tzinfo=UTC)  # 21:00 KST

    collector = _DbsecCollector(
        master=[_master_row(ticker) for ticker in TICKERS],
        minute={("005930", day): [_bar_row(day, "145900"), _bar_row(day, "153000")]},
    )
    ctx = _ctx(_runtime(tmp_path), _provider_with_lookback(1), collector=collector, now=evening)
    _persist_master(ctx, dbsec_jobs.DbsecTickerMasterJob(), day, collector)

    report, _ = _run(dbsec_jobs.DbsecClosingBarJob(), ctx)

    assert report.status == "complete"
    assert report.done == 2
    assert collector.health_checks == 1
    entries = ctx.catalog.latest(
        source=dbsec_jobs.DBSEC_CLOSING_BAR_SOURCE,
        natural_keys={f"{day.isoformat()}:{ticker}" for ticker in TICKERS},
    )
    assert entries[f"{day.isoformat()}:005930"].status is EvidenceStatus.SUCCESS
    assert entries[f"{day.isoformat()}:000020"].status is EvidenceStatus.EMPTY
    assert dbsec_jobs.DbsecClosingBarJob().pending(ctx) == ()


def test_malformed_master_envelope_fails_closed() -> None:
    import src.data.jobs.dbsec as dbsec_jobs
    from src.core.pit import PITDataError

    with pytest.raises(PITDataError, match="not valid JSON"):
        dbsec_jobs.dbsec_master_tickers(b"{oops", market_names=("거래소",))
    with pytest.raises(PITDataError, match="no rows list"):
        dbsec_jobs.dbsec_master_tickers(b'{"rows": "nope"}', market_names=("거래소",))
