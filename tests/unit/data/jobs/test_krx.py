"""KRX job invariants: completed sessions only, catalog resume, holiday fail-closed."""
from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest

SCOPE_CONFIG = Path("config/research/kr_swing_2019_v1.toml")
SESSIONS = (date(2026, 1, 2), date(2026, 1, 5), date(2026, 1, 6))
NOW_AFTER_CLOSE = datetime(2026, 1, 6, 9, 1, tzinfo=UTC)  # 18:01 KST, last completed 2026-01-06
NOW_BEFORE_CLOSE = datetime(2026, 1, 6, 6, 40, tzinfo=UTC)  # 15:40 KST, last completed 2026-01-05


def _runtime(tmp_path: Path):  # type: ignore[no-untyped-def]
    from src.data.runtime import load_data_runtime

    return load_data_runtime(scope_config=SCOPE_CONFIG, data_root=tmp_path / "data")


def _provider():  # type: ignore[no-untyped-def]
    from src.config import load_provider_policy, load_runtime_config

    return load_provider_policy(load_runtime_config())


def _stub_calendar(start: date, end: date) -> tuple[date, ...]:  # type: ignore[no-untyped-def]
    return tuple(day for day in SESSIONS if start <= day <= end)


class _KrxCollector:
    """Scripted stand-in for the KRX collector surface the jobs use."""

    def __init__(self, *, daily=None, master=None, error=None):  # type: ignore[no-untyped-def]
        self._daily = {day: list(rows) for day, rows in dict(daily or {}).items()}
        self._master = {day: list(rows) for day, rows in dict(master or {}).items()}
        self._error = error
        self.fetch_calls: list[tuple] = []
        self.health_checks = 0

    def fetch_daily_records(self, session):  # type: ignore[no-untyped-def]
        self.fetch_calls.append(("daily", session))
        if self._error is not None:
            raise self._error
        return [dict(row) for row in self._daily.get(session, ())]

    def fetch_master_records(self, session):  # type: ignore[no-untyped-def]
        self.fetch_calls.append(("master", session))
        if self._error is not None:
            raise self._error
        return [dict(row) for row in self._master.get(session, ())]

    def health_check(self) -> None:
        self.health_checks += 1


def _daily_row():  # type: ignore[no-untyped-def]
    return {"TDD_CLSPRC": "1000", "MKTCAP": "5000", "LIST_SHRS": "100"}


def _master_row():  # type: ignore[no-untyped-def]
    return {"ISU_SRT_CD": "005930", "ISU_CD": "KR7005930003"}


def _ctx(runtime, provider, *, collector=None, now=None):  # type: ignore[no-untyped-def]
    from src.data.jobs.krx import build_krx_job_context

    return build_krx_job_context(
        runtime=runtime, provider=provider, collector=collector, now=now or (lambda: NOW_AFTER_CLOSE)
    )


def _run(spec, ctx, **kwargs):  # type: ignore[no-untyped-def]
    from src.data.jobs.runner import run_job

    emitted: list[dict] = []
    params = {"chunk_size": 5, "max_chunks": None, "dry_run": False, "emit": emitted.append}
    params.update(kwargs)
    return run_job(spec, ctx, **params), emitted


def test_today_before_close_is_excluded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import src.data.jobs.krx as krx_jobs

    monkeypatch.setattr(krx_jobs, "_xkrx_sessions", _stub_calendar)
    runtime = _runtime(tmp_path)
    provider = _provider()

    before = krx_jobs.KrxDailyMarketJob().pending(_ctx(runtime, provider, now=lambda: NOW_BEFORE_CLOSE))
    after = krx_jobs.KrxDailyMarketJob().pending(_ctx(runtime, provider, now=lambda: NOW_AFTER_CLOSE))

    assert [unit.natural_key for unit in before] == ["2026-01-02", "2026-01-05"]
    assert [unit.natural_key for unit in after] == ["2026-01-02", "2026-01-05", "2026-01-06"]


def test_stored_sessions_are_not_refetched(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import src.data.jobs.krx as krx_jobs

    monkeypatch.setattr(krx_jobs, "_xkrx_sessions", _stub_calendar)
    runtime = _runtime(tmp_path)
    provider = _provider()
    collector = _KrxCollector(daily={day: [_daily_row()] for day in SESSIONS})
    ctx = _ctx(runtime, provider, collector=collector)
    ctx.writer.persist_many(
        tuple(
            krx_jobs.krx_daily_market_scoped_payload(
                records=[_daily_row()], session=day, retrieved_at=NOW_AFTER_CLOSE
            )
            for day in SESSIONS
        )
    )
    collector.fetch_calls.clear()

    report, _ = _run(krx_jobs.KrxDailyMarketJob(), ctx)

    assert report.status == "complete"
    assert report.done == 0
    assert collector.fetch_calls == []
    assert collector.health_checks == 0


def test_holiday_mismatch_fails_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import src.data.jobs.krx as krx_jobs
    from src.integrations.krx.client import KrxHolidayError

    monkeypatch.setattr(krx_jobs, "_xkrx_sessions", _stub_calendar)
    runtime = _runtime(tmp_path)
    provider = _provider()
    collector = _KrxCollector(
        daily={date(2026, 1, 2): [_daily_row()]},
        error=KrxHolidayError("KRX reports no trading for 2026-01-05 (휴장일); review required"),
    )
    ctx = _ctx(runtime, provider, collector=collector)

    with pytest.raises(KrxHolidayError, match="2026-01-05"):
        _run(krx_jobs.KrxDailyMarketJob(), ctx)


def test_empty_answer_is_recorded_as_empty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import src.data.jobs.krx as krx_jobs
    from src.data.receipt_catalog import EvidenceStatus

    monkeypatch.setattr(krx_jobs, "_xkrx_sessions", _stub_calendar)
    runtime = _runtime(tmp_path)
    provider = _provider()
    ctx = _ctx(runtime, provider, collector=_KrxCollector())

    report, _ = _run(krx_jobs.KrxDailyMarketJob(), ctx)

    assert report.status == "complete"
    assert report.done == 3
    assert report.pending_left == 0
    entries = ctx.catalog.latest(
        source=krx_jobs.KRX_DAILY_MARKET_SOURCE, natural_keys={day.isoformat() for day in SESSIONS}
    )
    assert {key: entry.status for key, entry in entries.items()} == {
        day.isoformat(): EvidenceStatus.EMPTY for day in SESSIONS
    }


def test_master_job_roundtrip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import src.data.jobs.krx as krx_jobs

    monkeypatch.setattr(krx_jobs, "_xkrx_sessions", _stub_calendar)
    runtime = _runtime(tmp_path)
    provider = _provider()
    collector = _KrxCollector(master={day: [_master_row()] for day in SESSIONS})
    ctx = _ctx(runtime, provider, collector=collector)

    report, _ = _run(krx_jobs.KrxSecurityMasterJob(), ctx)

    assert report.status == "complete"
    assert report.done == 3
    assert collector.health_checks == 1
    second, _ = _run(krx_jobs.KrxSecurityMasterJob(), ctx)
    assert second.done == 0
    assert len(collector.fetch_calls) == 3


def test_completion_cutoff_boundaries() -> None:
    import src.data.jobs.krx as krx_jobs

    assert krx_jobs.last_completed_session_day(datetime(2026, 1, 6, 9, 0, tzinfo=UTC)) == date(2026, 1, 6)
    assert krx_jobs.last_completed_session_day(datetime(2026, 1, 6, 8, 59, tzinfo=UTC)) == date(2026, 1, 5)
    naive_noon_kst = datetime(2026, 1, 6, 12, 0)
    assert krx_jobs.last_completed_session_day(naive_noon_kst) == date(2026, 1, 6)
    assert krx_jobs.completed_sessions(evidence_start=date(2026, 1, 9), now=NOW_AFTER_CLOSE, calendar=_stub_calendar) == ()
    assert krx_jobs.completed_sessions(
        evidence_start=date(2016, 1, 1), now=NOW_AFTER_CLOSE, calendar=_stub_calendar
    ) == SESSIONS


def test_unknown_job_fails_closed() -> None:
    import src.data.jobs.krx as krx_jobs
    from src.core.pit import PITDataError

    with pytest.raises(PITDataError, match="unknown KRX job"):
        krx_jobs.resolve_krx_job("krx_dividend")


def test_job_context_uses_krx_ledger(tmp_path: Path) -> None:

    ctx = _ctx(_runtime(tmp_path), _provider())

    assert ctx.runner.quota_provider == "KRX"
    assert ctx.runner.daily_budget == 10000
    assert ctx.key_env == "KRX_OPENAPI_KEY"
    assert ctx.collector is None


def test_pending_with_no_sessions_returns_empty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import src.data.jobs.krx as krx_jobs

    monkeypatch.setattr(krx_jobs, "_xkrx_sessions", lambda start, end: ())
    runtime = _runtime(tmp_path)
    provider = _provider()

    assert krx_jobs.KrxDailyMarketJob().pending(_ctx(runtime, provider)) == ()


def test_master_empty_answer_is_recorded_as_empty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import src.data.jobs.krx as krx_jobs
    from src.data.receipt_catalog import EvidenceStatus

    monkeypatch.setattr(krx_jobs, "_xkrx_sessions", _stub_calendar)
    runtime = _runtime(tmp_path)
    provider = _provider()
    ctx = _ctx(runtime, provider, collector=_KrxCollector())

    report, _ = _run(krx_jobs.KrxSecurityMasterJob(), ctx)

    assert report.status == "complete"
    assert report.done == 3
    entries = ctx.catalog.latest(
        source=krx_jobs.KRX_SECURITY_MASTER_SOURCE, natural_keys={day.isoformat() for day in SESSIONS}
    )
    assert {key: entry.status for key, entry in entries.items()} == {
        day.isoformat(): EvidenceStatus.EMPTY for day in SESSIONS
    }


def test_daily_records_are_persisted_as_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import src.data.jobs.krx as krx_jobs
    from src.data.receipt_catalog import EvidenceStatus

    monkeypatch.setattr(krx_jobs, "_xkrx_sessions", _stub_calendar)
    runtime = _runtime(tmp_path)
    provider = _provider()
    collector = _KrxCollector(daily={day: [_daily_row()] for day in SESSIONS})
    ctx = _ctx(runtime, provider, collector=collector)

    report, _ = _run(krx_jobs.KrxDailyMarketJob(), ctx)

    assert report.status == "complete"
    assert report.done == 3
    entries = ctx.catalog.latest(
        source=krx_jobs.KRX_DAILY_MARKET_SOURCE, natural_keys={day.isoformat() for day in SESSIONS}
    )
    assert {key: entry.status for key, entry in entries.items()} == {
        day.isoformat(): EvidenceStatus.SUCCESS for day in SESSIONS
    }


class _HedgeCollector:
    """Scripted stand-in for the hedge-series collector surface."""

    def __init__(self, pages=None):  # type: ignore[no-untyped-def]
        self._pages = {day: [dict(row) for row in rows] for day, rows in dict(pages or {}).items()}
        self.fetch_calls: list = []

    def fetch_hedge_records(self, session, *, etf_tickers, index_name, index_class="KOSDAQ"):  # type: ignore[no-untyped-def]
        self.fetch_calls.append((session, tuple(etf_tickers), index_name, index_class))
        return [dict(row) for row in self._pages.get(session, ())]

    def health_check(self) -> None:
        return None


def _hedge_index_record(session):  # type: ignore[no-untyped-def]
    return {
        "_endpoint": "index",
        "IDX_CLSS": "KOSDAQ",
        "IDX_NM": "코스닥 150",
        "BAS_DD": session.strftime("%Y%m%d"),
        "CLSPRC_IDX": "1000.5",
    }


def _hedge_etf_record(session):  # type: ignore[no-untyped-def]
    return {
        "_endpoint": "etf",
        "ISU_CD": "251340",
        "BAS_DD": session.strftime("%Y%m%d"),
        "TDD_CLSPRC": "5000",
    }


def test_hedge_pending_skips_answered_and_honors_start(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import src.data.jobs.krx as krx_jobs

    monkeypatch.setattr(krx_jobs, "_xkrx_sessions", _stub_calendar)
    runtime = _runtime(tmp_path)
    provider = _provider()
    ctx = _ctx(runtime, provider, collector=_HedgeCollector())

    units = krx_jobs.KrxHedgeSeriesJob().pending(ctx)

    assert [unit.natural_key for unit in units] == [day.isoformat() for day in SESSIONS]
    assert all(unit.max_requests == 2 for unit in units)
    ctx.writer.persist_many(
        (
            krx_jobs.krx_hedge_series_scoped_payload(
                records=[_hedge_index_record(SESSIONS[0])],
                session=SESSIONS[0],
                retrieved_at=NOW_AFTER_CLOSE,
            ),
        )
    )
    remaining = krx_jobs.KrxHedgeSeriesJob().pending(ctx)

    assert [unit.natural_key for unit in remaining] == [day.isoformat() for day in SESSIONS[1:]]


def test_hedge_empty_page_is_recorded_as_empty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import src.data.jobs.krx as krx_jobs
    from src.data.receipt_catalog import EvidenceStatus

    monkeypatch.setattr(krx_jobs, "_xkrx_sessions", _stub_calendar)
    runtime = _runtime(tmp_path)
    provider = _provider()
    ctx = _ctx(runtime, provider, collector=_HedgeCollector())

    report, _ = _run(krx_jobs.KrxHedgeSeriesJob(), ctx)

    assert report.status == "complete"
    assert report.done == 3
    entries = ctx.catalog.latest(
        source=krx_jobs.KRX_HEDGE_SERIES_SOURCE, natural_keys={day.isoformat() for day in SESSIONS}
    )
    assert {key: entry.status for key, entry in entries.items()} == {
        day.isoformat(): EvidenceStatus.EMPTY for day in SESSIONS
    }


def test_hedge_job_registered() -> None:
    import src.data.jobs.krx as krx_jobs
    from src.core.pit import PITDataError

    assert isinstance(krx_jobs.resolve_krx_job("krx_hedge_series"), krx_jobs.KrxHedgeSeriesJob)
    with pytest.raises(PITDataError, match="unknown KRX job"):
        krx_jobs.resolve_krx_job("krx_nope")


def test_hedge_pending_empty_before_collection_start(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import src.data.jobs.krx as krx_jobs

    monkeypatch.setattr(krx_jobs, "_xkrx_sessions", _stub_calendar)
    runtime = _runtime(tmp_path)
    provider = _provider()

    early = datetime(2017, 1, 1, 9, 1, tzinfo=UTC)
    assert krx_jobs.KrxHedgeSeriesJob().pending(_ctx(runtime, provider, now=lambda: early)) == ()


def test_hedge_fetch_persists_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import src.data.jobs.krx as krx_jobs
    from src.data.jobs.runner import JobUnit
    from src.data.receipt_catalog import EvidenceStatus

    monkeypatch.setattr(krx_jobs, "_xkrx_sessions", _stub_calendar)
    runtime = _runtime(tmp_path)
    provider = _provider()
    pages = {
        day: [_hedge_index_record(day), _hedge_etf_record(day)] for day in SESSIONS
    }
    collector = _HedgeCollector(pages=pages)
    ctx = _ctx(runtime, provider, collector=collector)

    report, _ = _run(krx_jobs.KrxHedgeSeriesJob(), ctx)

    assert report.status == "complete"
    assert report.done == 3
    assert collector.fetch_calls[0] == (SESSIONS[0], ("251340",), "코스닥 150", "KOSDAQ")
    entries = ctx.catalog.latest(
        source=krx_jobs.KRX_HEDGE_SERIES_SOURCE, natural_keys={day.isoformat() for day in SESSIONS}
    )
    assert {key: entry.status for key, entry in entries.items()} == {
        day.isoformat(): EvidenceStatus.SUCCESS for day in SESSIONS
    }
    payloads = krx_jobs.KrxHedgeSeriesJob().fetch(
        ctx,
        [
            JobUnit(
                source=krx_jobs.KRX_HEDGE_SERIES_SOURCE,
                natural_key=SESSIONS[0].isoformat(),
                payload={"session": SESSIONS[0].isoformat()},
                max_requests=2,
            )
        ],
    )
    assert [payload.source_label for payload in payloads] == [f"krx:hedge-series:{SESSIONS[0].isoformat()}"]


def _trend_runtime(tmp_path: Path):  # type: ignore[no-untyped-def]
    return _runtime(tmp_path)


def test_trend_series_payload_labels_and_source() -> None:
    import src.data.jobs.krx as krx_jobs

    payload = krx_jobs.krx_trend_series_scoped_payload(
        records=[_hedge_index_record(SESSIONS[0])],
        session=SESSIONS[0],
        retrieved_at=NOW_AFTER_CLOSE,
    )

    assert payload.source == krx_jobs.KRX_TREND_SERIES_SOURCE
    assert payload.source_label == f"krx:trend-series:{SESSIONS[0].isoformat()}"


def test_trend_job_registered_and_roundtrip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import src.data.jobs.krx as krx_jobs
    from src.data.jobs.runner import JobUnit

    monkeypatch.setattr(krx_jobs, "_xkrx_sessions", _stub_calendar)
    assert isinstance(krx_jobs.resolve_krx_job("krx_trend_series"), krx_jobs.KrxTrendSeriesJob)
    runtime = _trend_runtime(tmp_path)
    provider = _provider()

    pages = {
        day: [
            _hedge_index_record(day) | {"IDX_CLSS": "KOSPI", "IDX_NM": "코스피 200"},
            _hedge_etf_record(day) | {"ISU_CD": "114800"},
        ]
        for day in SESSIONS
    }
    collector = _HedgeCollector(pages=pages)
    ctx = _ctx(runtime, provider, collector=collector)
    units = krx_jobs.KrxTrendSeriesJob().pending(ctx)
    assert [unit.natural_key for unit in units] == [day.isoformat() for day in SESSIONS]
    payloads = krx_jobs.KrxTrendSeriesJob().fetch(
        ctx,
        [
            JobUnit(
                source=krx_jobs.KRX_TREND_SERIES_SOURCE,
                natural_key=SESSIONS[0].isoformat(),
                payload={"session": SESSIONS[0].isoformat()},
                max_requests=2,
            )
        ],
    )
    assert [payload.source_label for payload in payloads] == [f"krx:trend-series:{SESSIONS[0].isoformat()}"]
    assert collector.fetch_calls[0][2] == "코스피 200"
    assert collector.fetch_calls[0][3] == "KOSPI"
    assert collector.fetch_calls[0][1] == ("114800",)
    ctx.writer.persist_many(payloads)
    assert [unit.natural_key for unit in krx_jobs.KrxTrendSeriesJob().pending(ctx)] == [
        day.isoformat() for day in SESSIONS[1:]
    ]
    assert len(krx_jobs.KrxHedgeSeriesJob().pending(ctx)) == len(SESSIONS)
