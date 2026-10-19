"""Execution CLI entry points: auction-open guards and wiring."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pytest

from src.core.time import KRX_TZ


@pytest.fixture(autouse=True)
def fixed_today(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.execution.cli._today", lambda: date(2026, 10, 2))


def test_auction_open_cli_early_exits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import json as _json
    from datetime import datetime as _datetime

    import src.execution.cli as cli

    code = cli.main(["auction-open", "--decisions-dir", str(tmp_path / "dec"), "--shadow-dir", str(tmp_path / "sh"), "--out", str(tmp_path / "out")])
    assert code == 0
    assert _json.loads(capsys.readouterr().out.strip().splitlines()[-1])["symbols"] == 0

    from src.research.operate import SealedDecision

    sunday = date(2026, 10, 4)
    friday = date(2026, 10, 2)
    chain = (SealedDecision(
        data_session=friday, execution_session=sunday,
        sealed_at=_datetime(2026, 10, 2, 20, 0, tzinfo=KRX_TZ), spec_hash="s", models_id="m",
        inputs={}, targets={}, reference={}, overlay={}, state_digest="d", prev_hash="", hash="",
    ),)
    monkeypatch.setattr("src.research.operate.read_chain", lambda path: chain)
    code = cli.main(["auction-open", "--decisions-dir", str(tmp_path / "dec"), "--shadow-dir", str(tmp_path / "sh"), "--out", str(tmp_path / "out")])
    assert code == 0
    assert _json.loads(capsys.readouterr().out.strip().splitlines()[-1])["symbols"] == 0


def test_session_open_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.execution.cli import _session_open

    friday = date(2026, 10, 2)
    instant = _session_open(friday)
    assert instant is not None
    assert (instant.hour, instant.minute) == (9, 0)
    assert _session_open(date(2026, 10, 4)) is None
    monkeypatch.setattr("src.core.krx_calendar.xkrx_calendar_through", lambda day: (_ for _ in ()).throw(ValueError("no calendar")))
    assert _session_open(friday) is None


def test_net_holdings_from_fills(tmp_path: Path) -> None:
    import polars as pl

    from src.execution.cli import _net_holdings

    assert _net_holdings(tmp_path) == ()
    root = tmp_path / "official"
    root.mkdir(parents=True)
    pl.DataFrame(
        {"session": [], "decision_session": [], "instrument_id": [], "side": [], "quantity": [], "price": []},
        schema={"session": pl.Date, "decision_session": pl.Date, "instrument_id": pl.String, "side": pl.String, "quantity": pl.Int64, "price": pl.Int64},
    ).write_parquet(root / "fills.parquet")
    assert _net_holdings(tmp_path) == ()
    day = date(2026, 10, 2)
    pl.DataFrame(
        {"session": [day, day, day], "decision_session": [day, day, day],
         "instrument_id": ["KRX:005930", "KRX:005930", "KRX:000660"], "side": ["buy", "sell", "buy"],
         "quantity": [10, 10, 5], "price": [10000, 10000, 10000]},
        schema={"session": pl.Date, "decision_session": pl.Date, "instrument_id": pl.String, "side": pl.String, "quantity": pl.Int64, "price": pl.Int64},
    ).write_parquet(root / "fills.parquet")
    assert _net_holdings(tmp_path) == ("KRX:000660",)


def test_auction_open_live_wiring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import json as _json
    from datetime import datetime as _datetime

    import src.execution.cli as cli
    from src.execution.auction_capture import AuctionRow
    from src.research.operate import SealedDecision

    friday = date(2026, 10, 2)
    thursday = date(2026, 10, 1)
    chain = (SealedDecision(
        data_session=thursday, execution_session=friday,
        sealed_at=_datetime(2026, 10, 1, 20, 0, tzinfo=KRX_TZ), spec_hash="s", models_id="m",
        inputs={}, targets={"KRX:005930": 0.5}, reference={}, overlay={}, state_digest="d", prev_hash="", hash="",
    ),)
    monkeypatch.setattr("src.research.operate.read_chain", lambda path: chain)
    monkeypatch.setattr("src.integrations.kis.build_pool_client", lambda policy, admission_class=None: object())
    open_at = _datetime(2026, 10, 2, 9, 0, tzinfo=KRX_TZ)
    rows = (
        AuctionRow(friday, "005930", "expected", open_at, open_at, "{}", "[]"),
        AuctionRow(friday, "005930", "open", open_at, open_at, "{}", "{}"),
    )
    from src.execution.auction_capture import CaptureBatch, CaptureResult

    def fake_capture(client: object, symbols: object, **kwargs: object) -> CaptureResult:
        on_batch = kwargs.get("on_batch")
        from src.execution.auction_capture import AuctionCapturePolicy

        assert isinstance(kwargs.get("policy"), AuctionCapturePolicy)
        result = CaptureResult(rows=rows, failed={}, slots_expected=1, slots_captured=1)
        if on_batch is not None:
            assert callable(on_batch)
            on_batch(CaptureBatch(kind="expected", scheduled_at=open_at, rows=(rows[0],)))  # type: ignore[operator]
            import polars as pl

            path = tmp_path / "out" / "2026-10" / "2026-10-02.parquet"
            assert pl.read_parquet(path).height == 1
            on_batch(CaptureBatch(kind="open", scheduled_at=open_at, rows=(rows[1],)))  # type: ignore[operator]
            assert pl.read_parquet(path).height == 2
        return result

    monkeypatch.setattr("src.execution.auction_capture.capture_open_auction", fake_capture)
    code = cli.main(["auction-open", "--decisions-dir", str(tmp_path / "dec"), "--shadow-dir", str(tmp_path / "sh"), "--out", str(tmp_path / "out")])
    assert code == 0
    payload = _json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload == {"expected_rows": 1, "open_resolved": 1, "symbols": 1, "unresolved": [],
                       "failed": {}, "slots_expected": 1, "slots_captured": 1, "persist_error": None}
    assert (tmp_path / "out" / "2026-10" / "2026-10-02.parquet").is_file()


def test_auction_open_bad_limit_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import json as _json
    from datetime import datetime as _datetime

    import src.execution.cli as cli
    from src.research.operate import SealedDecision

    friday = date(2026, 10, 2)
    chain = (SealedDecision(
        data_session=date(2026, 10, 1), execution_session=friday,
        sealed_at=_datetime(2026, 10, 1, 20, 0, tzinfo=KRX_TZ), spec_hash="s", models_id="m",
        inputs={}, targets={}, reference={}, overlay={}, state_digest="d", prev_hash="", hash="",
    ),)
    monkeypatch.setattr("src.research.operate.read_chain", lambda path: chain)
    code = cli.main(["auction-open", "--decisions-dir", str(tmp_path / "dec"), "--shadow-dir", str(tmp_path / "sh"), "--out", str(tmp_path / "out"), "--limit", "0"])
    assert code == 1
    assert "limit" in _json.loads(capsys.readouterr().out.strip().splitlines()[-1])["error"]


def test_session_open_shifted_and_empty_symbols(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import json as _json
    from datetime import datetime as _datetime
    from types import SimpleNamespace

    import src.core.krx_calendar as _calendar
    import src.execution.cli as cli
    from src.execution.cli import _session_open
    from src.research.operate import SealedDecision

    friday = date(2026, 10, 2)
    real_calendar = _calendar.xkrx_calendar_through
    shifted = SimpleNamespace(sessions=(_datetime(2026, 10, 2, 10, 0, tzinfo=KRX_TZ),))
    monkeypatch.setattr("src.core.krx_calendar.xkrx_calendar_through", lambda day: shifted)
    assert _session_open(friday) is None
    monkeypatch.setattr("src.core.krx_calendar.xkrx_calendar_through", real_calendar)
    chain = (SealedDecision(
        data_session=date(2026, 10, 1), execution_session=friday,
        sealed_at=_datetime(2026, 10, 1, 20, 0, tzinfo=KRX_TZ), spec_hash="s", models_id="m",
        inputs={}, targets={}, reference={}, overlay={}, state_digest="d", prev_hash="", hash="",
    ),)
    monkeypatch.setattr("src.research.operate.read_chain", lambda path: chain)
    code = cli.main(["auction-open", "--decisions-dir", str(tmp_path / "dec"), "--shadow-dir", str(tmp_path / "sh"), "--out", str(tmp_path / "out")])
    assert code == 0
    assert _json.loads(capsys.readouterr().out.strip().splitlines()[-1])["symbols"] == 0


def test_today_and_closed_day(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import json
    from datetime import datetime

    import src.execution.cli as cli
    from src.research.operate import SealedDecision

    monkeypatch.undo()
    assert cli._today() == datetime.now(KRX_TZ).date()
    assert cli._now().tzinfo is not None
    monkeypatch.setattr(cli, "_today", lambda: date(2026, 10, 4))
    decision = SealedDecision(
        date(2026, 10, 1), date(2026, 10, 2), datetime(2026, 10, 1, 20, tzinfo=KRX_TZ),
        "s", "m", {}, {}, {}, {}, "d", "", "",
    )
    monkeypatch.setattr("src.research.operate.read_chain", lambda path: (decision,))
    assert cli.main(["auction-open", "--decisions-dir", str(tmp_path), "--shadow-dir", str(tmp_path), "--out", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["symbols"] == 0
@pytest.mark.parametrize("scenario", ["closed", "missing", "late_open", "ok", "unresolved", "error"])
def test_scheduled_capture_records_outcome(tmp_path, monkeypatch, scenario):  # type: ignore[no-untyped-def]
    from datetime import date
    from types import SimpleNamespace

    from src.execution import cli
    from src.ops.run_outcome import load_run_outcomes

    day = date(2026, 10, 8)
    monkeypatch.setattr(cli, "_today", lambda: day)
    monkeypatch.setattr("src.ops.guards.trading_day", lambda _: scenario != "closed")
    monkeypatch.setattr("src.research.operate.read_chain", lambda _: () if scenario == "missing" else (
        SimpleNamespace(execution_session=day),
    ))
    open_at = datetime(2026, 10, 8, 9, 0, tzinfo=KRX_TZ)
    monkeypatch.setattr(cli, "_session_open", lambda _: None if scenario == "late_open" else open_at)
    monkeypatch.setattr(cli, "_now", lambda: open_at)

    def capture(args):  # type: ignore[no-untyped-def]
        if scenario == "error":
            raise ValueError("secret-value")
        base: dict[str, object] = {"symbols": 1, "expected_rows": 1, "open_resolved": 0,
                                   "failed": {}, "slots_expected": 1, "slots_captured": 1, "persist_error": None}
        base["unresolved"] = ["005930"] if scenario == "unresolved" else []
        return base

    monkeypatch.setattr(cli, "_run_auction_open", capture)
    events = tmp_path / "events"
    rc = cli.main(["auction-open", "--decisions-dir", str(tmp_path), "--shadow-dir", str(tmp_path),
                   "--out", str(tmp_path / "out.json"), "--events-root", str(events)])
    expected = {"closed": "SKIPPED", "missing": "NO_DECISION", "late_open": "SKIPPED",
                "ok": "OK", "unresolved": "DEGRADED", "error": "DEGRADED"}[scenario]
    assert load_run_outcomes(events, day) == {"auction_open": expected}
    assert rc == (1 if expected in ("NO_DECISION", "DEGRADED") else 0)
    assert "secret-value" not in (events / "2026-10" / "2026-10-08.jsonl").read_text()


def test_scheduled_capture_waits_for_late_decision(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    from datetime import date, timedelta
    from types import SimpleNamespace

    from src.execution import cli
    from src.ops.run_outcome import load_run_outcomes

    day = date(2026, 10, 8)
    open_at = datetime(2026, 10, 8, 9, 0, tzinfo=KRX_TZ)
    clock = {"now": open_at - timedelta(minutes=20), "polls": 0}

    def chain(_):  # type: ignore[no-untyped-def]
        clock["polls"] += 1
        return (SimpleNamespace(execution_session=day),) if clock["polls"] > 3 else ()

    def sleep(seconds: float) -> None:
        clock["now"] += timedelta(seconds=seconds)

    monkeypatch.setattr(cli, "_today", lambda: day)
    monkeypatch.setattr("src.ops.guards.trading_day", lambda _: True)
    monkeypatch.setattr("src.research.operate.read_chain", chain)
    monkeypatch.setattr(cli, "_session_open", lambda _: open_at)
    monkeypatch.setattr(cli, "_now", lambda: clock["now"])
    monkeypatch.setattr(cli.time, "sleep", sleep)
    monkeypatch.setattr(cli, "_run_auction_open", lambda args: {"unresolved": [], "failed": {},
                                                                "slots_expected": 1, "slots_captured": 1,
                                                                "persist_error": None})
    events = tmp_path / "events"
    rc = cli.main(["auction-open", "--decisions-dir", str(tmp_path), "--shadow-dir", str(tmp_path),
                   "--out", str(tmp_path / "out.json"), "--events-root", str(events)])
    assert rc == 0
    assert load_run_outcomes(events, day) == {"auction_open": "OK"}
    assert clock["polls"] == 4

    clock.update(now=open_at - timedelta(minutes=20), polls=-10**6)
    rc = cli.main(["auction-open", "--decisions-dir", str(tmp_path), "--shadow-dir", str(tmp_path),
                   "--out", str(tmp_path / "out.json"), "--events-root", str(events)])
    assert rc == 1
    assert clock["now"] >= open_at - timedelta(minutes=1)


def test_rehearse_cli_success_and_missing_keys(tmp_path, monkeypatch, capsys):  # type: ignore[no-untyped-def]
    import json as _json
    from datetime import datetime as _datetime

    import src.execution.cli as cli
    from src.execution.application.rehearsal import RehearsalReport
    from src.research.operate import SealedDecision

    session = date(2026, 10, 2)
    decision = SealedDecision(
        data_session=date(2026, 10, 1), execution_session=session,
        sealed_at=_datetime(2026, 10, 1, 20, 0, tzinfo=KRX_TZ),
        spec_hash="s", models_id="m", inputs={}, targets={"KRX:005930": 0.5},
        reference={"KRX:005930": 50000.0}, overlay={}, state_digest="d", prev_hash="", hash="h" * 32,
    )
    monkeypatch.setattr("src.research.operate.read_chain", lambda path: (decision,))
    monkeypatch.setenv("KIS_MOCK_APP_KEY", "k")
    monkeypatch.setenv("KIS_MOCK_APP_SECRET", "s")
    monkeypatch.setenv("KIS_MOCK_ACCOUNT_NO", "12345678-01")
    monkeypatch.setenv("KIS_ENV", "vps")
    fake = RehearsalReport(
        session=session, intents=1, submitted=1, rejected=0,
        duplicates_blocked=0, reconciled=False, position_mismatches=(),
    )
    monkeypatch.setattr("src.execution.application.rehearsal.rehearse_open", lambda *a, **k: fake)
    store_path = tmp_path / "state.db"
    evidence_path = tmp_path / "evidence.json"
    code = cli.main([
        "rehearse", "--decisions-dir", str(tmp_path), "--store-path", str(store_path),
        "--evidence-path", str(evidence_path), "--nav-krw", "10000000",
        "--session", session.isoformat(),
    ])
    assert code == 0
    payload = _json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["session"] == session.isoformat()
    assert evidence_path.is_file()
    monkeypatch.delenv("KIS_MOCK_APP_KEY", raising=False)
    code = cli.main([
        "rehearse", "--decisions-dir", str(tmp_path), "--store-path", str(store_path),
        "--evidence-path", str(evidence_path), "--nav-krw", "10000000",
        "--session", session.isoformat(),
    ])
    assert code == 1
    monkeypatch.setenv("KIS_MOCK_APP_KEY", "k")
    monkeypatch.setattr("src.research.operate.read_chain", lambda path: ())
    code = cli.main([
        "rehearse", "--decisions-dir", str(tmp_path), "--store-path", str(store_path),
        "--evidence-path", str(evidence_path), "--nav-krw", "10000000",
        "--session", session.isoformat(),
    ])
    assert code == 1
    capsys.readouterr()
    monkeypatch.setattr("src.research.operate.read_chain", lambda path: (decision,))
    code = cli.main([
        "rehearse", "--decisions-dir", str(tmp_path), "--store-path", str(store_path),
        "--evidence-path", str(evidence_path), "--nav-krw", "10000000",
        "--session", "2026-10-05",
    ])
    assert code == 1
    capsys.readouterr()
    monkeypatch.setattr(cli, "_today", lambda: session)
    code = cli.main([
        "rehearse", "--decisions-dir", str(tmp_path), "--store-path", str(store_path),
        "--evidence-path", str(evidence_path), "--nav-krw", "10000000",
    ])
    assert code == 0
    capsys.readouterr()
    second_session = date(2026, 10, 5)
    decision2 = SealedDecision(
        data_session=date(2026, 10, 2), execution_session=second_session,
        sealed_at=_datetime(2026, 10, 2, 20, 0, tzinfo=KRX_TZ),
        spec_hash="s", models_id="m", inputs={}, targets={"KRX:005930": 0.5},
        reference={"KRX:005930": 50000.0}, overlay={}, state_digest="d", prev_hash="", hash="h" * 32,
    )
    fake2 = RehearsalReport(
        session=second_session, intents=1, submitted=1, rejected=0,
        duplicates_blocked=0, reconciled=False, position_mismatches=(),
    )
    monkeypatch.setattr(
        "src.research.operate.read_chain", lambda path: (decision, decision2)
    )
    monkeypatch.setattr("src.execution.application.rehearsal.rehearse_open", lambda *a, **k: fake2)
    code = cli.main([
        "rehearse", "--decisions-dir", str(tmp_path), "--store-path", str(store_path),
        "--evidence-path", str(evidence_path), "--nav-krw", "10000000",
        "--session", second_session.isoformat(),
    ])
    assert code == 0
    capsys.readouterr()
    evidence_path.write_text("corrupt{", encoding="utf-8")
    code = cli.main([
        "rehearse", "--decisions-dir", str(tmp_path), "--store-path", str(store_path),
        "--evidence-path", str(evidence_path), "--nav-krw", "10000000",
        "--session", second_session.isoformat(),
    ])
    assert code == 0


def _full_result(**overrides):  # type: ignore[no-untyped-def]
    base: dict[str, object] = {"symbols": 1, "expected_rows": 1, "open_resolved": 1, "unresolved": [],
                               "failed": {}, "slots_expected": 1, "slots_captured": 1, "persist_error": None}
    base.update(overrides)
    return base


def test_early_returns_carry_full_shape(tmp_path, monkeypatch, capsys):  # type: ignore[no-untyped-def]
    import json as _json

    import src.execution.cli as cli

    monkeypatch.setattr("src.research.operate.read_chain", lambda path: ())
    code = cli.main(["auction-open", "--decisions-dir", str(tmp_path), "--shadow-dir", str(tmp_path), "--out", str(tmp_path)])
    assert code == 0
    payload = _json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload == {"symbols": 0, "expected_rows": 0, "open_resolved": 0, "unresolved": [],
                       "failed": {}, "slots_expected": 0, "slots_captured": 0, "persist_error": None}


def test_killed_after_first_slot_keeps_rows(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    from datetime import datetime as _datetime

    import src.execution.cli as cli
    from src.execution.auction_capture import AuctionRow, CaptureBatch
    from src.ops.run_outcome import load_run_outcomes
    from src.research.operate import SealedDecision

    friday = date(2026, 10, 2)
    chain = (SealedDecision(data_session=date(2026, 10, 1), execution_session=friday,
             sealed_at=_datetime(2026, 10, 1, 20, 0, tzinfo=KRX_TZ), spec_hash="s", models_id="m",
             inputs={}, targets={"KRX:005930": 0.5}, reference={}, overlay={}, state_digest="d", prev_hash="", hash=""),)
    monkeypatch.setattr("src.research.operate.read_chain", lambda path: chain)
    monkeypatch.setattr("src.integrations.kis.build_pool_client", lambda policy, admission_class=None: object())
    open_at = _datetime(2026, 10, 2, 9, 0, tzinfo=KRX_TZ)
    row = AuctionRow(friday, "005930", "expected", open_at, open_at, "{}", "[]")

    def fake_capture(client, symbols, **kwargs):  # type: ignore[no-untyped-def]
        kwargs["on_batch"](CaptureBatch(kind="expected", scheduled_at=open_at, rows=(row,)))
        raise RuntimeError("boom")

    monkeypatch.setattr("src.execution.auction_capture.capture_open_auction", fake_capture)
    events = tmp_path / "events"
    rc = cli.main(["auction-open", "--decisions-dir", str(tmp_path), "--shadow-dir", str(tmp_path),
                   "--out", str(tmp_path / "out"), "--events-root", str(events)])
    assert rc == 1
    assert (tmp_path / "out" / "2026-10" / "2026-10-02.parquet").is_file()
    assert load_run_outcomes(events, friday) == {"auction_open": "DEGRADED"}


def test_fallback_when_capture_ignores_on_batch(tmp_path, monkeypatch, capsys):  # type: ignore[no-untyped-def]
    import json as _json
    from datetime import datetime as _datetime

    import src.execution.cli as cli
    from src.execution.auction_capture import AuctionRow, CaptureResult
    from src.research.operate import SealedDecision

    friday = date(2026, 10, 2)
    chain = (SealedDecision(data_session=date(2026, 10, 1), execution_session=friday,
             sealed_at=_datetime(2026, 10, 1, 20, 0, tzinfo=KRX_TZ), spec_hash="s", models_id="m",
             inputs={}, targets={"KRX:005930": 0.5}, reference={}, overlay={}, state_digest="d", prev_hash="", hash=""),)
    monkeypatch.setattr("src.research.operate.read_chain", lambda path: chain)
    monkeypatch.setattr("src.integrations.kis.build_pool_client", lambda policy, admission_class=None: object())
    open_at = _datetime(2026, 10, 2, 9, 0, tzinfo=KRX_TZ)
    rows = (AuctionRow(friday, "005930", "expected", open_at, open_at, "{}", "[]"),)
    monkeypatch.setattr("src.execution.auction_capture.capture_open_auction",
                        lambda client, symbols, **kwargs: CaptureResult(rows=rows, failed={}, slots_expected=1, slots_captured=1))
    assert cli.main(["auction-open", "--decisions-dir", str(tmp_path), "--shadow-dir", str(tmp_path), "--out", str(tmp_path / "out")]) == 0
    payload = _json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["failed"] == {}
    assert payload["persist_error"] is None
    assert (tmp_path / "out" / "2026-10" / "2026-10-02.parquet").is_file()


def test_persist_failure_isolated_and_reconciled(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    from datetime import datetime as _datetime

    import src.execution.auction_capture as ac
    import src.execution.cli as cli
    from src.execution.auction_capture import AuctionRow, CaptureBatch, CaptureResult
    from src.research.operate import SealedDecision

    friday = date(2026, 10, 2)
    chain = (SealedDecision(data_session=date(2026, 10, 1), execution_session=friday,
             sealed_at=_datetime(2026, 10, 1, 20, 0, tzinfo=KRX_TZ), spec_hash="s", models_id="m",
             inputs={}, targets={"KRX:005930": 0.5}, reference={}, overlay={}, state_digest="d", prev_hash="", hash=""),)
    monkeypatch.setattr("src.research.operate.read_chain", lambda path: chain)
    monkeypatch.setattr("src.integrations.kis.build_pool_client", lambda policy, admission_class=None: object())
    open_at = _datetime(2026, 10, 2, 9, 0, tzinfo=KRX_TZ)
    rows = (AuctionRow(friday, "005930", "expected", open_at, open_at, "{}", "[]"),
            AuctionRow(friday, "005930", "open", open_at, open_at, "{}", "{}"))

    def fake_capture(client, symbols, **kwargs):  # type: ignore[no-untyped-def]
        kwargs["on_batch"](CaptureBatch(kind="expected", scheduled_at=open_at, rows=(rows[0],)))
        kwargs["on_batch"](CaptureBatch(kind="open", scheduled_at=open_at, rows=(rows[1],)))
        return CaptureResult(rows=rows, failed={}, slots_expected=1, slots_captured=1)

    monkeypatch.setattr("src.execution.auction_capture.capture_open_auction", fake_capture)
    real_write = ac.write_auction_rows
    calls = {"n": 0}

    def flaky(root, r):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("disk")
        return real_write(root, r)

    monkeypatch.setattr("src.execution.auction_capture.write_auction_rows", flaky)
    events = tmp_path / "events"
    rc = cli.main(["auction-open", "--decisions-dir", str(tmp_path), "--shadow-dir", str(tmp_path),
                   "--out", str(tmp_path / "out"), "--events-root", str(events)])
    assert rc == 1
    import polars as pl
    assert pl.read_parquet(tmp_path / "out" / "2026-10" / "2026-10-02.parquet").height == 2

    monkeypatch.setattr("src.execution.auction_capture.write_auction_rows", lambda root, r: (_ for _ in ()).throw(OSError("disk")))
    rc = cli.main(["auction-open", "--decisions-dir", str(tmp_path), "--shadow-dir", str(tmp_path),
                   "--out", str(tmp_path / "out2"), "--events-root", str(events)])
    assert rc == 1


@pytest.mark.parametrize(("metrics", "reason"), [
    (_full_result(slots_captured=2, slots_expected=3), "slots_incomplete"),
    (_full_result(failed={"005930": "RuntimeError"}), "symbol_failed"),
    (_full_result(persist_error="OSError"), "persist_failed"),
    (_full_result(unresolved=["005930"]), "unresolved_auction"),
    (_full_result(slots_captured=0, slots_expected=2, failed={"005930": "RuntimeError"}, unresolved=["005930"]),
     "slots_incomplete,symbol_failed,unresolved_auction"),
])
def test_scheduled_degraded_reasons(tmp_path, monkeypatch, metrics, reason):  # type: ignore[no-untyped-def]
    import json as _json
    from datetime import datetime

    from src.execution import cli

    day = date(2026, 10, 8)
    open_at = datetime(2026, 10, 8, 9, 0, tzinfo=KRX_TZ)
    monkeypatch.setattr(cli, "_today", lambda: day)
    monkeypatch.setattr("src.ops.guards.trading_day", lambda _: True)
    monkeypatch.setattr(cli, "_session_open", lambda _: open_at)
    monkeypatch.setattr(cli, "_await_execution_decision", lambda *a: True)
    monkeypatch.setattr(cli, "_run_auction_open", lambda args: dict(metrics))
    events = tmp_path / "events"
    rc = cli.main(["auction-open", "--decisions-dir", str(tmp_path), "--shadow-dir", str(tmp_path),
                   "--out", str(tmp_path / "o"), "--events-root", str(events)])
    assert rc == 1
    line = (events / "2026-10" / "2026-10-08.jsonl").read_text().strip().splitlines()[-1]
    assert _json.loads(line)["reason"] == reason


def test_clean_run_stays_ok(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    from datetime import datetime

    from src.execution import cli
    from src.ops.run_outcome import load_run_outcomes

    day = date(2026, 10, 8)
    open_at = datetime(2026, 10, 8, 9, 0, tzinfo=KRX_TZ)
    monkeypatch.setattr(cli, "_today", lambda: day)
    monkeypatch.setattr("src.ops.guards.trading_day", lambda _: True)
    monkeypatch.setattr(cli, "_session_open", lambda _: open_at)
    monkeypatch.setattr(cli, "_await_execution_decision", lambda *a: True)
    emitted: list[dict] = []
    monkeypatch.setattr(cli, "_run_auction_open", lambda args: _full_result())
    monkeypatch.setattr(cli, "_emit", emitted.append)
    events = tmp_path / "events"
    assert cli.main(["auction-open", "--decisions-dir", str(tmp_path), "--shadow-dir", str(tmp_path),
                     "--out", str(tmp_path / "o"), "--events-root", str(events)]) == 0
    assert load_run_outcomes(events, day) == {"auction_open": "OK"}
    assert emitted
    assert emitted[0]["slots_captured"] == 1


def test_slot_margin_flag(tmp_path, monkeypatch, capsys):  # type: ignore[no-untyped-def]
    import json as _json
    from datetime import datetime as _datetime

    import src.execution.cli as cli
    from src.execution.auction_capture import CaptureResult
    from src.research.operate import SealedDecision

    friday = date(2026, 10, 2)
    chain = (SealedDecision(data_session=date(2026, 10, 1), execution_session=friday,
             sealed_at=_datetime(2026, 10, 1, 20, 0, tzinfo=KRX_TZ), spec_hash="s", models_id="m",
             inputs={}, targets={"KRX:005930": 0.5}, reference={}, overlay={}, state_digest="d", prev_hash="", hash=""),)
    monkeypatch.setattr("src.research.operate.read_chain", lambda path: chain)
    monkeypatch.setattr("src.integrations.kis.build_pool_client", lambda policy, admission_class=None: policy)
    seen: dict[str, object] = {}

    def fake_capture(client, symbols, **kwargs):  # type: ignore[no-untyped-def]
        seen.update({"margin": kwargs["policy"].slot_margin_s, "estimate": kwargs["policy"].call_estimate_s})
        assert kwargs["policy"].call_estimate_s == client.min_interval_seconds
        return CaptureResult(rows=(), failed={}, slots_expected=0, slots_captured=0)

    monkeypatch.setattr("src.execution.auction_capture.capture_open_auction", fake_capture)
    assert cli.main(["auction-open", "--decisions-dir", str(tmp_path), "--shadow-dir", str(tmp_path),
                     "--out", str(tmp_path / "o"), "--slot-margin-s", "2.5"]) == 0
    assert seen["margin"] == 2.5
    capsys.readouterr()
    assert cli.main(["auction-open", "--decisions-dir", str(tmp_path), "--shadow-dir", str(tmp_path),
                     "--out", str(tmp_path / "o"), "--slot-margin-s", "-1"]) == 1
    assert "slot_margin_s" in _json.loads(capsys.readouterr().out)["error"]


def test_persister_keeps_first_failure_and_empty_finalize(tmp_path):  # type: ignore[no-untyped-def]
    from src.execution.cli import _make_persister
    from src.execution.auction_capture import AuctionRow, CaptureBatch
    from datetime import datetime

    day = date(2026, 10, 2)
    open_at = datetime(2026, 10, 2, 9, 0, tzinfo=KRX_TZ)
    row = AuctionRow(day, "005930", "expected", open_at, open_at, "{}", "[]")
    import unittest.mock as mock

    import src.execution.auction_capture as ac

    with mock.patch.object(ac, "write_auction_rows", side_effect=OSError("disk")):
        persist = _make_persister(tmp_path / "out")
        persist.on_batch(CaptureBatch(kind="expected", scheduled_at=open_at, rows=(row,)))
        assert persist.persist_error == "OSError"
    with mock.patch.object(ac, "write_auction_rows", side_effect=ValueError("other")):
        persist.finalize([row])
        assert persist.persist_error == "OSError"
    with mock.patch.object(ac, "write_auction_rows", side_effect=ValueError("other")):
        persist3 = _make_persister(tmp_path / "out3")
        persist3.finalize([row])
        assert persist3.persist_error == "ValueError"
    persist2 = _make_persister(tmp_path / "out2")
    persist2.finalize([])
    assert persist2.persist_error is None


def test_scheduled_capture_rejects_invalid_slots(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    from datetime import datetime

    from src.execution import cli
    from src.ops.run_outcome import load_run_outcomes

    day = date(2026, 10, 8)
    open_at = datetime(2026, 10, 8, 9, 0, tzinfo=KRX_TZ)
    monkeypatch.setattr(cli, "_today", lambda: day)
    monkeypatch.setattr("src.ops.guards.trading_day", lambda _: True)
    monkeypatch.setattr(cli, "_session_open", lambda _: open_at)
    monkeypatch.setattr(cli, "_await_execution_decision", lambda *a: True)
    monkeypatch.setattr(cli, "_run_auction_open", lambda args: _full_result(slots_expected=None, slots_captured=None))
    events = tmp_path / "events"
    assert cli.main(["auction-open", "--decisions-dir", str(tmp_path), "--shadow-dir", str(tmp_path),
                     "--out", str(tmp_path / "o"), "--events-root", str(events)]) == 1
    assert load_run_outcomes(events, day) == {"auction_open": "DEGRADED"}


@pytest.mark.parametrize("margin", ["-1", "nan", "inf"])
def test_invalid_margin_rejected_before_empty_chain(tmp_path, monkeypatch, capsys, margin):  # type: ignore[no-untyped-def]
    import json

    from src.execution import cli

    monkeypatch.setattr("src.research.operate.read_chain", lambda _: ())
    assert cli.main([
        "auction-open", "--decisions-dir", str(tmp_path), "--shadow-dir", str(tmp_path),
        "--out", str(tmp_path / "out"), "--slot-margin-s", margin,
    ]) == 1
    assert "slot_margin_s" in json.loads(capsys.readouterr().out)["error"]
