"""Open-auction capture scheduling, payload preservation and symbol selection."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from src.core.time import KRX_TZ
from src.execution.auction_capture import (
    OPEN_SNAPSHOT_OFFSETS_MIN,
    SLOT_DEADLINE_ERROR,
    AuctionCapturePolicy,
    AuctionRow,
    capture_open_auction,
    capture_symbols,
    write_auction_rows,
)

_POLICY = AuctionCapturePolicy(call_estimate_s=1.0, slot_margin_s=0.0)


def _capture(client: Any, symbols: list[str], day: date, clock: Any, **kwargs: Any) -> Any:
    params: dict[str, Any] = {
        "session": day, "open_at": _open_at(day), "now": clock.now, "sleep": clock.sleep,
        "policy": _POLICY,
    }
    params.update(kwargs)
    return capture_open_auction(client, symbols, **params)


def _sealed(data: date, execution: date, targets: dict[str, float], sealed_at: datetime) -> Any:
    from src.research.operate import SealedDecision

    return SealedDecision(
        data_session=data, execution_session=execution, sealed_at=sealed_at, spec_hash="s",
        models_id="m", inputs={}, targets=dict(targets), reference={}, overlay={},
        state_digest="d", prev_hash="", hash="",
    )


def _morning(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, 8, 0, tzinfo=KRX_TZ)


@dataclass
class _Clock:
    current: datetime
    slept: list[float] = field(default_factory=list)

    def now(self) -> datetime:
        return self.current

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.current += timedelta(seconds=seconds)


@dataclass
class _StubClient:
    asking: dict[str, list[dict[str, Any]]] | None = None
    prices: dict[str, list[dict[str, Any]]] | None = None
    calls: list[tuple[str, str]] = field(default_factory=list)

    def inquire_asking_price_exp_ccn(self, symbol: str) -> dict[str, Any]:
        self.calls.append(("expected", symbol))
        queue = (self.asking or {}).get(symbol, [])
        if queue:
            return queue.pop(0)
        return {"output1": {"antc_cnpr": "10000"}, "output2": []}

    def inquire_price(self, symbol: str) -> dict[str, Any]:
        self.calls.append(("open", symbol))
        queue = (self.prices or {}).get(symbol, [])
        if queue:
            return queue.pop(0)
        return {"stck_oprc": "10000"}


def _open_at(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, 9, 0, tzinfo=KRX_TZ)


def test_rounds_follow_schedule() -> None:
    day = date(2026, 10, 5)
    clock = _Clock(_open_at(day) - timedelta(minutes=25))
    client = _StubClient()
    result = _capture(client, ["005930", "000660"], day, clock)
    rows = result.rows
    expected = [row for row in rows if row.kind == "expected"]
    assert len(expected) == 2 * len(OPEN_SNAPSHOT_OFFSETS_MIN)
    for offset in OPEN_SNAPSHOT_OFFSETS_MIN:
        batch = [row for row in expected if row.scheduled_at == _open_at(day) + timedelta(minutes=offset)]
        assert sorted(row.symbol for row in batch) == ["000660", "005930"]
    assert result.slots_expected == len(OPEN_SNAPSHOT_OFFSETS_MIN)
    assert result.slots_captured == len(OPEN_SNAPSHOT_OFFSETS_MIN)
    assert result.failed == {}


def test_late_start_skips_past_rounds() -> None:
    day = date(2026, 10, 5)
    clock = _Clock(_open_at(day) - timedelta(minutes=3))
    client = _StubClient()
    result = _capture(client, ["005930"], day, clock)
    rows = result.rows
    offsets = sorted(
        (row.scheduled_at - _open_at(day)).total_seconds() / 60.0
        for row in rows if row.kind == "expected"
    )
    assert offsets == [-2.0, -1.0]
    assert result.slots_expected == 2


def test_open_print_resolves_or_times_out() -> None:
    day = date(2026, 10, 5)
    clock = _Clock(_open_at(day) - timedelta(minutes=25))
    client = _StubClient(
        prices={
            "005930": [{"stck_oprc": "0"}, {"stck_oprc": "0"}, {"stck_oprc": "12350", "extra": 1}],
            "000660": [{"stck_oprc": "0"}] * 50,
        }
    )
    result = _capture(client, ["005930", "000660"], day, clock)
    rows = result.rows
    opened = {row.symbol: row for row in rows if row.kind == "open"}
    assert set(opened) == {"005930"}
    assert json.loads(opened["005930"].output1)["stck_oprc"] == "12350"
    assert clock.now() >= _open_at(day) + timedelta(seconds=180)


def test_raw_payload_preserved() -> None:
    day = date(2026, 10, 5)
    clock = _Clock(_open_at(day) - timedelta(minutes=25))
    block = {"stck_oprc": "9999", "mystery": {"nested": [1, 2, 3]}}
    client = _StubClient(
        asking={"005930": [{"output1": {"antc_cnpr": "9999", "unknown": "x"}, "output2": [{"p": 1}]}]},
        prices={"005930": [dict(block)]},
    )
    result = _capture(client, ["005930"], day, clock)
    rows = result.rows
    by_kind: dict[str, list[AuctionRow]] = {}
    for row in rows:
        by_kind.setdefault(row.kind, []).append(row)
    assert json.loads(by_kind["expected"][0].output1) == {"antc_cnpr": "9999", "unknown": "x"}
    assert json.loads(by_kind["expected"][0].output2) == [{"p": 1}]
    assert json.loads(by_kind["open"][0].output1) == block


def test_symbol_selection_deterministic_and_capped() -> None:
    from datetime import timedelta as _td

    days = [date(2026, 9, 28) + _td(days=i) for i in range(4)]
    old = {f"KRX:{i:06d}": 0.01 for i in range(100)}
    new = {f"KRX:{i:06d}": 0.01 + (i % 7) * 0.001 for i in range(100)}
    chain = (
        _sealed(days[0], days[1], old, _morning(days[1])),
        _sealed(days[1], days[2], new, _morning(days[2])),
    )
    holdings = [f"KRX:{i:06d}" for i in range(100, 130)]
    picked = capture_symbols(chain, holdings, session=days[2], limit=80)
    assert len(picked) == 80
    assert picked == tuple(sorted(picked, key=lambda name: (-abs(new[f"KRX:{name}"] - old[f"KRX:{name}"]), name)))
    again = capture_symbols(chain, holdings, session=days[2], limit=80)
    assert again == picked


def test_request_budget() -> None:
    day = date(2026, 10, 5)
    clock = _Clock(_open_at(day) - timedelta(minutes=25))
    symbols = [f"{i:06d}" for i in range(80)]
    client = _StubClient()
    _capture(client, symbols, day, clock)
    assert len(client.calls) <= 80 * (len(OPEN_SNAPSHOT_OFFSETS_MIN) + 6)


def test_write_auction_rows_appends_atomically(tmp_path: Path) -> None:
    import polars as pl

    day = date(2026, 10, 5)
    first = AuctionRow(day, "005930", "expected", _open_at(day) - timedelta(minutes=1), _open_at(day), "{}", "[]")
    path = write_auction_rows(tmp_path, [first])
    assert path == tmp_path / "2026-10" / "2026-10-05.parquet"
    second = AuctionRow(day, "000660", "open", _open_at(day), _open_at(day), "{}", "{}")
    assert write_auction_rows(tmp_path, [second]) == path
    frame = pl.read_parquet(path)
    assert frame.height == 2
    assert sorted(frame["symbol"].to_list()) == ["000660", "005930"]


def test_capture_symbols_limit_and_fallback() -> None:
    from datetime import timedelta as _td

    import pytest

    days = [date(2026, 9, 28) + _td(days=i) for i in range(4)]
    chain = (_sealed(days[0], days[1], {"KRX:000001": 0.5}, _morning(days[1])),)
    with pytest.raises(ValueError, match="limit"):
        capture_symbols(chain, [], session=days[1], limit=0)
    assert capture_symbols(chain, [], session=days[0], limit=80) == ()
    picked = capture_symbols(chain, ["KRX:000002", "KRX:000001"], session=days[2], limit=80)
    assert picked == ("000001", "000002")


def test_capture_open_auction_guards() -> None:
    import pytest

    from datetime import datetime as _datetime

    day = date(2026, 10, 5)
    clock = _Clock(_open_at(day) - timedelta(minutes=25))
    empty = _capture(_StubClient(), [], day, clock)
    assert empty.rows == ()
    assert empty.slots_expected == 0
    assert empty.slots_captured == 0
    with pytest.raises(ValueError, match="unique"):
        _capture(_StubClient(), ["005930", "005930"], day, clock)
    with pytest.raises(ValueError, match="timezone-aware"):
        capture_open_auction(
            _StubClient(), ["005930"], session=day,
            open_at=_datetime(2026, 10, 5, 9, 0), now=clock.now, sleep=clock.sleep,
            policy=_POLICY,
        )


def test_garbled_open_output_never_resolves() -> None:
    day = date(2026, 10, 5)
    clock = _Clock(_open_at(day) - timedelta(minutes=25))
    client = _StubClient(prices={"005930": [{"stck_oprc": "n/a"}, {"other": 1}] * 10})
    result = _capture(client, ["005930"], day, clock)
    rows = result.rows
    assert [row for row in rows if row.kind == "open"] == []
    assert clock.now() >= _open_at(day) + timedelta(seconds=180)


def test_non_mapping_payloads_rejected() -> None:
    day = date(2026, 10, 5)
    clock = _Clock(_open_at(day) - timedelta(minutes=25))

    class _BadAsk(_StubClient):
        def inquire_asking_price_exp_ccn(self, symbol: str) -> Any:
            self.calls.append(("expected", symbol))
            return ["not", "a", "mapping"]  # type: ignore[return-value]

    result = _capture(_BadAsk(), ["005930"], day, clock)
    assert result.failed == {"005930": "ValueError"}
    assert [row for row in result.rows if row.kind == "expected"] == []

    clock2 = _Clock(_open_at(day) - timedelta(minutes=25))

    class _BadPrint(_StubClient):
        def inquire_price(self, symbol: str) -> Any:
            self.calls.append(("open", symbol))
            return ["not", "a", "mapping"]  # type: ignore[return-value]

    result2 = _capture(_BadPrint(), ["005930"], day, clock2)
    assert result2.failed == {"005930": "ValueError"}
    assert [row for row in result2.rows if row.kind == "open"] == []


def test_deadline_breaks_mid_pass() -> None:
    day = date(2026, 10, 5)

    class _RushingClock(_Clock):
        def now(self) -> datetime:
            self.current += timedelta(seconds=60)
            return self.current

    clock = _RushingClock(_open_at(day) - timedelta(minutes=25))
    client = _StubClient(prices={"a": [{"stck_oprc": "0"}] * 50, "b": [{"stck_oprc": "0"}] * 50})
    result = _capture(client, ["a", "b"], day, clock)
    rows = result.rows
    assert [row for row in rows if row.kind == "open"] == []


def test_write_auction_rows_rejects_bad_input(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(ValueError, match="no auction rows"):
        write_auction_rows(tmp_path, [])
    other = date(2026, 10, 6)
    rows = [
        AuctionRow(date(2026, 10, 5), "005930", "expected", _open_at(date(2026, 10, 5)), _open_at(date(2026, 10, 5)), "{}", "[]"),
        AuctionRow(other, "005930", "expected", _open_at(other), _open_at(other), "{}", "[]"),
    ]
    with pytest.raises(ValueError, match="one session"):
        write_auction_rows(tmp_path, rows)


def test_deadline_breaks_inside_pass() -> None:
    day = date(2026, 10, 5)

    class _ScriptedClock:
        def __init__(self, stamps: list[datetime]) -> None:
            self._stamps = list(stamps)
            self.slept: list[float] = []

        def now(self) -> datetime:
            if len(self._stamps) > 1:
                return self._stamps.pop(0)
            return self._stamps[0]

        def sleep(self, seconds: float) -> None:
            self.slept.append(seconds)

    early = _open_at(day) - timedelta(minutes=25)
    stamps = [early] * 26 + [
        datetime(2026, 10, 5, 9, 0, 0, tzinfo=KRX_TZ),
        datetime(2026, 10, 5, 9, 0, 31, tzinfo=KRX_TZ),
        datetime(2026, 10, 5, 9, 0, 32, tzinfo=KRX_TZ),
        datetime(2026, 10, 5, 9, 3, 0, tzinfo=KRX_TZ),
    ]
    clock = _ScriptedClock(stamps)
    client = _StubClient(prices={"a": [{"stck_oprc": "0"}] * 50, "b": [{"stck_oprc": "0"}] * 50})
    result = _capture(client, ["a", "b"], day, clock)
    rows = result.rows
    assert [row for row in rows if row.kind == "open"] == []
    assert sum(1 for row in rows if row.kind == "expected") == 2 * len(OPEN_SNAPSHOT_OFFSETS_MIN)


def test_slot_batches_arrive_in_order_with_union_rows() -> None:
    day = date(2026, 10, 5)
    clock = _Clock(_open_at(day) - timedelta(minutes=25))
    client = _StubClient()
    log: list[str] = []
    batches: list[Any] = []
    orig_ask = client.inquire_asking_price_exp_ccn
    orig_px = client.inquire_price

    def ask(symbol: str) -> Any:
        log.append(f"call:{clock.now().isoformat()}:{symbol}")
        clock.current += timedelta(seconds=1)
        return orig_ask(symbol)

    def px(symbol: str) -> Any:
        return orig_px(symbol)

    client.inquire_asking_price_exp_ccn = ask  # type: ignore[method-assign]
    client.inquire_price = px  # type: ignore[method-assign]

    def on_batch(batch: Any) -> None:
        batches.append(batch)
        log.append(f"batch:{batch.kind}:{batch.scheduled_at.isoformat()}:{len(batch.rows)}")

    result = _capture(client, ["005930", "000660"], day, clock, on_batch=on_batch)
    assert [b.kind for b in batches] == ["expected"] * 5 + ["open"]
    assert [b.scheduled_at for b in batches[:5]] == sorted(b.scheduled_at for b in batches[:5])
    union = [row for b in batches for row in b.rows]
    assert len(union) == len(result.rows)
    assert {(r.symbol, r.kind, r.scheduled_at) for r in union} == {(r.symbol, r.kind, r.scheduled_at) for r in result.rows}
    assert log[-1].startswith("batch:open")
    for i, batch in enumerate(batches[:5]):
        assert log[3 * i:3 * i + 2] == [
            f"call:{batch.scheduled_at.isoformat()}:005930",
            f"call:{(batch.scheduled_at + timedelta(seconds=1)).isoformat()}:000660",
        ]
        assert log[3 * i + 2].startswith("batch:expected")


def test_open_batch_only_when_resolved() -> None:
    day = date(2026, 10, 5)
    clock = _Clock(_open_at(day) - timedelta(minutes=25))
    client = _StubClient(prices={"a": [{"stck_oprc": "0"}] * 50, "b": [{"stck_oprc": "0"}] * 50})
    seen: list[Any] = []
    result = _capture(client, ["a", "b"], day, clock, on_batch=seen.append)
    assert [b for b in seen if b.kind == "open"] == []
    assert len([b for b in seen if b.kind == "expected"]) == len(OPEN_SNAPSHOT_OFFSETS_MIN)


def test_expected_isolation_and_first_failure_wins() -> None:
    day = date(2026, 10, 5)
    clock = _Clock(_open_at(day) - timedelta(minutes=25))

    class _Flaky(_StubClient):
        def inquire_asking_price_exp_ccn(self, symbol: str) -> Any:
            self.calls.append(("expected", symbol))
            if symbol == "bad":
                raise RuntimeError("boom")
            return {"output1": {}, "output2": []}

    result = _capture(_Flaky(), ["good", "bad"], day, clock)
    assert sum(1 for r in result.rows if r.kind == "expected" and r.symbol == "good") == len(OPEN_SNAPSHOT_OFFSETS_MIN)
    assert result.failed == {"bad": "RuntimeError"}
    assert result.slots_captured == result.slots_expected

    clock2 = _Clock(_open_at(day) - timedelta(minutes=25))

    class _TwoErr(_StubClient):
        def __init__(self) -> None:
            super().__init__()
            self.n = 0

        def inquire_asking_price_exp_ccn(self, symbol: str) -> Any:
            self.calls.append(("expected", symbol))
            if symbol == "flaky":
                self.n += 1
                raise OSError("first") if self.n == 1 else RuntimeError("second")
            return {"output1": {}, "output2": []}

    result2 = _capture(_TwoErr(), ["flaky", "good"], day, clock2)
    assert result2.failed["flaky"] == "OSError"


def test_open_phase_isolation_and_transient_recovery() -> None:
    day = date(2026, 10, 5)
    clock = _Clock(_open_at(day) - timedelta(minutes=25))

    class _OpenFlaky(_StubClient):
        def inquire_price(self, symbol: str) -> Any:
            self.calls.append(("open", symbol))
            if symbol == "bad":
                raise ConnectionError("down")
            return {"stck_oprc": "100"}

    result = _capture(_OpenFlaky(), ["good", "bad"], day, clock)
    opened = {r.symbol for r in result.rows if r.kind == "open"}
    assert opened == {"good"}
    assert result.failed == {"bad": "ConnectionError"}

    clock2 = _Clock(_open_at(day) - timedelta(minutes=25))

    class _Transient(_StubClient):
        def __init__(self) -> None:
            super().__init__()
            self.n = 0

        def inquire_price(self, symbol: str) -> Any:
            self.calls.append(("open", symbol))
            if symbol == "flaky":
                self.n += 1
                if self.n == 1:
                    raise ConnectionError("once")
            return {"stck_oprc": "100"}

    result2 = _capture(_Transient(), ["flaky"], day, clock2)
    assert {r.symbol for r in result2.rows if r.kind == "open"} == {"flaky"}
    assert "flaky" not in result2.failed


def test_base_exception_propagates_with_batches_delivered() -> None:
    import pytest

    day = date(2026, 10, 5)
    clock = _Clock(_open_at(day) - timedelta(minutes=25))

    class _Killer(_StubClient):
        def inquire_asking_price_exp_ccn(self, symbol: str) -> Any:
            self.calls.append(("expected", symbol))
            if len(self.calls) > 1:
                raise KeyboardInterrupt()
            return {"output1": {}, "output2": []}

    seen: list[Any] = []
    with pytest.raises(KeyboardInterrupt):
        _capture(_Killer(), ["a"], day, clock, on_batch=seen.append)
    assert len(seen) == 1
    assert seen[0].rows[0].scheduled_at == _open_at(day) - timedelta(minutes=20)


@dataclass
class _AdvancingClient:
    seconds_per_call: float
    clock: _Clock
    calls: list[tuple[str, str]] = field(default_factory=list)

    def inquire_asking_price_exp_ccn(self, symbol: str) -> dict[str, Any]:
        self.calls.append(("expected", symbol))
        self.clock.current += timedelta(seconds=self.seconds_per_call)
        return {"output1": {}, "output2": []}

    def inquire_price(self, symbol: str) -> dict[str, Any]:
        self.calls.append(("open", symbol))
        return {"stck_oprc": "100"}


def test_overrun_truncates_tail() -> None:
    day = date(2026, 10, 5)
    clock = _Clock(datetime(2026, 10, 5, 8, 57, 30, tzinfo=KRX_TZ))
    symbols = [f"{i:06d}" for i in range(80)]
    client = _AdvancingClient(1.0, clock)
    result = capture_open_auction(
        client, symbols, session=day, open_at=_open_at(day), now=clock.now, sleep=clock.sleep,
        policy=AuctionCapturePolicy(call_estimate_s=1.0, slot_margin_s=0.0),
    )
    slot = _open_at(day) - timedelta(minutes=1)
    got = sorted(r.symbol for r in result.rows if r.kind == "expected" and r.scheduled_at == slot)
    assert got == symbols[:60]
    assert all(result.failed[s] == SLOT_DEADLINE_ERROR for s in symbols[60:])
    assert result.slots_expected == 2
    assert max(r.received_at for r in result.rows if r.kind == "expected") <= _open_at(day)


def test_long_slots_not_truncated() -> None:
    day = date(2026, 10, 5)
    clock = _Clock(_open_at(day) - timedelta(minutes=25))
    symbols = [f"{i:06d}" for i in range(58)]
    client = _AdvancingClient(1.0, clock)
    result = capture_open_auction(
        client, symbols, session=day, open_at=_open_at(day), now=clock.now, sleep=clock.sleep,
        policy=AuctionCapturePolicy(call_estimate_s=1.0, slot_margin_s=0.0),
    )
    assert result.failed == {}
    assert sum(1 for r in result.rows if r.kind == "expected") == 58 * result.slots_expected


def test_margin_shrinks_budget() -> None:
    day = date(2026, 10, 5)
    clock = _Clock(datetime(2026, 10, 5, 8, 57, 30, tzinfo=KRX_TZ))
    symbols = [f"{i:06d}" for i in range(60)]
    client = _AdvancingClient(1.0, clock)
    result = capture_open_auction(
        client, symbols, session=day, open_at=_open_at(day), now=clock.now, sleep=clock.sleep,
        policy=AuctionCapturePolicy(call_estimate_s=1.0, slot_margin_s=5.0),
    )
    slot = _open_at(day) - timedelta(minutes=1)
    got = [r for r in result.rows if r.kind == "expected" and r.scheduled_at == slot]
    assert len(got) == 55
    assert len([s for s in symbols if result.failed.get(s) == SLOT_DEADLINE_ERROR]) == 5


def test_measured_rate_raises_estimate() -> None:
    day = date(2026, 10, 5)
    clock = _Clock(datetime(2026, 10, 5, 8, 57, 30, tzinfo=KRX_TZ))
    symbols = [f"{i:06d}" for i in range(60)]
    client = _AdvancingClient(2.0, clock)
    result = capture_open_auction(
        client, symbols, session=day, open_at=_open_at(day), now=clock.now, sleep=clock.sleep,
        policy=AuctionCapturePolicy(call_estimate_s=1.0, slot_margin_s=0.0),
    )
    slot = _open_at(day) - timedelta(minutes=1)
    got = [r for r in result.rows if r.kind == "expected" and r.scheduled_at == slot]
    assert len(got) <= 30
    assert all(result.failed.get(s) == SLOT_DEADLINE_ERROR for s in symbols[len(got):])


def test_truncation_deterministic_priority_ordered() -> None:
    day = date(2026, 10, 5)

    def run() -> Any:
        clock = _Clock(datetime(2026, 10, 5, 8, 57, 30, tzinfo=KRX_TZ))
        client = _AdvancingClient(1.0, clock)
        symbols = [f"{i:06d}" for i in range(80)]
        return capture_open_auction(
            client, symbols, session=day, open_at=_open_at(day), now=clock.now, sleep=clock.sleep,
            policy=AuctionCapturePolicy(call_estimate_s=1.0, slot_margin_s=0.0),
        )

    first, second = run(), run()
    assert first.failed == second.failed
    assert {(r.symbol, r.scheduled_at) for r in first.rows} == {(r.symbol, r.scheduled_at) for r in second.rows}
    truncated = [s for s in [f"{i:06d}" for i in range(80)] if s in first.failed]
    assert truncated == [f"{i:06d}" for i in range(80)][len([f"{i:06d}" for i in range(80)]) - len(truncated):]


def test_missed_slot_lowers_captured() -> None:
    day = date(2026, 10, 5)
    clock = _Clock(_open_at(day) - timedelta(minutes=25))

    class _OneSlotDown(_StubClient):
        def __init__(self) -> None:
            super().__init__()
            self.n = 0

        def inquire_asking_price_exp_ccn(self, symbol: str) -> Any:
            self.calls.append(("expected", symbol))
            self.n += 1
            if self.n <= 2:
                raise RuntimeError("slot down")
            return {"output1": {}, "output2": []}

    result = _capture(_OneSlotDown(), ["a", "b"], day, clock)
    assert result.slots_captured == result.slots_expected - 1
    assert set(result.failed) == {"a", "b"}


def test_policy_validation() -> None:
    import pytest

    for kwargs in (
        {"call_estimate_s": 0, "slot_margin_s": 0.0},
        {"call_estimate_s": float("nan"), "slot_margin_s": 0.0},
        {"call_estimate_s": True, "slot_margin_s": 0.0},
        {"call_estimate_s": 1.0, "slot_margin_s": -1.0},
        {"call_estimate_s": 1.0, "slot_margin_s": True},
    ):
        with pytest.raises(ValueError, match=r"call_estimate_s|slot_margin_s"):
            AuctionCapturePolicy(**kwargs)  # type: ignore[arg-type]


def test_no_error_rows_or_schema_drift() -> None:
    day = date(2026, 10, 5)
    clock = _Clock(_open_at(day) - timedelta(minutes=25))

    class _Flaky(_StubClient):
        def inquire_asking_price_exp_ccn(self, symbol: str) -> Any:
            self.calls.append(("expected", symbol))
            if symbol == "bad":
                raise RuntimeError("x")
            return {"output1": {}, "output2": []}

    result = _capture(_Flaky(), ["good", "bad"], day, clock)
    assert {r.kind for r in result.rows} <= {"expected", "open"}


def test_dedup_repeat_write(tmp_path: Path) -> None:
    day = date(2026, 10, 5)
    row = AuctionRow(day, "005930", "expected", _open_at(day) - timedelta(minutes=1), _open_at(day), "{}", "[]")
    write_auction_rows(tmp_path, [row])
    write_auction_rows(tmp_path, [row])
    import polars as pl

    assert pl.read_parquet(tmp_path / "2026-10" / "2026-10-05.parquet").height == 1


def test_existing_row_wins(tmp_path: Path) -> None:
    import polars as pl

    day = date(2026, 10, 5)
    sched = _open_at(day) - timedelta(minutes=1)
    write_auction_rows(tmp_path, [AuctionRow(day, "005930", "expected", sched, _open_at(day), '{"a":1}', "[]")])
    write_auction_rows(tmp_path, [AuctionRow(day, "005930", "expected", sched, _open_at(day) + timedelta(seconds=5), '{"a":2}', "[]")])
    frame = pl.read_parquet(tmp_path / "2026-10" / "2026-10-05.parquet")
    assert frame.height == 1
    assert frame["output1"].to_list() == ['{"a":1}']


def test_distinct_keys_accumulate_sorted(tmp_path: Path) -> None:
    import polars as pl

    day = date(2026, 10, 5)
    base = _open_at(day)
    rows = [
        AuctionRow(day, "000660", "open", base, base, "{}", "{}"),
        AuctionRow(day, "005930", "expected", base - timedelta(minutes=2), base, "{}", "[]"),
        AuctionRow(day, "005930", "expected", base - timedelta(minutes=1), base, "{}", "[]"),
    ]
    for row in rows:
        write_auction_rows(tmp_path, [row])
    frame = pl.read_parquet(tmp_path / "2026-10" / "2026-10-05.parquet")
    assert frame.height == 3
    assert frame["scheduled_at"].to_list() == sorted(frame["scheduled_at"].to_list())


def test_superset_batch_idempotent(tmp_path: Path) -> None:
    import polars as pl

    day = date(2026, 10, 5)
    base = _open_at(day)
    first = AuctionRow(day, "a", "expected", base - timedelta(minutes=2), base, "{}", "[]")
    second = AuctionRow(day, "b", "expected", base - timedelta(minutes=1), base, "{}", "[]")
    write_auction_rows(tmp_path, [first])
    write_auction_rows(tmp_path, [first, second])
    frame = pl.read_parquet(tmp_path / "2026-10" / "2026-10-05.parquet")
    assert frame.height == 2
    assert frame.height - frame.select("symbol", "kind", "scheduled_at").n_unique() == 0


def test_legacy_duplicates_normalized(tmp_path: Path) -> None:
    import polars as pl

    day = date(2026, 10, 5)
    sched = _open_at(day) - timedelta(minutes=1)
    target = tmp_path / "2026-10" / "2026-10-05.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    dup = pl.DataFrame(
        {
            "session": [day, day],
            "symbol": ["005930", "005930"],
            "kind": ["expected", "expected"],
            "scheduled_at": [sched, sched],
            "received_at": [_open_at(day), _open_at(day)],
            "output1": ["{}", "{}"],
            "output2": ["[]", "[]"],
        },
        schema={
            "session": pl.Date, "symbol": pl.String, "kind": pl.String,
            "scheduled_at": pl.Datetime("us", "Asia/Seoul"), "received_at": pl.Datetime("us", "Asia/Seoul"),
            "output1": pl.String, "output2": pl.String,
        },
    )
    dup.write_parquet(target)
    write_auction_rows(tmp_path, [AuctionRow(day, "000660", "expected", sched, _open_at(day), "{}", "[]")])
    frame = pl.read_parquet(target)
    assert frame.height == 2
    assert frame.height - frame.select("symbol", "kind", "scheduled_at").n_unique() == 0


def test_schema_unchanged_and_atomic(tmp_path: Path) -> None:
    import polars as pl

    day = date(2026, 10, 5)
    write_auction_rows(tmp_path, [AuctionRow(day, "005930", "expected", _open_at(day), _open_at(day), "{}", "[]")])
    frame = pl.read_parquet(tmp_path / "2026-10" / "2026-10-05.parquet")
    assert frame.columns == ["session", "symbol", "kind", "scheduled_at", "received_at", "output1", "output2"]
    assert set(frame["kind"].to_list()) <= {"expected", "open"}
    assert list((tmp_path / "2026-10").iterdir()) == [tmp_path / "2026-10" / "2026-10-05.parquet"]
    assert frame.schema == {
        "session": pl.Date, "symbol": pl.String, "kind": pl.String,
        "scheduled_at": pl.Datetime("us", "Asia/Seoul"),
        "received_at": pl.Datetime("us", "Asia/Seoul"),
        "output1": pl.String, "output2": pl.String,
    }


def test_open_attempt_limit_is_per_symbol() -> None:
    day = date(2026, 10, 5)
    clock = _Clock(_open_at(day))

    class _MixedClient(_StubClient):
        def inquire_price(self, symbol: str) -> Any:
            self.calls.append(("open", symbol))
            count = self.calls.count(("open", symbol))
            if symbol == "slow" and count <= 4:
                return {"stck_oprc": "0"}
            raise ConnectionError("secret-value")

    client = _MixedClient()
    result = _capture(client, ["slow", "bad"], day, clock)
    assert client.calls.count(("open", "slow")) == 6
    assert client.calls.count(("open", "bad")) == 6
    assert result.failed == {"slow": "ConnectionError", "bad": "ConnectionError"}


def test_mapping_payloads_and_serialization_failures_are_isolated() -> None:
    from types import MappingProxyType

    day = date(2026, 10, 5)
    clock = _Clock(_open_at(day) - timedelta(minutes=25))

    class _Mappings(_StubClient):
        def inquire_asking_price_exp_ccn(self, symbol: str) -> Any:
            if symbol == "bad":
                return {"output1": {"invalid": object()}}
            return MappingProxyType({"output1": {"price": "100"}, "output2": []})

        def inquire_price(self, symbol: str) -> Any:
            if symbol == "bad":
                return {"stck_oprc": "100", "invalid": object()}
            return MappingProxyType({"stck_oprc": "100"})

    result = _capture(_Mappings(), ["bad", "good"], day, clock)
    assert result.failed == {"bad": "TypeError"}
    assert len(result.rows) == 6
    assert {row.symbol for row in result.rows} == {"good"}


def test_extreme_finite_budget_truncates_without_overflow() -> None:
    day = date(2026, 10, 5)
    clock = _Clock(_open_at(day) - timedelta(minutes=25))
    result = _capture(
        _StubClient(), ["a"], day, clock,
        policy=AuctionCapturePolicy(call_estimate_s=1e300, slot_margin_s=0),
    )
    assert result.slots_captured == 0
    assert result.failed == {"a": SLOT_DEADLINE_ERROR}


def test_failed_calls_raise_observed_estimate() -> None:
    day = date(2026, 10, 5)
    clock = _Clock(_open_at(day) - timedelta(seconds=61))

    class _SlowFailure(_StubClient):
        def inquire_asking_price_exp_ccn(self, symbol: str) -> Any:
            self.calls.append(("expected", symbol))
            clock.current += timedelta(seconds=2)
            raise OSError("secret-value")

    client = _SlowFailure()
    names = [str(i) for i in range(60)]
    result = _capture(client, names, day, clock)
    assert [symbol for kind, symbol in client.calls if kind == "expected"] == names[:30]
    assert {result.failed[symbol] for symbol in names[:30]} == {"OSError"}
    assert {result.failed[symbol] for symbol in names[30:]} == {SLOT_DEADLINE_ERROR}


def test_unexpected_overrun_keeps_row_and_truncates_tail() -> None:
    day = date(2026, 10, 5)
    clock = _Clock(_open_at(day) - timedelta(seconds=61))
    result = _capture(_AdvancingClient(61, clock), ["head", "tail"], day, clock)
    expected = [row for row in result.rows if row.kind == "expected"]
    assert len(expected) == 1
    assert expected[0].symbol == "head"
    assert expected[0].received_at == _open_at(day) + timedelta(seconds=1)
    assert result.failed == {"tail": SLOT_DEADLINE_ERROR}


def test_batch_callback_errors_propagate() -> None:
    import pytest

    day = date(2026, 10, 5)
    clock = _Clock(_open_at(day) - timedelta(minutes=25))

    def fail(batch: Any) -> None:
        raise RuntimeError("persist callback")

    with pytest.raises(RuntimeError, match="persist callback"):
        _capture(_StubClient(), ["a"], day, clock, on_batch=fail)
