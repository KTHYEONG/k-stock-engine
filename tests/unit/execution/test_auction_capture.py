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
    AuctionRow,
    capture_open_auction,
    capture_symbols,
    write_auction_rows,
)


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
    rows = capture_open_auction(
        client, ["005930", "000660"], session=day, open_at=_open_at(day), now=clock.now, sleep=clock.sleep
    )
    expected = [row for row in rows if row.kind == "expected"]
    assert len(expected) == 2 * len(OPEN_SNAPSHOT_OFFSETS_MIN)
    for offset in OPEN_SNAPSHOT_OFFSETS_MIN:
        batch = [row for row in expected if row.scheduled_at == _open_at(day) + timedelta(minutes=offset)]
        assert sorted(row.symbol for row in batch) == ["000660", "005930"]


def test_late_start_skips_past_rounds() -> None:
    day = date(2026, 10, 5)
    clock = _Clock(_open_at(day) - timedelta(minutes=3))
    client = _StubClient()
    rows = capture_open_auction(
        client, ["005930"], session=day, open_at=_open_at(day), now=clock.now, sleep=clock.sleep
    )
    offsets = sorted(
        (row.scheduled_at - _open_at(day)).total_seconds() / 60.0
        for row in rows if row.kind == "expected"
    )
    assert offsets == [-2.0, -1.0]


def test_open_print_resolves_or_times_out() -> None:
    day = date(2026, 10, 5)
    clock = _Clock(_open_at(day) - timedelta(minutes=25))
    client = _StubClient(
        prices={
            "005930": [{"stck_oprc": "0"}, {"stck_oprc": "0"}, {"stck_oprc": "12350", "extra": 1}],
            "000660": [{"stck_oprc": "0"}] * 50,
        }
    )
    rows = capture_open_auction(
        client, ["005930", "000660"], session=day, open_at=_open_at(day), now=clock.now, sleep=clock.sleep
    )
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
    rows = capture_open_auction(
        client, ["005930"], session=day, open_at=_open_at(day), now=clock.now, sleep=clock.sleep
    )
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
    capture_open_auction(client, symbols, session=day, open_at=_open_at(day), now=clock.now, sleep=clock.sleep)
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
    assert capture_open_auction(_StubClient(), [], session=day, open_at=_open_at(day), now=clock.now, sleep=clock.sleep) == ()
    with pytest.raises(ValueError, match="unique"):
        capture_open_auction(_StubClient(), ["005930", "005930"], session=day, open_at=_open_at(day), now=clock.now, sleep=clock.sleep)
    with pytest.raises(ValueError, match="timezone-aware"):
        capture_open_auction(
            _StubClient(), ["005930"], session=day,
            open_at=_datetime(2026, 10, 5, 9, 0), now=clock.now, sleep=clock.sleep,
        )


def test_garbled_open_output_never_resolves() -> None:
    day = date(2026, 10, 5)
    clock = _Clock(_open_at(day) - timedelta(minutes=25))
    client = _StubClient(prices={"005930": [{"stck_oprc": "n/a"}, {"other": 1}] * 10})
    rows = capture_open_auction(
        client, ["005930"], session=day, open_at=_open_at(day), now=clock.now, sleep=clock.sleep
    )
    assert [row for row in rows if row.kind == "open"] == []
    assert clock.now() >= _open_at(day) + timedelta(seconds=180)


def test_non_mapping_payloads_rejected() -> None:
    import pytest

    day = date(2026, 10, 5)
    clock = _Clock(_open_at(day) - timedelta(minutes=25))

    class _BadAsk(_StubClient):
        def inquire_asking_price_exp_ccn(self, symbol: str) -> Any:
            self.calls.append(("expected", symbol))
            return ["not", "a", "mapping"]  # type: ignore[return-value]

    with pytest.raises(ValueError, match="must be a mapping"):
        capture_open_auction(_BadAsk(), ["005930"], session=day, open_at=_open_at(day), now=clock.now, sleep=clock.sleep)

    class _BadPrint(_StubClient):
        def inquire_price(self, symbol: str) -> Any:
            self.calls.append(("open", symbol))
            return ["not", "a", "mapping"]  # type: ignore[return-value]

    with pytest.raises(ValueError, match="must be a mapping"):
        capture_open_auction(_BadPrint(), ["005930"], session=day, open_at=_open_at(day), now=clock.now, sleep=clock.sleep)


def test_deadline_breaks_mid_pass() -> None:
    day = date(2026, 10, 5)

    class _RushingClock(_Clock):
        def now(self) -> datetime:
            self.current += timedelta(seconds=60)
            return self.current

    clock = _RushingClock(_open_at(day) - timedelta(minutes=25))
    client = _StubClient(prices={"a": [{"stck_oprc": "0"}] * 50, "b": [{"stck_oprc": "0"}] * 50})
    rows = capture_open_auction(client, ["a", "b"], session=day, open_at=_open_at(day), now=clock.now, sleep=clock.sleep)
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
    stamps = [early] * 16 + [
        datetime(2026, 10, 5, 9, 0, 0, tzinfo=KRX_TZ),
        datetime(2026, 10, 5, 9, 0, 31, tzinfo=KRX_TZ),
        datetime(2026, 10, 5, 9, 0, 32, tzinfo=KRX_TZ),
        datetime(2026, 10, 5, 9, 3, 0, tzinfo=KRX_TZ),
    ]
    clock = _ScriptedClock(stamps)
    client = _StubClient(prices={"a": [{"stck_oprc": "0"}] * 50, "b": [{"stck_oprc": "0"}] * 50})
    rows = capture_open_auction(client, ["a", "b"], session=day, open_at=_open_at(day), now=clock.now, sleep=clock.sleep)
    assert [row for row in rows if row.kind == "open"] == []
    assert sum(1 for row in rows if row.kind == "expected") == 2 * len(OPEN_SNAPSHOT_OFFSETS_MIN)

