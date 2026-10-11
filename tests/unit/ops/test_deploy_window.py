"""Deploy blackout window invariants."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

import src.ops.deploy_window as window
from src.ops.deploy_window import BLACKOUT_EXIT, _wait_until_clear, in_blackout, main

KRX_TZ = ZoneInfo("Asia/Seoul")


def _at(day: int, hour: int, minute: int) -> datetime:
    return datetime(2026, 10, day, hour, minute, tzinfo=KRX_TZ)


def test_blackout_windows_are_half_open() -> None:
    assert in_blackout(_at(8, 8, 20)) is True
    assert in_blackout(_at(8, 9, 10)) is True
    assert in_blackout(_at(8, 8, 50)) is True
    assert in_blackout(_at(8, 9, 40)) is False
    assert in_blackout(_at(8, 15, 10)) is True
    assert in_blackout(_at(8, 15, 43)) is False
    assert in_blackout(_at(8, 18, 20)) is True
    assert in_blackout(_at(8, 21, 50)) is False
    assert in_blackout(_at(8, 12, 0)) is False


def test_blackout_skips_weekends_and_naive_input() -> None:
    assert in_blackout(datetime(2026, 10, 10, 9, 0, tzinfo=KRX_TZ)) is False
    assert in_blackout(datetime(2026, 10, 11, 19, 0, tzinfo=KRX_TZ)) is False
    assert in_blackout(datetime(2026, 10, 8, 8, 30)) is True


def test_wait_blocks_until_clear() -> None:
    plan = [_at(8, 8, 30), _at(8, 8, 31), _at(8, 12, 0)]
    calls: list[float] = []
    _wait_until_clear(lambda: plan.pop(0), calls.append)
    assert len(calls) == 2


def test_main_exit_codes(monkeypatch: pytest.MonkeyPatch) -> None:
    assert main([]) in (0, BLACKOUT_EXIT)
    calls: list[float] = []
    seq = [_at(8, 8, 30), _at(8, 12, 0)]

    class _Clock:
        @staticmethod
        def now(tz: object = None) -> datetime:
            return seq.pop(0)

    monkeypatch.setattr(window, "datetime", _Clock)
    monkeypatch.setattr(window.time, "sleep", calls.append)
    assert main(["--wait"]) == 0
    assert len(calls) == 1
