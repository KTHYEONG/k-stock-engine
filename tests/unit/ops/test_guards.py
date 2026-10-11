"""Calendar and headroom guard invariants."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.ops import guards
from src.ops.guards import GuardRefused, require_headroom, trading_day


def test_trading_day_distinguishes_session_and_holiday() -> None:
    assert trading_day(date(2026, 10, 8)) is True
    assert trading_day(date(2026, 10, 9)) is False


def test_trading_day_uses_calendar_sessions(monkeypatch: pytest.MonkeyPatch) -> None:
    from datetime import datetime

    from src.core.time import KRX_TZ

    sessions = (
        datetime(2026, 10, 8, 9, 0, tzinfo=KRX_TZ),
        datetime(2026, 10, 12, 9, 0, tzinfo=KRX_TZ),
    )
    monkeypatch.setattr(
        "src.core.krx_calendar.xkrx_calendar_through", lambda day: SimpleNamespace(sessions=sessions)
    )
    assert trading_day(date(2026, 10, 8)) is True
    assert trading_day(date(2026, 10, 9)) is False


def test_headroom_passes_and_refuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(guards, "_mem_available_mb", lambda: 8000.0)
    monkeypatch.setattr(guards, "_free_disk_gb", lambda path: 100.0)
    require_headroom(min_available_mb=3500, min_free_disk_gb=25.0, data_root=tmp_path)
    monkeypatch.setattr(guards, "_mem_available_mb", lambda: 100.0)
    with pytest.raises(GuardRefused, match="100 MB"):
        require_headroom(min_available_mb=3500, min_free_disk_gb=25.0, data_root=tmp_path)
    monkeypatch.setattr(guards, "_mem_available_mb", lambda: 8000.0)
    monkeypatch.setattr(guards, "_free_disk_gb", lambda path: 1.0)
    with pytest.raises(GuardRefused, match=r"1\.0 GiB"):
        require_headroom(min_available_mb=3500, min_free_disk_gb=25.0, data_root=tmp_path)


def test_headroom_reads_proc_meminfo() -> None:
    assert guards._mem_available_mb() > 0.0


def test_headroom_fails_closed_on_unreadable_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import builtins

    real_open = builtins.open

    def _boom(path: object, *args: object, **kwargs: object) -> object:
        if str(path) == "/proc/meminfo":
            raise OSError("denied")
        return real_open(str(path), *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "open", _boom)
    with pytest.raises(GuardRefused, match="MemAvailable"):
        require_headroom(min_available_mb=3500, min_free_disk_gb=25.0, data_root=tmp_path)
    monkeypatch.setattr(builtins, "open", real_open)
    monkeypatch.setattr(guards, "_mem_available_mb", lambda: 8000.0)
    with pytest.raises(GuardRefused, match="free disk"):
        require_headroom(min_available_mb=3500, min_free_disk_gb=25.0, data_root=tmp_path / "missing")
