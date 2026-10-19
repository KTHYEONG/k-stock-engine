"""Calendar and headroom guard invariants."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.ops import guards
from src.ops.guards import GuardRefused, exclusive_run_lock, require_headroom, trading_day


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


def test_exclusive_lock_acquired_and_released(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "morning.lock"
    with exclusive_run_lock(path) as first:
        assert first is True
    assert path.is_file()
    with exclusive_run_lock(path) as second:
        assert second is True
    assert path.is_file()


def test_exclusive_lock_second_holder_refused(tmp_path: Path) -> None:
    path = tmp_path / "morning.lock"
    with exclusive_run_lock(path) as outer:
        assert outer is True
        with exclusive_run_lock(path) as inner:
            assert inner is False


def test_exclusive_lock_released_on_exception(tmp_path: Path) -> None:
    path = tmp_path / "morning.lock"
    with pytest.raises(RuntimeError, match="boom"), exclusive_run_lock(path):
        raise RuntimeError("boom")
    with exclusive_run_lock(path) as again:
        assert again is True


def test_exclusive_lock_survives_no_unlink(tmp_path: Path) -> None:
    path = tmp_path / "morning.lock"
    with exclusive_run_lock(path):
        pass
    assert path.is_file()


def test_exclusive_lock_os_error_propagates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import fcntl as _fcntl

    path = tmp_path / "morning.lock"
    real_flock = _fcntl.flock

    def _boom(handle: object, flags: int) -> None:
        raise OSError("read-only filesystem")

    monkeypatch.setattr(_fcntl, "flock", _boom)
    with pytest.raises(OSError, match="read-only"), exclusive_run_lock(path):
        pass
    monkeypatch.setattr(_fcntl, "flock", real_flock)
    with exclusive_run_lock(path) as held:
        assert held is True
