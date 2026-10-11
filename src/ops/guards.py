"""Resource and calendar preconditions for scheduled operations jobs."""

from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path
from typing import Final

__all__ = [
    "DEFAULT_MIN_AVAILABLE_MB",
    "DEFAULT_MIN_FREE_DISK_GB",
    "GuardRefused",
    "require_headroom",
    "trading_day",
]

DEFAULT_MIN_AVAILABLE_MB: Final = 3500
DEFAULT_MIN_FREE_DISK_GB: Final = 25.0


class GuardRefused(RuntimeError):  # noqa: N818 - name fixed by the ops-7 spec
    """A resource or calendar precondition failed; the job must not start."""

    available_mb: float | None = None
    free_disk_gb: float | None = None


def trading_day(day: date) -> bool:
    """True when day is an XKRX session."""
    from src.core.krx_calendar import xkrx_calendar_through
    from src.core.time import KRX_TZ

    sessions = xkrx_calendar_through(day).sessions
    return any(moment.astimezone(KRX_TZ).date() == day for moment in sessions)


def _mem_available_mb() -> float:
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024.0
    except (OSError, ValueError):
        pass
    raise GuardRefused("MemAvailable is unreadable; refusing to start")


def _free_disk_gb(path: Path) -> float:
    try:
        return shutil.disk_usage(path).free / float(1024**3)
    except OSError as exc:
        raise GuardRefused(f"free disk is unreadable for {path}: {exc}") from exc


def require_headroom(*, min_available_mb: int, min_free_disk_gb: float, data_root: Path) -> None:
    """Raise GuardRefused unless MemAvailable and free disk on data_root's filesystem meet the minimums."""
    available_mb = _mem_available_mb()
    free_gb = _free_disk_gb(Path(data_root))
    if available_mb < float(min_available_mb) or free_gb < float(min_free_disk_gb):
        refusal = GuardRefused(
            f"insufficient headroom: MemAvailable {available_mb:.0f} MB "
            f"(need {min_available_mb}), free disk {free_gb:.1f} GiB "
            f"(need {min_free_disk_gb})"
        )
        refusal.available_mb = available_mb
        refusal.free_disk_gb = free_gb
        raise refusal
