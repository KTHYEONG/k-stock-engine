"""Deploy blackout gate for VPS releases on weekday mornings and evenings."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from datetime import datetime
from datetime import time as _time
from typing import Final

from src.core.time import KRX_TZ

__all__ = ["BLACKOUTS", "in_blackout", "main"]

BLACKOUTS: Final[tuple[tuple[_time, _time], ...]] = (
    (_time(8, 20), _time(9, 10)),
    (_time(18, 20), _time(21, 50)),
    (_time(8, 50), _time(9, 40)),
    (_time(15, 10), _time(15, 43)),
)

BLACKOUT_EXIT: Final = 3
_WAIT_POLL_S: Final = 30.0


def in_blackout(moment: datetime) -> bool:
    """True inside a weekday deploy blackout window (half-open, Asia/Seoul)."""
    local = moment if moment.tzinfo is not None else moment.replace(tzinfo=KRX_TZ)
    local = local.astimezone(KRX_TZ)
    if local.weekday() >= 5:
        return False
    mark = local.time()
    return any(start <= mark < end for start, end in BLACKOUTS)


def _wait_until_clear(now: Callable[[], datetime], sleep: Callable[[float], None]) -> None:
    while in_blackout(now()):
        sleep(_WAIT_POLL_S)


def main(argv: list[str] | None = None) -> int:
    """Exit 0 when deploys may proceed, 3 inside a blackout; --wait blocks until clear."""
    parser = argparse.ArgumentParser(prog="src.ops.deploy_window")
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args(argv)
    if args.wait:
        _wait_until_clear(lambda: datetime.now(KRX_TZ), time.sleep)
        return 0
    return BLACKOUT_EXIT if in_blackout(datetime.now(KRX_TZ)) else 0


if __name__ == "__main__":  # pragma: no cover - entrypoint glue
    raise SystemExit(main(sys.argv[1:]))
