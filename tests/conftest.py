"""Project-local temporary root for every test run.

Pytest's ``tmp_path``/``tmp_path_factory`` fixtures, the ``tempfile`` module,
and tools honoring ``TMPDIR`` are all pinned to ``<project>/tmp/pytest`` so no
test artifact is ever written to the system temp (``/tmp``). ``tmp/`` is
git-ignored and purged by the sync skill.
"""
from __future__ import annotations

import os
# 다중 프로젝트 및 로컬 동시성 환경 리소스 안전 가드:
# Polars, NumPy, OpenBLAS, MKL, Numba 등이 8코어 머신에서 스레드를 과도하게 점유하지 못하도록 상한선 강제
for _key, _val in (
    ("POLARS_MAX_THREADS", "2"),
    ("OMP_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("NUMBA_NUM_THREADS", "1"),
    ("RAY_ACCEL_NUM_WORKERS", "1"),
    ("PYTHONDONTWRITEBYTECODE", "1"),
):
    os.environ.setdefault(_key, _val)


import os
import tempfile
from pathlib import Path

from _pytest.config import Config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TEMP_ROOT = PROJECT_ROOT / "tmp" / "pytest"
_ACTIVE_TEMP_ROOT: Path | None = None


def pytest_configure(config: Config) -> None:
    """Pin the process temporary root to the project before any fixture runs.

    pytest resolves the ``tmp_path`` base lazily from ``PYTEST_DEBUG_TEMPROOT``
    on the first ``getbasetemp()`` call (after configuration), and xdist workers
    each run this hook in their own process, so the env inheritance holds there
    as well.
    """
    del config
    global _ACTIVE_TEMP_ROOT
    _ACTIVE_TEMP_ROOT = _TEMP_ROOT / f"session-{os.getpid()}"
    _ACTIVE_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ["PYTEST_DEBUG_TEMPROOT"] = str(_ACTIVE_TEMP_ROOT)
    os.environ["TMPDIR"] = str(_ACTIVE_TEMP_ROOT)
    tempfile.tempdir = str(_ACTIVE_TEMP_ROOT)


def pytest_sessionfinish(session: object, exitstatus: int) -> None:
    """Clean up project-local temporary files.

    Time Complexity: O(N) where N is the number of temporary entries.
    Space Complexity: O(1) auxiliary space.
    """
    del session
    del exitstatus
    if _ACTIVE_TEMP_ROOT is not None and _ACTIVE_TEMP_ROOT.exists():
        import shutil

        # Clean all items inside _TEMP_ROOT without removing the root itself
        for item in _ACTIVE_TEMP_ROOT.iterdir():
            if item.name == ".gitignore":
                continue
            try:
                if item.is_dir() and not item.is_symlink():
                    shutil.rmtree(item, ignore_errors=True)
                else:
                    item.unlink(missing_ok=True)
            except OSError:
                pass
