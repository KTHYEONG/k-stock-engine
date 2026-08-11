"""Project-local temporary root for every test run.

Pytest's ``tmp_path``/``tmp_path_factory`` fixtures, the ``tempfile`` module,
and tools honoring ``TMPDIR`` are all pinned to ``<project>/tmp/pytest`` so no
test artifact is ever written to the system temp (``/tmp``). ``tmp/`` is
git-ignored and purged by the sync skill.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from _pytest.config import Config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TEMP_ROOT = PROJECT_ROOT / "tmp" / "pytest"


def pytest_configure(config: Config) -> None:
    """Pin the process temporary root to the project before any fixture runs.

    pytest resolves the ``tmp_path`` base lazily from ``PYTEST_DEBUG_TEMPROOT``
    on the first ``getbasetemp()`` call (after configuration), and xdist workers
    each run this hook in their own process, so the env inheritance holds there
    as well.
    """
    del config
    _TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ["PYTEST_DEBUG_TEMPROOT"] = str(_TEMP_ROOT)
    os.environ["TMPDIR"] = str(_TEMP_ROOT)
    tempfile.tempdir = str(_TEMP_ROOT)
