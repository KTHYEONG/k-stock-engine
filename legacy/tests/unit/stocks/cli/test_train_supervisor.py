"""TrainSupervisor contract: sampled terminal outcomes for abnormal child exits."""
from __future__ import annotations

import json
import signal
from pathlib import Path

import pytest

from legacy.stocks.cli.train import RunExecutionJournal, TrainSupervisor

TERMINAL_OBS_04 = "TERMINAL_OBS_04_SUPERVISED_SIGNAL_TERMINAL_RECORD"


class _SigkilledChild:
    """Fake Popen handle: alive for one sample window, then SIGKILLed."""

    pid = 999_999_001

    def __init__(self) -> None:
        self._polls = 0

    def poll(self) -> int | None:
        self._polls += 1
        if self._polls == 1:
            return None
        return -int(signal.SIGKILL)


def test_terminal_obs_04_supervised_signal_terminal_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TERMINAL_OBS_04_SUPERVISED_SIGNAL_TERMINAL_RECORD.

    A SIGKILLed child yields a non-zero supervisor exit and exactly one
    atomically written terminal JSON carrying status='failed',
    signal='SIGKILL', sample_count >= 1, and the last durable child
    checkpoint.
    """
    monkeypatch.setattr("src.core.paths.RUN_DIAGNOSTIC_ROOT", tmp_path / "diag")
    run_id = "supa1"
    run_dir = tmp_path / "diag" / run_id
    journal_path = run_dir / "execution_journal.jsonl"
    child_journal = RunExecutionJournal(journal_path, run_id=run_id)
    child_journal.checkpoint("direct_preflight", {"rows": 5})

    outcome_path = run_dir / "supervisor_outcome.json"
    supervisor = TrainSupervisor(
        run_id=run_id,
        interval_seconds=0.05,
        popen=lambda argv: _SigkilledChild(),
        journal_path=journal_path,
        outcome_path=outcome_path,
    )

    rc = supervisor.run(["--artifact-id", run_id])

    assert rc == 128 + int(signal.SIGKILL)
    assert rc != 0

    json_files = sorted(path for path in run_dir.glob("*.json"))
    assert [path.name for path in json_files] == ["supervisor_outcome.json"]
    assert not list(run_dir.glob("*.tmp"))
    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    assert outcome["status"] == "failed"
    assert outcome["signal"] == "SIGKILL"
    assert outcome["exit_code"] is None
    assert outcome["sample_count"] >= 1
    last_checkpoint = outcome["last_child_checkpoint"]
    assert isinstance(last_checkpoint, dict)
    assert last_checkpoint["stage"] == "direct_preflight"


class _CleanExitChild:
    """Fake Popen handle that finishes successfully after one sample."""

    pid = 999_999_002

    def __init__(self) -> None:
        self._polled = False

    def poll(self) -> int | None:
        if not self._polled:
            self._polled = True
            return None
        return 0


def test_supervisor_normal_child_completes_zero(tmp_path: Path) -> None:
    """A zero-exit child publishes status='completed' and exits 0."""
    run_dir = tmp_path / "diag" / "clean1"
    journal_path = run_dir / "execution_journal.jsonl"
    outcome_path = run_dir / "supervisor_outcome.json"
    supervisor = TrainSupervisor(
        run_id="clean1",
        interval_seconds=0.05,
        popen=lambda argv: _CleanExitChild(),
        journal_path=journal_path,
        outcome_path=outcome_path,
    )

    rc = supervisor.run(["--artifact-id", "clean1"])

    assert rc == 0
    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    assert outcome["status"] == "completed"
    assert outcome["exit_code"] == 0
    assert outcome["signal"] is None
    assert outcome["sample_count"] >= 1


class _NonZeroChild:
    """Fake Popen handle exiting with a positive failure code."""

    pid = 999_999_003

    def poll(self) -> int | None:
        return 3


def test_supervisor_nonzero_child_is_failed_and_nonzero(tmp_path: Path) -> None:
    """A non-zero child exit stays failed and propagates the code."""
    run_dir = tmp_path / "diag" / "boom1"
    outcome_path = run_dir / "supervisor_outcome.json"
    supervisor = TrainSupervisor(
        run_id="boom1",
        interval_seconds=0.05,
        popen=lambda argv: _NonZeroChild(),
        journal_path=run_dir / "execution_journal.jsonl",
        outcome_path=outcome_path,
    )

    rc = supervisor.run(["--artifact-id", "boom1"])

    assert rc == 3
    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    assert outcome["status"] == "failed"
    assert outcome["exit_code"] == 3
    assert outcome["signal"] is None


def test_supervisor_rejects_interval_above_half_second(tmp_path: Path) -> None:
    """Sampling intervals must stay <=500 ms by contract."""
    with pytest.raises(ValueError, match="interval"):
        TrainSupervisor(
            run_id="slow1",
            interval_seconds=0.75,
            journal_path=tmp_path / "j.jsonl",
            outcome_path=tmp_path / "o.json",
        )
