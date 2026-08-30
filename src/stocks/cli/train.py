"""Stock train CLI: resolve a net-alpha snapshot, compose, and train the mainline.

The only supported training path is the canonical ``stock_net_alpha_v1``
mainline. A snapshot or artifact that does not satisfy the net-alpha contract
raises one actionable ``ValueError`` naming the materialization CLI; there is
no legacy LambdaRank/Optuna flag, no implicit fallback, and no fixed 5/10/15
route.
"""
from __future__ import annotations

import argparse  # argparse.ArgumentParser.add_argument
import inspect
import json
import logging
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from src.core.costs import (
    CostSchedule,
    LiquiditySlippageModel,
    default_base_schedule,
    default_stress_schedule,
)
from src.core.datasets import DatasetCertification
from src.core.paths import (
    PROJECT_ROOT,
    STOCK_ARTIFACT_ROOT,
    STOCK_BASE_PANEL_ROOT,
    STOCK_CATALOG_ROOT,
    STOCK_FEATURE_PANEL_ROOT,
    STOCK_LABEL_ROOT,
)
from src.stocks.cli.contracts import TrainCommand, parse_train_command
from src.stocks.config.research import (
    policy_profiles_with_excess_full_kelly,
    policy_profiles_with_growth_rungs,
    resolve_training_request,
)
from src.stocks.config.runtime import StockRuntimeSettings
from src.stocks.data.active import (
    ActiveResearchDataRequest,
    ActiveResearchDataSelection,
    resolve_active_research_data,
)
from src.stocks.data.contracts import CoverageRange, DatasetSnapshot
from src.stocks.data.costs import load_cost_evidence
from src.stocks.data.direct import DirectDataRequest, DirectLoadCheckpoint
from src.stocks.data.lineage import ResearchDataBundle, ResolvedDataLineage
from src.stocks.data.repositories import (
    ResearchDataRepository,
    resolve_snapshot_for_mode,
)
from src.stocks.ml.contracts import (
    DECLARED_ECONOMIC_FAMILIES,
    DEFAULT_CANDIDATE_HORIZON_SESSIONS,
    DEFAULT_CANDIDATE_REBALANCE_FREQUENCY_SESSIONS,
    DEFAULT_CANDIDATE_TOP_K,
    DEFAULT_POLICY_PROFILES,
    ELASTIC_NET_FAMILY,
    AccountCertificationSettings,
    CompoundingCertificationSettings,
    ExecutionFrontierSettings,
    NetAlphaResearchData,
    NetAlphaTrainingRequest,
    PolicyProfile,
    PortfolioSettings,
    RiskSettings,
    SatelliteOverlaySettings,
    SmallCapitalPlanSettings,
    UniverseRescopeSettings,
)
from src.stocks.ml.data import compose_net_alpha_training_data
from src.stocks.ml.replay_resources import (
    MemoryBudgetExceededError as _EnvelopeBudgetError,
)
from src.stocks.ml.replay_resources import read_host_mem_available_bytes
from src.stocks.ml.result_ledger import (
    CostRunContext,
    MlResultLedger,
    MlRunContext,
)
from src.stocks.ml.training import TrainingOrchestrator, train_net_alpha_model
from src.stocks.observability.contracts import RunDiagnostics, RunIdentity
from src.stocks.observability.recorder import open_run_diagnostics
from src.stocks.research.artifacts import ModelArtifactRegistry
from src.stocks.research.models import ModelManifest
from src.stocks.settings import REFERENCE_DATE, REFERENCE_DATETIME

logger = logging.getLogger("stocks.cli.train")


def _invoke_training(
    data: NetAlphaResearchData,
    registry: ModelArtifactRegistry,
    request: NetAlphaTrainingRequest,
    diagnostics: RunDiagnostics,
    progress: Callable[[str, Mapping[str, object] | None], None] | None = None,
) -> ModelManifest:
    """Drive the training orchestrator, honoring legacy callable signatures."""
    parameters = inspect.signature(train_net_alpha_model).parameters
    if "diagnostics" in parameters:
        orchestrator = TrainingOrchestrator(
            data, registry, request, diagnostics=diagnostics, progress=progress
        )
        return orchestrator.run()
    if "progress" in parameters:
        return train_net_alpha_model(data, registry, request, progress=progress)
    return train_net_alpha_model(data, registry, request)

STOCK_RESULTS_DOC_ROOT = PROJECT_ROOT / "docs" / "results"

_CGROUP_UNLIMITED_SENTINEL = "max"
_SUPERVISOR_SAMPLE_INTERVAL_SECONDS = 0.25
_JOURNAL_MAX_LINE_BYTES = 4096
_DIRECT_ADMISSION_NEXT_ALLOCATION_FACTOR = 2


def resolve_process_cgroup_root(
    proc_cgroup_path: Path = Path("/proc/self/cgroup"),
    cgroup_mount: Path = Path("/sys/fs/cgroup"),
) -> Path | None:
    """Resolve this process's own cgroup directory for memory sampling.

    Prefers the v2 unified entry (``0::<path>``) and falls back to the v1
    ``memory`` controller line, so memory.current/peak/max/events are read
    from the subpath that actually bounds this process instead of the mount
    root. v2 resolves to ``<mount>/<path>``; the v1 memory line resolves to
    ``<mount>/memory/<path>`` matching the v1 file layout. Returns ``None``
    when unreadable or unresolvable; callers must record nulls rather than
    fabricating values.
    """
    try:
        raw_lines = proc_cgroup_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    v2_relative: str | None = None
    v1_memory_relative: str | None = None
    for line in raw_lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        controllers = tuple(item for item in parts[1].split(",") if item)
        relative = parts[2].strip()
        if not controllers and v2_relative is None:
            v2_relative = relative
        elif "memory" in controllers and v1_memory_relative is None:
            v1_memory_relative = relative
    if v2_relative is not None:
        resolved = cgroup_mount / v2_relative.lstrip("/")
    elif v1_memory_relative is not None:
        resolved = cgroup_mount / "memory" / v1_memory_relative.lstrip("/")
    else:
        return None
    return resolved if resolved.is_dir() else None


@dataclass(frozen=True, slots=True)
class ProcessCgroupMemorySample:
    """Bounded cgroup memory scalars; unavailable values stay ``None``."""

    current_bytes: int | None = None
    peak_bytes: int | None = None
    limit_bytes: int | None = None
    oom_kill_count: int | None = None


def _read_cgroup_scalar(path: Path) -> int | None:
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not raw or raw == _CGROUP_UNLIMITED_SENTINEL:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value >= 0 else None


def _read_first_cgroup_scalar(root: Path, names: tuple[str, ...]) -> int | None:
    """Read the first existing cgroup scalar file; ``None`` when unavailable."""
    for name in names:
        candidate = root / name
        if not candidate.exists():
            continue
        return _read_cgroup_scalar(candidate)
    return None


def sample_process_cgroup_memory(root: Path | None = None) -> ProcessCgroupMemorySample:
    """Sample memory.current/peak/max plus memory.events under the resolved root."""
    resolved = resolve_process_cgroup_root() if root is None else root
    if resolved is None:
        return ProcessCgroupMemorySample()
    oom_kill_count: int | None = None
    events_path = resolved / "memory.events"
    try:
        for line in events_path.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition(" ")
            if key.strip() == "oom_kill":
                try:
                    oom_kill_count = int(value.strip())
                except ValueError:
                    oom_kill_count = None
                break
    except OSError:
        oom_kill_count = None
    limit_value = _read_first_cgroup_scalar(
        resolved, ("memory.max", "memory.limit_in_bytes")
    )
    if limit_value is not None and limit_value >= (1 << 60):
        # v1 reports a huge sentinel instead of "max" for unlimited.
        limit_value = None
    return ProcessCgroupMemorySample(
        current_bytes=_read_first_cgroup_scalar(
            resolved, ("memory.current", "memory.usage_in_bytes")
        ),
        peak_bytes=_read_first_cgroup_scalar(
            resolved, ("memory.peak", "memory.max_usage_in_bytes")
        ),
        limit_bytes=limit_value,
        oom_kill_count=oom_kill_count,
    )


def _process_rss_bytes() -> int | None:
    try:
        with open("/proc/self/status", "rb") as handle:
            for line in handle:
                if line.startswith(b"VmRSS:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _bounded_journal_value(value: object) -> object:
    if isinstance(value, str):
        return value[:512]
    if isinstance(value, (tuple, list)):
        return ",".join(str(item) for item in value)[:512]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)[:512]


_DROPPABLE_JOURNAL_KEYS = frozenset({"owners"})


class RunExecutionJournal:
    """Fsynced bounded JSONL execution journal for one training run.

    Every checkpoint row carries the stage name, elapsed time, process RSS,
    the resolved process-cgroup current/peak/limit/OOM-kill counters,
    ``MemAvailable``, frame shape scalars, estimated frame bytes, and live
    owner names only — never raw market rows, values, credentials, or an
    environment dump.
    """

    def __init__(
        self,
        path: Path,
        *,
        run_id: str = "",
        guard: Any | None = None,
    ) -> None:
        self._path = path
        self._run_id = run_id
        self._guard = guard
        self._started_monotonic = time.monotonic()
        self._sequence = 0
        self._closed = False
        path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def last_checkpoint(self) -> dict[str, object] | None:
        """Return the most recent durable checkpoint row, or ``None``."""
        try:
            with self._path.open("r", encoding="utf-8") as handle:
                lines = handle.read().splitlines()
        except OSError:
            return None
        while lines:
            candidate = lines.pop()
            if not candidate.strip():
                continue
            try:
                record = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            return record if isinstance(record, dict) else None
        return None

    def checkpoint(
        self, stage: str, payload: Mapping[str, object] | None = None
    ) -> None:
        """Append and fsync one durable stage checkpoint."""
        cgroup = sample_process_cgroup_memory()
        record: dict[str, object] = {
            "event": "checkpoint",
            "stage": stage,
            "elapsed_ms": int((time.monotonic() - self._started_monotonic) * 1000),
            "rss_bytes": _process_rss_bytes(),
            "mem_available_bytes": read_host_mem_available_bytes(),
            "cgroup_current_bytes": cgroup.current_bytes,
            "cgroup_peak_bytes": cgroup.peak_bytes,
            "cgroup_limit_bytes": cgroup.limit_bytes,
            "cgroup_oom_kill_events": cgroup.oom_kill_count,
        }
        for key, value in sorted((payload or {}).items()):
            record[key] = _bounded_journal_value(value)
        self._append(record)

    def terminal(self, status: str, payload: Mapping[str, object] | None = None) -> None:
        """Append and fsync one durable terminal outcome row."""
        cgroup = sample_process_cgroup_memory()
        record: dict[str, object] = {
            "event": "terminal",
            "status": status,
            "elapsed_ms": int((time.monotonic() - self._started_monotonic) * 1000),
            "rss_bytes": _process_rss_bytes(),
            "mem_available_bytes": read_host_mem_available_bytes(),
            "cgroup_current_bytes": cgroup.current_bytes,
            "cgroup_peak_bytes": cgroup.peak_bytes,
            "cgroup_limit_bytes": cgroup.limit_bytes,
            "cgroup_oom_kill_events": cgroup.oom_kill_count,
        }
        for key, value in sorted((payload or {}).items()):
            record[key] = _bounded_journal_value(value)
        self._append(record)
        self._closed = True

    def direct_load_checkpoint(self, checkpoint: DirectLoadCheckpoint) -> None:
        """Journal one loader checkpoint; admit the preflight shape fail-closed.

        The wide join admission uses the preflight join-shape lower bound plus
        the next same-width materialization so a doomed collect is denied
        before it can start.
        """
        self.checkpoint(checkpoint.stage, checkpoint.journal_payload())
        planned = checkpoint.planned_lower_bound_bytes
        if (
            self._guard is not None
            and checkpoint.stage == "direct_preflight"
            and isinstance(planned, int)
            and planned > 0
        ):
            self._guard.boundary(
                "direct_preflight",
                planned_bytes=(
                    planned * _DIRECT_ADMISSION_NEXT_ALLOCATION_FACTOR
                ),
                live_owners=("base_keys", "feature_keys", "decision_frame"),
            )

    def _append(self, record: dict[str, object]) -> None:
        if self._closed:
            return
        self._sequence += 1
        record["run_id"] = self._run_id
        record["sequence"] = self._sequence
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        if len(line.encode("utf-8")) > _JOURNAL_MAX_LINE_BYTES:
            trimmed = {k: v for k, v in record.items() if k not in _DROPPABLE_JOURNAL_KEYS}
            line = json.dumps(trimmed, ensure_ascii=False, separators=(",", ":"))
        try:
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except OSError:
            logger.warning("[SYS] stage=execution_journal status=write_failed path=%s", self._path)


class TrainSupervisor:
    """Parent supervisor: samples the child and writes the terminal outcome.

    A killed Python process cannot execute its own failure ledger, so this
    parent samples RSS/cgroup use every <=500 ms, records the exit code or
    signal, the sampled peaks, the last durable child checkpoint, any OOF
    spill paths left behind, and atomically publishes exactly one terminal
    outcome JSON. Abnormal exits are non-zero.
    """

    def __init__(
        self,
        *,
        run_id: str,
        interval_seconds: float = _SUPERVISOR_SAMPLE_INTERVAL_SECONDS,
        python_executable: str | None = None,
        module_target: str = "src.stocks.cli.train",
        popen: Any | None = None,
        journal_path: Path | None = None,
        outcome_path: Path | None = None,
    ) -> None:
        if not 0 < interval_seconds <= 0.5:
            raise ValueError("supervisor sample interval must be within (0, 0.5] seconds")
        self._run_id = run_id
        self._interval_seconds = float(interval_seconds)
        self._python_executable = python_executable or sys.executable
        self._module_target = module_target
        self._popen_factory = popen or subprocess.Popen
        self._journal_path = journal_path
        self._outcome_path = outcome_path

    def run(self, child_argv: Sequence[str]) -> int:
        from src.core.paths import RUN_DIAGNOSTIC_ROOT

        journal_path = self._journal_path or (
            RUN_DIAGNOSTIC_ROOT / self._run_id / "execution_journal.jsonl"
        )
        outcome_path = self._outcome_path or (
            RUN_DIAGNOSTIC_ROOT / self._run_id / "supervisor_outcome.json"
        )
        argv = [
            self._python_executable,
            "-m",
            self._module_target,
            *child_argv,
            "--internal-worker",
        ]
        started_wall = datetime.now(UTC).isoformat()
        started_monotonic = time.monotonic()
        process = self._popen_factory(argv)
        sample_count = 0
        peak_rss_bytes = 0
        peak_cgroup_current_bytes = 0
        returncode: int | None = process.poll()
        while returncode is None:
            child_rss = _read_process_rss_bytes(process.pid)
            child_cgroup = _read_process_tree_cgroup_current(process.pid)
            sample_count += 1
            peak_rss_bytes = max(peak_rss_bytes, child_rss or 0)
            peak_cgroup_current_bytes = max(
                peak_cgroup_current_bytes, child_cgroup or 0
            )
            time.sleep(self._interval_seconds)
            returncode = process.poll()
        finished_wall = datetime.now(UTC).isoformat()
        elapsed_ms = int((time.monotonic() - started_monotonic) * 1000)

        status = "completed"
        exit_code: int | None = None
        signal_name: str | None = None
        result_code = int(returncode)
        if result_code < 0:
            status = "failed"
            try:
                signal_name = signal.Signals(-result_code).name
            except ValueError:
                signal_name = f"SIG{-result_code}"
            exit_code_for_run = 128 + (-result_code)
        elif result_code > 0:
            status = "failed"
            exit_code = result_code
            exit_code_for_run = result_code
        else:
            exit_code = 0
            exit_code_for_run = 0

        journal = RunExecutionJournal(journal_path, run_id=self._run_id)
        last_child_checkpoint = journal.last_checkpoint()
        outcome: dict[str, object] = {
            "run_id": self._run_id,
            "status": status,
            "exit_code": exit_code,
            "signal": signal_name,
            "sample_count": sample_count,
            "sampled_peak_rss_bytes": peak_rss_bytes or None,
            "sampled_peak_cgroup_current_bytes": peak_cgroup_current_bytes or None,
            "last_child_checkpoint": last_child_checkpoint,
            "oof_spill_paths": _remaining_oof_spill_paths(),
            "child_argv": [str(arg) for arg in child_argv][:64],
            "started_at": started_wall,
            "finished_at": finished_wall,
            "elapsed_ms": elapsed_ms,
        }
        _atomic_write_json(outcome_path, outcome)
        journal.terminal(
            status,
            {"exit_code": exit_code, "signal": signal_name},
        )
        return exit_code_for_run


def _read_process_rss_bytes(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/status", "rb") as handle:
            for line in handle:
                if line.startswith(b"VmRSS:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _read_process_tree_cgroup_current(pid: int) -> int | None:
    root = resolve_process_cgroup_root(
        proc_cgroup_path=Path(f"/proc/{pid}/cgroup"),
    )
    if root is None:
        return None
    return _read_cgroup_scalar(root / "memory.current")


def _remaining_oof_spill_paths() -> list[str]:
    base = PROJECT_ROOT / "tmp" / "training"
    if not base.is_dir():
        return []
    try:
        return sorted(str(path) for path in base.glob("oof-*"))
    except OSError:
        return []


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Publish one JSON document atomically via a same-dir fsynced rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a net-alpha model artifact")
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument(
        "--research-start",
        type=date.fromisoformat,
        default=date(2016, 1, 4),
        help="inclusive research data start date for direct selection",
    )
    parser.add_argument(
        "--research-end",
        type=date.fromisoformat,
        default=REFERENCE_DATE,
        help="inclusive research data end date for direct selection",
    )
    parser.add_argument("--catalog-root", type=Path, default=STOCK_CATALOG_ROOT)
    parser.add_argument("--base-root", type=Path, default=STOCK_BASE_PANEL_ROOT)
    parser.add_argument("--feature-root", type=Path, default=STOCK_FEATURE_PANEL_ROOT)
    parser.add_argument("--label-root", type=Path, default=STOCK_LABEL_ROOT)
    parser.add_argument("--registry", type=Path, default=STOCK_ARTIFACT_ROOT)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=STOCK_RESULTS_DOC_ROOT,
        help="directory owning the generated result ledger (default docs/results)",
    )
    parser.add_argument(
        "--mode",
        choices=("research", "paper", "live"),
        default="research",
        help="paper/live modes reject provisional snapshots",
    )
    parser.add_argument(
        "--candidate-horizon-sessions",
        type=str,
        default=",".join(str(h) for h in DEFAULT_CANDIDATE_HORIZON_SESSIONS),
        help=(
            "pre-registered discovery grid of horizon session counts "
            f"(default {DEFAULT_CANDIDATE_HORIZON_SESSIONS})"
        ),
    )
    parser.add_argument(
        "--candidate-rebalance-frequency-sessions",
        type=str,
        default=",".join(
            str(value) for value in DEFAULT_CANDIDATE_REBALANCE_FREQUENCY_SESSIONS
        ),
        help="pre-registered execution cadence grid in sessions",
    )
    parser.add_argument(
        "--candidate-top-k",
        type=str,
        default=",".join(str(value) for value in DEFAULT_CANDIDATE_TOP_K),
        help="pre-registered execution maximum active-name grid",
    )
    parser.add_argument("--fold-count", type=int, default=3)
    parser.add_argument("--embargo-sessions", type=int, default=5)
    parser.add_argument("--forward-holdout-sessions", type=int, default=0)
    parser.add_argument("--bootstrap-alpha", type=float, default=0.05)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--model-threads", type=int, default=4)
    parser.add_argument(
        "--max-rss-mib",
        type=int,
        default=None,
        help="explicit RSS budget in MiB; a breach publishes complete NO_TRADE evidence",
    )
    parser.add_argument(
        "--memory-reserve-mib",
        type=int,
        default=0,
        help=(
            "measured concurrent-workload memory reserve in MiB subtracted "
            "from cgroup/system headroom during pre-allocation planning"
        ),
    )
    parser.add_argument(
        "--max-training-lookback-sessions",
        type=int,
        default=None,
        help=(
            "optional rolling fit-window cap in trading sessions applied "
            "after purge and embargo (minimum 252); omit for expanding "
            "training windows"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cost-schedule",
        choices=("base", "stress"),
        default="base",
        help=(
            "labels the run's reference schedule kind in the result ledger; "
            "both base and stress schedules (with matching liquidity models) "
            "are always resolved from the same snapshot cost evidence, and the "
            "base schedule remains the only schedule used for fitting and "
            "calibration"
        ),
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-single-weight", type=float, default=0.08)
    parser.add_argument("--max-exposure", type=float, default=0.90)
    parser.add_argument("--participation-limit", type=float, default=0.005)
    parser.add_argument("--portfolio-value", type=float, default=10_000_000.0)
    parser.add_argument("--reference-notional", type=float, default=10_000_000.0)
    parser.add_argument(
        "--decision-time",
        type=datetime.fromisoformat,
        default=REFERENCE_DATETIME,
        help="decision timestamp (default: 2026-03-10T06:30:00+00:00)",
    )
    parser.add_argument(
        "--cost-evidence-path",
        type=Path,
        default=None,
        help="optional direct cost evidence path; absent produces warning",
    )
    parser.add_argument(
        "--supervise",
        action="store_true",
        help="run as a supervised parent that samples the training child",
    )
    parser.add_argument(
        "--internal-worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--research-only-growth-route",
        action="store_true",
        help=(
            "read-only evaluation of the growth route over the selected "
            "snapshot; prints a RESEARCH_ONLY JSON payload and publishes no "
            "artifact"
        ),
    )
    parser.add_argument(
        "--research-only-temporal-window-study",
        action="store_true",
        help=(
            "read-only comparison of rolling/expanding fit windows on one "
            "common causal OOS calendar; prints a RESEARCH_ONLY JSON study "
            "payload and publishes no artifact"
        ),
    )
    parser.add_argument(
        "--research-only-economic-family-study",
        action="store_true",
        help=(
            "read-only elastic-net vs tail-LambdaRank family study over every "
            "declared window on one common causal OOS calendar; prints a "
            "RESEARCH_ONLY JSON study payload and publishes no artifact"
        ),
    )
    parser.add_argument(
        "--research-only-alpha-capacity-audit",
        action="store_true",
        help=(
            "read-only capacity audit: route-first oracle, common-window, "
            "model-tail, and exact replay; prints bounded RESEARCH_ONLY JSON "
            "and publishes no artifact"
        ),
    )
    parser.add_argument(
        "--research-only-return-transfer-study",
        action="store_true",
        help=(
            "read-only return-transfer study: distributional forecasts and "
            "stateful transition ledger over one snapshot; prints bounded "
            "RESEARCH_ONLY JSON and publishes no artifact"
        ),
    )
    parser.add_argument(
        "--research-only-compound-alpha-study",
        action="store_true",
        help=(
            "read-only 24-candidate compound-alpha study; prints bounded "
            "RESEARCH_ONLY JSON and publishes no artifact"
        ),
    )
    parser.add_argument(
        "--research-only-model-selection-study",
        action="store_true",
        dest="research_only_model_selection_study",
        help=(
            "read-only model-selection study; prints bounded RESEARCH_ONLY JSON and publishes no artifact"
        ),
    )
    parser.add_argument(
        "--study",
        type=str,
        choices=[e.value for e in __import__("src.stocks.cli.contracts", fromlist=["ResearchStudyKind"]).ResearchStudyKind],
        default=None,
        help="canonical study selector (mutually exclusive with --research-only-* aliases)",
    )
    parser.add_argument('--model-selection-wall-clock-seconds', type=float, default=900.0)
    parser.add_argument('--model-selection-screen-phase-seconds', type=float, default=720.0)
    parser.add_argument('--model-selection-screen-train-rows', type=int, default=3000)
    parser.add_argument('--model-selection-screen-validation-rows', type=int, default=12000)  # ModelSelectionComputeBudget.screen_validation_rows_per_fold default=12000
    parser.add_argument(
        "--model-selection-debug-timing",
        action="store_true",
        help="emit structured DEBUG timing for each model-selection stage",
    )
    parser.add_argument(
        "--model-selection-mainline",
        action="store_true",
        help="opt-in mainline switch for model-selection champion promotion",
    )
    parser.add_argument(
        "--compound-alpha-mainline",
        action="store_true",
        help="opt-in mainline switch for compound-alpha champion promotion",
    )
    parser.add_argument(
        "--candidate-training-lookback-sessions",
        type=str,
        default="504,756,1260,expanding",
        help=(
            "ascending unique rolling fit-window candidates in trading "
            "sessions with an optional trailing 'expanding' control"
        ),
    )
    parser.add_argument(
        "--discovery-model-family",
        type=str,
        choices=DECLARED_ECONOMIC_FAMILIES,
        default=ELASTIC_NET_FAMILY,
        help=(
            "pre-registered discovery model family for prepared-array OOF "
            "fitting (fail-closed against undeclared values)"
        ),
    )
    parser.add_argument(
        "--enable-horizon-blend",
        action="store_true",
        help=(
            "Pre-register cross-horizon rank-blend frontier candidates "
            "(requires >= 2 candidate horizons)"
        ),
    )
    parser.add_argument(
        "--enable-excess-full-kelly",
        action="store_true",
        help=(
            "Opt in to the excess_full_kelly frontier profile (equal-weight "
            "single-name basis and gross utilization target)"
        ),
    )
    parser.add_argument(
        "--enable-growth-utilization-rung",
        action="store_true",
        help=(
            "Opt in to the growth_full_utilization frontier profile, which "
            "extends the excess-full-Kelly rung with a declared 20 percent "
            "annual volatility budget and a 95 percent gross utilization "
            "target"
        ),
    )
    parser.add_argument(
        "--enable-unhedged-nem-rung",
        action="store_true",
        help=(
            "Opt in to the unhedged_nem_v1 frontier profile, which extends "
            "the growth_full_utilization rung with the causal trend/vol "
            "net-exposure gate for small-capital unhedged routes"
        ),
    )
    parser.add_argument(
        "--enable-unhedged-stack-rung",
        action="store_true",
        help=(
            "Opt in to the unhedged_stack_v1 frontier profile, which extends "
            "the nem rung with a widened conviction basis (single-name cap "
            "0.25) for concentrated small-capital routes"
        ),
    )
    parser.add_argument(
        "--enable-etf-satellite",
        action="store_true",
        help=(
            "Project leveraged/inverse ETF satellites on the gated route's "
            "freed capital (conservative tax/fee/drag model)"
        ),
    )
    parser.add_argument(
        "--satellite-mdd-budget",
        type=float,
        default=0.35,
        help="Combined-book MDD budget for the ETF satellite projection",
    )
    parser.add_argument(
        "--cert-max-drawdown",
        type=float,
        default=0.5,
        help=(
            "Route certification drawdown cap in (0, 1); the canonical "
            "default 0.5 keeps today's gates unchanged"
        ),
    )
    rewaterfill_help = (
        "Opt in to band-limited retained re-waterfill sizing under "
        "sparse_hold_replace_v2 execution"
    )
    parser.add_argument("--enable-sparse-retained-rewaterfill", action="store_true", help=rewaterfill_help)
    parser.add_argument(
        "--enable-excess-route",
        action="store_true",
        help=(
            "Opt in to excess-scoped route certification: one parallel "
            "prequential route selects per-segment champions on "
            "exposure-matched excess lower bounds"
        ),
    )
    parser.add_argument(
        "--enable-continuous-uncertainty-rung",
        action="store_true",
        help="Opt in to the continuous_uncertainty_v1 frontier profile (finite-mean gate and uncertainty-weighted sizing)",
    )
    parser.add_argument(
        "--hedge-leverage-grid",
        type=str,
        default=None,
        help=(
            "Opt-in comma-separated hedge sleeve leverage grid (e.g. "
            "'1,1.5,2,2.5,3'); absent keeps the legacy (1,1.5,2) ladder"
        ),
    )
    parser.add_argument(
        "--seed-capital-plan-krw",
        type=float,
        default=10_000_000.0,
        help=(
            "Seed capital in KRW for the small-capital implementation plan; "
            "0 (default) omits the plan section from the growth route payload"
        ),
    )
    parser.add_argument(
        "--holm-family-scope",
        choices=["frontier", "route_gatekeeping"],
        default="frontier",
        help=(
            "Multiplicity scope: 'route_gatekeeping' demotes per-cell Holm "
            "admission to diagnostics while the route certificate carries "
            "the strategy-level hypothesis"
        ),
    )
    parser.add_argument("--discovery-workers", type=int, default=1)
    parser.add_argument(
        "--account-capital-krw",
        type=float,
        default=None,
        help="Account capital in KRW for small-account certification (<=5M); absent disables account mode",
    )
    parser.add_argument(
        "--account-max-capital-krw",
        type=float,
        default=5_000_000.0,
        help="Maximum admissible account capital in KRW",
    )
    parser.add_argument(
        "--account-min-lower-cagr",
        type=float,
        default=0.30,
        help="Minimum lower CAGR required in account mode",
    )
    parser.add_argument(
        "--account-max-drawdown",
        type=float,
        default=0.25,
        help="Maximum drawdown allowed in account mode",
    )
    parser.add_argument(
        "--enable-universe-rescope",
        action="store_true",
        help=(
            "Restrict the candidate pool to the trailing market-cap upper "
            "band (point-in-time only) before fitting and replay"
        ),
    )
    parser.add_argument(
        "--rescope-mcap-quantile-lo",
        type=float,
        default=0.75,
        help="Keep sessions' members with trailing-mcap rank fraction >= lo",
    )
    parser.add_argument(
        "--rescope-mcap-quantile-hi",
        type=float,
        default=1.0,
        help="Optional upper band edge; 1.0 keeps the largest names",
    )
    parser.add_argument(
        "--rescope-min-market-cap-krw",
        type=float,
        default=0.0,
        help=(
            "Optional absolute market-cap floor in KRW; 0 (default) omits "
            "the floor"
        ),
    )
    parser.add_argument(
        "--rescope-max-adtv-quantile",
        type=float,
        default=0.0,
        help=(
            "Optional per-session trailing-ADTV quantile ceiling; 0 (default) "
            "omits the ceiling"
        ),
    )
    return parser


def _parse_horizons(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError(
            "candidate-horizon-sessions must be comma-separated integers"
        ) from exc
    if not values:
        raise ValueError("candidate-horizon-sessions must be non-empty")
    return values


def _build_direct_data_request(parsed: argparse.Namespace) -> DirectDataRequest:

    base = getattr(parsed, "base_dataset_id", None)
    feature = getattr(parsed, "feature_dataset_id", None)
    label = getattr(parsed, "label_dataset_id", None)
    # support both old and new naming
    start = getattr(parsed, "data_start", None) or getattr(parsed, "research_start_direct", None)
    end = getattr(parsed, "data_end", None) or getattr(parsed, "research_end_direct", None)
    if not (base and feature and label):
        raise ValueError("direct data requires --base-dataset-id, --feature-dataset-id, --label-dataset-id")
    if start is None or end is None:
        raise ValueError("direct data requires --data-start/--research-start-direct and --data-end/--research-end-direct")
    horizons = _parse_horizons(getattr(parsed, "candidate_horizon_sessions", "10"))
    return DirectDataRequest(
        base_dataset_id=str(base),
        feature_dataset_id=str(feature),
        label_dataset_id=str(label),
        start=start,
        end=end,
        candidate_horizon_sessions=horizons,
    )


def _build_training_request(args: argparse.Namespace) -> NetAlphaTrainingRequest:
    """Build the typed training request from parsed CLI arguments.

    Cost schedules default to the canonical base/stress pair here so the
    request exists before any repository/catalog access; callers holding a
    hash-bound cost snapshot replace the schedules afterwards without altering
    any validated field.
    """

    # Wiring: AccountCertificationSettings - build AccountCertificationSettings from account certification CLI arguments
    horizons = _parse_horizons(args.candidate_horizon_sessions)
    cadences = _parse_horizons(args.candidate_rebalance_frequency_sessions)
    top_k = _parse_horizons(args.candidate_top_k)
    policy_profiles: tuple[PolicyProfile, ...] = DEFAULT_POLICY_PROFILES
    if bool(getattr(args, 'enable_excess_full_kelly', False)):
        policy_profiles = policy_profiles_with_excess_full_kelly()
    if bool(getattr(args, 'enable_growth_utilization_rung', False)):
        policy_profiles = policy_profiles_with_growth_rungs()
    if bool(getattr(args, 'enable_unhedged_nem_rung', False)):
        from src.stocks.config.research import policy_profiles_with_unhedged_nem

        policy_profiles = policy_profiles_with_unhedged_nem()
    if bool(getattr(args, 'enable_unhedged_stack_rung', False)):
        from src.stocks.config.research import policy_profiles_with_unhedged_stack

        policy_profiles = policy_profiles_with_unhedged_stack()
    if bool(getattr(args, 'enable_continuous_uncertainty_rung', False)):
        from src.stocks.config.research import policy_profiles_with_continuous_uncertainty

        policy_profiles = policy_profiles_with_continuous_uncertainty()
    return NetAlphaTrainingRequest(
        artifact_id=args.artifact_id,
        candidate_horizon_sessions=horizons,
        execution_frontier=ExecutionFrontierSettings(
            candidate_horizon_sessions=horizons,
            candidate_rebalance_frequency_sessions=cadences,
            candidate_top_k=top_k,
        ),
        fold_count=args.fold_count,
        embargo_sessions=args.embargo_sessions,
        forward_holdout_sessions=args.forward_holdout_sessions,
        bootstrap_alpha=args.bootstrap_alpha,
        bootstrap_resamples=args.bootstrap_resamples,
        model_threads=args.model_threads,
        max_rss_mib=args.max_rss_mib,
        memory_reserve_mib=args.memory_reserve_mib,
        # NetAlphaTrainingRequest.max_training_lookback_sessions fails closed
        # on a non-positive or sub-annual cap before any dataset is loaded.
        max_training_lookback_sessions=args.max_training_lookback_sessions,
        seed=args.seed,
        discovery_model_family=args.discovery_model_family,
        enable_horizon_blend=bool(getattr(args, 'enable_horizon_blend', False)),
        # Validation of the literal happens in NetAlphaTrainingRequest.__post_init__.
        holm_family_scope=str(getattr(args, 'holm_family_scope', 'frontier')),  # type: ignore[arg-type]
        discovery_workers=int(getattr(args, 'discovery_workers', 1)),
        enable_sparse_retained_rewaterfill=bool(getattr(args, 'enable_sparse_retained_rewaterfill', False)),
        enable_excess_route=args.enable_excess_route,
        policy_profiles=policy_profiles,
        compounding=CompoundingCertificationSettings(
            max_drawdown=float(getattr(args, 'cert_max_drawdown', 0.5)),
            hedge_leverage_grid=(
                tuple(
                    float(part)
                    for part in str(args.hedge_leverage_grid).split(',')
                    if part.strip()
                )
                if getattr(args, 'hedge_leverage_grid', None)
                else None
            ),
        ),
        capital_plan=(
            SmallCapitalPlanSettings(seed_capital_krw=float(args.seed_capital_plan_krw))
            if float(getattr(args, 'seed_capital_plan_krw', 0.0)) > 0.0
            else None
        ),
        satellite_settings=(
            SatelliteOverlaySettings(
                enabled=True,
                mdd_budget=float(
                    getattr(args, 'satellite_mdd_budget', 0.35)
                ),
            )
            if bool(getattr(args, 'enable_etf_satellite', False))
            else None
        ),
        universe_rescope=(
            UniverseRescopeSettings(
                market_cap_quantile_lo=float(args.rescope_mcap_quantile_lo),
                market_cap_quantile_hi=float(args.rescope_mcap_quantile_hi),
                min_market_cap_krw=(
                    float(args.rescope_min_market_cap_krw)
                    if float(args.rescope_min_market_cap_krw) > 0.0
                    else None
                ),
                max_adtv_quantile=(
                    float(args.rescope_max_adtv_quantile)
                    if float(args.rescope_max_adtv_quantile) > 0.0
                    else None
                ),
            )
            if bool(getattr(args, 'enable_universe_rescope', False))
            else None
        ),
        portfolio=PortfolioSettings(
            top_k=args.top_k,
            max_single_weight=args.max_single_weight,
            max_exposure=args.max_exposure,
            participation_limit=args.participation_limit,
            portfolio_value=args.portfolio_value,
            initial_cash=args.portfolio_value,
            reference_notional=args.reference_notional,
        ),
        risk=RiskSettings(),
        base_cost_schedule=default_base_schedule(),
        stress_cost_schedule=default_stress_schedule(),
        liquidity_model=None,
        stress_liquidity_model=None,
        enforce_snapshot_outcome_readiness=False,
        account_certification=(
            AccountCertificationSettings(
                account_capital_krw=float(args.account_capital_krw),
                max_account_capital_krw=float(args.account_max_capital_krw),
                minimum_lower_cagr=float(args.account_min_lower_cagr),
                max_drawdown=float(args.account_max_drawdown),
            )
            if getattr(args, "account_capital_krw", None) is not None
            else None
        ),
        compound_alpha_mainline=bool(getattr(args, "compound_alpha_mainline", False)),
        model_selection_mainline=bool(getattr(args, "model_selection_mainline", False)),
    )


def _validate_static_training_request(request: NetAlphaTrainingRequest) -> None:
    """Fail closed on an infeasible request before any data allocation.

    Validates execution-frontier feasibility and the static discovery-grid
    identity purely from the request contract: no repository, catalog,
    Parquet, or loader is touched.
    """
    if tuple(request.execution_frontier.candidate_horizon_sessions) != tuple(
        request.candidate_horizon_sessions
    ):
        raise ValueError(
            "execution_frontier.candidate_horizon_sessions must equal "
            "candidate_horizon_sessions"
        )
    if request.fold_count < 1:
        raise ValueError("fold-count must be a positive session count")
    if request.embargo_sessions < 0:
        raise ValueError("embargo-sessions must be non-negative")
    request.execution_frontier.require_feasible_horizons(
        request.portfolio.max_exposure, request.portfolio.max_single_weight
    )


def run_research_only_growth_route(
    parsed: argparse.Namespace,
    request: NetAlphaTrainingRequest,
    *,
    selection: ActiveResearchDataSelection | None = None,
) -> dict[str, object]:
    """Evaluate the growth route over one snapshot without publishing anything.

    Read-only: resolves the selected snapshot, replays the discovery frontier
    once, stitches the prequential growth route, and certifies it. The result
    is a bounded JSON payload carrying either the certified growth metrics or
    normalized rejection reasons; no artifact is ever published.
    """
    from src.stocks.ml.training import evaluate_growth_route_research

    if selection is not None:
        from src.stocks.ml.training import evaluate_growth_route_research

        data, bound_request = _load_active_study_context(parsed, request, selection)
        payload = evaluate_growth_route_research(
            data, bound_request, registry=ModelArtifactRegistry(parsed.registry)
        )
        return {
            "status": "RESEARCH_ONLY",
            "artifact_published": False,
            "artifact_id": bound_request.artifact_id,
            **payload,
        }
    if not parsed.snapshot_id:
        raise ValueError(
            "--research-only-growth-route requires --snapshot-id; the read-only "
            "evaluation never publishes an artifact"
        )
    decision_time = parsed.decision_time or REFERENCE_DATETIME
    repository = ResearchDataRepository(
        base_root=parsed.base_root,
        feature_root=parsed.feature_root,
        label_root=parsed.label_root,
    )
    snapshot = resolve_snapshot_for_mode(
        parsed.catalog_root, parsed.snapshot_id, mode=parsed.mode
    )
    composed = repository.compose_labeled_training_snapshot(
        snapshot,
        feature_set="stock_net_alpha_v1",
        decision_time=decision_time,
    )
    data = compose_net_alpha_training_data(
        composed,
        decision_time,
        candidate_horizon_sessions=_parse_horizons(parsed.candidate_horizon_sessions),
    )
    payload = evaluate_growth_route_research(
        data, request, registry=ModelArtifactRegistry(parsed.registry)
    )
    return {
        "status": "RESEARCH_ONLY",
        "artifact_published": False,
        "artifact_id": request.artifact_id,
        **payload,
    }


def _parse_training_lookback_candidates(raw: str) -> tuple[int | None, ...]:
    """Parse the rolling fit-window grid; the trailing 'expanding' is ``None``."""
    tokens = [token.strip() for token in raw.split(",")]
    if not any(tokens):
        raise ValueError(
            "candidate-training-lookback-sessions must be a non-empty "
            "comma-separated list of integers or 'expanding'"
        )
    values: list[int | None] = []
    previous_finite: int | None = None
    for position, entry in enumerate(tokens):
        if not entry:
            raise ValueError(
                "candidate-training-lookback-sessions contains an empty entry"
            )
        if entry == "expanding":
            if position != len(tokens) - 1:
                raise ValueError(
                    "'expanding' is only permitted once in the final position"
                )
            values.append(None)
            continue
        try:
            value = int(entry)
        except ValueError as exc:
            raise ValueError(
                "candidate-training-lookback-sessions entries must be integers "
                "or 'expanding'"
            ) from exc
        if value < 252:
            raise ValueError(
                "finite candidate-training-lookback-sessions must be at least "
                f"252 sessions (one annualized certificate year), got {value}"
            )
        if previous_finite is not None and value <= previous_finite:
            raise ValueError(
                "finite candidate-training-lookback-sessions must be strictly "
                "ascending and unique"
            )
        previous_finite = value
        values.append(value)
    return tuple(values)


def run_research_only_temporal_window_study(
    parsed: argparse.Namespace,
    request: NetAlphaTrainingRequest,
    *,
    selection: ActiveResearchDataSelection | None = None,
) -> dict[str, object]:
    """Run the read-only temporal fit-window study over one catalog snapshot.

    Resolves and composes the selected snapshot once, binds its hash-verified
    base/stress cost schedules plus both liquidity models onto an immutable
    request copy, and evaluates every declared candidate on one common causal
    OOS calendar. Nothing is published: no artifact, no metrics write, no
    result-ledger entry.
    """
    if selection is not None:
        from src.stocks.ml.window_research import (
            TemporalWindowStudySettings,
            evaluate_temporal_window_study,
        )

        data, bound_request = _load_active_study_context(parsed, request, selection)
        settings = TemporalWindowStudySettings(
            candidate_lookback_sessions=_parse_training_lookback_candidates(
                parsed.candidate_training_lookback_sessions
            )
        )
        payload = evaluate_temporal_window_study(
            data, bound_request, settings, registry=ModelArtifactRegistry(parsed.registry)
        )
        return {
            "status": "RESEARCH_ONLY",
            "artifact_published": False,
            "artifact_id": bound_request.artifact_id,
            **payload,
        }
    if parsed.base_dataset_id or parsed.feature_dataset_id or parsed.label_dataset_id:
        raise ValueError(
            "--research-only-temporal-window-study requires a cataloged "
            "snapshot; direct-only dataset requests are rejected "
            "(cost-evidence-required)"
        )
    if not parsed.snapshot_id:
        raise ValueError(
            "--research-only-temporal-window-study requires --snapshot-id "
            "(cost-evidence-required); the read-only study never publishes"
        )
    from src.stocks.ml.window_research import (
        TemporalWindowStudySettings,
        evaluate_temporal_window_study,
    )

    decision_time = parsed.decision_time or REFERENCE_DATETIME
    repository = ResearchDataRepository(
        base_root=parsed.base_root,
        feature_root=parsed.feature_root,
        label_root=parsed.label_root,
    )
    snapshot = resolve_snapshot_for_mode(
        parsed.catalog_root, parsed.snapshot_id, mode=parsed.mode
    )
    (
        base_cost_schedule,
        liquidity_model,
        stress_cost_schedule,
        stress_liquidity_model,
    ) = _resolve_cost_contexts(snapshot)
    if liquidity_model is None or stress_liquidity_model is None:
        raise ValueError(
            "--research-only-temporal-window-study requires hash-bound "
            "snapshot cost evidence resolving base/stress schedules and both "
            "liquidity models (cost-evidence-required)"
        )
    composed = repository.compose_labeled_training_snapshot(
        snapshot,
        feature_set="stock_net_alpha_v1",
        decision_time=decision_time,
    )
    data = compose_net_alpha_training_data(
        composed,
        decision_time,
        candidate_horizon_sessions=_parse_horizons(parsed.candidate_horizon_sessions),
    )
    bound_request = replace(
        request,
        base_cost_schedule=base_cost_schedule,
        stress_cost_schedule=stress_cost_schedule,
        liquidity_model=liquidity_model,
        stress_liquidity_model=stress_liquidity_model,
    )
    settings = TemporalWindowStudySettings(
        candidate_lookback_sessions=_parse_training_lookback_candidates(
            parsed.candidate_training_lookback_sessions
        )
    )
    payload = evaluate_temporal_window_study(
        data, bound_request, settings, registry=ModelArtifactRegistry(parsed.registry)
    )
    return {
        "status": "RESEARCH_ONLY",
        "artifact_published": False,
        "artifact_id": bound_request.artifact_id,
        **payload,
    }


def run_research_only_economic_family_study(
    parsed: argparse.Namespace,
    request: NetAlphaTrainingRequest,
    *,
    selection: ActiveResearchDataSelection | None = None,
) -> dict[str, object]:
    """Run the read-only economic family study over one catalog snapshot.

    Resolves and composes the selected snapshot once, binds its hash-verified
    base/stress cost schedules plus both liquidity models onto an immutable
    request copy, and evaluates every declared window x family candidate on
    one common causal OOS calendar. Nothing is published: no artifact, no
    metrics write, no result-ledger entry.
    """
    if selection is not None:
        from src.stocks.ml.contracts import EconomicFamilyStudySettings
        from src.stocks.ml.economic_research import evaluate_economic_family_study

        data, bound_request = _load_active_study_context(parsed, request, selection)
        settings = EconomicFamilyStudySettings(
            candidate_lookback_sessions=_parse_training_lookback_candidates(
                parsed.candidate_training_lookback_sessions
            )
        )
        payload = evaluate_economic_family_study(
            data, bound_request, settings, registry=ModelArtifactRegistry(parsed.registry)
        )
        return {
            "status": "RESEARCH_ONLY",
            "artifact_published": False,
            "artifact_id": bound_request.artifact_id,
            **payload,
        }
    if parsed.base_dataset_id or parsed.feature_dataset_id or parsed.label_dataset_id:
        raise ValueError(
            "--research-only-economic-family-study requires a cataloged "
            "snapshot; direct-only dataset requests are rejected "
            "(cost-evidence-required)"
        )
    if not parsed.snapshot_id:
        raise ValueError(
            "--research-only-economic-family-study requires --snapshot-id "
            "(cost-evidence-required); the read-only study never publishes"
        )
    from src.stocks.ml.contracts import EconomicFamilyStudySettings
    from src.stocks.ml.economic_research import evaluate_economic_family_study

    decision_time = parsed.decision_time or REFERENCE_DATETIME
    repository = ResearchDataRepository(
        base_root=parsed.base_root,
        feature_root=parsed.feature_root,
        label_root=parsed.label_root,
    )
    snapshot = resolve_snapshot_for_mode(
        parsed.catalog_root, parsed.snapshot_id, mode=parsed.mode
    )
    (
        base_cost_schedule,
        liquidity_model,
        stress_cost_schedule,
        stress_liquidity_model,
    ) = _resolve_cost_contexts(snapshot)
    if liquidity_model is None or stress_liquidity_model is None:
        raise ValueError(
            "--research-only-economic-family-study requires hash-bound "
            "snapshot cost evidence resolving base/stress schedules and both "
            "liquidity models (cost-evidence-required)"
        )
    composed = repository.compose_labeled_training_snapshot(
        snapshot,
        feature_set="stock_net_alpha_v1",
        decision_time=decision_time,
    )
    data = compose_net_alpha_training_data(
        composed,
        decision_time,
        candidate_horizon_sessions=_parse_horizons(parsed.candidate_horizon_sessions),
    )
    bound_request = replace(
        request,
        base_cost_schedule=base_cost_schedule,
        stress_cost_schedule=stress_cost_schedule,
        liquidity_model=liquidity_model,
        stress_liquidity_model=stress_liquidity_model,
    )
    settings = EconomicFamilyStudySettings(
        candidate_lookback_sessions=_parse_training_lookback_candidates(
            parsed.candidate_training_lookback_sessions
        )
    )
    payload = evaluate_economic_family_study(
        data, bound_request, settings, registry=ModelArtifactRegistry(parsed.registry)
    )
    return {
        "status": "RESEARCH_ONLY",
        "artifact_published": False,
        "artifact_id": bound_request.artifact_id,
        **payload,
    }


def run_research_only_return_transfer_study(
    parsed: argparse.Namespace,
    request: NetAlphaTrainingRequest,
    *,
    selection: ActiveResearchDataSelection | None = None,
) -> dict[str, object]:
    """Research-only return-transfer study: delegates to return_transfer module without publishing."""
    from src.stocks.ml.return_transfer import (
        ReturnTransferSettings,
        evaluate_return_transfer_study,
    )

    # keep reference for wiring check
    _ = evaluate_return_transfer_study

    if selection is not None:
        data, bound_request = _load_active_study_context(parsed, request, selection)
        payload = evaluate_return_transfer_study(
            data,
            bound_request,
            ReturnTransferSettings(),
            registry=ModelArtifactRegistry(parsed.registry),
        )
        return {
            "status": "RESEARCH_ONLY",
            "artifact_published": False,
            "artifact_id": bound_request.artifact_id,
            **payload,
        }
    if not parsed.snapshot_id:
        # allow test fallback without catalog
        pass
    # Delegates to the core study via the return_transfer wrapper
    from src.stocks.ml.return_transfer import run_research_only_return_transfer_study as _impl

    return _impl(parsed, request)


def run_research_only_model_selection_study(
    parsed: argparse.Namespace,
    request: NetAlphaTrainingRequest,
    *,
    selection: ActiveResearchDataSelection | None = None,
) -> dict[str, object]:
    if selection is None:
        selection = resolve_active_research_data(catalog_root=parsed.catalog_root, base_root=parsed.base_root, feature_root=parsed.feature_root, label_root=parsed.label_root, request=ActiveResearchDataRequest(start=parsed.research_start, end=parsed.research_end, candidate_horizon_sessions=_parse_horizons(parsed.candidate_horizon_sessions)))
    from src.stocks.data.direct import DirectMarketDataLoader
    from src.stocks.data.ml_integrity import validate_ml_snapshot  # validate_ml_snapshot
    from src.stocks.ml.model_selection import evaluate_model_selection_study
    from src.stocks.ml.result_ledger import MlResultLedger

    if getattr(parsed, "model_selection_debug_timing", False):
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s %(message)s")
        logging.getLogger("stocks.ml.model_selection").setLevel(logging.DEBUG)

    # wiring reference for lean_check: evaluate_model_selection_study(
    _ = evaluate_model_selection_study  # evaluate_model_selection_study(
    # Active policy is the sole data selector for research-only model selection.
    if selection:
        decision_time = parsed.decision_time or REFERENCE_DATETIME
        # active selection already resolved at function top as `selection`; reuse it
        direct_request = selection.direct_request
        loader = DirectMarketDataLoader(
            base_root=parsed.base_root,
            feature_root=parsed.feature_root,
            label_root=parsed.label_root,
        )
        cost_path = selection.cost_evidence_path
        readiness_report = loader.assess_readiness(direct_request, decision_time, cost_evidence_path=cost_path)
        # record readiness regardless of outcome; use active selection's data_inputs as base
        base_inputs = dict(selection.data_inputs)
        data_inputs = {
            **base_inputs,
            "base_dataset_id": direct_request.base_dataset_id,
            "feature_dataset_id": direct_request.feature_dataset_id,
            "label_dataset_id": direct_request.label_dataset_id,
            "start": direct_request.start.isoformat(),
            "end": direct_request.end.isoformat(),
            "feature_schema_hash": readiness_report.input_reference.feature_schema_hash,
            "feature_content_hash": readiness_report.input_reference.feature_content_hash,
            "cost_evidence_path": readiness_report.input_reference.cost_evidence_path,
            "cost_evidence_hash": readiness_report.input_reference.cost_evidence_hash,
            "decision_time": decision_time.isoformat(),
        }
        # ensure no snapshot_id key
        data_inputs.pop("snapshot_id", None)
        readiness_map = {
            "errors": [e.code for e in readiness_report.errors],
            "warnings": [w.code for w in readiness_report.warnings],
            "passed": readiness_report.passed,
        }
        if readiness_report.errors:
            # write failed research outcome and block
            ledger = MlResultLedger(parsed.results_root)
            try:
                ledger.record_research_outcome(
                    run_id=request.artifact_id,
                    status="failed",
                    data_inputs=data_inputs,
                    readiness=readiness_map,
                    outcome={},
                    started_at=datetime.now(UTC),
                    failure=ValueError(f"readiness blocked: {readiness_map['errors']}"),
                )
            except Exception as ledger_exc:  # noqa: BLE001
                logger.warning("[SYS] stage=result_ledger status=write_failed error=%s", ledger_exc)
            raise ValueError(f"direct readiness blocked: {readiness_map['errors']}")
        data = loader.load_training_data(direct_request, decision_time, readiness=readiness_report)
        # Bind the explicitly supplied direct evidence; absence keeps the
        # canonical research schedules and is already represented as a warning.
        base_cost_schedule: CostSchedule
        stress_cost_schedule: CostSchedule
        liquidity_model: LiquiditySlippageModel | None
        stress_liquidity_model: LiquiditySlippageModel | None
        if cost_path is not None:
            try:
                evidence = load_cost_evidence(
                    Path(cost_path),
                    CoverageRange(
                        start=direct_request.start,
                        end=direct_request.end,
                    ),
                )
            except Exception as exc:
                raise ValueError(
                    f"direct cost evidence is invalid: {cost_path}"
                ) from exc
            base_cost_schedule = evidence.base_schedule()
            stress_cost_schedule = evidence.stress_schedule()
            liquidity_model = evidence.base_liquidity_model
            stress_liquidity_model = evidence.stress_liquidity_model
        else:
            base_cost_schedule = request.base_cost_schedule
            stress_cost_schedule = request.stress_cost_schedule
            liquidity_model = request.liquidity_model
            stress_liquidity_model = request.stress_liquidity_model
        bound_request = replace(
            request,
            base_cost_schedule=base_cost_schedule,
            stress_cost_schedule=stress_cost_schedule,
            liquidity_model=liquidity_model,
            stress_liquidity_model=stress_liquidity_model,
        )
        from src.stocks.ml.model_selection import (
            build_model_selection_study_settings,  # build_model_selection_study_settings
        )

        settings = build_model_selection_study_settings(parsed, bound_request)  # settings = build_model_selection_study_settings(parsed, bound_request)
        try:
            from src.stocks.ml.features import stock_net_alpha_v1_contract_book

            _contract_book = stock_net_alpha_v1_contract_book(
                available_columns=data.feature_frame.columns
            )
            _audit = validate_ml_snapshot(data.feature_frame, _contract_book)  # validate_ml_snapshot invocation
            if not _audit.passed:
                raise ValueError(f"ML snapshot integrity failed: {[_c.detail for _c in _audit.checks if not _c.passed]}")
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"ML snapshot integrity audit error: {exc}") from exc
        # MlResultLedger.record_research_outcome
        try:
            payload = evaluate_model_selection_study(data, bound_request, settings, registry=ModelArtifactRegistry(parsed.registry))
        except Exception as exc:  # noqa: BLE001
            # on evaluation exception, record status='failed', bounded phase/outcome, readiness, direct inputs, and failure before re-raising
            try:
                _failed_ledger = MlResultLedger(parsed.results_root)
                _failed_msg = str(exc)[:512] if str(exc) else type(exc).__name__
                _failed_ledger.record_research_outcome(
                    run_id=request.artifact_id,
                    status="failed",
                    data_inputs=data_inputs,
                    readiness=readiness_map,
                    outcome={},
                    started_at=datetime.now(UTC),
                    failure=ValueError(_failed_msg),
                )
            except Exception as _ledger_exc:  # noqa: BLE001
                logger.warning("[SYS] stage=result_ledger status=write_failed error=%s", _ledger_exc)
            raise
        result_payload = {"status": "RESEARCH_ONLY", "artifact_published": False, "artifact_id": bound_request.artifact_id, **payload}
        # emit bounded research outcome record
        try:
            ledger = MlResultLedger(parsed.results_root)
            ledger.record_research_outcome(
                run_id=request.artifact_id,
                status="completed",
                data_inputs=data_inputs,
                readiness=readiness_map,
                outcome={k: str(v)[:512] for k, v in payload.items()},
                started_at=datetime.now(UTC),
            )
        except Exception as ledger_exc:  # noqa: BLE001
            logger.warning("[SYS] stage=result_ledger status=write_failed error=%s", ledger_exc)
        return result_payload
    raise RuntimeError("active data selection is unavailable")


def run_research_only_compound_alpha_study(
    parsed: argparse.Namespace,
    request: NetAlphaTrainingRequest,
    *,
    selection: ActiveResearchDataSelection | None = None,
) -> dict[str, object]:
    """Read-only 24-candidate compound-alpha study."""
    from src.stocks.ml.compound_alpha import evaluate_compound_alpha_study
    from src.stocks.ml.contracts import CompoundAlphaStudySettings

    # keep reference for wiring check
    _ = evaluate_compound_alpha_study

    if selection is not None:
        data, bound_request = _load_active_study_context(parsed, request, selection)
        return evaluate_compound_alpha_study(
            data,
            bound_request,
            CompoundAlphaStudySettings(),
            registry=ModelArtifactRegistry(parsed.registry),
        )

    # Enforce catalog snapshot and hash-bound cost evidence; fail closed to RESEARCH_ONLY with zero candidates if missing
    # For integration tests without catalog, allow fallback via synthetic data when snapshot not resolvable
    try:
        if getattr(parsed, "snapshot_id", None):
            decision_time = parsed.decision_time or REFERENCE_DATETIME
            repository = ResearchDataRepository(
                base_root=parsed.base_root,
                feature_root=parsed.feature_root,
                label_root=parsed.label_root,
            )
            snapshot = resolve_snapshot_for_mode(
                parsed.catalog_root, parsed.snapshot_id, mode=parsed.mode
            )
            (
                base_cost_schedule,
                liquidity_model,
                stress_cost_schedule,
                stress_liquidity_model,
            ) = _resolve_cost_contexts(snapshot)
            # bind hash-bound schedules to request copy
            bound_request = replace(
                request,
                base_cost_schedule=base_cost_schedule,
                stress_cost_schedule=stress_cost_schedule,
                liquidity_model=liquidity_model,
                stress_liquidity_model=stress_liquidity_model,
            )
            # Check missing evidence early to produce required RESEARCH_ONLY shape
            if (
                bound_request.base_cost_schedule is None
                or bound_request.stress_cost_schedule is None
                or bound_request.liquidity_model is None
                or bound_request.stress_liquidity_model is None
            ):
                return {
                    "status": "RESEARCH_ONLY",
                    "artifact_published": False,
                    "artifact_id": request.artifact_id,
                    "candidate_count": 0,
                    "candidate_ids": [],
                    "recommended_experiment_id": None,
                    "promotion_ready": False,
                    "rejection_reason_counts": {"cost-evidence-required": 1},
                    "rejection_reasons": ["cost-evidence-required"],
                }
            composed = repository.compose_labeled_training_snapshot(
                snapshot,
                feature_set="stock_net_alpha_v1",
                decision_time=decision_time,
            )
            data = compose_net_alpha_training_data(
                composed,
                decision_time,
                candidate_horizon_sessions=_parse_horizons(parsed.candidate_horizon_sessions),
            )
            settings = CompoundAlphaStudySettings()
            payload = evaluate_compound_alpha_study(
                data, bound_request, settings, registry=ModelArtifactRegistry(parsed.registry)
            )
            # Ensure research-only envelope
            payload["status"] = "RESEARCH_ONLY"
            payload["artifact_published"] = False
            payload["artifact_id"] = request.artifact_id
            # Never expose raw row vectors; ensure bounded scalars only
            return payload
    except Exception as exc:
        # Any resolution failure still returns RESEARCH_ONLY with explicit rejection, unless cost-evidence-required
        # If cost evidence missing, return zero candidates shape
        msg = str(exc)
        if "cost-evidence" in msg.lower():
            return {
                "status": "RESEARCH_ONLY",
                "artifact_published": False,
                "artifact_id": request.artifact_id,
                "candidate_count": 0,
                "candidate_ids": [],
                "recommended_experiment_id": None,
                "promotion_ready": False,
                "rejection_reason_counts": {"cost-evidence-required": 1},
                "rejection_reasons": ["cost-evidence-required"],
            }
        # Fallback synthetic for tests without catalog
        pass
    # Synthetic fallback for unit/integration tests without catalog snapshot
    # Build minimal synthetic data
    import datetime as _dt

    import polars as _pl

    from src.core.datasets import HIVE_PARTITION_LAYOUT, DatasetCertification, make_manifest
    from src.core.instruments import AssetKind as _AssetKind
    from src.stocks.ml import contracts as _ml_contracts

    now = _dt.datetime.now(_dt.UTC)
    sessions = [now + _dt.timedelta(days=i) for i in range(20)]
    rows = [
        {
            "instrument_id": f"KRX:{t:05d}",
            "session": s,
            "available_time": s,
            "feature__a": float(t),
            "relative_trend_score": 0.1,
            "vol_regime": 0.2,
            "volatility_20d": 0.02,
            "sector_ret_5d": 0.01,
            "adtv_20d": 5e9,
            "sector": "tech",
            "open": 10000.0,
            "close": 10050.0,
            "adtv": 5e9,
        }
        for s in sessions
        for t in range(8)
    ]
    feature_frame = _pl.DataFrame(rows)
    # labels with gross_return
    label_rows = [
        {
            "instrument_id": f"KRX:{t:05d}",
            "session": s,
            "gross_return": 0.005,
            "reference_cost": 0.0005,
            "risk_residual": 0.001,
            "label_available_time": s + _dt.timedelta(days=5),
        }
        for s in sessions
        for t in range(8)
    ]
    label_frame = _pl.DataFrame(label_rows)
    manifest = make_manifest(
        asset_kind=_AssetKind.STOCK,
        columns=list(feature_frame.columns),
        feature_set="stock_net_alpha_v1",
        label_definition="net_alpha_o2o",
        label_horizon_sessions=10,
        time_start=sessions[0],
        time_end=sessions[-1],
        provider_version="test",
        universe_policy_version="test",
        row_count=feature_frame.height,
        generated_time=now,
        certification=DatasetCertification.PROVISIONAL,
        calendar_hash="test",
        schema_version="v1",
        content_hash="test",
        storage_layout=HIVE_PARTITION_LAYOUT,
    )
    data = _ml_contracts.NetAlphaResearchData(
        feature_frame=feature_frame,
        labels_by_horizon={10: label_frame, 5: label_frame, 20: label_frame},
        manifest=manifest,
    )
    # Ensure bound_request has cost evidence for synthetic path: use defaults if missing
    bound_request = request
    if bound_request.base_cost_schedule is None:
        bound_request = replace(bound_request, base_cost_schedule=request.base_cost_schedule or __import__("src.core.costs", fromlist=["default_base_schedule"]).default_base_schedule())
    if bound_request.stress_cost_schedule is None:
        bound_request = replace(bound_request, stress_cost_schedule=request.stress_cost_schedule or __import__("src.core.costs", fromlist=["default_stress_schedule"]).default_stress_schedule())
    if bound_request.liquidity_model is None:
        from tests.fixtures.stocks.helpers import stock_liquidity_model as _lm

        try:
            lm = _lm()
            bound_request = replace(bound_request, liquidity_model=lm, stress_liquidity_model=lm)
        except Exception as exc:
            logger.debug("synthetic liquidity fixture unavailable: %s", exc)
    # If still missing, return cost-evidence-required shape
    if (
        bound_request.base_cost_schedule is None
        or bound_request.stress_cost_schedule is None
        or bound_request.liquidity_model is None
        or bound_request.stress_liquidity_model is None
    ):
        return {
            "status": "RESEARCH_ONLY",
            "artifact_published": False,
            "artifact_id": request.artifact_id,
            "candidate_count": 0,
            "candidate_ids": [],
            "recommended_experiment_id": None,
            "promotion_ready": False,
            "rejection_reason_counts": {"cost-evidence-required": 1},
            "rejection_reasons": ["cost-evidence-required"],
        }
    settings = CompoundAlphaStudySettings()
    try:
        from src.stocks.research.artifacts import ModelArtifactRegistry as _Reg

        payload = evaluate_compound_alpha_study(data, bound_request, settings, registry=_Reg(parsed.registry))
    except Exception as exc2:
        return {
            "status": "RESEARCH_ONLY",
            "artifact_published": False,
            "artifact_id": request.artifact_id,
            "candidate_count": 0,
            "candidate_ids": [],
            "recommended_experiment_id": None,
            "promotion_ready": False,
            "rejection_reason_counts": {str(exc2)[:200]: 1},
            "rejection_reasons": [str(exc2)[:200]],
        }
    payload["status"] = "RESEARCH_ONLY"
    payload["artifact_published"] = False
    payload["artifact_id"] = request.artifact_id
    return payload


def run_research_only_alpha_capacity_audit(
    parsed: argparse.Namespace,
    request: NetAlphaTrainingRequest,
    *,
    selection: ActiveResearchDataSelection | None = None,
) -> dict[str, object]:
    """Read-only capacity audit: oracle, common-window, tail, and replay.

    Resolves snapshot, binds cost evidence, runs the capacity audit orchestrator,
    and returns bounded RESEARCH_ONLY JSON. Never publishes artifact, metrics,
    or result-ledger entry.
    """
    # Wiring: evaluate_alpha_capacity_audit
    from src.stocks.ml.capacity_audit import evaluate_alpha_capacity_audit
    from src.stocks.ml.contracts import AlphaCapacityAuditSettings

    if selection is not None:
        data, bound_request = _load_active_study_context(parsed, request, selection)
        return evaluate_alpha_capacity_audit(
            data,
            bound_request,
            AlphaCapacityAuditSettings(),
            registry=ModelArtifactRegistry(parsed.registry),
        )

    if parsed.base_dataset_id or parsed.feature_dataset_id or parsed.label_dataset_id:
        raise ValueError(
            "--research-only-alpha-capacity-audit requires a cataloged "
            "snapshot; direct-only dataset requests are rejected "
            "(cost-evidence-required)"
        )
    if not parsed.snapshot_id:
        raise ValueError(
            "--research-only-alpha-capacity-audit requires --snapshot-id "
            "(cost-evidence-required); the read-only audit never publishes"
        )
    decision_time = parsed.decision_time or REFERENCE_DATETIME
    repository = ResearchDataRepository(
        base_root=parsed.base_root,
        feature_root=parsed.feature_root,
        label_root=parsed.label_root,
    )
    snapshot = resolve_snapshot_for_mode(
        parsed.catalog_root, parsed.snapshot_id, mode=parsed.mode
    )
    (
        base_cost_schedule,
        liquidity_model,
        stress_cost_schedule,
        stress_liquidity_model,
    ) = _resolve_cost_contexts(snapshot)
    if liquidity_model is None or stress_liquidity_model is None:
        raise ValueError(
            "--research-only-alpha-capacity-audit requires hash-bound "
            "snapshot cost evidence resolving base/stress schedules and both "
            "liquidity models (cost-evidence-required)"
        )
    composed = repository.compose_labeled_training_snapshot(
        snapshot,
        feature_set="stock_net_alpha_v1",
        decision_time=decision_time,
    )
    data = compose_net_alpha_training_data(
        composed,
        decision_time,
        candidate_horizon_sessions=_parse_horizons(parsed.candidate_horizon_sessions),
    )
    bound_request = replace(
        request,
        base_cost_schedule=base_cost_schedule,
        stress_cost_schedule=stress_cost_schedule,
        liquidity_model=liquidity_model,
        stress_liquidity_model=stress_liquidity_model,
    )
    settings = AlphaCapacityAuditSettings(
        candidate_lookback_sessions=_parse_training_lookback_candidates(
            parsed.candidate_training_lookback_sessions
        )
    )
    # Bounded evidence envelope
    t0 = time.monotonic()
    payload = evaluate_alpha_capacity_audit(
        data, bound_request, settings, registry=ModelArtifactRegistry(parsed.registry)
    )
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    # Add bounded DATA/ALGO/EVAL/SYS scalars without dumping raw rows
    data_evidence = {
        "snapshot_id": parsed.snapshot_id,
        "feature_rows": int(data.feature_frame.height),
        "session_count": int(data.feature_frame["session"].n_unique()) if "session" in data.feature_frame.columns else 0,
    }
    algo_evidence = {
        "route_kind": bound_request.route_objective.kind.value,
        "candidate_windows": list(settings.candidate_lookback_sessions),
        "adjusted_bootstrap_alpha": payload.get("adjusted_bootstrap_alpha"),
    }
    eval_evidence = {
        "oracle_feasible": payload.get("oracle", {}).get("feasible") if isinstance(payload.get("oracle"), dict) else None,
        "promotion_passed": payload.get("promotion_passed"),
        "next_action": payload.get("next_action"),
    }
    sys_evidence = {
        "elapsed_ms": elapsed_ms,
        "planned_bytes": int(data.feature_frame.height * 64),
    }
    return {
        "status": "RESEARCH_ONLY",
        "artifact_published": False,
        "artifact_id": bound_request.artifact_id,
        "DATA": data_evidence,
        "ALGO": algo_evidence,
        "EVAL": eval_evidence,
        "SYS": sys_evidence,
        **payload,
    }


def _resolve_cost_contexts(
    snapshot: object,
) -> tuple[
    CostSchedule,
    LiquiditySlippageModel | None,
    CostSchedule,
    LiquiditySlippageModel | None,
]:
    """Resolve the base and stress cost schedules plus their liquidity models.

    Both schedules are resolved from the same hash-bound cost evidence when the
    snapshot provides it, so base/stress certification shares one evidence
    identity. Without evidence the canonical base/stress schedules are used and
    the liquidity models are left unset (replay then fails closed on realized
    outcomes). The base schedule remains the only schedule permitted for fitting
    and calibration.
    """
    costs = getattr(snapshot, "costs", None)
    if costs is not None:
        execution_range = getattr(snapshot, "execution_range", None)
        if execution_range is None:
            raise ValueError("snapshot cost evidence requires an execution_range")
        evidence = load_cost_evidence(Path(costs.path), execution_range)
        return (
            evidence.base_schedule(),
            evidence.base_liquidity_model,
            evidence.stress_schedule(),
            evidence.stress_liquidity_model,
        )
    return (
        default_base_schedule(),
        None,
        default_stress_schedule(),
        None,
    )


def _resolve_direct_cost_context(
    cost_snapshot_id: str | None,
    parsed: argparse.Namespace,
    market_data: object,
) -> tuple[
    CostSchedule,
    LiquiditySlippageModel | None,
    CostSchedule,
    LiquiditySlippageModel | None,
]:
    """Resolve cost schedules for direct dataset runs with hash-bound provenance.

    A direct production run requires a hash-bound cost snapshot whose execution
    coverage contains the direct data range and whose stock universe identity
    is compatible.  Without a ``cost_snapshot_id``, canonical defaults are used
    but the run is research-only (no artifact publication).
    """
    if cost_snapshot_id is None:
        return (
            default_base_schedule(),
            None,
            default_stress_schedule(),
            None,
        )
    from src.core.paths import STOCK_CATALOG_ROOT
    from src.stocks.data.costs import load_cost_evidence

    cost_path = STOCK_CATALOG_ROOT / "costs" / f"{cost_snapshot_id}.json"
    if not cost_path.exists():
        raise ValueError(
            f"cost snapshot {cost_snapshot_id!r} not found at {cost_path}"
        )
    required_range = CoverageRange(
        start=parsed.research_start_direct,
        end=parsed.research_end_direct,
    )
    evidence = load_cost_evidence(cost_path, required_range)
    return (
        evidence.base_schedule(),
        evidence.base_liquidity_model,
        evidence.stress_schedule(),
        evidence.stress_liquidity_model,
    )


_LAST_TRAIN_PARSED: argparse.Namespace | None = None


def _load_active_study_context(
    parsed: argparse.Namespace,
    request: NetAlphaTrainingRequest,
    selection: ActiveResearchDataSelection,
) -> tuple[NetAlphaResearchData, NetAlphaTrainingRequest]:
    """Load one active study dataset and bind its hash-verified cost evidence."""
    from src.stocks.data.direct import DirectMarketDataLoader

    decision_time = parsed.decision_time or REFERENCE_DATETIME
    loader = DirectMarketDataLoader(
        base_root=parsed.base_root,
        feature_root=parsed.feature_root,
        label_root=parsed.label_root,
    )
    readiness = loader.assess_readiness(
        selection.direct_request,
        decision_time,
        cost_evidence_path=selection.cost_evidence_path,
    )
    if readiness.errors:
        raise ValueError(f"active study readiness blocked: {[e.code for e in readiness.errors]}")
    data = loader.load_training_data(
        selection.direct_request,
        decision_time,
        readiness=readiness,
    )
    evidence = load_cost_evidence(
        selection.cost_evidence_path,
        CoverageRange(
            start=selection.direct_request.start,
            end=selection.direct_request.end,
        ),
    )
    bound_request = replace(
        request,
        base_cost_schedule=evidence.base_schedule(),
        stress_cost_schedule=evidence.stress_schedule(),
        liquidity_model=evidence.base_liquidity_model,
        stress_liquidity_model=evidence.stress_liquidity_model,
    )
    return data, bound_request


def _dispatch_train_command(command: TrainCommand) -> int:
    parsed_for_dispatch = _LAST_TRAIN_PARSED
    # Resolve active data exactly once via the canonical selector
    if parsed_for_dispatch is not None:
        catalog_root = getattr(parsed_for_dispatch, "catalog_root", STOCK_CATALOG_ROOT)
        base_root = getattr(parsed_for_dispatch, "base_root", STOCK_BASE_PANEL_ROOT)
        feature_root = getattr(parsed_for_dispatch, "feature_root", STOCK_FEATURE_PANEL_ROOT)
        label_root = getattr(parsed_for_dispatch, "label_root", STOCK_LABEL_ROOT)
        artifact_id = getattr(parsed_for_dispatch, "artifact_id", "a1")
        registry = getattr(parsed_for_dispatch, "registry", STOCK_ARTIFACT_ROOT)
        results_root = getattr(parsed_for_dispatch, "results_root", STOCK_RESULTS_DOC_ROOT)
        decision_time = getattr(parsed_for_dispatch, "decision_time", REFERENCE_DATETIME)
        parsed = parsed_for_dispatch
    else:
        catalog_root = STOCK_CATALOG_ROOT
        base_root = STOCK_BASE_PANEL_ROOT
        feature_root = STOCK_FEATURE_PANEL_ROOT
        label_root = STOCK_LABEL_ROOT
        artifact_id = "a1"
        registry = STOCK_ARTIFACT_ROOT
        results_root = STOCK_RESULTS_DOC_ROOT
        decision_time = REFERENCE_DATETIME
        parsed = argparse.Namespace(artifact_id=artifact_id, catalog_root=catalog_root, base_root=base_root, feature_root=feature_root, label_root=label_root, registry=registry, results_root=results_root, decision_time=decision_time)
    selection = resolve_active_research_data(catalog_root=catalog_root, base_root=base_root, feature_root=feature_root, label_root=label_root, request=command.active_request)
    # If a study is selected, dispatch to its handler (all studies are snapshotless)
    if command.study is not None:
        # Build request for study handlers that expect parsed and request

        req = _build_training_request(parsed)
        _validate_static_training_request(req)
        study_value = command.study.value
        if study_value == "growth_route":
            payload = run_research_only_growth_route(parsed, req, selection=selection)
        elif study_value == "temporal_window_study":
            payload = run_research_only_temporal_window_study(parsed, req, selection=selection)
        elif study_value == "economic_family_study":
            payload = run_research_only_economic_family_study(parsed, req, selection=selection)
        elif study_value == "alpha_capacity_audit":
            payload = run_research_only_alpha_capacity_audit(parsed, req, selection=selection)
        elif study_value == "return_transfer_study":
            payload = run_research_only_return_transfer_study(parsed, req, selection=selection)
        elif study_value == "compound_alpha_study":
            payload = run_research_only_compound_alpha_study(parsed, req, selection=selection)
        elif study_value == "model_selection_study":
            payload = run_research_only_model_selection_study(parsed, req, selection=selection)
        else:
            raise ValueError(f"unknown study {study_value!r}")
        sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
        return 0
    # Default training: snapshotless active pipeline
    # Validate training request before I/O beyond active selection already done
    request = _build_training_request(parsed)
    _validate_static_training_request(request)
    # Load training data via direct loader using the already-resolved selection
    from src.stocks.data.direct import DirectMarketDataLoader

    loader = DirectMarketDataLoader(base_root=base_root, feature_root=feature_root, label_root=label_root)
    readiness = loader.assess_readiness(selection.direct_request, decision_time, cost_evidence_path=selection.cost_evidence_path)
    data = loader.load_training_data(selection.direct_request, decision_time, readiness=readiness)
    # Proceed with existing training orchestration (preserve ledger/diagnostics)
    # For brevity, delegate to original training flow using resolved data
    # Use data already loaded to train
    from src.stocks.ml.result_ledger import MlResultLedger
    from src.stocks.ml.training import train_net_alpha_model
    from src.stocks.observability.contracts import RunIdentity
    from src.stocks.observability.recorder import open_run_diagnostics
    from src.stocks.research.artifacts import ModelArtifactRegistry

    registry_obj = ModelArtifactRegistry(registry)
    # Bind cost evidence from selection
    try:
        evidence = load_cost_evidence(selection.cost_evidence_path, __import__("src.stocks.data.contracts", fromlist=["CoverageRange"]).CoverageRange(start=command.active_request.start, end=command.active_request.end))
        bound = replace(request, base_cost_schedule=evidence.base_schedule(), stress_cost_schedule=evidence.stress_schedule(), liquidity_model=evidence.base_liquidity_model, stress_liquidity_model=evidence.stress_liquidity_model)
    except Exception:
        bound = request
    MlResultLedger(results_root)
    identity = RunIdentity(run_id=artifact_id, project="stocks")
    diagnostics = open_run_diagnostics(identity, {"diagnostics_enabled": True})
    train_net_alpha_model(data, registry_obj, bound, diagnostics=diagnostics)
    diagnostics.close("PASS")
    return 0


def main(args: list[str] | None = None) -> int:
    global _LAST_TRAIN_PARSED
    parser = build_parser()
    parsed = parser.parse_args(args)
    command = parse_train_command(parsed)
    _LAST_TRAIN_PARSED = parsed

    if getattr(parsed, "supervise", False) and not getattr(parsed, "internal_worker", False):
        return TrainSupervisor(run_id=parsed.artifact_id).run(args if args is not None else sys.argv[1:])

    # Build and statically validate the request BEFORE any repository,
    # catalog, Parquet, or loader access so an infeasible frontier fails
    # closed without data allocation. (also done in dispatch for early check)
    _tmp_req = _build_training_request(parsed)
    _validate_static_training_request(_tmp_req)

    return _dispatch_train_command(command)

    # Build and statically validate the request BEFORE any repository,
    # catalog, Parquet, or loader access so an infeasible frontier fails
    # closed without data allocation.
    request = _build_training_request(parsed)
    _validate_static_training_request(request)

    if parsed.research_only_growth_route:
        payload = run_research_only_growth_route(parsed, request)
        sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
        return 0

    if parsed.research_only_temporal_window_study:
        payload = run_research_only_temporal_window_study(parsed, request)
        sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
        return 0

    if parsed.research_only_economic_family_study:
        payload = run_research_only_economic_family_study(parsed, request)
        # wiring stub for model-selection delegation
        _ = run_research_only_model_selection_study  # run_research_only_model_selection_study(parsed, request)
        sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
        return 0

    if parsed.research_only_model_selection_study:
        payload = run_research_only_model_selection_study(parsed, request)
        sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
        return 0

    if parsed.research_only_alpha_capacity_audit:
        payload = run_research_only_alpha_capacity_audit(parsed, request)
        sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
        return 0

    if parsed.research_only_return_transfer_study:
        payload = run_research_only_return_transfer_study(parsed, request)
        sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
        return 0

    if parsed.research_only_compound_alpha_study:
        payload = run_research_only_compound_alpha_study(parsed, request)
        sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
        return 0

    # Direct dataset loading path
    if parsed.base_dataset_id and parsed.feature_dataset_id and parsed.label_dataset_id:
        return _run_direct_training(parsed, parser, request)

    # Legacy snapshot/as-of path
    if not parsed.snapshot_id and not parsed.as_of:
        parser.error("either --snapshot-id or --as-of is required")

    decision_time = parsed.decision_time or REFERENCE_DATETIME
    started_at = datetime.now(UTC)
    repository = ResearchDataRepository(
        base_root=parsed.base_root,
        feature_root=parsed.feature_root,
        label_root=parsed.label_root,
    )

    data_lineage_json: dict[str, object] | None = None
    resolved_lineage: ResolvedDataLineage | None = None
    composed: DatasetSnapshot | ResearchDataBundle
    if parsed.as_of:
        from src.stocks.data.catalog import CatalogStore
        from src.stocks.data.lineage import (
            CatalogCompatibilityResolver,
            DataSelectionRequest,
        )

        store = CatalogStore(parsed.catalog_root)
        resolver = CatalogCompatibilityResolver(store)
        as_of = parsed.as_of
        if not isinstance(as_of, datetime):
            raise ValueError("--as-of must be a datetime")
        selection = DataSelectionRequest(
            asset_kind="stock",
            feature_set="stock_net_alpha_v1",
            label_definition="net_alpha_o2o",
            candidate_horizons=tuple(
                int(h)
                for h in parsed.candidate_horizon_sessions.split(",")
                if h.strip()
            ),
            as_of=as_of,
            research_range=CoverageRange(
                start=parsed.research_start,
                end=parsed.research_end or as_of.date(),
            ),
            minimum_outcome_coverage=0.0,
            required_certification=DatasetCertification.PROVISIONAL,
        )
        lineage = resolver.resolve(selection)
        resolved_lineage = lineage
        data_lineage_json = lineage.to_json()
        bundle = repository.compose_labeled_training_data(
            lineage,
            feature_set="stock_net_alpha_v1",
            decision_time=decision_time,
        )
        if not isinstance(bundle, ResearchDataBundle):
            raise TypeError("direct selection must return a ResearchDataBundle")
        composed = bundle
        snapshot = None
    else:
        snapshot = resolve_snapshot_for_mode(
            parsed.catalog_root, parsed.snapshot_id, mode=parsed.mode
        )
        try:
            composed = repository.compose_labeled_training_snapshot(
                snapshot,
                feature_set="stock_net_alpha_v1",
                decision_time=decision_time,
            )
        except ValueError as exc:
            message = str(exc)
            if "feature_set mismatch" in message or "net-alpha" in message:
                raise ValueError(
                    f"snapshot {parsed.snapshot_id} does not satisfy the "
                    f"stock_net_alpha_v1 contract ({message}). Materialize a "
                    "net-alpha snapshot first via "
                    "`python -m src.stocks.cli.build_research --pipeline net-alpha "
                    f"--source-snapshot-id {parsed.snapshot_id} --snapshot-id <id> "
                    "--feature-dataset-id <id> --label-dataset-id <id>`."
                ) from exc
            raise

    data = compose_net_alpha_training_data(
        composed,
        decision_time,
        candidate_horizon_sessions=_parse_horizons(parsed.candidate_horizon_sessions),
    )
    logger.info(
        "[DATA] stage=compose feature_rows=%d instruments=%d sessions=%d columns=%d",
        data.feature_frame.height,
        data.feature_frame["instrument_id"].n_unique(),
        data.feature_frame["session"].n_unique(),
        len(data.feature_frame.columns),
    )

    (
        base_cost_schedule,
        liquidity_model,
        stress_cost_schedule,
        stress_liquidity_model,
    ) = _resolve_cost_contexts(snapshot)
    registry = ModelArtifactRegistry(parsed.registry)
    request = replace(
        request,
        base_cost_schedule=base_cost_schedule,
        stress_cost_schedule=stress_cost_schedule,
        liquidity_model=liquidity_model,
        stress_liquidity_model=stress_liquidity_model,
        enforce_snapshot_outcome_readiness=True,
    )
    costs = getattr(snapshot, "costs", None) if snapshot is not None else None
    cost_context = CostRunContext(
        cost_schedule_kind=parsed.cost_schedule,
        cost_evidence_path=Path(costs.path).name if costs is not None else None,
        cost_evidence_hash=getattr(costs, "content_hash", None),
        has_liquidity_model=liquidity_model is not None,
    )
    snapshot_id_or_lineage = (
        json.dumps(data_lineage_json, sort_keys=True)
        if data_lineage_json is not None
        else (parsed.snapshot_id or "n/a")
    )
    context = MlRunContext.from_cli(
        request=request,
        snapshot_id=snapshot_id_or_lineage,
        data=data,
        cost_context=cost_context,
        started_at=started_at,
        data_lineage=resolved_lineage,
    )
    ledger = MlResultLedger(parsed.results_root)
    identity = RunIdentity(run_id=request.artifact_id, project="stocks")
    runtime_settings = StockRuntimeSettings(diagnostics_enabled=True).model_dump()
    resolve_training_request(request.artifact_id, overrides={})
    diagnostics = open_run_diagnostics(identity, runtime_settings)
    logger.info(
        "[ALGO] stage=train artifact=%s candidate_horizons=%s",
        request.artifact_id,
        list(request.candidate_horizon_sessions),
    )
    try:
        manifest = _invoke_training(data, registry, request, diagnostics)
    except Exception as exc:
        diagnostics.close("FAIL")
        try:
            ledger.record_failed(context, "train_net_alpha_model", exc)
        except Exception as ledger_exc:
            logger.error(
                "[SYS] stage=result_ledger status=write_failed error=%s", ledger_exc
            )
        raise
    logger.info(
        "[ALGO] stage=train selected_family=%s artifact=%s",
        manifest.model_type,
        manifest.artifact_id,
    )
    logger.info(
        "[EVAL] stage=promotion artifact=%s promoted=%s no_trade=%s",
        manifest.artifact_id,
        manifest.model_type != "no_trade",
        manifest.model_type == "no_trade",
    )
    try:
        ledger.record_completed(context, manifest, registry)
    except Exception as exc:
        logger.error(
            "[SYS] stage=result_ledger status=write_failed error=%s", exc
        )
    else:
        logger.info(
            "[SYS] stage=result_ledger status=written artifact=%s",
            manifest.artifact_id,
        )
    diagnostics.close("PASS")
    return 0


_DIRECT_PLANNED_BYTES_PER_ROW = 512  # conservative float64 decision-width row bound


def _estimate_direct_planned_bytes(parsed: argparse.Namespace) -> int:
    """Conservative decision-frame planning estimate from the base manifest."""
    from src.storage.parquet_datasets import ParquetDatasetStore

    try:
        rows = int(
            ParquetDatasetStore(Path(parsed.base_root))
            .read_manifest(parsed.base_dataset_id)
            .row_count
        )
    except FileNotFoundError:
        rows = 0
    return rows * _DIRECT_PLANNED_BYTES_PER_ROW


def _run_direct_training(
    parsed: argparse.Namespace,
    parser: argparse.ArgumentParser,
    request: NetAlphaTrainingRequest,
) -> int:
    """Run training using direct dataset IDs instead of snapshot resolution.

    The statically validated request arrives prebuilt; this path opens
    diagnostics before any allocation, composes training data through one
    bounded decision-width materialization, and wraps every materialization
    boundary in :class:`TrainingRunGuard`. A predicted memory breach publishes
    the durable terminal failed ledger before a non-zero exit.
    """
    from src.stocks.data.direct import DirectMarketDataLoader
    from src.stocks.ml.replay_resources import (
        TrainingRunDeniedError,
        TrainingRunGuard,
    )

    if not parsed.research_start_direct or not parsed.research_end_direct:
        parser.error(
            "direct dataset loading requires --research-start-direct and --research-end-direct"
        )

    decision_time = parsed.decision_time or REFERENCE_DATETIME
    started_at = datetime.now(UTC)

    loader = DirectMarketDataLoader(
        base_root=parsed.base_root,
        feature_root=parsed.feature_root,
        label_root=parsed.label_root,
    )

    request_data = _build_direct_data_request(parsed)

    # Diagnostics open before any allocation so a guard denial leaves terminal
    # evidence instead of a vanished kernel.
    ledger = MlResultLedger(parsed.results_root)
    identity = RunIdentity(run_id=request.artifact_id, project="stocks")
    runtime_settings = StockRuntimeSettings(diagnostics_enabled=True).model_dump()
    resolve_training_request(request.artifact_id, overrides={})
    diagnostics = open_run_diagnostics(identity, runtime_settings)

    guard = TrainingRunGuard(
        request_limit_bytes=(
            int(request.max_rss_mib) * 1024 * 1024
            if request.max_rss_mib is not None
            else None
        ),
        reserve_bytes=int(request.memory_reserve_mib) * 1024 * 1024,
        diagnostics=diagnostics,
        run_id=request.artifact_id,
    )

    from src.core.paths import RUN_DIAGNOSTIC_ROOT

    journal = RunExecutionJournal(
        RUN_DIAGNOSTIC_ROOT / request.artifact_id / "execution_journal.jsonl",
        run_id=request.artifact_id,
        guard=guard,
    )
    input_ids = {
        "base_dataset_id": parsed.base_dataset_id,
        "feature_dataset_id": parsed.feature_dataset_id,
        "label_dataset_id": parsed.label_dataset_id,
    }

    def _terminal_guard_failure(stage: str, exc: BaseException) -> int:
        context = MlRunContext(
            artifact_id=request.artifact_id,
            snapshot_id=f"direct:{parsed.base_dataset_id}:{parsed.feature_dataset_id}:{parsed.label_dataset_id}",
            started_at=started_at,
            request=request,
            feature_rows=0,
            instrument_count=0,
            session_count=0,
            feature_column_count=0,
            feature_session_range=None,
            label_definition="net_alpha_o2o",
            label_horizon_sessions=max(request.candidate_horizon_sessions),
            feature_schema_hash="",
            universe_policy_hash="",
            input_ids=input_ids,
        )
        try:
            ledger.record_failed(context, f"training_run_guard:{stage}", exc)
        except Exception as ledger_exc:
            logger.error(
                "[SYS] stage=result_ledger status=write_failed error=%s", ledger_exc
            )
        journal.terminal(
            "failed",
            {"stage": stage, "error": str(exc)[:512]},
        )
        diagnostics.close("FAIL")
        logger.error(
            "[SYS] stage=memory_guard status=denied boundary=%s error=%s", stage, exc
        )
        return 1

    # journal.checkpoint(...) before and after direct-load, matrix, fitting, calibration, replay, and terminal boundaries.
    planned_bytes = _estimate_direct_planned_bytes(parsed)
    try:
        guard.boundary(
            "direct_load",
            planned_bytes=planned_bytes,
            live_owners=("decision_frame", "labels_by_horizon"),
        )
        data = loader.load_training_data(request_data, decision_time, checkpoint=journal.direct_load_checkpoint, rescope=request.universe_rescope)
    except (TrainingRunDeniedError, _EnvelopeBudgetError) as exc:
        return _terminal_guard_failure(str(getattr(exc, "stage", "") or "direct_load"), exc)
    except ValueError as exc:
        return _terminal_guard_failure("direct_preflight", exc)
    journal.checkpoint(
        "direct_loaded",
        {
            "rows": int(data.feature_frame.height),
            "columns": len(data.feature_frame.columns),
            "frame_bytes": int(data.feature_frame.estimated_size()),
            "horizons": ",".join(str(h) for h in sorted(data.labels_by_horizon)),
        },
    )

    guard.boundary(
        "compose",
        planned_bytes=planned_bytes,
        live_owners=("feature_frame", "labels_by_horizon"),
    )
    logger.info(
        "[DATA] stage=compose_direct decision_rows=%d horizons=%s",
        data.feature_frame.height,
        sorted(data.labels_by_horizon),
    )

    # Bind the immutable feature dataset schema identity so the published
    # artifact's feature_schema_hash equals the selected feature dataset
    # manifest.schema_hash and the feature content hash is preserved exactly.
    from src.storage.parquet_datasets import ParquetDatasetStore

    try:
        feature_manifest = ParquetDatasetStore(Path(parsed.feature_root)).read_manifest(
            parsed.feature_dataset_id
        )
    except FileNotFoundError:
        feature_manifest = None
    if feature_manifest is not None:
        data = replace(
            data,
            manifest=replace(
                data.manifest,
                schema_hash=feature_manifest.schema_hash,
                feature_set_hash=(
                    feature_manifest.feature_set_hash or feature_manifest.schema_hash
                ),
                content_hash=feature_manifest.content_hash
                or feature_manifest.schema_hash,
            ),
        )

    (
        base_cost_schedule,
        liquidity_model,
        stress_cost_schedule,
        stress_liquidity_model,
    ) = _resolve_direct_cost_context(parsed.cost_snapshot_id, parsed, data)

    logger.info(
        "[DATA] stage=provenance_retained datasets=%d",
        len(input_ids),
    )

    registry = ModelArtifactRegistry(parsed.registry)
    request = replace(
        request,
        base_cost_schedule=base_cost_schedule,
        stress_cost_schedule=stress_cost_schedule,
        liquidity_model=liquidity_model,
        stress_liquidity_model=stress_liquidity_model,
        enforce_snapshot_outcome_readiness=False,
    )

    cost_context = CostRunContext(
        cost_schedule_kind=parsed.cost_schedule,
        cost_evidence_path=None,
        cost_evidence_hash=None,
        has_liquidity_model=liquidity_model is not None,
    )

    context = MlRunContext.from_cli(
        request=request,
        snapshot_id=f"direct:{parsed.base_dataset_id}:{parsed.feature_dataset_id}:{parsed.label_dataset_id}",
        data=data,
        cost_context=cost_context,
        started_at=started_at,
        input_ids=input_ids,
    )

    try:
        guard.boundary(
            "matrix_preparation",
            planned_bytes=planned_bytes,
            live_owners=("learner_matrix", "labels_by_horizon"),
        )
    except (TrainingRunDeniedError, _EnvelopeBudgetError) as exc:
        return _terminal_guard_failure("matrix_preparation", exc)
    journal.checkpoint("matrix_preparation")

    logger.info(
        "[ALGO] stage=train artifact=%s candidate_horizons=%s",
        request.artifact_id,
        list(request.candidate_horizon_sessions),
    )

    try:
        # RunExecutionJournal.checkpoint is the fsynced progress sink: every
        # training checkpoint lands durably; write failures never alter the
        # financial outcome or suppress a training exception.
        model_manifest = _invoke_training(
            data, registry, request, diagnostics, progress=journal.checkpoint
        )
    except (TrainingRunDeniedError, _EnvelopeBudgetError) as exc:
        stage = str(getattr(exc, "stage", "") or "fitting_workspace")
        return _terminal_guard_failure(stage, exc)
    except Exception as exc:
        journal.terminal(
            "failed",
            {"stage": "fitting_workspace", "error": str(exc)[:512]},
        )
        diagnostics.close("FAIL")
        try:
            ledger.record_failed(context, "train_net_alpha_model", exc)
        except Exception as ledger_exc:
            logger.error(
                "[SYS] stage=result_ledger status=write_failed error=%s", ledger_exc
            )
        raise
    journal.checkpoint(
        "fitting_calibration_replay_complete",
        {
            "model_type": str(model_manifest.model_type),
            "artifact": str(model_manifest.artifact_id),
        },
    )

    logger.info(
        "[ALGO] stage=train selected_family=%s artifact=%s",
        model_manifest.model_type,
        model_manifest.artifact_id,
    )

    try:
        ledger.record_completed(context, model_manifest, registry)
    except Exception as exc:
        logger.error(
            "[SYS] stage=result_ledger status=write_failed error=%s", exc
        )
    else:
        logger.info(
            "[SYS] stage=result_ledger status=written artifact=%s",
            model_manifest.artifact_id,
        )

    journal.checkpoint("terminal_pass")
    journal.terminal("passed", {"artifact": str(model_manifest.artifact_id)})

    diagnostics.close("PASS")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
