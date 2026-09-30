"""Research segments, lockbox enforcement, and promotion policy bindings."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from src.config.errors import ConfigError
from src.core.pit import PITDataError
from src.data.research_scope import ResearchScope

__all__ = [
    "CriteriaBootstrap",
    "CriteriaC1",
    "CriteriaC2",
    "CriteriaC3",
    "CriteriaC4",
    "CriteriaPolicy",
    "FinalistRecord",
    "LockboxAuthorization",
    "LockboxError",
    "LockboxLedger",
    "ResearchProtocol",
    "Segment",
    "StatisticsPolicy",
    "load_research_protocol",
]

_LOG = logging.getLogger(__name__)


class Segment(StrEnum):
    DISCOVERY = "discovery"
    HOLDOUT = "holdout"
    FORWARD = "forward"


class LockboxError(PITDataError):
    """A run touched a sealed segment without a valid authorization."""


class StatisticsPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    bootstrap_block_sessions: int
    bootstrap_draws: int
    bootstrap_seed: int
    cscv_blocks: int


class CriteriaBootstrap(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    block_sessions: int
    draws: int
    seed: int
    horizon_sessions: int
    mdd_limit: float


class CriteriaC1(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    stress_extra_slippage: float
    stress_execution_delay: int
    max_p_cagr_le_zero: float
    perturbation_cuts: int
    perturbation_seed: int


class CriteriaC2(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    min_point_calmar: float
    min_p_calmar: float
    max_p_mdd_below_limit: float
    max_underwater_median_sessions: int
    max_underwater_p95_sessions: int
    min_worst_phase_calmar: float


class CriteriaC3(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    stress_min_point_calmar: float
    ledger_capitals: tuple[int, ...]
    ledger_min_calmar: float
    parity_max_growth_gap: float


class CriteriaC4(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    holdout_max_p_mean_le_zero: float
    holdout_min_point_calmar: float


class CriteriaPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    bootstrap: CriteriaBootstrap
    c1: CriteriaC1
    c2: CriteriaC2
    c3: CriteriaC3
    c4: CriteriaC4
    prior_trials: int
    prior_effective_trials: float
    prior_trial_sharpe_std_annual: float


class ResearchProtocol(BaseModel):
    """Segments and promotion policy of the research program.

    Holdout and forward boundaries come from the research scope so one calendar governs data,
    backtests and research; only the discovery start is research-specific.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str
    discovery_start: date
    discovery_end: date
    holdout_start: date
    holdout_end: date
    forward_start: date
    max_finalists: int
    prior_trials: int
    sessions_per_year: int
    primary_capital_krw: int
    statistics: StatisticsPolicy
    criteria: CriteriaPolicy

    def segment_of(self, day: date) -> Segment:
        if day <= self.discovery_end:
            return Segment.DISCOVERY
        if day <= self.holdout_end:
            return Segment.HOLDOUT
        return Segment.FORWARD

    def discovery_window(self) -> tuple[date, date]:
        return (self.discovery_start, self.discovery_end)

    def holdout_window(self) -> tuple[date, date]:
        return (self.holdout_start, self.holdout_end)

    @property
    def content_hash(self) -> str:
        payload = {
            "criteria": self.criteria.model_dump(mode="json"),
            "discovery_end": self.discovery_end.isoformat(),
            "discovery_start": self.discovery_start.isoformat(),
            "forward_start": self.forward_start.isoformat(),
            "holdout_end": self.holdout_end.isoformat(),
            "holdout_start": self.holdout_start.isoformat(),
            "max_finalists": self.max_finalists,
            "primary_capital_krw": self.primary_capital_krw,
            "prior_trials": self.prior_trials,
            "sessions_per_year": self.sessions_per_year,
            "statistics": self.statistics.model_dump(mode="json"),
            "version": self.version,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_research_protocol(path: Path, scope: ResearchScope) -> ResearchProtocol:
    """Load the protocol TOML and bind scope boundaries.

    Raises:
        ConfigError: file missing or invalid, unknown keys, ``discovery_start`` not after
            ``scope.evidence_start`` or not before ``scope.validation_end``, or any gate ratio
            outside its domain.
    """
    import tomllib

    try:
        with open(path, "rb") as handle:
            raw = tomllib.load(handle)
    except OSError as exc:
        raise ConfigError(f"research protocol is missing: {path}") from exc
    except ValueError as exc:
        raise ConfigError(f"research protocol is invalid TOML: {path}") from exc
    allowed_top = {
        "version",
        "discovery_start",
        "max_finalists",
        "prior_trials",
        "sessions_per_year",
        "primary_capital_krw",
        "statistics",
        "criteria",
    }
    unknown = set(raw) - allowed_top
    if unknown:
        raise ConfigError(f"research protocol has unknown keys: {sorted(unknown)}")
    try:
        discovery_start = date.fromisoformat(str(raw.get("discovery_start")))
    except (ValueError, TypeError) as exc:
        raise ConfigError(f"research protocol discovery_start is invalid: {raw.get('discovery_start')!r}") from exc
    if discovery_start <= scope.evidence_start:
        raise ConfigError("research protocol discovery_start must be after scope.evidence_start")
    if discovery_start > scope.validation_end:
        raise ConfigError("research protocol discovery_start must be on or before scope.validation_end")
    statistics_raw = raw.get("statistics")
    criteria_raw = raw.get("criteria")
    if not isinstance(statistics_raw, dict) or not isinstance(criteria_raw, dict):
        raise ConfigError("research protocol must declare [statistics] and [criteria]")
    try:
        statistics = StatisticsPolicy.model_validate(statistics_raw)
        criteria = CriteriaPolicy.model_validate(criteria_raw)
        protocol = ResearchProtocol(
            version=str(raw.get("version")),
            discovery_start=discovery_start,
            discovery_end=scope.validation_end,
            holdout_start=scope.holdout_start,
            holdout_end=scope.holdout_end,
            forward_start=scope.forward_start,
            max_finalists=int(raw["max_finalists"]),
            prior_trials=int(raw["prior_trials"]),
            sessions_per_year=int(raw["sessions_per_year"]),
            primary_capital_krw=int(raw["primary_capital_krw"]),
            statistics=statistics,
            criteria=criteria,
        )
    except (ValueError, TypeError, KeyError) as exc:
        raise ConfigError(f"research protocol is invalid: {exc}") from exc
    if protocol.max_finalists <= 0:
        raise ConfigError("research protocol max_finalists must be positive")
    if protocol.prior_trials < 0:
        raise ConfigError("research protocol prior_trials must be non-negative")
    if protocol.sessions_per_year <= 0:
        raise ConfigError("research protocol sessions_per_year must be positive")
    if protocol.primary_capital_krw <= 0:
        raise ConfigError("research protocol primary_capital_krw must be positive")
    if protocol.statistics.bootstrap_block_sessions <= 0 or protocol.statistics.bootstrap_draws <= 0:
        raise ConfigError("research protocol bootstrap policy must be positive")
    if protocol.statistics.cscv_blocks < 2 or protocol.statistics.cscv_blocks % 2 != 0:
        raise ConfigError("research protocol cscv_blocks must be an even integer >= 2")
    _check_criteria(protocol.criteria)
    return protocol


def _check_criteria_int(name: str, value: object, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"research protocol criteria {name!r} must be an int >= {minimum}")


def _check_criteria_number(name: str, value: object, *, minimum: float, above: bool = False) -> None:
    amount = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else math.nan
    if not math.isfinite(amount) or (amount <= minimum if above else amount < minimum):
        raise ConfigError(f"research protocol criteria {name!r} must be a finite number {'>' if above else '>='} {minimum}")


def _check_criteria(criteria: CriteriaPolicy) -> None:
    """Validate termination-criteria domains (ratios, positivity, session counts)."""
    positive_ints: list[tuple[str, object]] = [
        ("block_sessions", criteria.bootstrap.block_sessions),
        ("draws", criteria.bootstrap.draws),
        ("horizon_sessions", criteria.bootstrap.horizon_sessions),
        ("perturbation_cuts", criteria.c1.perturbation_cuts),
        ("max_underwater_median_sessions", criteria.c2.max_underwater_median_sessions),
        ("max_underwater_p95_sessions", criteria.c2.max_underwater_p95_sessions),
    ]
    for name, value in positive_ints:
        _check_criteria_int(name, value, minimum=1)
    nonneg_ints: list[tuple[str, object]] = [
        ("seed", criteria.bootstrap.seed),
        ("perturbation_seed", criteria.c1.perturbation_seed),
        ("stress_execution_delay", criteria.c1.stress_execution_delay),
        ("prior_trials", criteria.prior_trials),
    ]
    for name, value in nonneg_ints:
        _check_criteria_int(name, value, minimum=0)
    positive_numbers: list[tuple[str, object]] = [
        ("min_point_calmar", criteria.c2.min_point_calmar),
        ("min_worst_phase_calmar", criteria.c2.min_worst_phase_calmar),
        ("stress_min_point_calmar", criteria.c3.stress_min_point_calmar),
        ("ledger_min_calmar", criteria.c3.ledger_min_calmar),
        ("holdout_min_point_calmar", criteria.c4.holdout_min_point_calmar),
        ("prior_trial_sharpe_std_annual", criteria.prior_trial_sharpe_std_annual),
        ("prior_effective_trials", criteria.prior_effective_trials),
    ]
    for name, value in positive_numbers:
        _check_criteria_number(name, value, minimum=0.0, above=True)
    nonneg_numbers: list[tuple[str, object]] = [
        ("stress_extra_slippage", criteria.c1.stress_extra_slippage),
        ("parity_max_growth_gap", criteria.c3.parity_max_growth_gap),
    ]
    for name, value in nonneg_numbers:
        _check_criteria_number(name, value, minimum=0.0)
    for name, value in (
        ("max_p_cagr_le_zero", criteria.c1.max_p_cagr_le_zero),
        ("min_p_calmar", criteria.c2.min_p_calmar),
        ("max_p_mdd_below_limit", criteria.c2.max_p_mdd_below_limit),
        ("holdout_max_p_mean_le_zero", criteria.c4.holdout_max_p_mean_le_zero),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
            raise ConfigError(f"research protocol criteria {name!r} must be a number in [0, 1]")
    if not -1.0 <= float(criteria.bootstrap.mdd_limit) < 0.0:
        raise ConfigError("research protocol criteria 'mdd_limit' must satisfy -1 <= mdd_limit < 0")
    if not criteria.c3.ledger_capitals or any(
        isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in criteria.c3.ledger_capitals
    ):
        raise ConfigError("research protocol criteria 'ledger_capitals' must be non-empty positive ints")


@dataclass(frozen=True, slots=True)
class FinalistRecord:
    spec_hash: str
    family: str
    discovery_trial_id: str
    gate_report_digest: str
    registered_at: datetime


@dataclass(frozen=True, slots=True)
class LockboxAuthorization:
    """Permission for one run window.

    ``evidence`` is False for runs on an already-opened segment by a non-finalist: such a run may be
    inspected but can never support a promotion decision.
    """

    segment: Segment
    start: date
    end: date
    spec_hash: str | None
    evidence: bool


_LOCKBOX_NAME = "lockbox.json"


class LockboxLedger:
    """Durable, append-only record of finalists and one-shot segment openings.

    State lives in ``<state_root>/research/lockbox.json``, written atomically. Openings are
    irreversible: once the holdout has been seen, no further finalist can be registered, because
    any later choice would be informed by holdout outcomes.
    """

    def __init__(self, *, state_root: Path, protocol: ResearchProtocol, now: Callable[[], datetime]) -> None:
        self._dir = Path(state_root) / "research"
        self._path = self._dir / _LOCKBOX_NAME
        self._protocol = protocol
        self._now = now
        self._state = self._load()

    def finalists(self) -> tuple[FinalistRecord, ...]:
        return tuple(self._state["finalists"])

    def is_open(self, segment: Segment) -> bool:
        if segment is Segment.DISCOVERY:
            return True
        if segment is Segment.HOLDOUT:
            return self._state["holdout_opened_at"] is not None
        return self._state["forward_opened_at"] is not None

    def register_finalists(self, finalists: Sequence[FinalistRecord]) -> None:
        candidates = list(finalists)
        if not candidates:
            return
        if self._state["holdout_opened_at"] is not None:
            raise LockboxError("holdout already open; no further finalists can be registered")
        existing_hashes = {record.spec_hash for record in self._state["finalists"]}
        batch_hashes: set[str] = set()
        for record in candidates:
            if record.spec_hash in existing_hashes or record.spec_hash in batch_hashes:
                raise LockboxError(f"duplicate finalist spec_hash: {record.spec_hash}")
            batch_hashes.add(record.spec_hash)
        if len(self._state["finalists"]) + len(candidates) > self._protocol.max_finalists:
            raise LockboxError("registering finalists would exceed max_finalists")
        state = self._clone_state()
        for record in candidates:
            state["finalists"].append(record)
            state["audit"].append({"type": "finalist_registered", "at": record.registered_at.isoformat(), "spec_hash": record.spec_hash})
        self._persist(state)
        for record in candidates:
            _LOG.info("[RISK] lockbox finalist registered spec_hash=%s family=%s", record.spec_hash, record.family)

    def authorize(self, *, start: date, end: date, spec_hash: str | None) -> LockboxAuthorization:
        if start > end:
            raise ValueError(f"run window [{start}, {end}] must satisfy start <= end")
        segment = self._protocol.segment_of(end)
        if segment is Segment.DISCOVERY:
            return LockboxAuthorization(segment=segment, start=start, end=end, spec_hash=spec_hash, evidence=True)
        finalist_hashes = {record.spec_hash for record in self._state["finalists"]}
        if segment is Segment.HOLDOUT:
            if spec_hash is not None and spec_hash in finalist_hashes:
                if self._state["holdout_opened_at"] is None:
                    state = self._clone_state()
                    opened_at = self._now().isoformat()
                    state["holdout_opened_at"] = opened_at
                    state["audit"].append({"type": "holdout_opened", "at": opened_at, "spec_hash": spec_hash})
                    self._persist(state)
                    _LOG.info("[RISK] lockbox holdout opened spec_hash=%s", spec_hash)
                return LockboxAuthorization(segment=segment, start=start, end=end, spec_hash=spec_hash, evidence=True)
            if self._state["holdout_opened_at"] is None:
                raise LockboxError("holdout is sealed; register a finalist to open it")
            return LockboxAuthorization(segment=segment, start=start, end=end, spec_hash=spec_hash, evidence=False)
        if spec_hash is not None and self.holdout_passed(spec_hash):
            if self._state["forward_opened_at"] is None:
                state = self._clone_state()
                opened_at = self._now().isoformat()
                state["forward_opened_at"] = opened_at
                state["audit"].append({"type": "forward_opened", "at": opened_at, "spec_hash": spec_hash})
                self._persist(state)
                _LOG.info("[RISK] lockbox forward opened spec_hash=%s", spec_hash)
            return LockboxAuthorization(segment=segment, start=start, end=end, spec_hash=spec_hash, evidence=True)
        if self._state["forward_opened_at"] is None:
            raise LockboxError("forward is sealed; a passed holdout verdict is required")
        return LockboxAuthorization(segment=segment, start=start, end=end, spec_hash=spec_hash, evidence=False)

    def record_holdout_verdict(self, *, spec_hash: str, passed: bool, report_digest: str) -> None:
        finalist_hashes = {record.spec_hash for record in self._state["finalists"]}
        if spec_hash not in finalist_hashes:
            raise LockboxError(f"holdout verdict for unknown finalist: {spec_hash}")
        if self._state["holdout_opened_at"] is None:
            raise LockboxError("holdout verdict requires an opened holdout")
        if spec_hash in self._state["verdicts"]:
            raise LockboxError(f"holdout verdict already recorded: {spec_hash}")
        state = self._clone_state()
        state["verdicts"][spec_hash] = {
            "passed": bool(passed),
            "report_digest": report_digest,
            "at": self._now().isoformat(),
        }
        state["audit"].append({"type": "holdout_verdict", "at": state["verdicts"][spec_hash]["at"], "spec_hash": spec_hash})
        self._persist(state)

    def holdout_passed(self, spec_hash: str) -> bool:
        verdict = self._state["verdicts"].get(spec_hash)
        return bool(verdict is not None and verdict.get("passed") is True)

    def _clone_state(self) -> dict[str, Any]:
        return {
            "protocol_version": self._state["protocol_version"],
            "finalists": list(self._state["finalists"]),
            "holdout_opened_at": self._state["holdout_opened_at"],
            "forward_opened_at": self._state["forward_opened_at"],
            "verdicts": {key: dict(value) for key, value in self._state["verdicts"].items()},
            "audit": [dict(event) for event in self._state["audit"]],
        }

    def _load(self) -> dict[str, Any]:
        if not self._path.exists():
            return {
                "protocol_version": self._protocol.version,
                "finalists": [],
                "holdout_opened_at": None,
                "forward_opened_at": None,
                "verdicts": {},
                "audit": [],
            }
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise LockboxError(f"unreadable lockbox ledger: {self._path}") from exc
        if not isinstance(raw, dict):
            raise LockboxError(f"invalid lockbox ledger: {self._path}")
        if raw.get("protocol_version") != self._protocol.version:
            raise LockboxError(f"lockbox protocol version mismatch: {self._path}")
        try:
            finalists = tuple(
                FinalistRecord(
                    spec_hash=str(item["spec_hash"]),
                    family=str(item["family"]),
                    discovery_trial_id=str(item["discovery_trial_id"]),
                    gate_report_digest=str(item["gate_report_digest"]),
                    registered_at=datetime.fromisoformat(str(item["registered_at"])),
                )
                for item in raw.get("finalists", [])
            )
            holdout_opened_at = raw.get("holdout_opened_at")
            forward_opened_at = raw.get("forward_opened_at")
            verdicts_raw = raw.get("verdicts", {})
            audit_raw = raw.get("audit", [])
            if not isinstance(verdicts_raw, dict) or not isinstance(audit_raw, list):
                raise LockboxError(f"invalid lockbox ledger: {self._path}")
            verdicts = {
                str(key): {
                    "passed": bool(value["passed"]),
                    "report_digest": str(value["report_digest"]),
                    "at": str(value["at"]),
                }
                for key, value in verdicts_raw.items()
            }
            audit = [
                {"type": str(event["type"]), "at": str(event["at"]), "spec_hash": event.get("spec_hash")}
                for event in audit_raw
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise LockboxError(f"invalid lockbox ledger: {self._path}") from exc
        return {
            "protocol_version": raw.get("protocol_version"),
            "finalists": list(finalists),
            "holdout_opened_at": holdout_opened_at,
            "forward_opened_at": forward_opened_at,
            "verdicts": verdicts,
            "audit": audit,
        }

    def _persist(self, state: dict[str, Any]) -> None:
        payload = {
            "protocol_version": state["protocol_version"],
            "finalists": [
                {
                    "spec_hash": record.spec_hash,
                    "family": record.family,
                    "discovery_trial_id": record.discovery_trial_id,
                    "gate_report_digest": record.gate_report_digest,
                    "registered_at": record.registered_at.isoformat(),
                }
                for record in state["finalists"]
            ],
            "holdout_opened_at": state["holdout_opened_at"],
            "forward_opened_at": state["forward_opened_at"],
            "verdicts": state["verdicts"],
            "audit": state["audit"],
        }
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            encoded = json.dumps(payload, sort_keys=True, indent=2) + "\n"
            descriptor, temporary_name = tempfile.mkstemp(prefix=".lockbox.", suffix=".tmp", dir=str(self._dir))
            temporary_path = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary_path, self._path)
            finally:
                temporary_path.unlink(missing_ok=True)
        except OSError as exc:
            raise LockboxError(f"cannot write lockbox ledger: {self._path}") from exc
        self._state = {
            "protocol_version": state["protocol_version"],
            "finalists": list(state["finalists"]),
            "holdout_opened_at": state["holdout_opened_at"],
            "forward_opened_at": state["forward_opened_at"],
            "verdicts": dict(state["verdicts"]),
            "audit": list(state["audit"]),
        }
