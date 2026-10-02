"""Append-only run registry with bit-identical return verification."""
from __future__ import annotations

import fcntl
import hashlib
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray

from src.core.pit import PITDataError

__all__ = ["RunRecord", "RunRegistry", "RunReturns"]

_LOG = logging.getLogger(__name__)

#: Current index-line format. A line carrying another schema (or none) is provenance of an older format: it is
#: never rewritten or deleted, and the reader skips it instead of failing on a format it no longer writes.
_INDEX_SCHEMA = 2


@dataclass(frozen=True, slots=True)
class RunRecord:
    run_id: str
    family: str
    spec_hash: str
    spec_json: str
    scenario: str
    capital_krw: int
    start: date
    end: date
    sim_config_json: str
    cube_id: str
    protocol_hash: str
    engine_config_hash: str
    report_digest: str
    created_at: datetime
    metrics: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class RunReturns:
    sessions: tuple[date, ...]
    net: NDArray[np.float64]
    benchmarks: Mapping[str, NDArray[np.float64]]


def _canonical_record_id(
    *, run_id: str, scenario: str, start: date, end: date, cube_id: str
) -> str:
    payload = {
        "cube_id": cube_id,
        "end": end.isoformat(),
        "run_id": run_id,
        "scenario": scenario,
        "start": start.isoformat(),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]


class RunRegistry:
    """Append-only provenance of every evaluated run and its daily log returns. Used to reproduce
    any reported number and to pair streams for champion decisions; it is not a multiplicity counter."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._index = self._root / "index.jsonl"
        self._returns_dir = self._root / "returns"

    def record(
        self,
        *,
        family: str,
        spec_hash: str,
        spec_json: str,
        run_id: str,
        scenario: str,
        capital_krw: int,
        sim_config_json: str,
        cube_id: str,
        protocol_hash: str,
        engine_config_hash: str,
        report_digest: str,
        returns: RunReturns,
        metrics: Mapping[str, float],
        now: datetime,
    ) -> RunRecord:
        sessions = tuple(returns.sessions)
        net = np.asarray(returns.net, dtype=np.float64)
        benchmarks = {key: np.asarray(value, dtype=np.float64) for key, value in returns.benchmarks.items()}
        self._check_arrays(sessions, net, benchmarks)
        start, end = sessions[0], sessions[-1]
        record_id = _canonical_record_id(
            run_id=run_id, scenario=scenario, start=start, end=end, cube_id=cube_id
        )
        existing = self._find_record(record_id)
        if existing is not None:
            stored = self.returns(record_id)
            if (
                stored.sessions != sessions
                or not np.array_equal(stored.net, net)
                or set(stored.benchmarks) != set(benchmarks)
                or any(not np.array_equal(stored.benchmarks[key], benchmarks[key]) for key in benchmarks)
            ):
                raise PITDataError(f"non-deterministic run returns: {record_id}")
            return existing
        record = RunRecord(
            run_id=run_id,
            family=family,
            spec_hash=spec_hash,
            spec_json=spec_json,
            scenario=scenario,
            capital_krw=int(capital_krw),
            start=start,
            end=end,
            sim_config_json=sim_config_json,
            cube_id=cube_id,
            protocol_hash=protocol_hash,
            engine_config_hash=engine_config_hash,
            report_digest=report_digest,
            created_at=now,
            metrics=dict(metrics),
        )
        self._write_returns(record_id, sessions, net, benchmarks)
        self._append_index(record_id, record)
        return record

    def runs(
        self, *, run_id: str | None = None, family: str | None = None
    ) -> tuple[RunRecord, ...]:
        records: list[RunRecord] = []
        for line in self._read_lines():
            _record_id, record = self._parse_line(line)
            if run_id is not None and record.run_id != run_id:
                continue
            if family is not None and record.family != family:
                continue
            records.append(record)
        return tuple(records)

    def returns(self, record_id: str) -> RunReturns:
        path = self._returns_dir / f"{record_id}.parquet"
        try:
            frame = pl.read_parquet(path)
        except (OSError, ValueError) as exc:
            raise PITDataError(f"unreadable run returns: {record_id}") from exc
        columns = frame.columns
        if "session" not in columns or "net" not in columns:
            raise PITDataError(f"invalid run returns: {record_id}")
        sessions = tuple(date.fromisoformat(str(value)) for value in frame["session"].to_list())
        net = np.asarray(frame["net"].to_numpy(), dtype=np.float64)
        benchmarks: dict[str, NDArray[np.float64]] = {}
        for column in columns:
            if column.startswith("bench__"):
                benchmarks[column[len("bench__") :]] = np.asarray(frame[column].to_numpy(), dtype=np.float64)
        self._check_arrays(sessions, net, benchmarks)
        return RunReturns(sessions=sessions, net=net, benchmarks=benchmarks)

    @staticmethod
    def _check_arrays(
        sessions: tuple[date, ...], net: NDArray[np.float64], benchmarks: Mapping[str, NDArray[np.float64]]
    ) -> None:
        if not sessions:
            raise PITDataError("run returns must have at least one session")
        if net.shape != (len(sessions),):
            raise PITDataError("run net returns length must match sessions")
        for key, values in benchmarks.items():
            if values.shape != (len(sessions),):
                raise PITDataError(f"run benchmark length must match sessions: {key}")
        if not np.all(np.isfinite(net)):
            raise PITDataError("run net returns must be finite")
        for key, values in benchmarks.items():
            if not np.all(np.isfinite(values)):
                raise PITDataError(f"run benchmark returns must be finite: {key}")

    def _find_record(self, record_id: str) -> RunRecord | None:
        for line in self._read_lines():
            stored_id, record = self._parse_line(line)
            if stored_id == record_id:
                return record
        return None

    def _read_lines(self) -> list[str]:
        if not self._index.exists():
            return []
        try:
            text = self._index.read_text(encoding="utf-8")
        except OSError as exc:
            raise PITDataError(f"unreadable run index: {self._index}") from exc
        current: list[str] = []
        legacy = 0
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PITDataError(f"invalid run index line: {self._index}") from exc
            if not isinstance(raw, dict) or raw.get("schema") != _INDEX_SCHEMA:
                legacy += 1
                continue
            current.append(line)
        if legacy:
            _LOG.warning(
                "[DATA] run index carries %d line(s) of an older schema; they are kept but not read", legacy
            )
        return current

    @staticmethod
    def _parse_line(line: str) -> tuple[str, RunRecord]:
        try:
            raw = json.loads(line)
            return (
                str(raw["record_id"]),
                RunRecord(
                    run_id=str(raw["run_id"]),
                    family=str(raw["family"]),
                    spec_hash=str(raw["spec_hash"]),
                    spec_json=str(raw["spec_json"]),
                    scenario=str(raw["scenario"]),
                    capital_krw=int(raw["capital_krw"]),
                    start=date.fromisoformat(str(raw["start"])),
                    end=date.fromisoformat(str(raw["end"])),
                    sim_config_json=str(raw["sim_config_json"]),
                    cube_id=str(raw["cube_id"]),
                    protocol_hash=str(raw["protocol_hash"]),
                    engine_config_hash=str(raw["engine_config_hash"]),
                    report_digest=str(raw["report_digest"]),
                    created_at=datetime.fromisoformat(str(raw["created_at"])),
                    metrics=dict(raw["metrics"]),
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PITDataError(f"invalid run index line: {exc}") from exc

    def _write_returns(
        self,
        record_id: str,
        sessions: tuple[date, ...],
        net: NDArray[np.float64],
        benchmarks: Mapping[str, NDArray[np.float64]],
    ) -> None:
        self._returns_dir.mkdir(parents=True, exist_ok=True)
        target = self._returns_dir / f"{record_id}.parquet"
        if target.exists():
            return
        data: dict[str, object] = {
            "session": pl.Series("session", list(sessions), dtype=pl.Date),
            "net": pl.Series("net", np.asarray(net, dtype=np.float64), dtype=pl.Float64),
        }
        for key in sorted(benchmarks):
            data[f"bench__{key}"] = pl.Series(
                f"bench__{key}", np.asarray(benchmarks[key], dtype=np.float64), dtype=pl.Float64
            )
        frame = pl.DataFrame(data)
        temporary = self._returns_dir / f".{record_id}.tmp.parquet"
        frame.write_parquet(temporary)
        temporary.replace(target)

    def _append_index(self, record_id: str, record: RunRecord) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": _INDEX_SCHEMA,
            "record_id": record_id,
            "run_id": record.run_id,
            "family": record.family,
            "spec_hash": record.spec_hash,
            "spec_json": record.spec_json,
            "scenario": record.scenario,
            "capital_krw": record.capital_krw,
            "start": record.start.isoformat(),
            "end": record.end.isoformat(),
            "sim_config_json": record.sim_config_json,
            "cube_id": record.cube_id,
            "protocol_hash": record.protocol_hash,
            "engine_config_hash": record.engine_config_hash,
            "report_digest": record.report_digest,
            "created_at": record.created_at.isoformat(),
            "metrics": dict(record.metrics),
        }
        line = json.dumps(payload, sort_keys=True) + "\n"
        self._index.touch(exist_ok=True)
        with self._index.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                handle.write(line)
                handle.flush()
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
