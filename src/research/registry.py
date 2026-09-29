"""Append-only trial registry with bit-identical return verification."""
from __future__ import annotations

import fcntl
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray

from src.core.pit import PITDataError
from src.data.research_protocol import Segment

__all__ = ["TrialRecord", "TrialRegistry", "TrialReturns"]


@dataclass(frozen=True, slots=True)
class TrialRecord:
    trial_id: str
    family: str
    spec_hash: str
    spec_json: str
    segment: Segment
    start: date
    end: date
    sim_config_json: str
    cube_id: str
    created_at: datetime
    metrics: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class TrialReturns:
    sessions: tuple[date, ...]
    net: NDArray[np.float64]
    benchmarks: Mapping[str, NDArray[np.float64]]


def _canonical_trial_id(
    *, spec_json: str, segment: Segment, start: date, end: date, sim_config_json: str, cube_id: str
) -> str:
    payload = {
        "cube_id": cube_id,
        "end": end.isoformat(),
        "segment": segment.value,
        "sim_config_json": sim_config_json,
        "spec_json": spec_json,
        "start": start.isoformat(),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]


class TrialRegistry:
    """Append-only record of every evaluated configuration and its daily returns.

    Multiplicity statistics are only honest when every evaluation is counted, including
    failures, so recording is part of evaluation, not an optional export.
    """

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
        segment: Segment,
        sim_config_json: str,
        cube_id: str,
        returns: TrialReturns,
        metrics: Mapping[str, float],
        now: datetime,
    ) -> TrialRecord:
        sessions = tuple(returns.sessions)
        net = np.asarray(returns.net, dtype=np.float64)
        benchmarks = {key: np.asarray(value, dtype=np.float64) for key, value in returns.benchmarks.items()}
        self._check_arrays(sessions, net, benchmarks)
        start, end = sessions[0], sessions[-1]
        trial_id = _canonical_trial_id(
            spec_json=spec_json,
            segment=segment,
            start=start,
            end=end,
            sim_config_json=sim_config_json,
            cube_id=cube_id,
        )
        existing = self._find_record(trial_id)
        if existing is not None:
            stored = self.returns(trial_id)
            if (
                stored.sessions != sessions
                or not np.array_equal(stored.net, net)
                or set(stored.benchmarks) != set(benchmarks)
                or any(not np.array_equal(stored.benchmarks[key], benchmarks[key]) for key in benchmarks)
            ):
                raise PITDataError(f"non-deterministic trial returns: {trial_id}")
            return existing
        record = TrialRecord(
            trial_id=trial_id,
            family=family,
            spec_hash=spec_hash,
            spec_json=spec_json,
            segment=segment,
            start=start,
            end=end,
            sim_config_json=sim_config_json,
            cube_id=cube_id,
            created_at=now,
            metrics=dict(metrics),
        )
        self._write_returns(trial_id, sessions, net, benchmarks)
        self._append_index(record)
        return record

    def trials(self, *, segment: Segment | None = None, family: str | None = None) -> tuple[TrialRecord, ...]:
        records: list[TrialRecord] = []
        for line in self._read_lines():
            record = self._parse_line(line)
            if segment is not None and record.segment is not segment:
                continue
            if family is not None and record.family != family:
                continue
            records.append(record)
        return tuple(records)

    def returns(self, trial_id: str) -> TrialReturns:
        path = self._returns_dir / f"{trial_id}.parquet"
        try:
            frame = pl.read_parquet(path)
        except (OSError, ValueError) as exc:
            raise PITDataError(f"unreadable trial returns: {trial_id}") from exc
        columns = frame.columns
        if "session" not in columns or "net" not in columns:
            raise PITDataError(f"invalid trial returns: {trial_id}")
        sessions = tuple(date.fromisoformat(str(value)) for value in frame["session"].to_list())
        net = np.asarray(frame["net"].to_numpy(), dtype=np.float64)
        benchmarks: dict[str, NDArray[np.float64]] = {}
        for column in columns:
            if column.startswith("bench__"):
                benchmarks[column[len("bench__") :]] = np.asarray(frame[column].to_numpy(), dtype=np.float64)
        self._check_arrays(sessions, net, benchmarks)
        return TrialReturns(sessions=sessions, net=net, benchmarks=benchmarks)

    @staticmethod
    def _check_arrays(
        sessions: tuple[date, ...], net: NDArray[np.float64], benchmarks: Mapping[str, NDArray[np.float64]]
    ) -> None:
        if not sessions:
            raise PITDataError("trial returns must have at least one session")
        if net.shape != (len(sessions),):
            raise PITDataError("trial net returns length must match sessions")
        for key, values in benchmarks.items():
            if values.shape != (len(sessions),):
                raise PITDataError(f"trial benchmark length must match sessions: {key}")
        if not np.all(np.isfinite(net)):
            raise PITDataError("trial net returns must be finite")
        for key, values in benchmarks.items():
            if not np.all(np.isfinite(values)):
                raise PITDataError(f"trial benchmark returns must be finite: {key}")

    def _find_record(self, trial_id: str) -> TrialRecord | None:
        for line in self._read_lines():
            record = self._parse_line(line)
            if record.trial_id == trial_id:
                return record
        return None

    def _read_lines(self) -> list[str]:
        if not self._index.exists():
            return []
        try:
            text = self._index.read_text(encoding="utf-8")
        except OSError as exc:
            raise PITDataError(f"unreadable trial index: {self._index}") from exc
        lines = [line for line in text.splitlines() if line.strip()]
        for line in lines:
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                raise PITDataError(f"invalid trial index line: {self._index}") from exc
        return lines

    @staticmethod
    def _parse_line(line: str) -> TrialRecord:
        try:
            raw = json.loads(line)
            return TrialRecord(
                trial_id=str(raw["trial_id"]),
                family=str(raw["family"]),
                spec_hash=str(raw["spec_hash"]),
                spec_json=str(raw["spec_json"]),
                segment=Segment(str(raw["segment"])),
                start=date.fromisoformat(str(raw["start"])),
                end=date.fromisoformat(str(raw["end"])),
                sim_config_json=str(raw["sim_config_json"]),
                cube_id=str(raw["cube_id"]),
                created_at=datetime.fromisoformat(str(raw["created_at"])),
                metrics=dict(raw["metrics"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PITDataError(f"invalid trial index line: {exc}") from exc

    def _write_returns(
        self,
        trial_id: str,
        sessions: tuple[date, ...],
        net: NDArray[np.float64],
        benchmarks: Mapping[str, NDArray[np.float64]],
    ) -> None:
        self._returns_dir.mkdir(parents=True, exist_ok=True)
        target = self._returns_dir / f"{trial_id}.parquet"
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
        temporary = self._returns_dir / f".{trial_id}.tmp.parquet"
        frame.write_parquet(temporary)
        temporary.replace(target)

    def _append_index(self, record: TrialRecord) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        payload = {
            "trial_id": record.trial_id,
            "family": record.family,
            "spec_hash": record.spec_hash,
            "spec_json": record.spec_json,
            "segment": record.segment.value,
            "start": record.start.isoformat(),
            "end": record.end.isoformat(),
            "sim_config_json": record.sim_config_json,
            "cube_id": record.cube_id,
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
