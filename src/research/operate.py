"""Operational daily decisions over a tail research cube with persistent sleeve state."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

import numpy as np
from numpy.typing import NDArray

from src.core.pit import PITDataError
from src.research.book import sleeve_capital_krw
from src.research.cube import ResearchCube
from src.research.model import FrozenScorer, score_rows
from src.research.panel import FeaturePanel
from src.research.policy import select_row, universe_mask

if TYPE_CHECKING:
    from src.research.pipeline import StrategySpec

__all__ = [
    "OPERATE_SCHEMA",
    "OperateState",
    "SealedDecision",
    "SleeveState",
    "advance",
    "bootstrap_state",
    "load_state",
    "overlay_signals",
    "read_chain",
    "save_state",
    "seal",
    "state_digest",
]

OPERATE_SCHEMA: Final = "operate-v1"

_GENESIS_HASH = "0" * 64


@dataclass(frozen=True, slots=True)
class SleeveState:
    phase: int
    incumbents: tuple[str, ...]
    weights: Mapping[str, float]
    decided_session: date | None

    def __post_init__(self) -> None:
        if tuple(sorted(set(self.incumbents))) != self.incumbents:
            raise ValueError("sleeve incumbents must be sorted and unique")
        weights = dict(self.weights)
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in weights.values()):
            raise ValueError("sleeve weights must be finite fractions")
        object.__setattr__(self, "weights", MappingProxyType(weights))


@dataclass(frozen=True, slots=True)
class OperateState:
    spec_hash: str
    anchor_session: date
    last_session: date
    sessions_since_anchor: int
    sleeves: tuple[SleeveState, ...]

    def __post_init__(self) -> None:
        count = self.sessions_since_anchor
        if isinstance(count, bool) or not isinstance(count, int) or count < 0 or self.last_session < self.anchor_session:
            raise ValueError("state requires a non-negative session count and an ordered timeline")


@dataclass(frozen=True, slots=True)
class SealedDecision:
    data_session: date
    execution_session: date
    sealed_at: datetime
    spec_hash: str
    models_id: str
    inputs: Mapping[str, str]
    targets: Mapping[str, float]
    reference: Mapping[str, float]
    overlay: Mapping[str, float]
    state_digest: str
    prev_hash: str
    hash: str


def _canonical(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sleeve_payload(sleeve: SleeveState) -> dict[str, Any]:
    return {
        "decided_session": None if sleeve.decided_session is None else sleeve.decided_session.isoformat(),
        "incumbents": list(sleeve.incumbents),
        "phase": int(sleeve.phase),
        "weights": {key: float(value) for key, value in sorted(sleeve.weights.items())},
    }


def _state_payload(state: OperateState) -> dict[str, Any]:
    return {
        "anchor_session": state.anchor_session.isoformat(),
        "last_session": state.last_session.isoformat(),
        "schema": OPERATE_SCHEMA,
        "sessions_since_anchor": int(state.sessions_since_anchor),
        "sleeves": [_sleeve_payload(sleeve) for sleeve in state.sleeves],
        "spec_hash": state.spec_hash,
    }


def state_digest(state: OperateState) -> str:
    """SHA-256 of the canonical JSON of the state."""
    return hashlib.sha256(_canonical(_state_payload(state)).encode("utf-8")).hexdigest()


def _decision_payload(decision: SealedDecision) -> dict[str, Any]:
    return {
        "data_session": decision.data_session.isoformat(),
        "execution_session": decision.execution_session.isoformat(),
        "inputs": {key: str(value) for key, value in sorted(decision.inputs.items())},
        "models_id": decision.models_id,
        "overlay": {key: float(value) for key, value in sorted(decision.overlay.items())},
        "prev_hash": decision.prev_hash,
        "reference": {key: float(value) for key, value in sorted(decision.reference.items())},
        "sealed_at": decision.sealed_at.isoformat(),
        "spec_hash": decision.spec_hash,
        "state_digest": decision.state_digest,
        "targets": {key: float(value) for key, value in sorted(decision.targets.items())},
    }


def _decision_hash(prev_hash: str, payload: Mapping[str, Any]) -> str:
    return hashlib.sha256((str(prev_hash) + _canonical(dict(payload))).encode("utf-8")).hexdigest()


def bootstrap_state(
    spec: StrategySpec,
    cube: ResearchCube,
    sleeve_targets: Sequence[Mapping[int, NDArray[np.float64]]],
    incumbents: Sequence[frozenset[int]],
    *,
    lo: int,
    last_row: int,
) -> OperateState:
    """Operational state equivalent to the research path having run through last_row."""
    sessions = list(cube.sessions)
    total = len(sessions)
    n_sleeves = int(spec.book.sleeves)
    sleeves_in = list(sleeve_targets)
    incumb_in = list(incumbents)
    if len(sleeves_in) != n_sleeves:
        raise ValueError(f"sleeve_targets hold {len(sleeves_in)} sleeves, need {n_sleeves}")
    if len(incumb_in) != n_sleeves:
        raise ValueError(f"incumbents hold {len(incumb_in)} sleeves, need {n_sleeves}")
    if isinstance(lo, bool) or not isinstance(lo, int) or not 1 <= lo < total:
        raise ValueError(f"lo {lo!r} is not within [1, {total})")
    if isinstance(last_row, bool) or not isinstance(last_row, int) or not lo - 1 <= last_row < total:
        raise ValueError(f"last_row {last_row!r} is not within [{lo - 1}, {total})")
    sleeves = list(sleeve_targets)
    incumb_list = list(incumbents)
    insts = list(cube.instrument_ids)
    out: list[SleeveState] = []
    for phase in range(n_sleeves):
        mapping = dict(sleeves[phase])
        rows = sorted(row for row in mapping if isinstance(row, int) and not isinstance(row, bool) and row <= last_row)
        if rows:
            last = int(rows[-1])
            weights = {
                insts[col]: float(value)
                for col, value in enumerate(np.asarray(mapping[last], dtype=np.float64).tolist())
                if 0 <= col < len(insts) and math.isfinite(float(value)) and float(value) != 0.0
            }
            decided: date | None = sessions[last]
        else:
            weights = {}
            decided = None
        ids = sorted(
            {insts[col] for col in incumb_list[phase] if isinstance(col, int) and 0 <= int(col) < len(insts)}
        )
        out.append(
            SleeveState(
                phase=int(phase),
                incumbents=tuple(ids),
                weights=dict(weights),
                decided_session=decided,
            )
        )
    return OperateState(
        spec_hash=spec.spec_hash,
        anchor_session=sessions[lo - 1],
        last_session=sessions[last_row],
        sessions_since_anchor=int(last_row - (lo - 1)),
        sleeves=tuple(out),
    )


def overlay_signals(
    spec: StrategySpec, *, trend_levels: NDArray[np.float64] | None, hedge_levels: NDArray[np.float64]
) -> dict[str, float]:
    """Signal-level overlay values from index levels ending at the decision session."""
    hedge = np.asarray(hedge_levels, dtype=np.float64)
    trend = None if trend_levels is None else np.asarray(trend_levels, dtype=np.float64)
    overlay = getattr(spec, "trend_overlay", None)
    regime = getattr(spec, "regime_hedge", None)
    if overlay is None:
        signal = 0.0
        scale = 1.0
    else:
        if trend is None or trend.size == 0:
            raise PITDataError("trend index level is missing at the decision session")
        idx = int(trend.shape[0]) - 1
        raw: float | None = None
        if str(getattr(overlay, "signal", "ma")) == "tsmom":
            from fractions import Fraction

            from src.research.trend_overlay import tsmom_signal

            value: Fraction | None = tsmom_signal(trend, idx, tuple(int(h) for h in overlay.tsmom_horizons))
            raw = None if value is None else float(value)
        else:
            window = int(overlay.ma_sessions)
            span = trend[max(idx - window + 1, 0) : idx + 1]
            if int(np.count_nonzero(np.isfinite(span) & (span > 0.0))) < window:
                raw = None
            else:
                top = float(np.max(span))
                raw = 1.0 if float(trend[idx]) / top > float(np.mean(span / top)) else -1.0
        signal = 0.0 if raw is None else float(raw)
        if getattr(overlay, "vol_cap", None) is None:
            scale = 1.0
        else:
            from src.research.trend_overlay import realised_vol

            sigma = realised_vol(trend, idx, int(overlay.vol_window_sessions))
            if sigma is None or not math.isfinite(sigma):
                signal = 0.0
                scale = 1.0
            else:
                scale = 1.0 if sigma <= float(overlay.vol_cap) else float(overlay.vol_cap) / sigma
    if regime is None:
        fraction = 0.0
    else:
        from src.research.trend_overlay import realised_vol, tsmom_signal

        idx = int(hedge.shape[0]) - 1
        value = tsmom_signal(hedge, idx, tuple(int(h) for h in regime.tsmom_horizons))
        if value is None or float(value) >= 0.0:
            fraction = 0.0
        else:
            sigma = realised_vol(hedge, idx, int(regime.vol_window_sessions))
            if sigma is None or not math.isfinite(sigma):
                fraction = 0.0
            else:
                fraction = (
                    float(regime.max_fraction) if sigma <= 0.0
                    else min(float(regime.max_fraction), float(regime.target_vol) / sigma)
                ) * abs(float(value))
    return {
        "regime_fraction": float(fraction),
        "trend_signal": float(signal),
        "trend_vol_scale": float(scale),
    }


def _advance_with_states(
    state: OperateState,
    spec: StrategySpec,
    cube: ResearchCube,
    panel: FeaturePanel,
    scorer: FrozenScorer,
    *,
    capital_krw: int,
    cash_buffer: float,
    series: Mapping[str, NDArray[np.float64]],
) -> tuple[OperateState, tuple[tuple[date, Mapping[str, float]], ...], tuple[OperateState, ...]]:
    if state.spec_hash != spec.spec_hash:
        raise ValueError("state spec hash differs from the strategy spec")
    if isinstance(capital_krw, bool) or not isinstance(capital_krw, int) or capital_krw < 1:
        raise ValueError(f"capital_krw must be an int >= 1, got {capital_krw!r}")
    buffer = float(cash_buffer)
    if not math.isfinite(buffer) or not 0.0 <= buffer < 1.0:
        raise ValueError(f"cash_buffer must satisfy 0 <= cash_buffer < 1, got {cash_buffer!r}")
    every = int(spec.policy.rebalance_every_sessions)
    if every < 1:
        raise ValueError(f"rebalance_every_sessions must be >= 1, got {every!r}")
    sessions = list(cube.sessions)
    try:
        start = sessions.index(state.last_session)
    except ValueError:
        raise PITDataError(
            f"state last session {state.last_session.isoformat()} is not in the cube"
        ) from None
    if (
        state.anchor_session in cube.sessions
        and start - cube.sessions.index(state.anchor_session) != state.sessions_since_anchor
    ):
        raise PITDataError("state session count differs from the cube timeline")
    if start + 1 < len(sessions):
        from src.core.krx_calendar import xkrx_calendar_through
        from src.core.time import KRX_TZ

        expected_sessions = tuple(
            instant.astimezone(KRX_TZ).date()
            for instant in xkrx_calendar_through(sessions[-1]).sessions
            if state.last_session < instant.astimezone(KRX_TZ).date() <= sessions[-1]
        )
        if tuple(sessions[start + 1 :]) != expected_sessions:
            raise PITDataError("cube skips an XKRX session after the state last session")
    if len(state.sleeves) != int(spec.book.sleeves):
        raise ValueError("state sleeves differ from the strategy book")
    for sleeve, phase in zip(state.sleeves, range(len(state.sleeves)), strict=True):
        if int(sleeve.phase) != int(phase):
            raise ValueError("state sleeve phases must tile range(sleeves)")
    if panel.last_row != len(sessions) - 1:
        raise ValueError(f"panel last_row {panel.last_row} differs from the cube end {len(sessions) - 1}")
    from src.research.panel import feature_names_for as _names_for

    feat_mats = [np.asarray(panel.features[name]) for name in _names_for(scorer.config.feature_set)]
    n_rows = feat_mats[0].shape[0] if feat_mats else 0
    if n_rows != len(sessions):
        raise ValueError("panel features are not aligned to the cube sessions")
    universe = np.ascontiguousarray(np.asarray(universe_mask(cube, spec.policy.universe), dtype=bool))
    close = np.asarray(cube.arrays["close"], dtype=np.float64)
    dev_enabled = spec.policy.trend_min_dev_ma20 is not None
    ret_enabled = spec.policy.trend_min_ret21 is not None
    dev = np.asarray(panel.features["dev_ma20"], dtype=np.float64) if dev_enabled else None
    ret = np.asarray(panel.features["ret_21"], dtype=np.float64) if ret_enabled else None
    slot_krw = float(sleeve_capital_krw(int(capital_krw), spec.book)) * (1.0 - buffer) / float(spec.policy.n)
    index_of = {inst: col for col, inst in enumerate(cube.instrument_ids)}
    current_weights: list[dict[str, float]] = [
        {inst: weight for inst, weight in sleeve.weights.items() if inst in index_of}
        for sleeve in state.sleeves
    ]
    current_incumbents: list[frozenset[int]] = [
        frozenset(index_of[inst] for inst in sleeve.incumbents if inst in index_of)
        for sleeve in state.sleeves
    ]
    decided: list[date | None] = [sleeve.decided_session for sleeve in state.sleeves]
    k = int(state.sessions_since_anchor)
    pending = range(start + 1, len(sessions))
    required = ["cash_returns", "hedge_levels", "hedge_inverse"]
    if spec.trend_overlay is not None:
        required.append("trend_levels")
    for name in required:
        values = np.asarray(series.get(name), dtype=np.float64)
        if values.shape != (len(sessions),):
            raise PITDataError(f"{name} is missing or not aligned to cube sessions")
        valid = np.isfinite(values[start + 1 :])
        if name != "cash_returns":
            valid &= values[start + 1 :] > 0.0
        if not bool(np.all(valid)):
            raise PITDataError(f"{name} is missing or invalid after {state.last_session.isoformat()}")
    decisions: list[tuple[date, Mapping[str, float]]] = []
    snapshots: list[OperateState] = []
    for pos in pending:
        k += 1
        day = sessions[pos]
        phase = k % every
        if int(scorer.year) != int(day.year):
            raise ValueError(f"frozen scorer year {scorer.year} != decision year {day.year}")
        scored = np.asarray(score_rows(feat_mats, universe, [pos], scorer), dtype=np.float64)[0]
        weights, nxt = select_row(
            spec.policy,
            score_row=scored,
            universe_row=universe[pos],
            close_row=close[pos],
            dev_row=None if dev is None else dev[pos],
            ret_row=None if ret is None else ret[pos],
            incumbents=current_incumbents[phase],
            slot_krw=slot_krw,
        )
        current_incumbents[phase] = frozenset(int(c) for c in nxt)
        current_weights[phase] = {
            cube.instrument_ids[col]: float(value)
            for col, value in enumerate(np.asarray(weights, dtype=np.float64).tolist())
            if math.isfinite(float(value)) and float(value) != 0.0
        }
        decided[phase] = day
        combined: dict[str, float] = {}
        for book in current_weights:
            for key, value in book.items():
                combined[key] = float(combined.get(key, 0.0)) + float(value)
        combined = {key: value / float(len(current_weights)) for key, value in combined.items() if value != 0.0}
        decisions.append((day, combined))
        snapshots.append(
            OperateState(
                spec_hash=state.spec_hash,
                anchor_session=state.anchor_session,
                last_session=day,
                sessions_since_anchor=k,
                sleeves=tuple(
                    SleeveState(
                        phase=int(sleeve.phase),
                        incumbents=tuple(sorted({cube.instrument_ids[c] for c in current_incumbents[idx]})),
                        weights=dict(current_weights[idx]),
                        decided_session=decided[idx],
                    )
                    for idx, sleeve in enumerate(state.sleeves)
                ),
            )
        )
    final = snapshots[-1] if snapshots else state
    return final, tuple(decisions), tuple(snapshots)


def advance(
    state: OperateState,
    spec: StrategySpec,
    cube: ResearchCube,
    panel: FeaturePanel,
    scorer: FrozenScorer,
    *,
    capital_krw: int,
    cash_buffer: float,
    series: Mapping[str, NDArray[np.float64]],
) -> tuple[OperateState, tuple[tuple[date, Mapping[str, float]], ...]]:
    """Advance through every cube session after state.last_session; returns the new state and each session's combined targets."""
    final, decisions, _ = _advance_with_states(
        state, spec, cube, panel, scorer, capital_krw=capital_krw, cash_buffer=cash_buffer, series=series
    )
    return final, decisions


def _chain_path(decisions_dir: Path) -> Path:
    return Path(decisions_dir) / "chain.jsonl"


def _decision_file(decisions_dir: Path, day: date) -> Path:
    return Path(decisions_dir) / f"{day.isoformat()}.json"


def _decision_from_payload(payload: Mapping[str, Any]) -> SealedDecision:
    try:
        return SealedDecision(
            data_session=date.fromisoformat(str(payload["data_session"])),
            execution_session=date.fromisoformat(str(payload["execution_session"])),
            sealed_at=datetime.fromisoformat(str(payload["sealed_at"])),
            spec_hash=str(payload["spec_hash"]),
            models_id=str(payload["models_id"]),
            inputs={str(k): str(v) for k, v in dict(payload["inputs"]).items()},
            targets={str(k): float(v) for k, v in dict(payload["targets"]).items()},
            reference={str(k): float(v) for k, v in dict(payload["reference"]).items()},
            overlay={str(k): float(v) for k, v in dict(payload["overlay"]).items()},
            state_digest=str(payload["state_digest"]),
            prev_hash=str(payload["prev_hash"]),
            hash=str(payload["hash"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"malformed sealed decision: {exc}") from exc


def read_chain(decisions_dir: Path) -> tuple[SealedDecision, ...]:
    """All sealed decisions in order after verifying the hash chain; raises ValueError on a broken link."""
    path = _chain_path(decisions_dir)
    if not path.is_file():
        return ()
    out: list[SealedDecision] = []
    expected = _GENESIS_HASH
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"unreadable decision chain: {exc}") from exc
    for line in lines:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"malformed decision chain line: {exc}") from exc
        decision = _decision_from_payload(payload)
        if out and decision.data_session <= out[-1].data_session:
            raise ValueError("decision sessions must be strictly increasing")
        if decision.prev_hash != expected:
            raise ValueError(f"broken decision chain before {decision.data_session.isoformat()}")
        if _decision_hash(decision.prev_hash, _decision_payload(decision)) != decision.hash:
            raise ValueError(f"broken decision chain at {decision.data_session.isoformat()}")
        try:
            raw = _decision_file(path.parent, decision.data_session).read_text(encoding="utf-8")
            filed = _decision_from_payload(json.loads(raw))
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"sealed decision file is missing for {decision.data_session.isoformat()}: {exc}"
            ) from exc
        if _decision_payload(filed) != _decision_payload(decision) or filed.hash != decision.hash:
            raise ValueError(f"sealed decision file was modified for {decision.data_session.isoformat()}")
        expected = decision.hash
        out.append(decision)
    return tuple(out)


def seal(decisions_dir: Path, decision: SealedDecision) -> SealedDecision:
    """Append one decision to the hash chain; returns it with prev_hash and hash set. Raises on a conflicting re-seal."""
    from src.core.krx_calendar import xkrx_calendar_through
    from src.core.time import KRX_TZ

    day = decision.data_session
    execution = xkrx_calendar_through(day).advance(
        datetime(day.year, day.month, day.day, tzinfo=KRX_TZ), 1
    ).astimezone(KRX_TZ).date()
    if decision.execution_session != execution or decision.sealed_at.utcoffset() is None:
        raise ValueError("decision requires the next XKRX execution session and an aware sealed_at")
    root = Path(decisions_dir)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".chain.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            return _seal(root, decision)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _seal(root: Path, decision: SealedDecision) -> SealedDecision:
    chain = read_chain(root)
    prev = chain[-1].hash if chain else _GENESIS_HASH
    target = _decision_file(root, decision.data_session)
    if target.is_file():
        from_actual = _decision_payload(decision)
        try:
            existing = _decision_from_payload(json.loads(target.read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            raise ValueError(f"unreadable sealed decision for {decision.data_session.isoformat()}: {exc}") from exc
        old = _decision_payload(existing)
        if (
            {k: v for k, v in old.items() if k != "prev_hash"}
            == {k: v for k, v in from_actual.items() if k != "prev_hash"}
            and existing.hash == _decision_hash(existing.prev_hash, old)
        ):
            if not any(item.data_session == existing.data_session for item in chain):
                if existing.prev_hash != prev or (chain and existing.data_session <= chain[-1].data_session):
                    raise ValueError("orphan decision does not extend the chain")
                record = dict(old, hash=existing.hash)
                _append_record(root, record)
            return existing
        raise ValueError(f"conflicting seal for {decision.data_session.isoformat()}")
    if chain and decision.data_session <= chain[-1].data_session:
        raise ValueError("decision sessions must be strictly increasing")
    staged = SealedDecision(
        data_session=decision.data_session,
        execution_session=decision.execution_session,
        sealed_at=decision.sealed_at,
        spec_hash=decision.spec_hash,
        models_id=decision.models_id,
        inputs=dict(decision.inputs),
        targets=dict(decision.targets),
        reference=dict(decision.reference),
        overlay=dict(decision.overlay),
        state_digest=decision.state_digest,
        prev_hash=prev,
        hash="",
    )
    digest = _decision_hash(prev, _decision_payload(staged))
    sealed = SealedDecision(
        data_session=staged.data_session,
        execution_session=staged.execution_session,
        sealed_at=staged.sealed_at,
        spec_hash=staged.spec_hash,
        models_id=staged.models_id,
        inputs=dict(staged.inputs),
        targets=dict(staged.targets),
        reference=dict(staged.reference),
        overlay=dict(staged.overlay),
        state_digest=staged.state_digest,
        prev_hash=prev,
        hash=digest,
    )
    record = dict(_decision_payload(sealed))
    record["hash"] = sealed.hash
    _write_json(target, record)
    _append_record(root, record)
    return sealed


def _append_record(root: Path, record: Mapping[str, Any]) -> None:
    with _chain_path(root).open("a", encoding="utf-8") as handle:
        handle.write(_canonical(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def save_state(path: Path, state: OperateState) -> None:
    """Atomically persist the operational state as canonical JSON."""
    _write_json(path, _state_payload(state))


def _write_json(path: Path, record: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical(record) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            os.replace(temporary, target)
        finally:
            with contextlib.suppress(FileNotFoundError):
                temporary.unlink()


def load_state(path: Path) -> OperateState:
    """Load an operational state written by save_state."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("operate state must be an object")
    except (OSError, ValueError) as exc:
        raise ValueError(f"unreadable operate state: {exc}") from exc
    try:
        if payload.get("schema") != OPERATE_SCHEMA:
            raise ValueError(f"unsupported operate schema {payload.get('schema')!r}")
        sleeves = tuple(
            SleeveState(
                phase=int(item["phase"]),
                incumbents=tuple(str(v) for v in item["incumbents"]),
                weights={str(k): float(v) for k, v in dict(item["weights"]).items()},
                decided_session=None
                if item["decided_session"] is None
                else date.fromisoformat(str(item["decided_session"])),
            )
            for item in payload["sleeves"]
        )
        return OperateState(
            spec_hash=str(payload["spec_hash"]),
            anchor_session=date.fromisoformat(str(payload["anchor_session"])),
            last_session=date.fromisoformat(str(payload["last_session"])),
            sessions_since_anchor=payload["sessions_since_anchor"],
            sleeves=sleeves,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"malformed operate state: {exc}") from exc
