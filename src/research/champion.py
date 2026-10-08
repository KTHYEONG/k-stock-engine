"""Champion/challenger promotion rule and the append-only champion store.

Why a paired comparison: research iterates without limit, so the number of ideas tried cannot be the
safeguard. Every challenger is instead measured against the incumbent on the very same sessions and the same
stress stream with a paired block bootstrap, so shared market noise cancels and the decision is far more
sensitive than either absolute statistic. A wrong promotion can therefore only swap in a strategy that is
statistically indistinguishable from the champion; the neighbor-plateau rule keeps numeric knobs off knife
edges and the report card's cost grid covers execution-cost uncertainty.

Two promotion paths share the report, J, neighbor-coverage, alpha and horizon gates. Superiority needs the
paired growth lower bound above 0. Non-inferiority (enabled by a declared margin δ) needs the growth lower bound
above -δ and the paired monthly-block CVaR lower bound above 0, because the superiority test alone can never
promote a change that keeps growth while cutting tail risk. Superiority wins when both pass.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal, cast, get_args

import numpy as np
from numpy.typing import NDArray

from src.core.pit import PITDataError
from src.data.research_protocol import ResearchProtocol
from src.research.evaluation import EvaluationEvidence, EvaluationPolicy
from src.research.pipeline import EvaluationRun, StrategySpec, strategy_spec_from_canonical_json
from src.research.stats import PairedDelta, PairedRiskDelta, paired_growth_tail_delta

__all__ = [
    "ChallengeDecision",
    "ChampionRecord",
    "ChampionStore",
    "champion_record_fields",
    "decide_challenge",
    "decision_from_canonical_json",
    "knob_changes",
]

STRESS_STREAMS = ("stress_slippage", "stress_delay")
PromotionPath = Literal["superiority", "noninferiority", "none"]
TREND_CONTRACT_KNOBS = frozenset({
    "trend_overlay.contract_multiplier_krw",
    "trend_overlay.initial_margin_rate",
    "trend_overlay.margin_buffer_rate",
    "trend_overlay.margin_topup_trigger_fraction",
    "trend_overlay.futures_cost_rate",
    "trend_overlay.futures_tax_rate",
    "trend_overlay.futures_annual_deduction_krw",
    "trend_overlay.vol_window_sessions",
})
REGIME_HEDGE_EXEMPT_KNOBS = frozenset({
    "regime_hedge.contract_multiplier_krw",
    "regime_hedge.initial_margin_rate",
    "regime_hedge.margin_buffer_rate",
    "regime_hedge.margin_topup_trigger_fraction",
    "regime_hedge.futures_cost_rate",
    "regime_hedge.futures_tax_rate",
    "regime_hedge.futures_annual_deduction_krw",
    "regime_hedge.vol_window_sessions",
    "regime_hedge.rebalance_every_sessions",
    "regime_hedge.max_fraction",
})


@dataclass(frozen=True, slots=True)
class ChampionRecord:
    """The spec that is currently authorized to trade, plus how it got there."""

    spec_hash: str
    spec_path: str  # repo-relative TOML path at promotion time; the TOML may move or change later
    spec_json: str  # canonical JSON is the identity, the path is only a hint for the operator
    run_id: str
    report_digest: str
    objective_j: float
    promoted_at: datetime
    reason: str  # "bootstrap" | "challenge" | "policy_override"
    decision_digest: str | None
    rationale: str | None = None  # operator's stated reason; only a "policy_override" record carries one


@dataclass(frozen=True, slots=True)
class ChallengeDecision:
    """One champion/challenger comparison on one window; the audit record of a promotion.

    ``paired`` and every neighbor delta are annualised paired growth differences (challenger minus champion) on
    the stress stream named by the challenger's report card, so a pair is only ever formed on shared sessions.
    """

    challenger_hash: str
    champion_hash: str
    run_ids: tuple[str, str]  # (challenger, champion) on the same window
    window: tuple[date, date]
    paired: PairedDelta
    challenger_j: float
    champion_j: float
    neighbors: tuple[tuple[str, PairedDelta], ...]  # (neighbor spec_hash, Δg vs champion)
    knob_changes: tuple[str, ...]  # dotted paths of differing numeric policy/book/hedge/scorer leaves
    reasons: tuple[str, ...]  # failed rules; empty when promotable
    alpha_effective: float
    paired_horizon_sessions: int
    path: PromotionPath
    tail: PairedDelta | None  # challenger - champion CVaR delta; None when the path is disabled
    noninferiority_margin: float | None
    prior_decisions: int = 0

    @property
    def promotable(self) -> bool:
        return self.path != "none"

    def canonical_json(self) -> str:
        """Key-sorted compact JSON; byte-identical for identical evidence."""
        payload = {
            "alpha_effective": _canon(self.alpha_effective),
            "challenger_hash": self.challenger_hash,
            "challenger_j": _canon(self.challenger_j),
            "champion_hash": self.champion_hash,
            "champion_j": _canon(self.champion_j),
            "knob_changes": list(self.knob_changes),
            "neighbors": [[spec_hash, _delta_fields(delta)] for spec_hash, delta in self.neighbors],
            "noninferiority_margin": (
                _canon(self.noninferiority_margin) if self.noninferiority_margin is not None else None
            ),
            "paired": _delta_fields(self.paired),
            "paired_horizon_sessions": int(self.paired_horizon_sessions),
            "path": self.path,
            "prior_decisions": self.prior_decisions,
            "promotable": self.promotable,
            "reasons": list(self.reasons),
            "run_ids": list(self.run_ids),
            "tail": _delta_fields(self.tail) if self.tail is not None else None,
            "window": [self.window[0].isoformat(), self.window[1].isoformat()],
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @property
    def digest(self) -> str:
        """SHA-256 hex of the canonical JSON."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def decide_challenge(
    *,
    challenger: EvaluationRun,
    champion: EvaluationRun,
    neighbors: Sequence[EvaluationRun],
    challenger_spec: StrategySpec,
    champion_spec: StrategySpec,
    protocol: ResearchProtocol,
    policy: EvaluationPolicy,
    prior_decisions: int = 0,
) -> ChallengeDecision:
    """Apply the promotion rule.

    Promotable iff all hold:
      1. ``challenger.report.passed``;
      2. paired Δg (challenger minus champion) lower bound at ``alpha_eff`` > 0, computed on the
         stress stream named by the challenger's ``objective_stream`` for both runs, where ``alpha_eff``
         is ``alpha / (1 + prior_decisions)`` under ``multiplicity="bonferroni_decisions"`` else ``alpha``,
         and the paired bootstrap horizon is ``len(sessions)`` under ``paired_horizon="full"`` else
         ``policy.horizon_sessions``;
      3. ``challenger J ≥ champion J``;
      4. when ``knob_changes`` is non-empty and ``require_neighbors``: at least one neighbor per changed knob
         was supplied and every neighbor has mean Δg vs champion > 0. Exchange contract terms and overlay
         activation are exempt; trend MA and long/short fractions still require neighbors on introduction.

    Raises ValueError when the two runs' sessions differ (never compares different windows), when a run was
    not produced by the spec it is paired with, or when ``prior_decisions`` is negative.
    """
    if isinstance(prior_decisions, bool) or not isinstance(prior_decisions, int) or prior_decisions < 0:
        raise ValueError(f"prior_decisions must be an int >= 0, got {prior_decisions!r}")
    sessions = tuple(champion.evidence.sessions)
    if not sessions:
        raise ValueError("champion evidence sessions must be non-empty")
    if tuple(challenger.evidence.sessions) != sessions:
        raise ValueError("challenger and champion must be evaluated on identical sessions")
    if challenger.report.spec_hash != challenger_spec.spec_hash:
        raise ValueError("challenger run was not produced by challenger_spec")
    if champion.report.spec_hash != champion_spec.spec_hash:
        raise ValueError("champion run was not produced by champion_spec")
    neighbor_runs = tuple(neighbors)
    for run in neighbor_runs:
        if tuple(run.evidence.sessions) != sessions:
            raise ValueError("every neighbor must be evaluated on the champion sessions")

    stream = str(challenger.report.objective_stream)
    if stream not in STRESS_STREAMS:
        raise ValueError(f"objective stream must be one of {STRESS_STREAMS}, got {stream!r}")
    if protocol.champion.multiplicity == "bonferroni_decisions":
        alpha_effective = float(protocol.champion.alpha) / (1 + prior_decisions)
    else:
        alpha_effective = float(protocol.champion.alpha)
    if protocol.champion.paired_horizon == "full":
        paired_horizon_sessions = len(sessions)
    else:
        paired_horizon_sessions = int(policy.horizon_sessions)
    paired_risk = _paired_risk_delta(
        challenger, champion, stream=stream, policy=policy, protocol=protocol,
        alpha=alpha_effective, horizon=paired_horizon_sessions,
    )
    paired = paired_risk.growth
    neighbor_risk_deltas = tuple(
        (
            run.report.spec_hash,
            _paired_risk_delta(
                run, champion, stream=stream, policy=policy, protocol=protocol,
                alpha=alpha_effective, horizon=paired_horizon_sessions,
            ),
        )
        for run in neighbor_runs
    )
    neighbor_deltas = tuple((spec_hash, risk.growth) for spec_hash, risk in neighbor_risk_deltas)
    changes = knob_changes(challenger_spec, champion_spec)

    reasons: list[str] = []
    if not challenger.report.passed:
        reasons.append("report_failed")
    challenger_j = float(challenger.report.objective_j)
    champion_j = float(champion.report.objective_j)
    if not (math.isfinite(challenger_j) and math.isfinite(champion_j) and challenger_j >= champion_j):
        reasons.append("objective_j")

    if protocol.champion.require_neighbors and changes:
        covered = _knob_coverage(challenger_spec, neighbor_runs)
        exempt = set(TREND_CONTRACT_KNOBS)
        exempt.update(REGIME_HEDGE_EXEMPT_KNOBS)
        if (challenger_spec.trend_overlay is None) != (champion_spec.trend_overlay is None):
            # One-overlay accounting forbids a hedge-ratio neighbor while the trend overlay is active.
            exempt.update({"hedge.hedge_ratio", "trend_overlay.rebalance_every_sessions"})
        if challenger_spec.trend_overlay is None:
            exempt.update(knob for knob in changes if knob.startswith("trend_overlay."))
        elif (
            champion_spec.trend_overlay is not None
            and challenger_spec.trend_overlay.signal != champion_spec.trend_overlay.signal
        ):
            exempt.add("trend_overlay.ma_sessions")
        if challenger_spec.regime_hedge is None:
            exempt.update(knob for knob in changes if knob.startswith("regime_hedge."))
        for knob in changes:
            if knob.startswith("scorer.") or knob in covered or knob in exempt:
                continue
            reasons.append(f"neighbors_missing:{knob}")

    superiority_failures: list[str] = []
    if not (math.isfinite(paired.lower) and paired.lower > 0.0):
        superiority_failures.append("superiority:paired_lower_bound")
    if protocol.champion.require_neighbors and changes:
        for spec_hash, delta in neighbor_deltas:
            if not (math.isfinite(delta.mean) and delta.mean > 0.0):
                superiority_failures.append(f"superiority:neighbor_not_better:{spec_hash}")

    delta_margin = protocol.champion.noninferiority_margin
    noninferiority_failures: list[str] = []
    if delta_margin is not None:
        tail_delta = paired_risk.tail
        if not (math.isfinite(paired.lower) and paired.lower > -float(delta_margin)):
            noninferiority_failures.append("noninferiority:growth_margin")
        if not (math.isfinite(tail_delta.lower) and tail_delta.lower > 0.0):
            noninferiority_failures.append("noninferiority:tail_lower_bound")
        if protocol.champion.require_neighbors and changes:
            for spec_hash, risk in neighbor_risk_deltas:
                if not (math.isfinite(risk.growth.mean) and risk.growth.mean > -float(delta_margin)):
                    noninferiority_failures.append(f"noninferiority:neighbor_growth_margin:{spec_hash}")
                if not (math.isfinite(risk.tail.mean) and risk.tail.mean > 0.0):
                    noninferiority_failures.append(f"noninferiority:neighbor_tail_not_better:{spec_hash}")

    superiority_passed = (len(reasons) == 0 and len(superiority_failures) == 0)
    noninferiority_passed = (
        delta_margin is not None
        and len(reasons) == 0
        and len(noninferiority_failures) == 0
    )

    path: PromotionPath
    if superiority_passed:
        path = "superiority"
        all_reasons = tuple(reasons)
    elif noninferiority_passed:
        path = "noninferiority"
        all_reasons = tuple(reasons)
    else:
        path = "none"
        combined_reasons = list(reasons)
        combined_reasons.extend(superiority_failures)
        if delta_margin is not None:
            combined_reasons.extend(noninferiority_failures)
        all_reasons = tuple(combined_reasons)

    tail = paired_risk.tail if delta_margin is not None else None

    return ChallengeDecision(
        challenger_hash=challenger.report.spec_hash,
        champion_hash=champion.report.spec_hash,
        run_ids=(challenger.report.run_id, champion.report.run_id),
        window=(sessions[0], sessions[-1]),
        paired=paired,
        challenger_j=challenger_j,
        champion_j=champion_j,
        neighbors=neighbor_deltas,
        knob_changes=changes,
        reasons=all_reasons,
        alpha_effective=alpha_effective,
        paired_horizon_sessions=paired_horizon_sessions,
        path=path,
        tail=tail,
        noninferiority_margin=float(delta_margin) if delta_margin is not None else None,
        prior_decisions=prior_decisions,
    )


def knob_changes(challenger: StrategySpec, champion: StrategySpec) -> tuple[str, ...]:
    """Sorted dotted paths of the numeric leaves that differ between two specs (``policy.n``, ``hedge.…``).

    Bools, strings and tuples are not knobs: they have no numeric neighborhood, so ``scorer.*`` changes are
    reported but exempt from the neighbor rule, and a change of e.g. ``hedge.use_futures`` never demands a
    plateau probe.
    """
    left = _numeric_leaves_of(challenger)
    right = _numeric_leaves_of(champion)
    return tuple(sorted(path for path in set(left) | set(right) if left.get(path) != right.get(path)))


class ChampionStore:
    """Current champion and append-only promotion history under ``<state>/research/champion/``.

    ``current.json`` is replaced atomically (``os.replace``); ``history.jsonl`` and ``decisions/<digest>.json``
    are append-only. A promotion therefore writes in the order decision file → current → history row: a crash
    can only leave a decision without a promotion, never a champion without its audit trail.
    """

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._current_path = self._root / "current.json"
        self._history_path = self._root / "history.jsonl"
        self._decisions_dir = self._root / "decisions"

    def current(self) -> ChampionRecord | None:
        """The champion in force, or None when the store has never been bootstrapped."""
        if not self._current_path.exists():
            return None
        try:
            raw = json.loads(self._current_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise PITDataError(f"unreadable champion record: {self._current_path}") from exc
        if not isinstance(raw, dict):  # pragma: no cover - json object is written by this module
            raise PITDataError(f"invalid champion record: {self._current_path}")
        return _record_from_fields(raw)

    def history(self) -> tuple[ChampionRecord, ...]:
        """Every promotion in append order."""
        if not self._history_path.exists():
            return ()
        try:
            text = self._history_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise PITDataError(f"unreadable champion history: {self._history_path}") from exc
        out: list[ChampionRecord] = []
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except ValueError as exc:
                raise PITDataError(f"invalid champion history line: {self._history_path}") from exc
            if not isinstance(raw, dict):  # pragma: no cover - json object is written by this module
                raise PITDataError(f"invalid champion history line: {self._history_path}")
            out.append(_record_from_fields(raw))
        return tuple(out)

    def decisions(self) -> tuple[ChallengeDecision, ...]:
        """Every saved challenge decision, ordered by file name (digest)."""
        if not self._decisions_dir.is_dir():
            return ()
        out: list[ChallengeDecision] = []
        for path in sorted(self._decisions_dir.glob("*.json")):
            try:
                payload = path.read_text(encoding="utf-8")
            except OSError as exc:
                raise PITDataError(f"unreadable champion decision: {path}") from exc
            out.append(decision_from_canonical_json(payload))
        return tuple(out)

    def decision_path(self, digest: str) -> Path:
        return self._decisions_dir / f"{digest}.json"

    def save_decision(self, decision: ChallengeDecision) -> Path:
        """Write ``decisions/<digest>.json`` (idempotent) and return its path; never touches ``current.json``."""
        path = self.decision_path(decision.digest)
        if path.is_file():
            return path
        self._decisions_dir.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(decision.canonical_json() + "\n", encoding="utf-8")
        os.replace(temporary, path)
        return path

    def bootstrap(self, *, run: EvaluationRun, spec: StrategySpec, spec_path: Path, now: datetime) -> ChampionRecord:
        """First champion. Raises ValueError if a champion exists or the report did not pass."""
        if self.current() is not None:
            raise ValueError("a champion already exists; promote a challenger instead")
        if not run.report.passed:
            raise ValueError("report card did not pass; refusing to bootstrap a champion")
        record = self._record(
            run=run,
            spec=spec,
            spec_path=spec_path,
            now=now,
            reason="bootstrap",
            decision_digest=None,
        )
        self._replace_current(record)
        self._append_history(record)
        return record

    def promote(
        self,
        *,
        decision: ChallengeDecision,
        run: EvaluationRun,
        spec: StrategySpec,
        spec_path: Path,
        now: datetime,
    ) -> ChampionRecord:
        """Raises ValueError unless the decision is promotable, was saved, names the current champion and the
        given run is the decision's challenger run."""
        if not decision.promotable:
            raise ValueError(f"decision {decision.digest[:12]} is not promotable: {list(decision.reasons)}")
        self._verify_decision_context(decision, run=run, spec=spec)
        record = self._record(
            run=run,
            spec=spec,
            spec_path=spec_path,
            now=now,
            reason="challenge",
            decision_digest=decision.digest,
        )
        self._replace_current(record)
        self._append_history(record)
        return record

    def adopt_policy(
        self,
        *,
        decision: ChallengeDecision,
        run: EvaluationRun,
        spec: StrategySpec,
        spec_path: Path,
        now: datetime,
        rationale: str,
    ) -> ChampionRecord:
        """Operator decision to adopt a risk policy whose growth margin could not be proven statistically.

        Only the statistical growth tests may be waived (superiority and ``noninferiority:growth_margin``), and
        only when the saved decision shows the point estimate inside the margin, a tail-risk improvement with a
        positive lower bound, ``J`` not below the champion's and every neighbor's tail and growth in order.
        Why: the paired growth bound has no power to certify a few-percentage-point loss on one 9-year path, so a
        risk limit the data clearly favours would otherwise be unadoptable. The record carries the failed
        decision's digest and the rationale so the exception stays auditable.

        Raises ValueError for a blank rationale, a decision that is not saved or does not match the current
        champion/run/spec/window, or one whose failures include anything other than the waivable growth tests.
        """
        if not rationale.strip():
            raise ValueError("a policy override needs a non-empty rationale")
        if decision.promotable:
            raise ValueError(f"decision {decision.digest[:12]} is promotable; use promote")
        blocking = [
            reason
            for reason in decision.reasons
            if not (reason.startswith("superiority:") or reason == "noninferiority:growth_margin")
        ]
        if blocking:
            raise ValueError(f"decision {decision.digest[:12]} fails non-waivable rules: {blocking}")
        margin = decision.noninferiority_margin
        tail = decision.tail
        if margin is None or tail is None:
            raise ValueError("a policy override needs the non-inferiority evidence (margin and tail) in the decision")
        if not (math.isfinite(decision.paired.mean) and decision.paired.mean > -margin):
            raise ValueError("paired growth point estimate is outside the declared margin")
        if not (math.isfinite(tail.lower) and tail.lower > 0.0):
            raise ValueError("tail-risk improvement is not significant (lower bound <= 0)")
        self._verify_decision_context(decision, run=run, spec=spec)
        record = self._record(
            run=run,
            spec=spec,
            spec_path=spec_path,
            now=now,
            reason="policy_override",
            decision_digest=decision.digest,
            rationale=rationale.strip(),
        )
        self._replace_current(record)
        self._append_history(record)
        return record

    def _verify_decision_context(self, decision: ChallengeDecision, *, run: EvaluationRun, spec: StrategySpec) -> None:
        if not self.decision_path(decision.digest).is_file():
            raise ValueError(f"decision {decision.digest[:12]} was never saved")
        current = self.current()
        if current is None:
            raise ValueError("no champion to promote over")
        if current.spec_hash != decision.champion_hash:
            raise ValueError(
                f"decision names champion {decision.champion_hash[:12]} but the current champion is "
                f"{current.spec_hash[:12]}"
            )
        if run.report.spec_hash != decision.challenger_hash or run.report.run_id != decision.run_ids[0]:
            raise ValueError("run is not the decision's challenger run")
        if spec.spec_hash != decision.challenger_hash:
            raise ValueError("spec is not the decision's challenger spec")
        if _window_of(run) != decision.window:
            raise ValueError("decision was decided on a different window")

    @staticmethod
    def _record(
        *,
        run: EvaluationRun,
        spec: StrategySpec,
        spec_path: Path,
        now: datetime,
        reason: str,
        decision_digest: str | None,
        rationale: str | None = None,
    ) -> ChampionRecord:
        if run.report.spec_hash != spec.spec_hash:
            raise ValueError("run was not produced by this spec")
        return ChampionRecord(
            spec_hash=spec.spec_hash,
            spec_path=_repo_relative(spec_path),
            spec_json=spec.canonical_json(),
            run_id=run.report.run_id,
            report_digest=run.report.digest,
            objective_j=float(run.report.objective_j),
            promoted_at=now,
            reason=reason,
            decision_digest=decision_digest,
            rationale=rationale,
        )

    def _replace_current(self, record: ChampionRecord) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(champion_record_fields(record), sort_keys=True, separators=(",", ":")) + "\n"
        temporary = self._current_path.with_name(f".current.{os.getpid()}.tmp")
        with open(temporary, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self._current_path)

    def _append_history(self, record: ChampionRecord) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        line = json.dumps(champion_record_fields(record), sort_keys=True, separators=(",", ":")) + "\n"
        self._history_path.touch(exist_ok=True)
        with self._history_path.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                handle.write(line)
                handle.flush()
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


def champion_record_fields(record: ChampionRecord) -> dict[str, Any]:
    """JSON-safe payload of one champion record (the shape stored in ``current.json`` and ``history.jsonl``)."""
    fields: dict[str, Any] = {
        "decision_digest": record.decision_digest,
        "objective_j": _canon(record.objective_j),
        "promoted_at": record.promoted_at.isoformat(),
        "reason": record.reason,
        "report_digest": record.report_digest,
        "run_id": record.run_id,
        "spec_hash": record.spec_hash,
        "spec_json": record.spec_json,
        "spec_path": record.spec_path,
    }
    if record.rationale is not None:
        fields["rationale"] = record.rationale  # absent otherwise so pre-override records keep their bytes
    return fields


def decision_from_canonical_json(
    payload: str, *, alpha: float = 0.05, horizon_sessions: int = 1260,
) -> ChallengeDecision:
    """Rebuild a decision from its canonical JSON (the saved ``decisions/<digest>.json`` content)."""
    try:
        raw = json.loads(payload)
        alpha_effective = (
            _num(raw["alpha_effective"]) if "alpha_effective" in raw else alpha
        )
        paired_horizon_sessions = (
            int(raw["paired_horizon_sessions"])
            if "paired_horizon_sessions" in raw
            else horizon_sessions
        )
        reasons = tuple(str(item) for item in raw["reasons"])
        raw_path = str(raw["path"]) if "path" in raw else ("superiority" if len(reasons) == 0 else "none")
        if raw_path not in get_args(PromotionPath):
            raise ValueError(f"unknown promotion path {raw_path!r}")
        path = cast(PromotionPath, raw_path)
        tail = _delta_from_fields(raw["tail"]) if raw.get("tail") is not None else None
        noninferiority_margin = (
            _num(raw["noninferiority_margin"]) if raw.get("noninferiority_margin") is not None else None
        )
        return ChallengeDecision(
            challenger_hash=str(raw["challenger_hash"]),
            champion_hash=str(raw["champion_hash"]),
            run_ids=(str(raw["run_ids"][0]), str(raw["run_ids"][1])),
            window=(date.fromisoformat(str(raw["window"][0])), date.fromisoformat(str(raw["window"][1]))),
            paired=_delta_from_fields(raw["paired"]),
            challenger_j=_num(raw["challenger_j"]),
            champion_j=_num(raw["champion_j"]),
            neighbors=tuple((str(item[0]), _delta_from_fields(item[1])) for item in raw["neighbors"]),
            knob_changes=tuple(str(item) for item in raw["knob_changes"]),
            reasons=reasons,
            alpha_effective=alpha_effective,
            paired_horizon_sessions=paired_horizon_sessions,
            path=path,
            tail=tail,
            noninferiority_margin=noninferiority_margin,
            prior_decisions=raw.get("prior_decisions", 0),
        )
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid challenge decision payload: {exc}") from exc


def _numeric_leaves_of(spec: StrategySpec) -> dict[str, float]:
    sections = {
        "book": json.loads(spec.book.canonical_json()),
        "hedge": json.loads(spec.hedge.canonical_json()),
        "policy": json.loads(spec.policy.canonical_json()),
        "scorer": json.loads(spec.scorer.canonical_json()),
    }
    if spec.trend_overlay is not None:
        sections["trend_overlay"] = json.loads(spec.trend_overlay.canonical_json())
    if spec.regime_hedge is not None:
        sections["regime_hedge"] = json.loads(spec.regime_hedge.canonical_json())
    out: dict[str, float] = {}
    for name in sorted(sections):
        out.update(_numeric_leaves(sections[name], name))
    return out


def _numeric_leaves(node: Any, prefix: str) -> dict[str, float]:
    if isinstance(node, Mapping):
        out: dict[str, float] = {}
        for key in sorted(node):
            out.update(_numeric_leaves(node[key], f"{prefix}.{key}"))
        return out
    if isinstance(node, bool) or not isinstance(node, (int, float)):
        return {}
    return {prefix: float(node)}


def _knob_coverage(challenger: StrategySpec, neighbors: Sequence[EvaluationRun]) -> set[str]:
    """Knobs probed by at least one neighbor that differs from the challenger in that knob alone.

    A neighbor without a stored ``spec_json`` cannot be proven to be a single-knob variant, so it covers
    nothing and the knob stays uncovered.
    """
    covered: set[str] = set()
    for run in neighbors:
        if run.spec_json is None:
            continue
        try:
            neighbor = strategy_spec_from_canonical_json(run.spec_json)
        except ValueError:
            continue
        difference = knob_changes(challenger, neighbor)
        if len(difference) == 1:
            covered.add(difference[0])
    return covered


def _paired_risk_delta(
    challenger: EvaluationRun,
    champion: EvaluationRun,
    *,
    stream: str,
    policy: EvaluationPolicy,
    protocol: ResearchProtocol,
    alpha: float,
    horizon: int,
) -> PairedRiskDelta:
    return paired_growth_tail_delta(
        _stress_stream(challenger.evidence, stream),
        _stress_stream(champion.evidence, stream),
        block=int(policy.block_sessions),
        draws=int(policy.draws),
        seed=int(policy.seed),
        horizon=int(horizon),
        sessions_per_year=int(policy.sessions_per_year),
        alpha=float(alpha),
        tail_block_sessions=int(protocol.champion.tail_block_sessions),
        tail_quantile=float(protocol.champion.tail_quantile),
    )


def _stress_stream(evidence: EvaluationEvidence, stream: str) -> NDArray[np.float64]:
    """Log returns of one stress stream; ``stream`` is validated by the caller."""
    outcomes = {"stress_slippage": evidence.stress_slippage, "stress_delay": evidence.stress_delay}
    return np.asarray(outcomes[stream].log_returns, dtype=np.float64)


def _window_of(run: EvaluationRun) -> tuple[date, date]:
    sessions = tuple(run.evidence.sessions)
    if not sessions:
        raise ValueError("run evidence sessions must be non-empty")
    return (sessions[0], sessions[-1])


def _delta_fields(delta: PairedDelta) -> dict[str, Any]:
    return {
        "lower": _canon(delta.lower),
        "mean": _canon(delta.mean),
        "p_positive": _canon(delta.p_positive),
        "sessions": int(delta.sessions),
        "upper": _canon(delta.upper),
    }


def _delta_from_fields(raw: Any) -> PairedDelta:
    return PairedDelta(
        lower=_num(raw["lower"]),
        mean=_num(raw["mean"]),
        p_positive=_num(raw["p_positive"]),
        sessions=int(raw["sessions"]),
        upper=_num(raw["upper"]),
    )


def _record_from_fields(raw: Mapping[str, Any]) -> ChampionRecord:
    try:
        return ChampionRecord(
            spec_hash=str(raw["spec_hash"]),
            spec_path=str(raw["spec_path"]),
            spec_json=str(raw["spec_json"]),
            run_id=str(raw["run_id"]),
            report_digest=str(raw["report_digest"]),
            objective_j=_num(raw["objective_j"]),
            promoted_at=datetime.fromisoformat(str(raw["promoted_at"])),
            reason=str(raw["reason"]),
            decision_digest=None if raw["decision_digest"] is None else str(raw["decision_digest"]),
            rationale=None if raw.get("rationale") is None else str(raw["rationale"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PITDataError(f"invalid champion record: {exc}") from exc


def _repo_relative(spec_path: Path) -> str:
    """Repo-relative POSIX path when the file lives under the working directory, else the given path."""
    path = Path(spec_path)
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except (OSError, ValueError):
        return path.as_posix()


def _canon(value: float) -> Any:
    amount = float(value)
    if math.isnan(amount):
        return "nan"
    if math.isinf(amount):
        return "inf" if amount > 0 else "-inf"
    return amount


def _num(value: Any) -> float:
    """Canonical JSON round-trip: non-finite floats are stored as the strings ``nan``/``inf``/``-inf``."""
    return float(value)
